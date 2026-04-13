"""Massive S3 flat file downloader — NO rate limits.

Downloads gzip-compressed CSV files directly from S3.
Path pattern: us_stocks_sip/minute_aggs_v1/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz

Each file contains ALL tickers for that day. We filter to our tickers.
No API rate limits — download as fast as your connection allows.

Requires: boto3, pandas
Credentials: MASSIVE_ACCESS_KEY, MASSIVE_SECRET_KEY in .env

Data available:
- us_stocks_sip/minute_aggs_v1/ — 1-min stock bars
- us_options_opra/minute_aggs_v1/ — 1-min options bars
- us_indices_sip/minute_aggs_v1/ — 1-min index bars
"""

from __future__ import annotations

import gzip
import io
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

S3_ENDPOINT = "https://files.massive.com"
S3_BUCKET = "flatfiles"

# S3 prefixes for different data types
STOCKS_MINUTE_PREFIX = "us_stocks_sip/minute_aggs_v1"
OPTIONS_MINUTE_PREFIX = "us_options_opra/minute_aggs_v1"
INDICES_MINUTE_PREFIX = "us_indices_sip/minute_aggs_v1"

CACHE_DIR = Path("data/flat_files_s3")


class MassiveS3Downloader:
    """Download bulk historical data from Massive S3 flat files.

    No rate limits. Each daily file contains all tickers — we filter
    to the ones we need and cache locally.
    """

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._access_key = access_key or os.getenv("MASSIVE_ACCESS_KEY", "")
        self._secret_key = secret_key or os.getenv("MASSIVE_SECRET_KEY", "")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config
            self._client = boto3.client(
                "s3",
                endpoint_url=S3_ENDPOINT,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                config=Config(signature_version="s3v4"),
            )
        return self._client

    def _s3_key_for_date(self, d: date, prefix: str) -> str:
        """Build S3 key path for a given date."""
        return f"{prefix}/{d.year}/{d.month:02d}/{d.isoformat()}.csv.gz"

    # ── Download Single Day ──────────────────────────────────────

    def download_day(
        self,
        target_date: date,
        tickers: list[str],
        prefix: str = STOCKS_MINUTE_PREFIX,
    ) -> list[dict[str, Any]]:
        """Download and filter one day of minute bars.

        Args:
            target_date: The date to download.
            tickers: List of tickers to filter for.
            prefix: S3 prefix (stocks, options, or indices).

        Returns:
            List of bar dicts with keys: ticker, timestamp, o, h, l, c, v, vw, n
        """
        s3 = self._get_client()
        key = self._s3_key_for_date(target_date, prefix)

        try:
            response = s3.get_object(Bucket=S3_BUCKET, Key=key)
        except Exception as e:
            logger.debug(
                "s3_day_not_found",
                date=target_date.isoformat(),
                key=key,
                error=str(e)[:80],
            )
            return []

        # Decompress gzip and parse CSV
        import csv

        raw = response["Body"].read()
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            text = gz.read().decode("utf-8")

        ticker_set = set(tickers)
        bars: list[dict[str, Any]] = []

        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            if row.get("ticker") not in ticker_set:
                continue

            # Parse timestamp from nanoseconds
            ts_ns = int(row.get("window_start", 0))
            ts = datetime.fromtimestamp(ts_ns / 1_000_000_000)

            bars.append({
                "ticker": row["ticker"],
                "timestamp": ts.isoformat(),
                "date": target_date.isoformat(),
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": int(float(row.get("volume", 0))),
                "vwap": float(row.get("vwap", 0)),
                "transactions": int(float(row.get("transactions", 0))),
            })

        logger.info(
            "s3_day_downloaded",
            date=target_date.isoformat(),
            bars=len(bars),
            tickers=len(ticker_set),
        )
        return bars

    # ── Download Date Range ──────────────────────────────────────

    def download_range(
        self,
        from_date: date,
        to_date: date,
        tickers: list[str],
        prefix: str = STOCKS_MINUTE_PREFIX,
    ) -> dict[str, list[dict[str, Any]]]:
        """Download a date range and return bars grouped by ticker.

        Returns:
            Dict mapping ticker → list of bar dicts, sorted by timestamp.
        """
        all_bars: dict[str, list[dict[str, Any]]] = {t: [] for t in tickers}
        current = from_date
        total_bars = 0
        days_processed = 0

        while current <= to_date:
            # Skip weekends
            if current.weekday() < 5:
                day_bars = self.download_day(current, tickers, prefix)
                for bar in day_bars:
                    tk = bar["ticker"]
                    if tk in all_bars:
                        all_bars[tk].append(bar)
                total_bars += len(day_bars)
                days_processed += 1

                if days_processed % 20 == 0:
                    logger.info(
                        "s3_download_progress",
                        days=days_processed,
                        total_bars=total_bars,
                        through=current.isoformat(),
                    )

            current += timedelta(days=1)

        logger.info(
            "s3_download_complete",
            days=days_processed,
            total_bars=total_bars,
            tickers={t: len(b) for t, b in all_bars.items() if b},
        )
        return all_bars

    # ── Cache to Disk ──────────────────────────────────────────

    def save_to_cache(
        self,
        bars_by_ticker: dict[str, list[dict[str, Any]]],
        timeframe: str = "1min",
    ) -> list[Path]:
        """Save downloaded bars to local JSON cache."""
        paths: list[Path] = []
        for ticker, bars in bars_by_ticker.items():
            if not bars:
                continue
            cache_dir = CACHE_DIR / ticker / timeframe
            cache_dir.mkdir(parents=True, exist_ok=True)

            first = bars[0].get("date", "unknown")
            last = bars[-1].get("date", "unknown")
            filepath = cache_dir / f"{ticker}_{timeframe}_{first}_{last}.json"
            filepath.write_text(json.dumps(bars, default=str))
            paths.append(filepath)
            logger.info(
                "s3_cache_saved",
                ticker=ticker,
                bars=len(bars),
                path=str(filepath),
            )
        return paths

    def load_from_cache(
        self,
        ticker: str,
        timeframe: str = "1min",
    ) -> list[dict[str, Any]]:
        """Load cached bars from disk, deduplicated and sorted.

        Multiple cache files may overlap in date range.  We deduplicate
        by (date, timestamp) and sort chronologically so downstream code
        (binary search, 5-min chunk building) gets clean data.
        """
        cache_dir = CACHE_DIR / ticker / timeframe
        if not cache_dir.exists():
            return []

        all_bars: list[dict[str, Any]] = []
        for filepath in sorted(cache_dir.glob("*.json")):
            data = json.loads(filepath.read_text())
            all_bars.extend(data)

        # Sort by nanosecond timestamp (handles mixed ts/timestamp keys)
        def _ts_key(b: dict[str, Any]) -> int:
            raw = b.get("ts", b.get("timestamp", "0"))
            try:
                return int(raw)
            except (ValueError, TypeError):
                return 0

        all_bars.sort(key=_ts_key)

        # Deduplicate: same (date, ts) = same bar from overlapping files
        seen: set[tuple[str, int]] = set()
        deduped: list[dict[str, Any]] = []
        for b in all_bars:
            key = (str(b.get("date", b.get("d", ""))), _ts_key(b))
            if key not in seen:
                seen.add(key)
                deduped.append(b)

        if len(deduped) < len(all_bars):
            logger.info(
                "s3_cache_deduped",
                ticker=ticker,
                before=len(all_bars),
                after=len(deduped),
            )

        logger.info("s3_cache_loaded", ticker=ticker, bars=len(deduped))
        return deduped

    # ── Convenience ──────────────────────────────────────────────

    def download_and_cache(
        self,
        from_date: date,
        to_date: date,
        tickers: list[str],
        prefix: str = STOCKS_MINUTE_PREFIX,
        timeframe: str = "1min",
    ) -> dict[str, list[dict[str, Any]]]:
        """Download, cache, and return bars. Uses cache if available."""
        # Check cache first
        cached: dict[str, list[dict[str, Any]]] = {}
        all_cached = True
        for ticker in tickers:
            bars = self.load_from_cache(ticker, timeframe)
            if bars:
                cached[ticker] = bars
            else:
                all_cached = False

        if all_cached and cached:
            logger.info("using_s3_cache", tickers=len(cached))
            return cached

        # Download fresh
        bars_by_ticker = self.download_range(from_date, to_date, tickers, prefix)
        self.save_to_cache(bars_by_ticker, timeframe)
        return bars_by_ticker
