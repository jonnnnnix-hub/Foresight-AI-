"""OPRA options minute bar downloader via Massive S3 flat files.

Downloads gzip-compressed CSV files from S3:
  us_options_opra/minute_aggs_v1/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz

Each file contains ALL option contracts for ALL underlyings (~2.9M rows/day).
We filter at download time to keep only:
  - Target underlyings (PLTR, NVDA, QQQ, etc.)
  - Calls only (scalp model is long-call)
  - 0-2 DTE (maximum gamma)
  - Strikes within 5% of underlying price (near-ATM)

Filtered bars are cached per underlying per day as JSON.

Requires: boto3
Credentials: MASSIVE_ACCESS_KEY, MASSIVE_SECRET_KEY in .env
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from flowedge.scanner.data_feeds.massive_s3 import (
    OPTIONS_MINUTE_PREFIX,
    S3_BUCKET,
    S3_ENDPOINT,
)

logger = structlog.get_logger()

CACHE_DIR = Path("data/flat_files_s3")

# OCC symbol format: O:PLTR250412C00085000
# After stripping "O:" prefix: PLTR250412C00085000
_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def parse_occ_symbol(
    raw_symbol: str,
) -> tuple[str, date, str, float] | None:
    """Parse an OCC option symbol into components.

    Args:
        raw_symbol: OCC symbol, optionally prefixed with ``O:``.

    Returns:
        Tuple of (underlying, expiration, 'C'|'P', strike) or None if
        the symbol cannot be parsed.
    """
    symbol = raw_symbol.removeprefix("O:")
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    underlying = m.group(1)
    exp = datetime.strptime(m.group(2), "%y%m%d").date()
    cp = m.group(3)
    strike = int(m.group(4)) / 1000.0
    return underlying, exp, cp, strike


MAX_RETRIES = 3


def _load_env(env_path: str | Path | None = None) -> None:
    """Load .env file into os.environ if not already loaded."""
    if os.getenv("MASSIVE_ACCESS_KEY"):
        return
    # Walk up from this file to find .env (handles worktrees)
    here = Path(__file__).resolve()
    parent_envs = [here.parents[i] / ".env" for i in range(4, 10)]
    candidates = [
        Path(env_path) if env_path else None,
        Path(".env"),
        *parent_envs,
    ]
    for p in candidates:
        if p and p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())
            logger.debug("env_loaded", path=str(p))
            return


class OptionsS3Downloader:
    """Download and cache OPRA options minute bars from Massive S3.

    Parallel to :class:`MassiveS3Downloader` for equities, but filters
    aggressively so only near-ATM, short-DTE options for target
    underlyings are retained.
    """

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
        env_path: str | Path | None = None,
    ) -> None:
        _load_env(env_path)
        self._access_key = access_key or os.getenv("MASSIVE_ACCESS_KEY", "")
        self._secret_key = secret_key or os.getenv("MASSIVE_SECRET_KEY", "")
        self._endpoint = os.getenv("MASSIVE_ENDPOINT", S3_ENDPOINT)
        self._bucket = os.getenv("MASSIVE_BUCKET", S3_BUCKET)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3  # type: ignore[import-not-found]
            from botocore.config import Config  # type: ignore[import-not-found]

            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                config=Config(signature_version="s3v4"),
            )
        return self._client

    # ── Download Single Day ────────────────────────────────────

    def download_options_day(
        self,
        target_date: date,
        underlying_tickers: list[str],
        underlying_prices: dict[str, float],
        max_dte: int = 5,
        strike_range_pct: float = 0.05,
        calls_only: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Download one day of OPRA data, filtered to near-ATM contracts.

        Args:
            target_date: Trading date to download.
            underlying_tickers: Underlyings to keep (e.g. ``["PLTR", "NVDA"]``).
            underlying_prices: Mapping of underlying ticker to closing price
                for strike-range filtering.
            max_dte: Maximum days to expiration to retain.
            strike_range_pct: Keep strikes within this fraction of underlying
                price (0.05 = 5%).
            calls_only: If True, discard puts.

        Returns:
            Dict mapping underlying ticker to list of option bar dicts.
        """
        import time as _time

        s3 = self._get_client()
        key = (
            f"{OPTIONS_MINUTE_PREFIX}"
            f"/{target_date.year}/{target_date.month:02d}"
            f"/{target_date.isoformat()}.csv.gz"
        )

        raw: bytes | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = s3.get_object(Bucket=self._bucket, Key=key)
                raw = response["Body"].read()
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "options_s3_retry",
                        date=target_date.isoformat(),
                        attempt=attempt + 1,
                        wait=wait,
                        error=str(e)[:80],
                    )
                    _time.sleep(wait)
                else:
                    logger.debug(
                        "options_s3_day_not_found",
                        date=target_date.isoformat(),
                        key=key,
                        error=str(e)[:80],
                    )
                    return {}

        if raw is None:
            return {}

        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            text = gz.read().decode("utf-8")

        ticker_set = set(underlying_tickers)
        bars_by_underlying: dict[str, list[dict[str, Any]]] = defaultdict(list)
        kept = 0
        skipped = 0

        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            raw_sym = row.get("ticker", "")
            parsed = parse_occ_symbol(raw_sym)
            if parsed is None:
                skipped += 1
                continue

            underlying, exp, cp, strike = parsed

            # Filter: target underlyings only
            if underlying not in ticker_set:
                skipped += 1
                continue

            # Filter: calls only
            if calls_only and cp != "C":
                skipped += 1
                continue

            # Filter: DTE
            dte = (exp - target_date).days
            if dte < 0 or dte > max_dte:
                skipped += 1
                continue

            # Filter: near-ATM
            ref_price = underlying_prices.get(underlying, 0)
            if ref_price > 0:
                dist = abs(strike - ref_price) / ref_price
                if dist > strike_range_pct:
                    skipped += 1
                    continue

            ts_ns = int(row.get("window_start", 0))
            ts = datetime.fromtimestamp(ts_ns / 1_000_000_000)

            bars_by_underlying[underlying].append({
                "contract": raw_sym,
                "underlying": underlying,
                "expiration": exp.isoformat(),
                "strike": strike,
                "option_type": cp,
                "dte": dte,
                "ts": str(ts_ns),
                "timestamp": ts.isoformat(),
                "date": target_date.isoformat(),
                "o": float(row.get("open", 0)),
                "h": float(row.get("high", 0)),
                "l": float(row.get("low", 0)),
                "c": float(row.get("close", 0)),
                "v": int(float(row.get("volume", 0))),
                "vw": float(row.get("vwap", 0)),
                "n": int(float(row.get("transactions", 0))),
            })
            kept += 1

        logger.info(
            "options_s3_day_downloaded",
            date=target_date.isoformat(),
            kept=kept,
            skipped=skipped,
            underlyings=list(bars_by_underlying.keys()),
        )
        return dict(bars_by_underlying)

    # ── Download Date Range ────────────────────────────────────

    def download_options_range(
        self,
        from_date: date,
        to_date: date,
        underlying_tickers: list[str],
        underlying_prices_by_date: dict[str, dict[str, float]] | None = None,
        max_dte: int = 5,
        strike_range_pct: float = 0.05,
    ) -> dict[str, int]:
        """Download a date range, caching each day to disk.

        Args:
            from_date: Start date (inclusive).
            to_date: End date (inclusive).
            underlying_tickers: Tickers to filter for.
            underlying_prices_by_date: Optional pre-computed prices.
                If None, reads from stock bar cache.
            max_dte: Max DTE filter.
            strike_range_pct: Strike distance filter.

        Returns:
            Dict mapping underlying to total cached bar count.
        """
        if underlying_prices_by_date is None:
            underlying_prices_by_date = get_underlying_closes_from_cache(
                underlying_tickers,
            )

        totals: dict[str, int] = {t: 0 for t in underlying_tickers}
        current = from_date
        days_processed = 0

        while current <= to_date:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            date_str = current.isoformat()

            # Skip if already cached for all tickers
            all_cached = all(
                self._cache_path(t, date_str).exists()
                for t in underlying_tickers
            )
            if all_cached:
                current += timedelta(days=1)
                days_processed += 1
                continue

            prices = underlying_prices_by_date.get(date_str, {})
            if not prices:
                current += timedelta(days=1)
                continue

            day_bars = self.download_options_day(
                current,
                underlying_tickers,
                prices,
                max_dte=max_dte,
                strike_range_pct=strike_range_pct,
            )

            for ticker, bars in day_bars.items():
                self.save_options_to_cache(ticker, date_str, bars)
                totals[ticker] = totals.get(ticker, 0) + len(bars)

            days_processed += 1
            if days_processed % 20 == 0:
                logger.info(
                    "options_download_progress",
                    days=days_processed,
                    through=date_str,
                    totals={k: v for k, v in totals.items() if v > 0},
                )

            current += timedelta(days=1)

        logger.info(
            "options_download_complete",
            days=days_processed,
            totals=totals,
        )
        return totals

    # ── Cache ──────────────────────────────────────────────────

    def _cache_path(self, underlying: str, date_str: str) -> Path:
        return (
            CACHE_DIR
            / underlying
            / "options_1min"
            / f"{underlying}_options_1min_{date_str}.json"
        )

    def save_options_to_cache(
        self,
        underlying: str,
        date_str: str,
        bars: list[dict[str, Any]],
    ) -> Path:
        """Write filtered option bars to local JSON cache."""
        path = self._cache_path(underlying, date_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bars, default=str))
        logger.debug(
            "options_cache_saved",
            underlying=underlying,
            date=date_str,
            bars=len(bars),
            path=str(path),
        )
        return path

    def load_options_from_cache(
        self,
        underlying: str,
        date_str: str,
    ) -> list[dict[str, Any]]:
        """Load cached option bars for one underlying on one date."""
        path = self._cache_path(underlying, date_str)
        if not path.exists():
            return []
        data: list[dict[str, Any]] = json.loads(path.read_text())
        return data


def get_underlying_closes_from_cache(
    tickers: list[str],
) -> dict[str, dict[str, float]]:
    """Build date→{ticker: close} mapping from cached stock bars.

    Reads the existing 1-min stock bar JSON files and extracts the
    last bar's close price for each trading day.

    Returns:
        Dict mapping date string to {ticker: closing_price}.
    """
    prices: dict[str, dict[str, float]] = defaultdict(dict)

    for ticker in tickers:
        min_dir = CACHE_DIR / ticker / "1min"
        if not min_dir.exists():
            logger.warning("no_stock_cache", ticker=ticker)
            continue

        for f in sorted(min_dir.glob("*.json")):
            bars: list[dict[str, Any]] = json.loads(f.read_text())
            by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for b in bars:
                d = str(b.get("d", b.get("date", "")))
                if d:
                    by_date[d].append(b)

            for d, day_bars in by_date.items():
                if day_bars:
                    last_close = float(
                        day_bars[-1].get("c", day_bars[-1].get("close", 0))
                    )
                    if last_close > 0:
                        prices[d][ticker] = last_close

    logger.info(
        "underlying_closes_loaded",
        tickers=len(tickers),
        dates=len(prices),
    )
    return dict(prices)
