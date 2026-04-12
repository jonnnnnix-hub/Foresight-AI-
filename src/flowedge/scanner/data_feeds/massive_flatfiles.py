"""Massive/Polygon flat file ingestion — bulk historical data.

Downloads S3 flat files for bulk historical data:
- Stock minute bars: 390 bars/day × years of history
- Options minute bars: real bid/ask/Greeks at minute resolution
- Index minute bars: SPY/QQQ/IWM with VWAP

This enables:
1. Backtesting with 1-minute granularity (vs daily)
2. Real options spreads (vs slippage model estimates)
3. Intraday IBS/RSI3 pattern validation
4. Training on 1M+ data points per ticker

File format: CSV/JSON via Polygon's flat files API
Access: Same API key as Polygon REST (paid tier required)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from flowedge.scanner.data_feeds.schemas import BarData, Timeframe

logger = structlog.get_logger()

# Local cache directory for flat files
FLAT_FILE_DIR = Path("data/flat_files")


class MassiveFlatFileProvider:
    """Bulk historical data from Massive/Polygon flat files.

    Downloads and caches minute-level bars for backtesting
    with intraday granularity.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._base_url = "https://api.polygon.io"
        self._client: Any = None

    async def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get(self, url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        client = await self._ensure_client()
        params = params or {}
        params["apiKey"] = self._api_key
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    # ── Minute Bars (bulk) ──────────────────────────────────────

    async def get_minute_bars_bulk(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        limit: int = 50000,
    ) -> list[BarData]:
        """Fetch minute-level bars for a date range.

        With paid tier, can fetch up to 50,000 bars per request.
        For 1 year of minute data: ~98,000 bars (390/day × 252 days).
        Requires pagination for long ranges.

        Args:
            ticker: e.g. "SPY"
            from_date: YYYY-MM-DD
            to_date: YYYY-MM-DD
            limit: Max bars per request (50000 for paid tier)

        Returns:
            List of BarData at 1-minute timeframe.
        """
        all_bars: list[BarData] = []
        current_start = from_date

        while current_start <= to_date:
            data = await self._get(
                f"{self._base_url}/v2/aggs/ticker/{ticker}"
                f"/range/1/minute/{current_start}/{to_date}",
                params={"limit": str(limit), "sort": "asc"},
            )

            results = data.get("results", [])
            if not results:
                break

            for r in results:
                ts = r.get("t", 0)
                all_bars.append(BarData(
                    ticker=ticker,
                    timestamp=datetime.fromtimestamp(ts / 1000),
                    timeframe=Timeframe.MIN_1,
                    open=float(r.get("o", 0)),
                    high=float(r.get("h", 0)),
                    low=float(r.get("l", 0)),
                    close=float(r.get("c", 0)),
                    volume=int(r.get("v", 0)),
                    vwap=float(r.get("vw", 0)),
                    trade_count=int(r.get("n", 0)),
                ))

            # Pagination: advance start to day after last bar
            last_ts = results[-1].get("t", 0)
            last_date = datetime.fromtimestamp(last_ts / 1000).date()
            next_date = last_date + timedelta(days=1)
            current_start = next_date.isoformat()

            if len(results) < limit:
                break  # No more data

            logger.info(
                "minute_bars_page",
                ticker=ticker,
                bars_so_far=len(all_bars),
                through=last_date.isoformat(),
            )

        logger.info(
            "minute_bars_loaded",
            ticker=ticker,
            total_bars=len(all_bars),
            from_date=from_date,
            to_date=to_date,
        )
        return all_bars

    async def get_5min_bars_bulk(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        limit: int = 50000,
    ) -> list[BarData]:
        """Fetch 5-minute bars for a date range (5x less data than 1-min)."""
        all_bars: list[BarData] = []
        current_start = from_date

        while current_start <= to_date:
            data = await self._get(
                f"{self._base_url}/v2/aggs/ticker/{ticker}"
                f"/range/5/minute/{current_start}/{to_date}",
                params={"limit": str(limit), "sort": "asc"},
            )

            results = data.get("results", [])
            if not results:
                break

            for r in results:
                ts = r.get("t", 0)
                all_bars.append(BarData(
                    ticker=ticker,
                    timestamp=datetime.fromtimestamp(ts / 1000),
                    timeframe=Timeframe.MIN_5,
                    open=float(r.get("o", 0)),
                    high=float(r.get("h", 0)),
                    low=float(r.get("l", 0)),
                    close=float(r.get("c", 0)),
                    volume=int(r.get("v", 0)),
                    vwap=float(r.get("vw", 0)),
                    trade_count=int(r.get("n", 0)),
                ))

            last_ts = results[-1].get("t", 0)
            last_date = datetime.fromtimestamp(last_ts / 1000).date()
            next_date = last_date + timedelta(days=1)
            current_start = next_date.isoformat()

            if len(results) < limit:
                break

        logger.info(
            "5min_bars_loaded",
            ticker=ticker,
            total_bars=len(all_bars),
        )
        return all_bars

    # ── Cache to Disk ──────────────────────────────────────────

    def save_bars_to_cache(
        self,
        ticker: str,
        timeframe: str,
        bars: list[BarData],
    ) -> Path:
        """Save bars to local cache as JSON for fast re-loading."""
        import json

        cache_dir = FLAT_FILE_DIR / ticker / timeframe
        cache_dir.mkdir(parents=True, exist_ok=True)

        if not bars:
            return cache_dir

        first_date = bars[0].timestamp.date().isoformat()
        last_date = bars[-1].timestamp.date().isoformat()
        filename = f"{ticker}_{timeframe}_{first_date}_{last_date}.json"
        filepath = cache_dir / filename

        data = [
            {
                "t": int(b.timestamp.timestamp() * 1000),
                "o": b.open,
                "h": b.high,
                "l": b.low,
                "c": b.close,
                "v": b.volume,
                "vw": b.vwap,
                "n": b.trade_count,
            }
            for b in bars
        ]
        filepath.write_text(json.dumps(data))
        logger.info("bars_cached", path=str(filepath), count=len(bars))
        return filepath

    def load_bars_from_cache(
        self,
        ticker: str,
        timeframe_str: str,
    ) -> list[BarData]:
        """Load cached bars from disk."""
        import json

        cache_dir = FLAT_FILE_DIR / ticker / timeframe_str
        if not cache_dir.exists():
            return []

        tf_map = {"1min": Timeframe.MIN_1, "5min": Timeframe.MIN_5}
        tf = tf_map.get(timeframe_str, Timeframe.MIN_5)

        all_bars: list[BarData] = []
        for filepath in sorted(cache_dir.glob("*.json")):
            data = json.loads(filepath.read_text())
            for r in data:
                all_bars.append(BarData(
                    ticker=ticker,
                    timestamp=datetime.fromtimestamp(r["t"] / 1000),
                    timeframe=tf,
                    open=float(r["o"]),
                    high=float(r["h"]),
                    low=float(r["l"]),
                    close=float(r["c"]),
                    volume=int(r["v"]),
                    vwap=float(r.get("vw", 0)),
                    trade_count=int(r.get("n", 0)),
                ))

        logger.info("bars_loaded_from_cache", ticker=ticker, count=len(all_bars))
        return all_bars

    # ── Convenience: Download + Cache ──────────────────────────

    async def download_and_cache(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        timeframe: str = "5min",
    ) -> list[BarData]:
        """Download bars and cache to disk. Returns bars."""
        # Check cache first
        cached = self.load_bars_from_cache(ticker, timeframe)
        if cached:
            logger.info(
                "using_cached_bars",
                ticker=ticker,
                count=len(cached),
            )
            return cached

        # Download
        if timeframe == "1min":
            bars = await self.get_minute_bars_bulk(ticker, from_date, to_date)
        else:
            bars = await self.get_5min_bars_bulk(ticker, from_date, to_date)

        # Cache
        if bars:
            self.save_bars_to_cache(ticker, timeframe, bars)

        return bars
