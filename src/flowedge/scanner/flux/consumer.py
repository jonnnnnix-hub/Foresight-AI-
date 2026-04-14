"""Polygon REST consumer for trade ticks and NBBO quotes.

Fetches individual trades and quotes from Polygon's v3 endpoints
for Lee-Ready classification and order flow analysis.

Polygon endpoints used:
- GET /v3/trades/{ticker} — individual trades (price, size, conditions)
- GET /v3/quotes/{ticker} — NBBO quotes (bid, ask, bid_size, ask_size)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from flowedge.scanner.flux.schemas import NBBOQuote, TradeTick

logger = structlog.get_logger()

# Polygon trade conditions to exclude (odd lots, out-of-sequence, etc.)
# See: https://polygon.io/glossary/us/stocks/conditions-indicators
_EXCLUDE_CONDITIONS = frozenset({
    2,   # Average Price Trade
    7,   # Qualified Contingent Trade
    10,  # Cross Trade
    15,  # Opening/Reopening Trade Detail
    16,  # Closing Trade Detail
    22,  # Corrected Consolidated Close
    29,  # Derivatively Priced
    33,  # Market Center Official Open
    38,  # Prior Reference Price
    52,  # Contingent Trade
    53,  # Qualified Contingent Trade
})


class PolygonTradeConsumer:
    """Fetches trade ticks and NBBO quotes from Polygon REST API.

    Designed for the 5-min scanner cadence: fetches windowed data
    on demand rather than maintaining a persistent WebSocket.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.polygon.io",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._session: Any = None

    async def _ensure_client(self) -> Any:
        if self._session is None:
            import httpx
            self._session = httpx.AsyncClient(timeout=30.0)
        return self._session

    async def close(self) -> None:
        if self._session:
            await self._session.aclose()
            self._session = None

    async def _get(
        self, url: str, params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        client = await self._ensure_client()
        params = params or {}
        params["apiKey"] = self._api_key
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    # ── Trades ────────────────────────────────────────────────────

    async def get_trades(
        self,
        ticker: str,
        window_minutes: int = 5,
        limit: int = 5000,
    ) -> list[TradeTick]:
        """Fetch individual trades for the last N minutes.

        Uses Polygon /v3/trades endpoint with timestamp filtering.
        Filters out odd-lot and non-standard trade conditions.

        Args:
            ticker: e.g. "SPY"
            window_minutes: How far back to look (default 5 min)
            limit: Max trades to return (Polygon max is 50000)

        Returns:
            List of TradeTick sorted by timestamp ascending.
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=window_minutes)

        # Polygon uses RFC3339 nanosecond timestamps
        start_ns = str(int(start.timestamp() * 1_000_000_000))
        end_ns = str(int(now.timestamp() * 1_000_000_000))

        data = await self._get(
            f"{self._base_url}/v3/trades/{ticker}",
            params={
                "timestamp.gte": start_ns,
                "timestamp.lte": end_ns,
                "limit": str(min(limit, 50000)),
                "sort": "timestamp",
                "order": "asc",
            },
        )

        ticks: list[TradeTick] = []
        for r in data.get("results", []):
            conditions = r.get("conditions", []) or []
            # Skip non-standard trades
            if any(c in _EXCLUDE_CONDITIONS for c in conditions):
                continue

            size = int(r.get("size", 0))
            if size <= 0:
                continue

            ticks.append(TradeTick(
                price=float(r.get("price", 0)),
                size=size,
                timestamp=int(r.get("sip_timestamp", r.get("participant_timestamp", 0))),
                conditions=conditions,
                exchange=int(r.get("exchange", 0)),
            ))

        logger.debug(
            "trades_fetched",
            ticker=ticker,
            window_min=window_minutes,
            raw=len(data.get("results", [])),
            filtered=len(ticks),
        )
        return ticks

    async def get_session_trades(
        self,
        ticker: str,
        limit: int = 50000,
    ) -> list[TradeTick]:
        """Fetch all trades since market open (9:30 ET) today.

        Used for session-level cumulative delta calculation.
        Caution: can be large for liquid tickers — use sparingly.
        """
        now = datetime.now(timezone.utc)
        # Approximate 9:30 ET = 13:30 UTC (EDT) or 14:30 UTC (EST)
        market_open = now.replace(hour=13, minute=30, second=0, microsecond=0)
        if now < market_open:
            # Before market open, use yesterday (shouldn't happen in scanner)
            market_open -= timedelta(days=1)

        start_ns = str(int(market_open.timestamp() * 1_000_000_000))
        end_ns = str(int(now.timestamp() * 1_000_000_000))

        data = await self._get(
            f"{self._base_url}/v3/trades/{ticker}",
            params={
                "timestamp.gte": start_ns,
                "timestamp.lte": end_ns,
                "limit": str(min(limit, 50000)),
                "sort": "timestamp",
                "order": "asc",
            },
        )

        ticks: list[TradeTick] = []
        for r in data.get("results", []):
            conditions = r.get("conditions", []) or []
            if any(c in _EXCLUDE_CONDITIONS for c in conditions):
                continue
            size = int(r.get("size", 0))
            if size <= 0:
                continue
            ticks.append(TradeTick(
                price=float(r.get("price", 0)),
                size=size,
                timestamp=int(r.get("sip_timestamp", r.get("participant_timestamp", 0))),
                conditions=conditions,
                exchange=int(r.get("exchange", 0)),
            ))

        logger.debug(
            "session_trades_fetched",
            ticker=ticker,
            count=len(ticks),
        )
        return ticks

    # ── Quotes (NBBO) ────────────────────────────────────────────

    async def get_quotes(
        self,
        ticker: str,
        window_minutes: int = 5,
        limit: int = 5000,
    ) -> list[NBBOQuote]:
        """Fetch NBBO quotes for the last N minutes.

        Uses Polygon /v3/quotes endpoint. Returns bid, ask, sizes,
        timestamps for Lee-Ready midpoint calculation.

        Args:
            ticker: e.g. "SPY"
            window_minutes: How far back to look
            limit: Max quotes to return

        Returns:
            List of NBBOQuote sorted by timestamp ascending.
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=window_minutes)

        start_ns = str(int(start.timestamp() * 1_000_000_000))
        end_ns = str(int(now.timestamp() * 1_000_000_000))

        data = await self._get(
            f"{self._base_url}/v3/quotes/{ticker}",
            params={
                "timestamp.gte": start_ns,
                "timestamp.lte": end_ns,
                "limit": str(min(limit, 50000)),
                "sort": "timestamp",
                "order": "asc",
            },
        )

        quotes: list[NBBOQuote] = []
        for r in data.get("results", []):
            bid = float(r.get("bid_price", 0))
            ask = float(r.get("ask_price", 0))
            if bid <= 0 or ask <= 0 or ask < bid:
                continue

            quotes.append(NBBOQuote(
                bid=bid,
                bid_size=int(r.get("bid_size", 0)),
                ask=ask,
                ask_size=int(r.get("ask_size", 0)),
                timestamp=int(r.get("sip_timestamp", r.get("participant_timestamp", 0))),
            ))

        logger.debug(
            "quotes_fetched",
            ticker=ticker,
            window_min=window_minutes,
            count=len(quotes),
        )
        return quotes
