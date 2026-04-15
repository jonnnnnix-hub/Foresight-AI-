"""WebSocket-backed bar provider — drop-in replacement for PolygonIntradayProvider.

Reads bars from the shared MassiveDataFeed WebSocket buffer instead of
making REST API calls. For today's data, reads are instant (synchronous
buffer reads). For historical data or options chains, delegates to the
REST-based PolygonIntradayProvider as fallback.

Usage:
    feed = MassiveDataFeed(api_key, tickers)
    await feed.start()
    provider = WebSocketBarProvider(feed, fallback_api_key=api_key)

    # These read from WebSocket buffer (instant):
    bars = await provider.get_today_bars("SPY")
    snap = await provider.get_snapshot("SPY")

    # These fall back to REST (options are REST-only):
    chain = await provider.get_options_chain("SPY", ...)
    atm = await provider.get_nearest_atm_option("SPY", 550.0)
"""

from __future__ import annotations

from datetime import date
from typing import Any

import structlog

from flowedge.scanner.data_feeds.schemas import BarData, OptionQuote, Timeframe

logger = structlog.get_logger()


class WebSocketBarProvider:
    """Drop-in replacement for PolygonIntradayProvider using WebSocket data.

    For intraday bars: reads from MassiveDataFeed buffer (instant).
    For options: delegates to REST-based PolygonIntradayProvider.
    """

    def __init__(
        self,
        data_feed: Any,  # MassiveDataFeed instance
        fallback_api_key: str = "",
        fallback_base_url: str = "https://api.polygon.io",
    ) -> None:
        self._feed = data_feed
        self._fallback_api_key = fallback_api_key
        self._fallback_base_url = fallback_base_url
        self._rest_provider: Any = None  # Lazy-loaded for REST fallback

    async def _get_rest_fallback(self) -> Any:
        """Lazy-load REST provider for options and historical data."""
        if self._rest_provider is None:
            from flowedge.scanner.data_feeds.polygon_intraday import (
                PolygonIntradayProvider,
            )
            self._rest_provider = PolygonIntradayProvider(
                self._fallback_api_key, self._fallback_base_url,
            )
        return self._rest_provider

    async def close(self) -> None:
        """Close REST fallback if it was used."""
        if self._rest_provider:
            await self._rest_provider.close()
            self._rest_provider = None

    # ── Bars (WebSocket buffer) ──────────────────────────────────

    async def get_intraday_bars(
        self,
        ticker: str,
        timeframe: Timeframe,
        from_date: str,
        to_date: str,
        limit: int = 5000,
    ) -> list[BarData]:
        """Get intraday bars — from WebSocket buffer if today, else REST.

        For today's data: instant read from the AM.* buffer.
        For historical: falls back to REST API.
        """
        today = date.today().isoformat()

        # If requesting today's data, use WebSocket buffer
        if from_date == today or to_date == today:
            bars = self._feed.get_bars(ticker, count=limit)
            if bars:
                logger.debug(
                    "ws_bars_served",
                    ticker=ticker,
                    count=len(bars),
                    source="websocket",
                )
                return bars  # type: ignore[no-any-return]

        # Fall back to REST for historical data
        rest = await self._get_rest_fallback()
        return await rest.get_intraday_bars(  # type: ignore[no-any-return]
            ticker, timeframe, from_date, to_date, limit,
        )

    async def get_today_bars(
        self,
        ticker: str,
        timeframe: Timeframe = Timeframe.MIN_5,
    ) -> list[BarData]:
        """Get today's bars from WebSocket buffer.

        Note: WebSocket provides 1-min bars. For 5-min timeframe,
        we return all 1-min bars and let the caller aggregate
        (or fall back to REST for native 5-min bars).
        """
        bars = self._feed.get_bars(ticker, count=500)
        if bars:
            logger.debug(
                "ws_today_bars_served",
                ticker=ticker,
                count=len(bars),
                source="websocket",
            )
            return bars  # type: ignore[no-any-return]

        # Fallback: REST
        rest = await self._get_rest_fallback()
        return await rest.get_today_bars(ticker, timeframe)  # type: ignore[no-any-return]

    # ── Options (REST fallback — no WebSocket for options) ───────

    async def get_options_chain(
        self,
        ticker: str,
        expiration_gte: str | None = None,
        expiration_lte: str | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
        contract_type: str | None = None,
        limit: int = 50,
    ) -> list[OptionQuote]:
        """Get options chain — always REST (no WebSocket for options)."""
        rest = await self._get_rest_fallback()
        return await rest.get_options_chain(  # type: ignore[no-any-return]
            ticker, expiration_gte, expiration_lte,
            strike_price_gte, strike_price_lte,
            contract_type, limit,
        )

    async def get_nearest_atm_option(
        self,
        ticker: str,
        current_price: float,
        option_type: str = "call",
        min_dte: int = 5,
        max_dte: int = 30,
    ) -> OptionQuote | None:
        """Find nearest ATM option — always REST."""
        rest = await self._get_rest_fallback()
        return await rest.get_nearest_atm_option(  # type: ignore[no-any-return]
            ticker, current_price, option_type, min_dte, max_dte,
        )

    # ── Snapshots (WebSocket buffer) ─────────────────────────────

    async def get_snapshot(self, ticker: str) -> dict[str, Any]:
        """Get real-time snapshot — from WebSocket last price + bar."""
        price = self._feed.get_latest_price(ticker)
        bar = self._feed.get_latest_bar(ticker)

        if price > 0 and bar:
            return {
                "ticker": ticker,
                "lastTrade": {"p": price},
                "min": {
                    "o": bar.open, "h": bar.high,
                    "l": bar.low, "c": bar.close,
                    "v": bar.volume, "vw": bar.vwap,
                },
                "source": "websocket",
            }

        # Fallback to REST
        rest = await self._get_rest_fallback()
        return await rest.get_snapshot(ticker)  # type: ignore[no-any-return]

    async def get_snapshots(
        self, tickers: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get snapshots for multiple tickers from WebSocket buffer."""
        result: dict[str, dict[str, Any]] = {}
        need_rest: list[str] = []

        for ticker in tickers:
            price = self._feed.get_latest_price(ticker)
            bar = self._feed.get_latest_bar(ticker)
            if price > 0 and bar:
                result[ticker] = {
                    "ticker": ticker,
                    "lastTrade": {"p": price},
                    "min": {
                        "o": bar.open, "h": bar.high,
                        "l": bar.low, "c": bar.close,
                        "v": bar.volume, "vw": bar.vwap,
                    },
                    "source": "websocket",
                }
            else:
                need_rest.append(ticker)

        if need_rest:
            rest = await self._get_rest_fallback()
            rest_snaps = await rest.get_snapshots(need_rest)
            result.update(rest_snaps)

        return result
