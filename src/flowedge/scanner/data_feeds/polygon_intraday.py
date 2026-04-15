"""Polygon.io paid tier — intraday bars + options chains.

Paid tier unlocks:
- 1-min / 5-min / 15-min aggregated bars
- Real-time options chain snapshots with Greeks
- Unlimited API requests (no 5/min limit)
- WebSocket streaming for real-time updates
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any

import structlog

from flowedge.scanner.data_feeds.schemas import (
    BarData,
    OptionQuote,
    Timeframe,
)

logger = structlog.get_logger()

_ORATS_API_KEY = os.getenv("ORATS_API_KEY", "")
_ORATS_BASE = "https://api.orats.io"

# Polygon timeframe mapping
_TF_MAP: dict[Timeframe, tuple[int, str]] = {
    Timeframe.MIN_1: (1, "minute"),
    Timeframe.MIN_5: (5, "minute"),
    Timeframe.MIN_15: (15, "minute"),
    Timeframe.HOUR_1: (1, "hour"),
    Timeframe.DAILY: (1, "day"),
}


class PolygonIntradayProvider:
    """Polygon paid-tier data provider for intraday bars + options."""

    def __init__(self, api_key: str, base_url: str = "https://api.polygon.io") -> None:
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

    async def _get(self, url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        client = await self._ensure_client()
        params = params or {}
        params["apiKey"] = self._api_key
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    # ── Intraday Bars ──────────────────────────────────────────────

    async def get_intraday_bars(
        self,
        ticker: str,
        timeframe: Timeframe,
        from_date: str,
        to_date: str,
        limit: int = 5000,
    ) -> list[BarData]:
        """Fetch intraday bars for a ticker at a given timeframe.

        Args:
            ticker: e.g. "SPY"
            timeframe: MIN_1, MIN_5, MIN_15, HOUR_1, DAILY
            from_date: YYYY-MM-DD
            to_date: YYYY-MM-DD
            limit: Max bars to return

        Returns:
            List of BarData sorted by timestamp ascending.
        """
        multiplier, span = _TF_MAP.get(timeframe, (1, "day"))

        data = await self._get(
            f"{self._base_url}/v2/aggs/ticker/{ticker}"
            f"/range/{multiplier}/{span}/{from_date}/{to_date}",
            params={"limit": str(limit), "sort": "asc"},
        )

        bars: list[BarData] = []
        for r in data.get("results", []):
            ts = r.get("t", 0)
            bars.append(BarData(
                ticker=ticker,
                timestamp=datetime.fromtimestamp(ts / 1000),
                timeframe=timeframe,
                open=float(r.get("o", 0)),
                high=float(r.get("h", 0)),
                low=float(r.get("l", 0)),
                close=float(r.get("c", 0)),
                volume=int(r.get("v", 0)),
                vwap=float(r.get("vw", 0)),
                trade_count=int(r.get("n", 0)),
            ))

        logger.info(
            "intraday_bars_loaded",
            ticker=ticker,
            timeframe=timeframe.value,
            count=len(bars),
        )
        return bars

    async def get_today_bars(
        self,
        ticker: str,
        timeframe: Timeframe = Timeframe.MIN_5,
    ) -> list[BarData]:
        """Fetch today's intraday bars."""
        today = date.today().isoformat()
        return await self.get_intraday_bars(ticker, timeframe, today, today)

    # ── Options Chain ──────────────────────────────────────────────

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
        """Fetch options chain snapshot with Greeks.

        Args:
            ticker: Underlying ticker
            expiration_gte/lte: Filter by expiration date range
            strike_price_gte/lte: Filter by strike range
            contract_type: "call" or "put"
            limit: Max contracts to return

        Returns:
            List of OptionQuote with real-time bid/ask/Greeks.
        """
        params: dict[str, str] = {
            "underlying_ticker": ticker,
            "limit": str(limit),
        }
        if expiration_gte:
            params["expiration_date.gte"] = expiration_gte
        if expiration_lte:
            params["expiration_date.lte"] = expiration_lte
        if strike_price_gte is not None:
            params["strike_price.gte"] = str(strike_price_gte)
        if strike_price_lte is not None:
            params["strike_price.lte"] = str(strike_price_lte)
        if contract_type:
            params["contract_type"] = contract_type

        data = await self._get(
            f"{self._base_url}/v3/snapshot/options/{ticker}",
            params=params,
        )

        quotes: list[OptionQuote] = []
        for result in data.get("results", []):
            details = result.get("details", {})
            greeks = result.get("greeks", {})
            day = result.get("day", {})
            last_quote = result.get("last_quote", {})

            bid = float(last_quote.get("bid", 0))
            ask = float(last_quote.get("ask", 0))
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
            spread = ask - bid if bid > 0 else 0.0
            spread_pct = (spread / mid * 100) if mid > 0 else 0.0

            quotes.append(OptionQuote(
                underlying=ticker,
                contract_symbol=details.get("ticker", ""),
                expiration=details.get("expiration_date", ""),
                strike=float(details.get("strike_price", 0)),
                option_type=details.get("contract_type", "call"),
                bid=bid,
                ask=ask,
                mid=round(mid, 4),
                last=float(day.get("last_trade_price", 0) if day else 0),
                delta=float(greeks.get("delta", 0)),
                gamma=float(greeks.get("gamma", 0)),
                theta=float(greeks.get("theta", 0)),
                vega=float(greeks.get("vega", 0)),
                iv=float(greeks.get("implied_volatility", 0) if greeks else 0),
                volume=int(day.get("volume", 0) if day else 0),
                open_interest=int(result.get("open_interest", 0)),
                bid_ask_spread=round(spread, 4),
                spread_pct=round(spread_pct, 2),
            ))

        logger.info(
            "options_chain_loaded",
            ticker=ticker,
            contracts=len(quotes),
        )
        return quotes

    async def get_nearest_atm_option(
        self,
        ticker: str,
        current_price: float,
        option_type: str = "call",
        min_dte: int = 5,
        max_dte: int = 30,
    ) -> OptionQuote | None:
        """Find the nearest ATM option with appropriate DTE.

        Tries ORATS live strikes first (real bid/ask + greeks), then
        falls back to Polygon options snapshot.
        """
        # ── Try ORATS first (real-time bid/ask) ─────────────────
        orats_key = _ORATS_API_KEY or os.getenv("ORATS_API_KEY", "")
        if orats_key:
            result = await self._orats_atm_lookup(
                ticker, current_price, option_type, min_dte, max_dte, orats_key,
            )
            if result:
                return result
            logger.info("orats_fallback_to_polygon", ticker=ticker)

        # ── Polygon fallback ────────────────────────────────────
        today = date.today()
        exp_gte = (today + timedelta(days=min_dte)).isoformat()
        exp_lte = (today + timedelta(days=max_dte)).isoformat()

        chain = await self.get_options_chain(
            ticker,
            expiration_gte=exp_gte,
            expiration_lte=exp_lte,
            strike_price_gte=current_price * 0.97,
            strike_price_lte=current_price * 1.03,
            contract_type=option_type,
            limit=20,
        )

        if not chain:
            return None

        best: OptionQuote | None = None
        best_distance = 999.0
        for q in chain:
            if q.bid <= 0 or q.ask <= 0:
                continue
            distance = abs(q.strike - current_price)
            if distance < best_distance:
                best_distance = distance
                best = q

        return best

    async def _orats_atm_lookup(
        self,
        ticker: str,
        current_price: float,
        option_type: str,
        min_dte: int,
        max_dte: int,
        orats_key: str,
    ) -> OptionQuote | None:
        """Look up nearest ATM option from ORATS live strikes.

        ORATS returns real bid/ask/IV/greeks updated every ~20 seconds.
        """
        try:
            import urllib.request
            import json as _json
            import asyncio

            strike_lo = current_price * 0.97
            strike_hi = current_price * 1.03
            url = (
                f"{_ORATS_BASE}/datav2/live/strikes"
                f"?token={orats_key}"
                f"&ticker={ticker}"
                f"&dte.gte={min_dte}&dte.lte={max_dte}"
                f"&strike.gte={strike_lo:.2f}&strike.lte={strike_hi:.2f}"
            )

            loop = asyncio.get_event_loop()
            req = urllib.request.Request(url, headers={"User-Agent": "FlowEdge/1.0"})
            resp = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(req, timeout=10),
            )
            data = _json.loads(resp.read())
            rows = data.get("data", [])

            if not rows:
                return None

            side = "call" if option_type == "call" else "put"
            best: OptionQuote | None = None
            best_distance = 999.0

            for row in rows:
                strike = float(row.get("strike", 0))
                bid = float(row.get(f"{side}BidPrice", 0))
                ask = float(row.get(f"{side}AskPrice", 0))
                if bid <= 0 or ask <= 0:
                    continue

                distance = abs(strike - current_price)
                if distance < best_distance:
                    best_distance = distance
                    mid = round((bid + ask) / 2, 4)
                    spread = round(ask - bid, 4)
                    spread_pct = round(spread / mid * 100, 2) if mid > 0 else 0.0

                    exp_str = str(row.get("expirDate", ""))
                    dte = int(row.get("dte", 0))

                    # Build OCC-style symbol: SPY260415C00550000
                    exp_compact = exp_str.replace("-", "")[2:]  # YYMMDD
                    sym = f"{ticker}{exp_compact}{side[0].upper()}{int(strike * 1000):08d}"

                    best = OptionQuote(
                        underlying=ticker,
                        contract_symbol=sym,
                        expiration=exp_str,
                        strike=strike,
                        option_type=option_type,
                        bid=bid,
                        ask=ask,
                        mid=mid,
                        last=float(row.get(f"{side}Value", mid)),
                        delta=float(row.get("delta", 0)),
                        gamma=float(row.get("gamma", 0)),
                        theta=float(row.get("theta", 0)),
                        vega=float(row.get("vega", 0)),
                        iv=float(row.get(f"{side}MidIv", row.get("smvVol", 0))),
                        volume=int(row.get(f"{side}Volume", 0)),
                        open_interest=int(row.get(f"{side}OpenInt", 0)),
                        bid_ask_spread=spread,
                        spread_pct=spread_pct,
                    )

            if best:
                logger.info(
                    "orats_option_found",
                    ticker=ticker,
                    strike=best.strike,
                    bid=best.bid,
                    ask=best.ask,
                    iv=round(best.iv, 3),
                    dte=rows[0].get("dte"),
                    source="orats",
                )
            return best

        except Exception as e:
            logger.warning("orats_lookup_failed", ticker=ticker, error=str(e))
            return None

    # ── Real-Time Snapshots ───────────────────────────────────────

    async def get_snapshot(self, ticker: str) -> dict[str, Any]:
        """Get real-time snapshot for a ticker (last trade, quote, min bar).

        Returns dict with keys: last_trade, last_quote, min, prevDay, etc.
        Much faster than pulling full bar history — use for exit monitoring.
        """
        data = await self._get(
            f"{self._base_url}/v2/snapshot/locale/us/markets/stocks"
            f"/tickers/{ticker}",
        )
        return data.get("ticker", {})

    async def get_snapshots(
        self, tickers: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get real-time snapshots for multiple tickers in one call.

        Polygon supports comma-separated tickers for batch snapshot.
        Returns {ticker: snapshot_data} dict.
        """
        ticker_str = ",".join(tickers)
        data = await self._get(
            f"{self._base_url}/v2/snapshot/locale/us/markets/stocks/tickers",
            params={"tickers": ticker_str},
        )
        result: dict[str, dict[str, Any]] = {}
        for snap in data.get("tickers", []):
            t = snap.get("ticker", "")
            if t:
                result[t] = snap
        return result
