"""Unusual Whales provider — options flow, dark pool, greeks, and GEX."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from flowedge.config.settings import Settings
from flowedge.scanner.providers.base import FlowAlertProvider
from flowedge.scanner.schemas.flow import (
    DarkPoolTrade,
    FlowAlert,
    FlowSentiment,
    FlowType,
)
from flowedge.scanner.schemas.options import OptionType


class UnusualWhalesProvider(FlowAlertProvider):
    """Unusual Whales options flow, dark pool, greeks, and GEX provider.

    Available endpoints on current plan:
    - /api/stock/{ticker}/options-volume — call/put volume, premium
    - /api/stock/{ticker}/option-contracts — per-contract IV, OI, volume
    - /api/stock/{ticker}/option-chains — full chain (5k+ items)
    - /api/darkpool/{ticker} — dark pool trades with NBBO
    - /api/stock/{ticker}/greeks — per-strike greeks by expiry
    - /api/stock/{ticker}/max-pain — max pain by expiration
    - /api/stock/{ticker}/net-prem-ticks — intraday GEX/premium ticks
    - /api/stock/{ticker}/earnings — historical earnings
    - /api/stock/{ticker}/financials — balance sheets, income, cash flows
    """

    name = "unusual_whales"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._base_url = settings.unusual_whales_base_url
        self._headers = {
            "Authorization": f"Bearer {settings.unusual_whales_api_key}",
            "Accept": "application/json",
        }

    async def health_check(self) -> bool:
        try:
            await self._get(
                f"{self._base_url}/api/stock/AAPL/options-volume",
                headers=self._headers,
            )
            return True
        except Exception:
            return False

    async def get_flow_alerts(
        self, ticker: str | None = None
    ) -> list[FlowAlert]:
        """Build flow alerts from options contracts data.

        Uses /option-contracts for per-contract volume/OI/IV
        and /options-volume for aggregate call/put flow.
        """
        if not ticker:
            return []

        # Get per-contract data
        data = await self._get(
            f"{self._base_url}/api/stock/{ticker}/option-contracts",
            headers=self._headers,
        )

        alerts: list[FlowAlert] = []
        now = datetime.now()

        for item in data.get("data", []):
            if not isinstance(item, dict):
                continue

            # Determine option type from symbol
            option_symbol = str(item.get("option_symbol", ""))
            opt_type = OptionType.CALL
            if "P" in option_symbol[-10:-8]:
                opt_type = OptionType.PUT

            vol = int(item.get("volume", 0))
            oi = int(item.get("open_interest", 0))
            premium = float(item.get("avg_price", 0)) * vol * 100

            # Skip low-activity contracts
            if vol < 10:
                continue

            # Determine sentiment from bid/ask side volume
            ask_vol = int(item.get("ask_volume", 0))
            bid_vol = int(item.get("bid_volume", 0))
            if ask_vol > bid_vol * 1.5:
                sentiment = (
                    FlowSentiment.BULLISH
                    if opt_type == OptionType.CALL
                    else FlowSentiment.BEARISH
                )
            elif bid_vol > ask_vol * 1.5:
                sentiment = (
                    FlowSentiment.BEARISH
                    if opt_type == OptionType.CALL
                    else FlowSentiment.BULLISH
                )
            else:
                sentiment = FlowSentiment.NEUTRAL

            # Classify flow type
            if premium >= 100_000:
                flow_type = FlowType.BLOCK
            elif ask_vol > bid_vol * 3:
                flow_type = FlowType.SWEEP
            else:
                flow_type = FlowType.REGULAR

            # Parse expiration from option symbol (OCC format)
            exp_date = date.today()
            try:
                # Standard OCC: TSLA260417C00250000
                if len(option_symbol) >= 15:
                    date_part = option_symbol[-15:-9]
                    exp_date = datetime.strptime(date_part, "%y%m%d").date()
            except (ValueError, IndexError):
                pass

            # Parse strike from option symbol
            strike = 0.0
            try:
                if len(option_symbol) >= 8:
                    strike = int(option_symbol[-8:]) / 1000
            except (ValueError, IndexError):
                pass

            alerts.append(
                FlowAlert(
                    ticker=ticker,
                    option_symbol=option_symbol,
                    option_type=opt_type,
                    strike=strike,
                    expiration=exp_date,
                    flow_type=flow_type,
                    sentiment=sentiment,
                    premium=premium,
                    volume=vol,
                    open_interest=oi,
                    volume_oi_ratio=round(vol / oi, 2) if oi > 0 else 0.0,
                    underlying_price=float(item.get("last_price", 0)),
                    timestamp=now,
                    source="unusual_whales",
                )
            )

        return alerts

    async def get_dark_pool_trades(
        self, ticker: str
    ) -> list[DarkPoolTrade]:
        """Fetch dark pool prints with NBBO context."""
        data = await self._get(
            f"{self._base_url}/api/darkpool/{ticker}",
            headers=self._headers,
        )
        trades: list[DarkPoolTrade] = []

        for item in data.get("data", []):
            if not isinstance(item, dict):
                continue

            ts_str = str(item.get("executed_at", ""))
            try:
                ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now()
            except ValueError:
                ts = datetime.now()

            price = float(item.get("price", 0))
            size = int(item.get("size", 0) or item.get("volume", 0) or 0)

            trades.append(
                DarkPoolTrade(
                    ticker=ticker,
                    price=price,
                    size=size,
                    notional=round(price * size, 2),
                    timestamp=ts,
                    exchange=str(item.get("sale_cond_codes", "")),
                )
            )

        return trades

    # ---- Enhanced UW endpoints ----

    async def get_options_volume(self, ticker: str) -> dict[str, Any]:
        """Get aggregate call/put volume and premium for today."""
        data = await self._get(
            f"{self._base_url}/api/stock/{ticker}/options-volume",
            headers=self._headers,
        )
        items = data.get("data", [])
        if items and isinstance(items, list):
            result: dict[str, Any] = items[0]
            return result
        return {}

    async def get_greeks_by_strike(
        self, ticker: str
    ) -> list[dict[str, Any]]:
        """Get per-strike greeks (delta, gamma, charm, vanna) by expiry."""
        data = await self._get(
            f"{self._base_url}/api/stock/{ticker}/greeks",
            headers=self._headers,
        )
        results = data.get("data", [])
        return results if isinstance(results, list) else []

    async def get_max_pain(
        self, ticker: str
    ) -> list[dict[str, Any]]:
        """Get max pain levels by expiration."""
        data = await self._get(
            f"{self._base_url}/api/stock/{ticker}/max-pain",
            headers=self._headers,
        )
        results = data.get("data", [])
        return results if isinstance(results, list) else []

    async def get_net_premium_ticks(
        self, ticker: str
    ) -> list[dict[str, Any]]:
        """Get intraday net premium ticks (GEX proxy)."""
        data = await self._get(
            f"{self._base_url}/api/stock/{ticker}/net-prem-ticks",
            headers=self._headers,
        )
        results = data.get("data", [])
        return results if isinstance(results, list) else []

    async def get_earnings_history(
        self, ticker: str
    ) -> list[dict[str, Any]]:
        """Get historical earnings with surprise data."""
        data = await self._get(
            f"{self._base_url}/api/stock/{ticker}/earnings",
            headers=self._headers,
        )
        results = data.get("data", [])
        return results if isinstance(results, list) else []
