"""Unusual Whales provider — options flow alerts and dark pool data."""

from __future__ import annotations

from datetime import date, datetime

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
    """Unusual Whales options flow and dark pool provider."""

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
                f"{self._base_url}/api/stock/AAPL/quote",
                headers=self._headers,
            )
            return True
        except Exception:
            return False

    async def get_flow_alerts(
        self, ticker: str | None = None
    ) -> list[FlowAlert]:
        """Fetch unusual options flow alerts."""
        url = f"{self._base_url}/api/option-trade/flow-alerts"
        params: dict[str, str] = {}
        if ticker:
            params["ticker"] = ticker

        data = await self._get(url, params=params, headers=self._headers)
        alerts: list[FlowAlert] = []

        for item in data.get("data", []):
            exp_str = item.get("expiry", item.get("expiration_date", ""))
            try:
                exp_date = (
                    date.fromisoformat(exp_str) if exp_str else date.today()
                )
            except ValueError:
                continue

            # Parse flow type
            flow_type_str = str(item.get("execution_estimate", "")).lower()
            if "sweep" in flow_type_str:
                flow_type = FlowType.SWEEP
            elif "block" in flow_type_str:
                flow_type = FlowType.BLOCK
            elif "split" in flow_type_str:
                flow_type = FlowType.SPLIT
            else:
                flow_type = FlowType.REGULAR

            # Parse sentiment
            sent_str = str(item.get("sentiment", "")).lower()
            if "bullish" in sent_str:
                sentiment = FlowSentiment.BULLISH
            elif "bearish" in sent_str:
                sentiment = FlowSentiment.BEARISH
            else:
                sentiment = FlowSentiment.NEUTRAL

            # Parse option type
            put_call = str(item.get("put_call", item.get("option_type", ""))).lower()
            option_type = (
                OptionType.CALL if put_call in ("call", "c") else OptionType.PUT
            )

            vol = int(item.get("volume", 0))
            oi = int(item.get("open_interest", 0))

            alerts.append(
                FlowAlert(
                    ticker=item.get("ticker", ticker or ""),
                    option_symbol=item.get("option_symbol", ""),
                    option_type=option_type,
                    strike=float(item.get("strike", 0)),
                    expiration=exp_date,
                    flow_type=flow_type,
                    sentiment=sentiment,
                    premium=float(item.get("premium", 0)),
                    volume=vol,
                    open_interest=oi,
                    volume_oi_ratio=round(vol / oi, 2) if oi > 0 else 0.0,
                    underlying_price=float(
                        item.get("underlying_price", item.get("stock_price", 0))
                    ),
                    timestamp=datetime.now(),
                    source="unusual_whales",
                )
            )

        return alerts

    async def get_dark_pool_trades(
        self, ticker: str
    ) -> list[DarkPoolTrade]:
        """Fetch dark pool prints for a ticker."""
        url = f"{self._base_url}/api/darkpool/{ticker}"
        data = await self._get(url, headers=self._headers)
        trades: list[DarkPoolTrade] = []

        for item in data.get("data", []):
            ts_str = item.get("executed_at", item.get("timestamp", ""))
            try:
                ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now()
            except ValueError:
                ts = datetime.now()

            price = float(item.get("price", 0))
            size = int(item.get("size", item.get("volume", 0)))

            trades.append(
                DarkPoolTrade(
                    ticker=ticker,
                    price=price,
                    size=size,
                    notional=round(price * size, 2),
                    timestamp=ts,
                    exchange=item.get("exchange", ""),
                )
            )

        return trades
