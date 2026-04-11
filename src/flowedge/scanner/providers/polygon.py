"""Polygon.io provider — options chains and equity snapshots."""

from __future__ import annotations

from datetime import date, datetime

from flowedge.config.settings import Settings
from flowedge.scanner.providers.base import OptionsChainProvider
from flowedge.scanner.schemas.options import (
    OptionContract,
    OptionsChain,
    OptionType,
)


class PolygonProvider(OptionsChainProvider):
    """Polygon.io options chain and market data provider."""

    name = "polygon"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._base_url = "https://api.polygon.io"
        self._api_key = settings.polygon_api_key

    async def health_check(self) -> bool:
        try:
            await self._get(
                f"{self._base_url}/v2/aggs/ticker/AAPL/prev",
                params={"apiKey": self._api_key},
            )
            return True
        except Exception:
            return False

    async def get_options_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> OptionsChain:
        """Fetch options chain snapshot from Polygon."""
        params: dict[str, str] = {"apiKey": self._api_key}
        if expiration:
            params["expiration_date"] = expiration.isoformat()

        url = f"{self._base_url}/v3/snapshot/options/{symbol}"
        data = await self._get(url, params=params)

        contracts: list[OptionContract] = []
        now = datetime.now()

        for result in data.get("results", []):
            details = result.get("details", {})
            greeks = result.get("greeks", {})
            quote = result.get("last_quote", {})
            trade = result.get("last_trade", {})

            exp_str = details.get("expiration_date", "")
            try:
                exp_date = date.fromisoformat(exp_str) if exp_str else date.today()
            except ValueError:
                continue

            bid = float(quote.get("bid", 0))
            ask = float(quote.get("ask", 0))

            contracts.append(
                OptionContract(
                    symbol=details.get("ticker", ""),
                    underlying=symbol,
                    option_type=(
                        OptionType.CALL
                        if details.get("contract_type", "").lower() == "call"
                        else OptionType.PUT
                    ),
                    strike=float(details.get("strike_price", 0)),
                    expiration=exp_date,
                    bid=bid,
                    ask=ask,
                    mid=round((bid + ask) / 2, 2) if (bid + ask) > 0 else 0.0,
                    last=float(trade.get("price", 0)) if trade.get("price") else None,
                    volume=int(result.get("day", {}).get("volume", 0)),
                    open_interest=int(result.get("open_interest", 0)),
                    implied_volatility=float(result.get("implied_volatility", 0))
                    if result.get("implied_volatility")
                    else None,
                    delta=greeks.get("delta"),
                    gamma=greeks.get("gamma"),
                    theta=greeks.get("theta"),
                    vega=greeks.get("vega"),
                    rho=greeks.get("rho"),
                    days_to_expiration=(exp_date - date.today()).days,
                    source="polygon",
                    fetched_at=now,
                )
            )

        # Get underlying price from first result
        underlying_price = 0.0
        if data.get("results"):
            underlying_price = float(
                data["results"][0]
                .get("underlying_asset", {})
                .get("price", 0)
            )

        return OptionsChain(
            underlying=symbol,
            underlying_price=underlying_price,
            contracts=contracts,
            fetched_at=now,
            source="polygon",
        )
