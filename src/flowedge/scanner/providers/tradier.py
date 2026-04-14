"""Tradier provider — options chains with greeks."""

from __future__ import annotations

from datetime import date, datetime

from flowedge.config.settings import Settings
from flowedge.scanner.providers.base import OptionsChainProvider
from flowedge.scanner.schemas.options import (
    OptionContract,
    OptionsChain,
    OptionType,
)


class TradierProvider(OptionsChainProvider):
    """Tradier options chain and greeks provider."""

    name = "tradier"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._base_url = settings.tradier_base_url
        self._headers = {
            "Authorization": f"Bearer {settings.tradier_api_key}",
            "Accept": "application/json",
        }

    async def health_check(self) -> bool:
        try:
            await self._get(
                f"{self._base_url}/v1/markets/quotes",
                params={"symbols": "AAPL"},
                headers=self._headers,
            )
            return True
        except Exception:
            return False

    async def get_options_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> OptionsChain:
        """Fetch options chain with greeks from Tradier."""
        params: dict[str, str] = {"symbol": symbol, "greeks": "true"}
        if expiration:
            params["expiration"] = expiration.isoformat()

        url = f"{self._base_url}/v1/markets/options/chains"
        data = await self._get(url, params=params, headers=self._headers)

        # Get underlying price
        quote_data = await self._get(
            f"{self._base_url}/v1/markets/quotes",
            params={"symbols": symbol},
            headers=self._headers,
        )
        underlying_price = 0.0
        quotes = quote_data.get("quotes", {})
        quote = quotes.get("quote", {})
        if isinstance(quote, dict):
            underlying_price = float(quote.get("last", 0))

        contracts: list[OptionContract] = []
        now = datetime.now()

        options = data.get("options", {})
        option_list = options.get("option", [])
        if isinstance(option_list, dict):
            option_list = [option_list]

        for opt in option_list:
            exp_str = opt.get("expiration_date", "")
            try:
                exp_date = date.fromisoformat(exp_str) if exp_str else date.today()
            except ValueError:
                continue

            greeks = opt.get("greeks", {}) or {}
            bid = float(opt.get("bid", 0))
            ask = float(opt.get("ask", 0))

            contracts.append(
                OptionContract(
                    symbol=opt.get("symbol", ""),
                    underlying=symbol,
                    option_type=(
                        OptionType.CALL
                        if opt.get("option_type", "").lower() == "call"
                        else OptionType.PUT
                    ),
                    strike=float(opt.get("strike", 0)),
                    expiration=exp_date,
                    bid=bid,
                    ask=ask,
                    mid=round((bid + ask) / 2, 2) if (bid + ask) > 0 else 0.0,
                    last=float(opt["last"]) if opt.get("last") else None,
                    volume=int(opt.get("volume", 0)),
                    open_interest=int(opt.get("open_interest", 0)),
                    implied_volatility=(
                        float(greeks["mid_iv"])
                        if greeks.get("mid_iv")
                        else None
                    ),
                    delta=greeks.get("delta"),
                    gamma=greeks.get("gamma"),
                    theta=greeks.get("theta"),
                    vega=greeks.get("vega"),
                    rho=greeks.get("rho"),
                    days_to_expiration=(exp_date - date.today()).days,
                    source="tradier",
                    fetched_at=now,
                )
            )

        return OptionsChain(
            underlying=symbol,
            underlying_price=underlying_price,
            contracts=contracts,
            fetched_at=now,
            source="tradier",
        )
