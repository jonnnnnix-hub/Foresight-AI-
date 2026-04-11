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

    async def get_intraday_bars(
        self,
        symbol: str,
        multiplier: int = 1,
        timespan: str = "minute",
        from_date: str = "",
        to_date: str = "",
        limit: int = 500,
    ) -> list[dict[str, object]]:
        """Fetch intraday bars from Polygon aggregates."""
        if not from_date:
            from datetime import date as d
            from_date = d.today().isoformat()
        if not to_date:
            to_date = from_date

        data = await self._get(
            f"{self._base_url}/v2/aggs/ticker/{symbol}"
            f"/range/{multiplier}/{timespan}/{from_date}/{to_date}",
            params={"apiKey": self._api_key, "limit": str(limit), "sort": "asc"},
        )

        return [
            {
                "timestamp": r.get("t", 0),
                "open": float(r.get("o", 0)),
                "high": float(r.get("h", 0)),
                "low": float(r.get("l", 0)),
                "close": float(r.get("c", 0)),
                "volume": int(r.get("v", 0)),
                "vwap": float(r.get("vw", 0)),
            }
            for r in data.get("results", [])
        ]

    async def get_technical_indicator(
        self,
        symbol: str,
        indicator: str = "sma",
        window: int = 20,
        timespan: str = "day",
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """Fetch technical indicator (SMA, RSI, MACD, EMA) from Polygon."""
        params: dict[str, str] = {
            "apiKey": self._api_key,
            "timespan": timespan,
            "limit": str(limit),
            "sort": "desc",
        }
        if indicator in ("sma", "ema", "rsi"):
            params["window"] = str(window)

        data = await self._get(
            f"{self._base_url}/v1/indicators/{indicator}/{symbol}",
            params=params,
        )

        return [
            {
                "timestamp": r.get("timestamp", 0),
                "value": r.get("value", r.get("values", {})),
            }
            for r in data.get("results", {}).get("values", [])
        ]

    async def get_news(
        self, ticker: str, limit: int = 10
    ) -> list[dict[str, object]]:
        """Fetch recent news articles for a ticker."""
        data = await self._get(
            f"{self._base_url}/v2/reference/news",
            params={
                "apiKey": self._api_key,
                "ticker": ticker,
                "limit": str(limit),
                "sort": "published_utc",
                "order": "desc",
            },
        )
        raw = data.get("results", [])
        results: list[dict[str, object]] = raw if isinstance(raw, list) else []
        return results

    async def get_previous_close(
        self, symbol: str
    ) -> dict[str, object]:
        """Fetch previous day's OHLCV."""
        data = await self._get(
            f"{self._base_url}/v2/aggs/ticker/{symbol}/prev",
            params={"apiKey": self._api_key},
        )
        results = data.get("results", [])
        if results:
            r = results[0]
            return {
                "open": float(r.get("o", 0)),
                "high": float(r.get("h", 0)),
                "low": float(r.get("l", 0)),
                "close": float(r.get("c", 0)),
                "volume": int(r.get("v", 0)),
                "vwap": float(r.get("vw", 0)),
            }
        return {}

    async def get_related_companies(
        self, ticker: str
    ) -> list[str]:
        """Fetch related/peer companies for a ticker."""
        data = await self._get(
            f"{self._base_url}/v1/related-companies/{ticker}",
            params={"apiKey": self._api_key},
        )
        results = data.get("results", [])
        tickers: list[str] = [
            r.get("ticker", "") for r in results if isinstance(r, dict)
        ]
        return tickers
