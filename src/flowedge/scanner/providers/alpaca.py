"""Alpaca provider — equity bars and paper trading."""

from __future__ import annotations

from typing import Any

from flowedge.config.settings import Settings
from flowedge.scanner.providers.base import EquityDataProvider


class AlpacaProvider(EquityDataProvider):
    """Alpaca equity data and paper trading provider."""

    name = "alpaca"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._base_url = settings.alpaca_base_url
        self._data_url = "https://data.alpaca.markets"
        self._headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key_id,
            "APCA-API-SECRET-KEY": settings.alpaca_api_secret_key,
        }

    async def health_check(self) -> bool:
        try:
            # Check both trading account AND data access
            await self._get(
                f"{self._base_url}/v2/account",
                headers=self._headers,
            )
            # Verify data feed access with a lightweight bars request
            await self._get(
                f"{self._data_url}/v2/stocks/SPY/bars",
                params={"timeframe": "1Day", "limit": "1", "feed": "iex"},
                headers=self._headers,
            )
            return True
        except Exception:
            return False

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch historical bars from Alpaca."""
        data = await self._get(
            f"{self._data_url}/v2/stocks/{symbol}/bars",
            params={
                "timeframe": timeframe,
                "limit": str(limit),
                "feed": "iex",  # IEX feed (free) — SIP requires paid subscription
            },
            headers=self._headers,
        )

        bars: list[dict[str, Any]] = []
        for bar in data.get("bars", []):
            bars.append({
                "timestamp": bar.get("t", ""),
                "open": float(bar.get("o", 0)),
                "high": float(bar.get("h", 0)),
                "low": float(bar.get("l", 0)),
                "close": float(bar.get("c", 0)),
                "volume": int(bar.get("v", 0)),
            })

        return bars
