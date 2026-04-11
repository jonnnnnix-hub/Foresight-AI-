"""Abstract base classes for all data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import httpx
import structlog

from flowedge.config.settings import Settings
from flowedge.scanner.schemas.catalyst import (
    EarningsEvent,
    ExpectedMove,
    InsiderTrade,
)
from flowedge.scanner.schemas.flow import DarkPoolTrade, FlowAlert
from flowedge.scanner.schemas.iv import IVRankData, TermStructurePoint
from flowedge.scanner.schemas.options import OptionsChain


class BaseProvider(ABC):
    """Base class for all data providers."""

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None
        self._logger = structlog.get_logger().bind(provider=self.name)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._settings.scanner_http_timeout
            )
        return self._client

    async def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated GET request."""
        client = await self._get_client()
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the provider connection is working."""
        ...


class OptionsChainProvider(BaseProvider):
    """Provider that can fetch options chains."""

    @abstractmethod
    async def get_options_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> OptionsChain: ...


class FlowAlertProvider(BaseProvider):
    """Provider that can fetch options flow data."""

    @abstractmethod
    async def get_flow_alerts(
        self, ticker: str | None = None
    ) -> list[FlowAlert]: ...

    @abstractmethod
    async def get_dark_pool_trades(
        self, ticker: str
    ) -> list[DarkPoolTrade]: ...


class IVDataProvider(BaseProvider):
    """Provider that can fetch IV rank and historical IV data."""

    @abstractmethod
    async def get_iv_rank(self, ticker: str) -> IVRankData: ...

    @abstractmethod
    async def get_historical_iv(
        self, ticker: str, days: int = 252
    ) -> list[TermStructurePoint]: ...

    @abstractmethod
    async def get_expected_move(
        self, ticker: str
    ) -> ExpectedMove | None: ...


class EarningsProvider(BaseProvider):
    """Provider that can fetch earnings calendar data."""

    @abstractmethod
    async def get_earnings_calendar(
        self, from_date: date, to_date: date
    ) -> list[EarningsEvent]: ...


class InsiderTradeProvider(BaseProvider):
    """Provider that can fetch insider trade filings."""

    @abstractmethod
    async def get_insider_trades(
        self, ticker: str, days_back: int = 90
    ) -> list[InsiderTrade]: ...


class EquityDataProvider(BaseProvider):
    """Provider for equity price data."""

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...
