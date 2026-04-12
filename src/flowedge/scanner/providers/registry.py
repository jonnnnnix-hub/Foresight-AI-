"""Provider registry — factory and lifecycle management."""

from __future__ import annotations

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.providers.alpaca import AlpacaProvider
from flowedge.scanner.providers.base import (
    BaseProvider,
    EarningsProvider,
    EquityDataProvider,
    FlowAlertProvider,
    InsiderTradeProvider,
    IVDataProvider,
    OptionsChainProvider,
)
from flowedge.scanner.providers.fmp import FMPProvider
from flowedge.scanner.providers.orats import OratsProvider
from flowedge.scanner.providers.polygon import PolygonProvider
from flowedge.scanner.providers.sec_edgar import SECEdgarProvider
from flowedge.scanner.providers.tradier import TradierProvider
from flowedge.scanner.providers.unusual_whales import UnusualWhalesProvider

logger = structlog.get_logger()


class ProviderRegistry:
    """Creates and manages data provider instances."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._providers: dict[str, BaseProvider] = {}

    def _get_or_create(
        self, name: str, factory: type[BaseProvider]
    ) -> BaseProvider:
        if name not in self._providers:
            self._providers[name] = factory(self._settings)
        return self._providers[name]

    def get_options_chain_provider(
        self, preferred: str = "orats"
    ) -> OptionsChainProvider:
        """Get options chain provider.

        Default: Orats (2,000+ strikes with full greeks and IV).
        Fallback: Tradier → Polygon.
        """
        if preferred == "orats" and self._settings.orats_api_key:
            p = self._get_or_create("orats", OratsProvider)
            assert isinstance(p, OptionsChainProvider)
            return p
        if preferred == "tradier" and self._settings.tradier_api_key:
            p = self._get_or_create("tradier", TradierProvider)
            assert isinstance(p, OptionsChainProvider)
            return p
        if self._settings.polygon_api_key:
            p = self._get_or_create("polygon", PolygonProvider)
            assert isinstance(p, OptionsChainProvider)
            return p
        # Final fallback to orats
        p = self._get_or_create("orats", OratsProvider)
        assert isinstance(p, OptionsChainProvider)
        return p

    def get_flow_provider(self) -> FlowAlertProvider:
        """Get options flow provider (Unusual Whales)."""
        p = self._get_or_create("unusual_whales", UnusualWhalesProvider)
        assert isinstance(p, FlowAlertProvider)
        return p

    def get_iv_provider(self) -> IVDataProvider:
        """Get IV data provider (Orats)."""
        p = self._get_or_create("orats", OratsProvider)
        assert isinstance(p, IVDataProvider)
        return p

    def get_earnings_provider(self) -> EarningsProvider:
        """Get earnings calendar provider (FMP)."""
        p = self._get_or_create("fmp", FMPProvider)
        assert isinstance(p, EarningsProvider)
        return p

    def get_insider_provider(self) -> InsiderTradeProvider:
        """Get insider trade provider (SEC EDGAR)."""
        p = self._get_or_create("sec_edgar", SECEdgarProvider)
        assert isinstance(p, InsiderTradeProvider)
        return p

    def get_equity_provider(self) -> EquityDataProvider:
        """Get equity data provider (Alpaca)."""
        p = self._get_or_create("alpaca", AlpacaProvider)
        assert isinstance(p, EquityDataProvider)
        return p

    async def close_all(self) -> None:
        """Close all provider HTTP clients."""
        for name, provider in self._providers.items():
            try:
                await provider.close()
            except Exception as e:
                logger.warning("provider_close_error", provider=name, error=str(e))
        self._providers.clear()
