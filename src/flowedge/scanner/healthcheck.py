"""Startup health check — reports which APIs are connected and functional."""

from __future__ import annotations

from typing import Any

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.providers.registry import ProviderRegistry

logger = structlog.get_logger()


class ProviderStatus:
    """Health status for a single provider."""

    def __init__(self, name: str, configured: bool, healthy: bool, error: str = "") -> None:
        self.name = name
        self.configured = configured
        self.healthy = healthy
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "configured": self.configured,
            "healthy": self.healthy,
            "error": self.error,
        }


async def check_all_providers(
    settings: Settings | None = None,
) -> list[ProviderStatus]:
    """Check connectivity for all configured data providers."""
    settings = settings or get_settings()
    results: list[ProviderStatus] = []

    checks = [
        ("Polygon.io", bool(settings.polygon_api_key), "polygon"),
        ("Tradier", bool(settings.tradier_api_key), "tradier"),
        ("Alpaca", bool(settings.alpaca_api_key_id and settings.alpaca_api_secret_key), "alpaca"),
        ("Unusual Whales", bool(settings.unusual_whales_api_key), "unusual_whales"),
        ("Orats", bool(settings.orats_api_key), "orats"),
        ("FMP", bool(settings.fmp_api_key), "fmp"),
        ("SEC EDGAR", True, "sec_edgar"),  # No key needed
        ("Anthropic (CIPHER)", bool(settings.anthropic_api_key), "anthropic"),
    ]

    registry = ProviderRegistry(settings)

    for display_name, configured, provider_key in checks:
        if not configured:
            results.append(ProviderStatus(display_name, False, False, "No API key"))
            continue

        # Skip Anthropic — it's not a scanner provider
        if provider_key == "anthropic":
            results.append(ProviderStatus(display_name, True, True))
            continue

        try:
            provider_map = {
                "polygon": registry.get_options_chain_provider("polygon"),
                "tradier": registry.get_options_chain_provider("tradier"),
                "alpaca": registry.get_equity_provider(),
                "unusual_whales": registry.get_flow_provider(),
                "orats": registry.get_iv_provider(),
                "fmp": registry.get_earnings_provider(),
                "sec_edgar": registry.get_insider_provider(),
            }
            provider = provider_map.get(provider_key)
            if provider:
                healthy = await provider.health_check()
                results.append(
                    ProviderStatus(display_name, True, healthy,
                                   "" if healthy else "Health check failed")
                )
            else:
                results.append(ProviderStatus(display_name, True, False, "Provider not found"))
        except Exception as e:
            results.append(ProviderStatus(display_name, True, False, str(e)[:100]))

    await registry.close_all()
    return results


def log_provider_status(statuses: list[ProviderStatus]) -> None:
    """Log provider status at startup."""
    connected = sum(1 for s in statuses if s.healthy)
    total = len(statuses)

    logger.info(
        "provider_health_check",
        connected=connected,
        total=total,
    )
    for status in statuses:
        if status.healthy:
            logger.info("provider_ok", provider=status.name)
        elif status.configured:
            logger.warning(
                "provider_unhealthy",
                provider=status.name,
                error=status.error,
            )
        else:
            logger.info("provider_not_configured", provider=status.name)
