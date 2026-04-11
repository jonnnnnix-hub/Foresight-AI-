"""Integration test — full scanner pipeline with live APIs.

These tests call real APIs and require valid keys in .env.
Skip with: pytest -m 'not integration'
"""

from __future__ import annotations

import pytest

from flowedge.config.settings import Settings

# Skip all tests if no API keys configured
_settings = Settings()
_has_keys = bool(_settings.polygon_api_key and _settings.orats_api_key)
pytestmark = pytest.mark.skipif(
    not _has_keys,
    reason="Live API keys not configured — set in .env to run integration tests",
)


@pytest.mark.integration
async def test_orats_iv_rank_live() -> None:
    """Orats IV rank returns valid data for a real ticker."""
    from flowedge.scanner.providers.orats import OratsProvider

    provider = OratsProvider(_settings)
    try:
        iv = await provider.get_iv_rank("AAPL")
        assert iv.ticker == "AAPL"
        assert 0 <= iv.iv_rank <= 100
        assert iv.current_iv > 0
        assert iv.source == "orats"
    finally:
        await provider.close()


@pytest.mark.integration
async def test_polygon_options_chain_live() -> None:
    """Polygon returns a non-empty options chain."""
    from flowedge.scanner.providers.polygon import PolygonProvider

    provider = PolygonProvider(_settings)
    try:
        chain = await provider.get_options_chain("AAPL")
        assert chain.underlying == "AAPL"
        assert len(chain.contracts) > 0
        assert chain.source == "polygon"
    finally:
        await provider.close()


@pytest.mark.integration
async def test_fmp_earnings_calendar_live() -> None:
    """FMP returns earnings events."""
    from datetime import date, timedelta

    from flowedge.scanner.providers.fmp import FMPProvider

    provider = FMPProvider(_settings)
    try:
        today = date.today()
        events = await provider.get_earnings_calendar(
            today, today + timedelta(days=30)
        )
        assert len(events) > 0
        assert events[0].source == "fmp"
    finally:
        await provider.close()


@pytest.mark.integration
async def test_unusual_whales_flow_live() -> None:
    """Unusual Whales returns flow data for a real ticker."""
    from flowedge.scanner.providers.unusual_whales import UnusualWhalesProvider

    if not _settings.unusual_whales_api_key:
        pytest.skip("No Unusual Whales key")

    provider = UnusualWhalesProvider(_settings)
    try:
        alerts = await provider.get_flow_alerts("TSLA")
        # May be empty outside market hours, but should not error
        assert isinstance(alerts, list)
    finally:
        await provider.close()


@pytest.mark.integration
async def test_full_scan_pipeline_live() -> None:
    """End-to-end: scan 2 tickers through all engines and get scored results."""
    from flowedge.scanner.catalyst.engine import scan_catalysts
    from flowedge.scanner.iv_rank.engine import scan_iv
    from flowedge.scanner.providers.registry import ProviderRegistry
    from flowedge.scanner.scorer.engine import score_lottos
    from flowedge.scanner.uoa.engine import scan_uoa

    tickers = ["AAPL", "TSLA"]
    registry = ProviderRegistry(_settings)

    try:
        # Each engine should not crash
        iv_signals = await scan_iv(registry, tickers, _settings)
        assert len(iv_signals) > 0
        assert all(s.ticker in tickers for s in iv_signals)

        catalyst_signals = await scan_catalysts(registry, tickers, _settings)
        # May be empty if no catalysts, but should not crash

        uoa_signals = await scan_uoa(registry, tickers, _settings)
        # May be empty if no UW key

        # Scorer should always return results
        result = score_lottos(uoa_signals, iv_signals, catalyst_signals, _settings)
        assert result.scan_id != ""
        assert result.tickers_scanned > 0

        # Every opportunity should have valid scores
        for opp in result.opportunities:
            assert 0 <= opp.composite_score <= 10
            assert 0 <= opp.iv_score <= 10
    finally:
        await registry.close_all()


@pytest.mark.integration
async def test_health_check_reports_providers() -> None:
    """Health check correctly identifies configured vs unconfigured providers."""
    from flowedge.scanner.healthcheck import check_all_providers

    statuses = await check_all_providers(_settings)
    assert len(statuses) > 0

    # At minimum Polygon and Orats should be configured
    names = {s.name for s in statuses if s.configured}
    assert "Polygon.io" in names
    assert "Orats" in names
