"""SPECTER — detects invisible institutional options flow."""

from __future__ import annotations

from datetime import datetime

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.providers.registry import ProviderRegistry
from flowedge.scanner.schemas.flow import (
    FlowAlert,
    FlowSentiment,
    FlowType,
    UOASignal,
)

logger = structlog.get_logger()


def _classify_signal_type(alerts: list[FlowAlert], settings: Settings) -> str:
    """Determine the dominant signal type from a group of alerts."""
    sweeps = [a for a in alerts if a.flow_type == FlowType.SWEEP]
    blocks = [a for a in alerts if a.premium >= settings.uoa_min_premium]
    vol_spikes = [
        a for a in alerts if a.volume_oi_ratio >= settings.uoa_volume_oi_threshold
    ]

    if len(sweeps) >= 3:
        return "sweep_cluster"
    if blocks:
        return "block_trade"
    if vol_spikes:
        return "volume_spike"
    return "skew_shift"


def _compute_direction(alerts: list[FlowAlert]) -> FlowSentiment:
    """Determine net directional bias from flow alerts."""
    bullish = sum(1 for a in alerts if a.sentiment == FlowSentiment.BULLISH)
    bearish = sum(1 for a in alerts if a.sentiment == FlowSentiment.BEARISH)

    if bullish > bearish * 1.5:
        return FlowSentiment.BULLISH
    if bearish > bullish * 1.5:
        return FlowSentiment.BEARISH
    return FlowSentiment.NEUTRAL


def _score_uoa(
    alerts: list[FlowAlert], signal_type: str, settings: Settings
) -> float:
    """Score UOA signal strength from 0-10.

    Scores based on:
    - How many contracts have unusual volume/OI ratios (not just total count)
    - Concentration of premium in sweeps/blocks (quality over quantity)
    - Call/put skew (directional conviction)
    """
    if not alerts:
        return 0.0

    score = 0.0

    # 1. Volume/OI outlier count (0-3 points)
    # Only contracts where volume > threshold * OI are genuinely unusual
    unusual_count = sum(
        1 for a in alerts if a.volume_oi_ratio >= settings.uoa_volume_oi_threshold
    )
    if unusual_count >= 20:
        score += 3.0
    elif unusual_count >= 10:
        score += 2.0
    elif unusual_count >= 3:
        score += 1.0

    # 2. Block/sweep premium concentration (0-3 points)
    # Big money is sweeps + blocks with premium > threshold
    significant_alerts = [
        a for a in alerts if a.premium >= settings.uoa_min_premium
    ]
    sig_premium = sum(a.premium for a in significant_alerts)
    if sig_premium >= settings.uoa_min_premium * 40:
        score += 3.0
    elif sig_premium >= settings.uoa_min_premium * 20:
        score += 2.0
    elif sig_premium >= settings.uoa_min_premium * 5:
        score += 1.0

    # 3. Call/put skew (0-2 points) — strong directional conviction
    call_vol = sum(a.volume for a in alerts if a.option_type.value == "call")
    put_vol = sum(a.volume for a in alerts if a.option_type.value == "put")
    total_vol = call_vol + put_vol
    if total_vol > 0:
        ratio = call_vol / max(put_vol, 1)
        if ratio >= 3.0 or ratio <= 0.33:
            score += 2.0  # Extreme skew — strong conviction
        elif ratio >= 2.0 or ratio <= 0.5:
            score += 1.0  # Moderate skew

    # 4. Signal type bonus (0-2 points)
    if signal_type == "sweep_cluster":
        score += 2.0
    elif signal_type == "block_trade":
        score += 1.5
    elif signal_type == "volume_spike":
        score += 1.0

    return min(score, 10.0)


async def scan_uoa(
    registry: ProviderRegistry,
    tickers: list[str] | None = None,
    settings: Settings | None = None,
) -> list[UOASignal]:
    """Scan for unusual options activity.

    Fetches flow alerts, groups by ticker, detects patterns,
    and scores each signal.
    """
    settings = settings or get_settings()
    flow_provider = registry.get_flow_provider()

    # Fetch flow alerts
    if tickers:
        all_alerts: list[FlowAlert] = []
        for ticker in tickers:
            try:
                alerts = await flow_provider.get_flow_alerts(ticker)
                all_alerts.extend(alerts)
            except Exception as e:
                logger.warning("uoa_fetch_failed", ticker=ticker, error=str(e))
    else:
        all_alerts = await flow_provider.get_flow_alerts()

    if not all_alerts:
        return []

    # Group by ticker
    by_ticker: dict[str, list[FlowAlert]] = {}
    for alert in all_alerts:
        by_ticker.setdefault(alert.ticker, []).append(alert)

    signals: list[UOASignal] = []

    for ticker, alerts in by_ticker.items():
        signal_type = _classify_signal_type(alerts, settings)
        direction = _compute_direction(alerts)
        strength = _score_uoa(alerts, signal_type, settings)

        call_vol = sum(a.volume for a in alerts if a.option_type.value == "call")
        put_vol = sum(a.volume for a in alerts if a.option_type.value == "put")
        total_premium = sum(a.premium for a in alerts)

        # Fetch dark pool data for strong signals
        dark_pool_trades = []
        if strength >= 5.0:
            try:
                dark_pool_trades = await flow_provider.get_dark_pool_trades(ticker)
            except Exception as e:
                logger.debug("dark_pool_fetch_failed", ticker=ticker, error=str(e))

        rationale_parts = [
            f"{signal_type}: {len(alerts)} alerts",
            f"total premium ${total_premium:,.0f}",
        ]
        if call_vol + put_vol > 0:
            ratio = round(call_vol / put_vol, 2) if put_vol > 0 else float("inf")
            rationale_parts.append(f"C/P ratio {ratio}")

        signals.append(
            UOASignal(
                ticker=ticker,
                signal_type=signal_type,
                direction=direction,
                strength=strength,
                alerts=alerts,
                dark_pool_trades=dark_pool_trades,
                call_volume=call_vol,
                put_volume=put_vol,
                call_put_ratio=(
                    round(call_vol / put_vol, 2) if put_vol > 0 else 0.0
                ),
                total_premium=total_premium,
                rationale="; ".join(rationale_parts),
                detected_at=datetime.now(),
            )
        )

    signals.sort(key=lambda s: s.strength, reverse=True)
    logger.info("uoa_scan_complete", signal_count=len(signals))
    return signals
