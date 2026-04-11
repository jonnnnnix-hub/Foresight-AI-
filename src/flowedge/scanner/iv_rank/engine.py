"""ORACLE — predicts when option premiums are historically mispriced."""

from __future__ import annotations

from datetime import datetime

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.providers.registry import ProviderRegistry
from flowedge.scanner.schemas.iv import IVRankData, IVRegime, IVSignal

logger = structlog.get_logger()


def classify_regime(iv_rank: float, settings: Settings) -> IVRegime:
    """Classify IV into a regime bucket."""
    if iv_rank < settings.iv_rank_low_threshold:
        return IVRegime.LOW
    if iv_rank < settings.iv_rank_high_threshold:
        return IVRegime.NORMAL
    if iv_rank < 90.0:
        return IVRegime.ELEVATED
    return IVRegime.EXTREME


def _is_cheap_premium(iv_data: IVRankData, settings: Settings) -> bool:
    """Determine if premiums are historically cheap."""
    if iv_data.iv_rank >= settings.iv_rank_low_threshold:
        return False
    # Extra confirmation: IV below recent HV means market is underpricing vol
    if iv_data.iv_hv_spread is not None and iv_data.iv_hv_spread < 0:
        return True
    # IV rank alone below threshold qualifies
    return iv_data.iv_rank < settings.iv_rank_low_threshold * 0.8


def _score_iv(
    iv_data: IVRankData,
    regime: IVRegime,
    is_cheap: bool,
    is_contango: bool,
    settings: Settings,
) -> float:
    """Score IV signal strength for lotto opportunity (0-10).

    Higher scores for cheap premium (low IV rank) because
    lottos need room for IV expansion.
    """
    score = 0.0

    # IV rank component (0-4 points) — lower is better for lottos
    if iv_data.iv_rank < 15:
        score += 4.0
    elif iv_data.iv_rank < 25:
        score += 3.0
    elif iv_data.iv_rank < settings.iv_rank_low_threshold:
        score += 2.0
    elif iv_data.iv_rank < 50:
        score += 1.0
    # High IV rank = expensive premium = bad for lottos

    # HV/IV spread component (0-3 points)
    if iv_data.iv_hv_spread is not None:
        if iv_data.iv_hv_spread < -0.10:
            score += 3.0  # IV way below HV — very cheap
        elif iv_data.iv_hv_spread < -0.05:
            score += 2.0
        elif iv_data.iv_hv_spread < 0:
            score += 1.0

    # Cheap premium bonus (0-2 points)
    if is_cheap:
        score += 2.0

    # Contango penalty (0-1 point)
    if not is_contango:
        score += 1.0  # Backwardation = near-term fear = potential catalyst

    return min(score, 10.0)


async def scan_iv(
    registry: ProviderRegistry,
    tickers: list[str],
    settings: Settings | None = None,
) -> list[IVSignal]:
    """Scan tickers for IV rank and classify volatility regime."""
    settings = settings or get_settings()
    iv_provider = registry.get_iv_provider()

    signals: list[IVSignal] = []

    for ticker in tickers:
        try:
            iv_data = await iv_provider.get_iv_rank(ticker)
            term_structure = await iv_provider.get_historical_iv(ticker)

            regime = classify_regime(iv_data.iv_rank, settings)
            is_cheap = _is_cheap_premium(iv_data, settings)

            # Check term structure for contango/backwardation
            is_contango = True
            if len(term_structure) >= 2:
                near = term_structure[0].iv
                far = term_structure[-1].iv
                is_contango = near <= far

            strength = _score_iv(iv_data, regime, is_cheap, is_contango, settings)

            rationale_parts = [
                f"IV Rank: {iv_data.iv_rank:.1f}%",
                f"Regime: {regime.value}",
            ]
            if iv_data.iv_hv_spread is not None:
                rationale_parts.append(f"IV-HV spread: {iv_data.iv_hv_spread:+.4f}")
            if is_cheap:
                rationale_parts.append("Premium is historically cheap")
            if not is_contango:
                rationale_parts.append("Term structure in backwardation")

            signals.append(
                IVSignal(
                    ticker=ticker,
                    iv_rank=iv_data,
                    term_structure=term_structure,
                    regime=regime,
                    is_cheap_premium=is_cheap,
                    is_contango=is_contango,
                    strength=strength,
                    rationale="; ".join(rationale_parts),
                    detected_at=datetime.now(),
                )
            )

        except Exception as e:
            logger.warning("iv_scan_failed", ticker=ticker, error=str(e))

    signals.sort(key=lambda s: s.strength, reverse=True)
    logger.info("iv_scan_complete", signal_count=len(signals))
    return signals
