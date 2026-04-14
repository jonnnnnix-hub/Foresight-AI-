"""Offline GEX (Gamma Exposure) proxy for backtesting.

Synthesizes a GEX-like regime and score from historical price/volume data.
Real GEX requires live options chain data (OI + greeks per strike). For
backtesting, we approximate dealer gamma regime using observable price
behavior patterns that correlate with GEX states:

Negative GEX proxies (dealers amplify moves):
- High volume + large ranges = dealer hedging causes momentum
- Price far from recent mean = less gamma pinning
- Increasing ATR = volatility expanding (negative gamma effect)

Positive GEX proxies (dealers dampen moves):
- Low volume + small ranges = pinning near gamma-heavy strikes
- Price near recent mean = gamma pinning in effect
- Decreasing ATR = volatility contracting
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from flowedge.scanner.backtest.strategies import Indicators


class GEXRegimeProxy(StrEnum):
    POSITIVE = "positive"  # Dealers dampen moves — bad for lottos
    NEGATIVE = "negative"  # Dealers amplify moves — good for lottos
    NEUTRAL = "neutral"


def _compute_range_expansion(bars: list[dict[str, Any]], period: int = 10) -> float:
    """Ratio of recent range to average range. >1.3 = expanding."""
    if len(bars) < period + 5:
        return 1.0

    recent_ranges = []
    for b in bars[-period:]:
        h = float(b.get("high", 0))
        lo = float(b.get("low", 0))
        if lo > 0:
            recent_ranges.append((h - lo) / lo * 100)

    older_ranges = []
    for b in bars[-(period + 5) : -period]:
        h = float(b.get("high", 0))
        lo = float(b.get("low", 0))
        if lo > 0:
            older_ranges.append((h - lo) / lo * 100)

    if not recent_ranges or not older_ranges:
        return 1.0

    avg_recent = sum(recent_ranges) / len(recent_ranges)
    avg_older = sum(older_ranges) / len(older_ranges)

    return avg_recent / avg_older if avg_older > 0 else 1.0


def _compute_mean_distance(bars: list[dict[str, Any]], period: int = 20) -> float:
    """How far current price is from recent mean, as percentage.

    Large distance = less pinning = more likely negative GEX.
    """
    if len(bars) < period:
        return 0.0

    closes = [float(b.get("close", 0)) for b in bars[-period:]]
    mean = sum(closes) / len(closes)
    current = closes[-1]

    return abs(current - mean) / mean * 100 if mean > 0 else 0.0


def classify_gex_proxy(
    ind: Indicators,
    bars: list[dict[str, Any]],
) -> tuple[GEXRegimeProxy, float]:
    """Classify synthetic GEX regime from price/volume behavior.

    Returns (regime, score) where score is 0-10 for lotto favorability.
    Mirrors VORTEX _score_gex_for_lottos() logic with proxy inputs.
    """
    score = 0.0
    parts: list[str] = []

    # 1. Range expansion (proxy for dealer hedging activity)
    range_exp = _compute_range_expansion(bars)
    if range_exp > 1.5:
        score += 2.0
        parts.append("high_range_expansion")
    elif range_exp > 1.2:
        score += 1.0
        parts.append("moderate_range_expansion")

    # 2. ATR ratio (increasing ATR = negative GEX environment)
    if ind.atr_ratio > 1.3:
        score += 2.0
        parts.append("atr_expanding")
    elif ind.atr_ratio > 1.0:
        score += 0.5

    # 3. Volume surge (high volume = dealer hedging)
    if ind.vol_ratio > 2.0:
        score += 1.5
        parts.append("volume_surge")
    elif ind.vol_ratio > 1.5:
        score += 0.8
        parts.append("elevated_volume")

    # 4. Distance from mean (far = less pinning = negative GEX)
    mean_dist = _compute_mean_distance(bars)
    if mean_dist > 5.0:
        score += 2.0
        parts.append("far_from_mean")
    elif mean_dist > 3.0:
        score += 1.0
        parts.append("moderate_mean_distance")

    # 5. Bollinger Band position (near extremes = breakout zone)
    if ind.close > ind.bb_upper or ind.close < ind.bb_lower:
        score += 1.5
        parts.append("outside_bands")
    elif ind.bb_width_pct < 3.0:
        # Compressed bands = squeeze = potential negative gamma ahead
        score += 1.0
        parts.append("bb_squeeze")

    score = min(score, 10.0)

    # Classify regime
    if score >= 5.0:
        regime = GEXRegimeProxy.NEGATIVE
    elif score >= 2.5:
        regime = GEXRegimeProxy.NEUTRAL
    else:
        regime = GEXRegimeProxy.POSITIVE

    return regime, score


def compute_gex_adjustment(
    regime: GEXRegimeProxy,
    gex_score: float,
    direction: str,
) -> float:
    """Compute conviction adjustment based on GEX regime.

    Negative GEX = dealers amplify moves = good for directional lottos.
    Positive GEX = dealers dampen moves = bad for lottos (pinning).

    Returns adjustment from -1.5 to +1.5.
    """
    if regime == GEXRegimeProxy.NEGATIVE:
        # Dealers amplify moves — favorable for lottos
        if gex_score >= 7.0:
            return 1.5
        if gex_score >= 5.0:
            return 1.0
        return 0.5
    if regime == GEXRegimeProxy.POSITIVE:
        # Dealers dampen moves — unfavorable for lottos
        if gex_score <= 1.5:
            return -1.5
        return -1.0
    # Neutral
    return 0.0
