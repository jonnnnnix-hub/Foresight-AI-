"""Offline PULSE momentum scoring for backtesting.

Mirrors the live PULSE engine logic but computes from historical bars
instead of making API calls. Scores momentum alignment on a 0-10 scale.

Factors scored:
- RSI position (oversold/overbought extremes)
- MACD histogram direction and crossover proximity
- Price vs SMA20/SMA50 (trend alignment)
- EMA9/EMA21 crossover
- Multi-indicator agreement (confluence bonus)
"""

from __future__ import annotations

from enum import StrEnum

from flowedge.scanner.backtest.strategies import Indicators


class MomentumBias(StrEnum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


def _ema(values: list[float], period: int) -> float:
    """Exponential moving average of the last values."""
    if len(values) < period:
        return values[-1] if values else 0.0

    k = 2.0 / (period + 1)
    ema_val = sum(values[:period]) / period

    for v in values[period:]:
        ema_val = v * k + ema_val * (1 - k)

    return ema_val


def _macd_histogram(closes: list[float]) -> float:
    """MACD histogram: EMA12 - EMA26 - signal(EMA9 of MACD)."""
    if len(closes) < 35:
        return 0.0

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26

    # Approximate signal line from recent MACD values
    macd_values: list[float] = []
    for i in range(max(0, len(closes) - 15), len(closes)):
        sub = closes[: i + 1]
        if len(sub) >= 26:
            e12 = _ema(sub, 12)
            e26 = _ema(sub, 26)
            macd_values.append(e12 - e26)

    signal = _ema(macd_values, 9) if len(macd_values) >= 9 else macd_line
    return macd_line - signal


def classify_momentum_bias(
    ind: Indicators,
    closes: list[float],
) -> tuple[MomentumBias, float]:
    """Classify momentum bias and score from indicators + close history.

    Returns (bias, score) where score is 0-10.
    Mirrors PULSE engine logic for offline backtesting.
    """
    bullish_signals = 0
    bearish_signals = 0
    total_signals = 4

    # 1. RSI position
    if ind.rsi14 > 60:
        bullish_signals += 1
    elif ind.rsi14 < 40:
        bearish_signals += 1

    # 2. MACD histogram
    macd_hist = _macd_histogram(closes)
    if macd_hist > 0:
        bullish_signals += 1
    elif macd_hist < 0:
        bearish_signals += 1

    # 3. Price vs SMA20
    if ind.close > ind.sma20:
        bullish_signals += 1
    elif ind.close < ind.sma20:
        bearish_signals += 1

    # 4. EMA9/EMA21 crossover
    ema9 = _ema(closes, 9) if len(closes) >= 9 else ind.close
    ema21 = _ema(closes, 21) if len(closes) >= 21 else ind.close
    if ema9 > ema21:
        bullish_signals += 1
    elif ema9 < ema21:
        bearish_signals += 1

    # Classify bias
    if bullish_signals >= total_signals:
        bias = MomentumBias.STRONG_BULLISH
    elif bearish_signals >= total_signals:
        bias = MomentumBias.STRONG_BEARISH
    elif bullish_signals > bearish_signals:
        bias = MomentumBias.BULLISH
    elif bearish_signals > bullish_signals:
        bias = MomentumBias.BEARISH
    else:
        bias = MomentumBias.NEUTRAL

    # Score momentum (0-10), mirrors PULSE _score_momentum()
    score = 0.0

    # Bias strength (0-4)
    if bias in (MomentumBias.STRONG_BULLISH, MomentumBias.STRONG_BEARISH):
        score += 4.0
    elif bias in (MomentumBias.BULLISH, MomentumBias.BEARISH):
        score += 2.5
    else:
        score += 1.0

    # Trend alignment bonus (0-3): strong bias = trend aligned
    if bias in (MomentumBias.STRONG_BULLISH, MomentumBias.STRONG_BEARISH):
        score += 3.0

    # RSI extremes (0-2)
    if ind.rsi14 < 30 or ind.rsi14 > 70:
        score += 2.0
    elif ind.rsi14 < 40 or ind.rsi14 > 60:
        score += 1.0

    # MACD crossover proximity (0-1)
    if abs(macd_hist) < abs(ind.close * 0.001):
        score += 1.0

    return bias, min(score, 10.0)


def momentum_direction_matches(
    bias: MomentumBias,
    direction: str,
) -> bool:
    """Check if momentum bias supports the trade direction."""
    if direction == "bullish":
        return bias in (MomentumBias.STRONG_BULLISH, MomentumBias.BULLISH)
    if direction == "bearish":
        return bias in (MomentumBias.STRONG_BEARISH, MomentumBias.BEARISH)
    return False


def compute_momentum_adjustment(
    bias: MomentumBias,
    momentum_score: float,
    direction: str,
) -> float:
    """Compute conviction adjustment based on momentum alignment.

    Returns a value from -2.0 to +2.0 that modifies entry conviction.
    Positive = momentum supports trade, negative = momentum opposes.
    """
    aligned = momentum_direction_matches(bias, direction)

    if aligned:
        if momentum_score >= 7.0:
            return 2.0  # Strong momentum confirmation
        if momentum_score >= 5.0:
            return 1.0  # Moderate confirmation
        return 0.5  # Weak confirmation

    # Momentum opposes direction — moderate penalty only
    # (Capped to prevent over-filtering correct-direction setups)
    opposing = False
    if direction == "bullish" and bias in (
        MomentumBias.STRONG_BEARISH,
        MomentumBias.BEARISH,
    ):
        opposing = True
    if direction == "bearish" and bias in (
        MomentumBias.STRONG_BULLISH,
        MomentumBias.BULLISH,
    ):
        opposing = True

    if opposing:
        if momentum_score >= 7.0:
            return -1.5  # Strong opposition — moderate penalty
        if momentum_score >= 5.0:
            return -1.0
        return -0.5

    return 0.0  # Neutral momentum — no adjustment
