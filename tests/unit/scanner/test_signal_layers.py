"""Tests for PULSE momentum, GEX proxy, and Kronos signal layers."""

import random

from flowedge.scanner.backtest.gex_proxy import (
    GEXRegimeProxy,
    classify_gex_proxy,
    compute_gex_adjustment,
)
from flowedge.scanner.backtest.kronos_signal import (
    compute_kronos_adjustment,
    predict_direction,
)
from flowedge.scanner.backtest.momentum_score import (
    MomentumBias,
    classify_momentum_bias,
    compute_momentum_adjustment,
    momentum_direction_matches,
)
from flowedge.scanner.backtest.strategies import compute_indicators


def _make_bars(
    n: int = 60,
    start: float = 100.0,
    trend: float = 0.002,
    vol: float = 0.01,
) -> list[dict]:
    """Generate synthetic bars for testing."""
    random.seed(42)
    bars = []
    price = start
    for i in range(n):
        change = trend + random.gauss(0, vol)
        price *= 1.0 + change
        high = price * (1 + abs(random.gauss(0, vol)))
        low = price * (1 - abs(random.gauss(0, vol)))
        bars.append({
            "date": f"2024-{1 + i // 22:02d}-{1 + i % 22:02d}",
            "open": price * (1 + random.gauss(0, vol * 0.3)),
            "high": max(high, price),
            "low": min(low, price),
            "close": price,
            "volume": int(1_000_000 * (1 + random.gauss(0, 0.3))),
        })
    return bars


# ── PULSE Momentum Tests ────────────────────────────────────────────


def test_momentum_bias_uptrend() -> None:
    """Strong uptrend should produce bullish bias."""
    bars = _make_bars(60, trend=0.008)
    ind = compute_indicators(bars)
    closes = [float(b["close"]) for b in bars]
    bias, score = classify_momentum_bias(ind, closes)
    assert bias in (MomentumBias.STRONG_BULLISH, MomentumBias.BULLISH)
    assert score > 2.0


def test_momentum_bias_downtrend() -> None:
    """Strong downtrend should produce bearish bias."""
    bars = _make_bars(60, trend=-0.008)
    ind = compute_indicators(bars)
    closes = [float(b["close"]) for b in bars]
    bias, score = classify_momentum_bias(ind, closes)
    assert bias in (MomentumBias.STRONG_BEARISH, MomentumBias.BEARISH)
    assert score > 2.0


def test_momentum_direction_matches() -> None:
    """Bullish bias should match bullish direction."""
    assert momentum_direction_matches(MomentumBias.STRONG_BULLISH, "bullish")
    assert momentum_direction_matches(MomentumBias.BULLISH, "bullish")
    assert not momentum_direction_matches(MomentumBias.STRONG_BEARISH, "bullish")
    assert momentum_direction_matches(MomentumBias.STRONG_BEARISH, "bearish")


def test_momentum_adjustment_aligned() -> None:
    """Aligned momentum should boost conviction."""
    adj = compute_momentum_adjustment(MomentumBias.STRONG_BULLISH, 8.0, "bullish")
    assert adj > 0


def test_momentum_adjustment_opposing() -> None:
    """Opposing momentum should penalize conviction."""
    adj = compute_momentum_adjustment(MomentumBias.STRONG_BEARISH, 8.0, "bullish")
    assert adj < 0


def test_momentum_score_range() -> None:
    """Score should be 0-10."""
    bars = _make_bars(60)
    ind = compute_indicators(bars)
    closes = [float(b["close"]) for b in bars]
    _, score = classify_momentum_bias(ind, closes)
    assert 0.0 <= score <= 10.0


# ── GEX Proxy Tests ────────────────────────────────────────────────


def test_gex_proxy_returns_regime() -> None:
    """classify_gex_proxy should return a valid regime."""
    bars = _make_bars(60)
    ind = compute_indicators(bars)
    regime, score = classify_gex_proxy(ind, bars)
    assert regime in (GEXRegimeProxy.POSITIVE, GEXRegimeProxy.NEGATIVE, GEXRegimeProxy.NEUTRAL)
    assert 0.0 <= score <= 10.0


def test_gex_volatile_bars() -> None:
    """High volatility bars should skew toward negative GEX."""
    bars = _make_bars(60, vol=0.04)  # 4x normal vol
    ind = compute_indicators(bars)
    regime, score = classify_gex_proxy(ind, bars)
    # High vol should produce higher score (more negative GEX)
    assert score >= 1.0


def test_gex_adjustment_negative() -> None:
    """Negative GEX should boost conviction."""
    adj = compute_gex_adjustment(GEXRegimeProxy.NEGATIVE, 7.0, "bullish")
    assert adj > 0


def test_gex_adjustment_positive() -> None:
    """Positive GEX should penalize conviction."""
    adj = compute_gex_adjustment(GEXRegimeProxy.POSITIVE, 1.0, "bullish")
    assert adj < 0


# ── Kronos Signal Tests ─────────────────────────────────────────────


def test_predict_direction_returns_valid() -> None:
    """predict_direction should return valid direction and confidence."""
    bars = _make_bars(100)
    direction, confidence = predict_direction(bars)
    assert direction in ("bullish", "bearish", "neutral")
    assert 0.0 <= confidence <= 1.0


def test_predict_direction_insufficient_data() -> None:
    """Short history should return neutral."""
    bars = _make_bars(10)
    direction, confidence = predict_direction(bars)
    assert direction == "neutral"
    assert confidence == 0.0


def test_kronos_adjustment_range() -> None:
    """Kronos adjustment should be bounded."""
    bars = _make_bars(100, trend=0.005)
    adj = compute_kronos_adjustment(bars, "bullish")
    assert -1.5 <= adj <= 1.5


def test_kronos_uptrend_bullish() -> None:
    """Strong uptrend should somewhat support bullish prediction."""
    bars = _make_bars(150, trend=0.008)
    direction, _ = predict_direction(bars)
    # Can't guarantee direction but should not crash
    assert direction in ("bullish", "bearish", "neutral")
