"""Tests for the adaptive conviction scorer."""

from __future__ import annotations

from flowedge.scanner.backtest.adaptive_scorer import (
    ScorerFeatures,
    ScorerWeights,
    compute_adaptive_conviction,
    extract_features,
    update_weights_from_trades,
)
from flowedge.scanner.backtest.strategies import Indicators, MarketRegime


def _make_indicators(**kwargs: float) -> Indicators:
    defaults = {
        "close": 450.0, "sma20": 448.0, "sma50": 445.0,
        "rsi14": 55.0, "adx14": 30.0, "atr14": 5.0,
        "bb_upper": 460.0, "bb_mid": 450.0, "bb_lower": 440.0,
        "bb_width_pct": 4.4, "vol_ratio": 1.2, "range_ratio": 1.0,
        "high_20": 465.0, "low_20": 435.0, "high_40": 470.0,
        "low_40": 430.0, "momentum_5": 1.5, "momentum_10": 3.0,
        "atr_ratio": 1.0, "close_in_range": 0.6,
    }
    defaults.update(kwargs)
    return Indicators(**defaults)


# ── Feature Extraction ──


def test_extract_features_basic():
    ind = _make_indicators()
    bars = [{"high": 455, "low": 445, "close": 452, "volume": 1000000}]
    features = extract_features(
        "SPY", "bullish", "ibs_reversion", ind,
        MarketRegime.STRONG_UPTREND, bars,
    )
    assert features.ticker == "SPY"
    assert features.ticker_historical_wr == 0.75
    assert features.strategy_historical_wr == 0.50
    assert features.direction == "bullish"


def test_extract_features_bearish():
    ind = _make_indicators(sma20=440.0, sma50=445.0)
    bars = [{"high": 445, "low": 435, "close": 436, "volume": 1000000}]
    features = extract_features(
        "TSLA", "bearish", "trend_pullback", ind,
        MarketRegime.STRONG_DOWNTREND, bars,
    )
    assert features.ticker == "TSLA"
    assert features.direction == "bearish"
    assert features.trend_alignment > 0  # Bearish + SMA20 < SMA50 = aligned


def test_extract_features_unknown_ticker():
    ind = _make_indicators()
    features = extract_features(
        "UNKNOWN", "bullish", "ibs_reversion", ind,
        MarketRegime.STRONG_UPTREND, [],
    )
    assert features.ticker_historical_wr == 0.30  # Default


# ── Conviction Scoring ──


def test_conviction_spy_ibs():
    """SPY + IBS reversion should get high conviction (75% + 50% WR)."""
    features = ScorerFeatures(
        ticker="SPY",
        ticker_historical_wr=0.75,
        strategy="ibs_reversion",
        strategy_historical_wr=0.50,
        regime="strong_uptrend",
        regime_historical_wr=0.36,
        rsi14=20.0,  # Oversold
        adx14=30.0,
        ibs=0.08,  # Extreme low IBS
        volume_ratio=1.5,
        momentum_alignment=0.5,
        trend_alignment=1.0,
    )
    conviction, breakdown = compute_adaptive_conviction(features)
    assert conviction >= 8.0  # Should be high
    assert "ticker_wr" in breakdown
    assert breakdown["ticker_wr"] > 0


def test_conviction_amzn_trend():
    """AMZN + trend pullback should get lower conviction (10% + 26% WR)."""
    features = ScorerFeatures(
        ticker="AMZN",
        ticker_historical_wr=0.25,
        strategy="trend_pullback",
        strategy_historical_wr=0.31,
        regime="strong_uptrend",
        regime_historical_wr=0.36,
        rsi14=55.0,
        adx14=20.0,
        ibs=0.5,
        volume_ratio=0.8,
    )
    conviction, breakdown = compute_adaptive_conviction(features)
    assert conviction < 8.0  # Should be lower than SPY+IBS


def test_conviction_monotonic_with_ticker_wr():
    """Higher ticker WR should produce higher conviction."""
    base = ScorerFeatures(
        strategy="trend_pullback",
        strategy_historical_wr=0.30,
        regime="strong_uptrend",
        regime_historical_wr=0.35,
    )

    # SPY (75% WR) vs AMZN (25% WR)
    spy = ScorerFeatures(**{**base.__dict__, "ticker": "SPY", "ticker_historical_wr": 0.75})
    amzn = ScorerFeatures(**{**base.__dict__, "ticker": "AMZN", "ticker_historical_wr": 0.25})

    spy_conv, _ = compute_adaptive_conviction(spy)
    amzn_conv, _ = compute_adaptive_conviction(amzn)
    assert spy_conv > amzn_conv


def test_conviction_clamped():
    """Conviction should stay in 0-10 range."""
    extreme_good = ScorerFeatures(
        ticker_historical_wr=1.0,
        strategy_historical_wr=1.0,
        regime_historical_wr=1.0,
        rsi14=10.0,
        adx14=50.0,
        ibs=0.01,
        volume_ratio=3.0,
    )
    conv, _ = compute_adaptive_conviction(extreme_good)
    assert 0 <= conv <= 10

    extreme_bad = ScorerFeatures(
        ticker_historical_wr=0.0,
        strategy_historical_wr=0.0,
        regime_historical_wr=0.0,
    )
    conv2, _ = compute_adaptive_conviction(extreme_bad)
    assert 0 <= conv2 <= 10


def test_conviction_breakdown_has_all_components():
    features = ScorerFeatures()
    _, breakdown = compute_adaptive_conviction(features)
    expected_keys = [
        "ticker_wr", "strategy_wr", "regime_wr",
        "rsi_extreme", "adx_strength", "volume",
        "momentum", "trend_alignment", "ibs_extreme",
        "bias", "conviction",
    ]
    for key in expected_keys:
        assert key in breakdown, f"Missing {key} in breakdown"


# ── Weight Updates ──


def test_update_weights_basic():
    weights = ScorerWeights()
    trades = [
        {"ticker": "SPY", "strategy": "ibs_reversion",
         "regime": "strong_uptrend", "pnl_pct": 50.0}
    ] * 15 + [
        {"ticker": "AMZN", "strategy": "trend_pullback",
         "regime": "strong_downtrend", "pnl_pct": -40.0}
    ] * 10
    updated = update_weights_from_trades(trades, weights)
    assert updated.version == weights.version + 1
    assert updated.trained_on_trades == 25


def test_update_weights_insufficient_trades():
    weights = ScorerWeights()
    updated = update_weights_from_trades([], weights)
    assert updated.version == weights.version  # No change


def test_update_preserves_structure():
    weights = ScorerWeights(ticker_wr_weight=3.5)
    trades = [
        {"ticker": "SPY", "pnl_pct": 50.0, "strategy": "x", "regime": "y"}
    ] * 25
    updated = update_weights_from_trades(trades, weights)
    assert updated.ticker_wr_weight == 3.5  # Preserved
