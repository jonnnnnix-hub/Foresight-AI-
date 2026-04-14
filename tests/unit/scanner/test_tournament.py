"""Tests for the multi-model tournament engine."""

from __future__ import annotations

import random
from typing import Any

from flowedge.scanner.backtest.strategies import (
    Indicators,
    MarketRegime,
    compute_indicators,
    detect_regime,
)
from flowedge.scanner.tournament.engine import (
    _compute_max_drawdown,
    _compute_sharpe,
    run_tournament_on_bars,
)
from flowedge.scanner.tournament.models import (
    ScoreCategory,
    build_all_models,
    build_contrarian_edge,
    build_edge_core,
    build_flow_hunter,
    build_momentum_alpha,
    build_regime_chameleon,
    compute_category_scores,
)
from flowedge.scanner.tournament.schemas import (
    ConsensusEntry,
    ModelName,
    ModelPerformance,
    TournamentResult,
)

# ── Helpers ─────────────────────────────────────────────────────────


def _make_bars(
    n: int = 80,
    start: float = 100.0,
    trend: float = 0.002,
    vol: float = 0.01,
) -> list[dict[str, Any]]:
    """Generate synthetic bars for testing."""
    rng = random.Random(42)
    bars: list[dict[str, Any]] = []
    price = start
    for i in range(n):
        change = trend + rng.gauss(0, vol)
        price *= 1.0 + change
        high = price * (1 + abs(rng.gauss(0, vol)))
        low = price * (1 - abs(rng.gauss(0, vol)))
        bars.append({
            "date": f"2024-{1 + i // 22:02d}-{1 + i % 22:02d}",
            "open": price * (1 + rng.gauss(0, vol * 0.3)),
            "high": max(high, price),
            "low": min(low, price),
            "close": price,
            "volume": int(1_000_000 * (1 + rng.gauss(0, 0.3))),
        })
    return bars


def _make_indicators() -> Indicators:
    """Create a representative set of indicators for unit tests."""
    return Indicators(
        close=105.0,
        sma20=103.0,
        sma50=100.0,
        rsi14=55.0,
        atr14=2.5,
        adx14=28.0,
        bb_upper=108.0,
        bb_mid=103.0,
        bb_lower=98.0,
        bb_width_pct=9.7,
        vol_ratio=1.3,
        range_ratio=1.1,
        high_20=107.0,
        low_20=99.0,
        high_40=110.0,
        low_40=95.0,
        momentum_5=2.0,
        momentum_10=4.0,
        atr_ratio=0.9,
        close_in_range=0.7,
    )


# ── Schema Tests ───────────────────────────────────────────────────


def test_model_performance_schema() -> None:
    perf = ModelPerformance(
        model_name="edge_core",
        total_return_pct=15.5,
        sharpe_ratio=1.2,
        win_rate=0.45,
        total_trades=20,
    )
    assert perf.model_name == "edge_core"
    dumped = perf.model_dump_json()
    rebuilt = ModelPerformance.model_validate_json(dumped)
    assert rebuilt.total_return_pct == 15.5


def test_tournament_result_schema() -> None:
    result = TournamentResult(
        run_id="test123",
        tickers=["AAPL", "TSLA"],
        lookback_days=100,
        starting_capital=10_000.0,
    )
    assert result.run_id == "test123"
    dumped = result.model_dump_json()
    rebuilt = TournamentResult.model_validate_json(dumped)
    assert rebuilt.tickers == ["AAPL", "TSLA"]


def test_consensus_entry_schema() -> None:
    entry = ConsensusEntry(
        ticker="AAPL",
        date="2024-01-15",
        models_agreeing=["edge_core", "momentum_alpha"],
        models_disagreeing=["contrarian_edge"],
        consensus_score=75.0,
        consensus_level=2,
        direction="bullish",
    )
    assert entry.consensus_level == 2
    dumped = entry.model_dump_json()
    rebuilt = ConsensusEntry.model_validate_json(dumped)
    assert rebuilt.ticker == "AAPL"


# ── Model Definition Tests ─────────────────────────────────────────


def test_build_all_models() -> None:
    models = build_all_models(MarketRegime.UPTREND)
    assert len(models) == 5
    names = {m.name for m in models}
    assert ModelName.EDGE_CORE in names
    assert ModelName.MOMENTUM_ALPHA in names
    assert ModelName.FLOW_HUNTER in names
    assert ModelName.CONTRARIAN_EDGE in names
    assert ModelName.REGIME_CHAMELEON in names


def test_model_weights_sum_to_100() -> None:
    """Each model's weights should sum to 100."""
    for model in build_all_models(MarketRegime.SIDEWAYS):
        total = sum(model.weights.values())
        assert total == 100, f"{model.name} weights sum to {total}, expected 100"


def test_edge_core_balanced() -> None:
    model = build_edge_core()
    assert model.threshold == 72.0
    assert model.weights[ScoreCategory.TREND_STRUCTURE] == 15


def test_momentum_alpha_trend_heavy() -> None:
    model = build_momentum_alpha()
    assert model.threshold == 68.0
    assert model.weights[ScoreCategory.TREND_STRUCTURE] == 30
    assert model.hard_gates.get("adx14") == 25.0
    assert model.ema9_exit is True


def test_flow_hunter_volume_heavy() -> None:
    model = build_flow_hunter()
    assert model.weights[ScoreCategory.VOLUME_FLOW] == 40
    assert model.no_followthrough_days == 3


def test_contrarian_edge_inverts_rsi() -> None:
    model = build_contrarian_edge()
    assert model.invert_rsi_at_extremes is True
    assert model.skip_adx_above == 30.0
    assert model.threshold == 75.0


def test_regime_chameleon_adapts() -> None:
    """Chameleon should pick different weights per regime."""
    strong_trend = build_regime_chameleon(MarketRegime.STRONG_UPTREND)
    sideways = build_regime_chameleon(MarketRegime.SIDEWAYS)
    # In strong trend, should adopt MOMENTUM_ALPHA weights (trend=30)
    assert strong_trend.weights[ScoreCategory.TREND_STRUCTURE] == 30
    # In sideways, should adopt CONTRARIAN_EDGE weights (trend=5)
    assert sideways.weights[ScoreCategory.TREND_STRUCTURE] == 5


# ── Category Score Tests ───────────────────────────────────────────


def test_category_scores_range() -> None:
    """All category scores should be between 0 and 100."""
    ind = _make_indicators()
    scores = compute_category_scores(ind, MarketRegime.UPTREND)
    assert len(scores) == 8
    for cat, val in scores.items():
        assert 0.0 <= val <= 100.0, f"{cat} score {val} out of range"


def test_category_scores_from_bars() -> None:
    bars = _make_bars(80, trend=0.003)
    ind = compute_indicators(bars)
    regime = detect_regime(ind)
    scores = compute_category_scores(ind, regime)
    assert len(scores) == 8
    # Uptrending bars should have decent trend_structure score
    assert scores[ScoreCategory.TREND_STRUCTURE] > 30


# ── Scoring Tests ──────────────────────────────────────────────────


def test_score_setup_returns_0_to_100() -> None:
    ind = _make_indicators()
    for model in build_all_models(MarketRegime.UPTREND):
        score = model.score_setup(ind, MarketRegime.UPTREND)
        assert 0.0 <= score <= 100.0, f"{model.name} score {score} out of range"


def test_models_produce_different_scores() -> None:
    """Different models should produce meaningfully different scores."""
    ind = _make_indicators()
    regime = MarketRegime.UPTREND
    scores = {
        m.name.value: m.score_setup(ind, regime)
        for m in build_all_models(regime)
    }
    unique_scores = set(scores.values())
    # At least 3 distinct scores among 5 models
    assert len(unique_scores) >= 3, f"Too few distinct scores: {scores}"


def test_should_enter_respects_threshold() -> None:
    model = build_edge_core()
    ind = _make_indicators()
    # Score below threshold → no entry
    assert model.should_enter(50.0, ind) is False
    # Score above threshold → entry
    assert model.should_enter(80.0, ind) is True


def test_momentum_alpha_hard_gate() -> None:
    """MOMENTUM_ALPHA requires ADX > 25."""
    model = build_momentum_alpha()
    ind = _make_indicators()
    # ADX=28 → should pass gate
    assert model.should_enter(80.0, ind) is True
    # Low ADX → should fail gate
    low_adx = Indicators(
        close=105.0, sma20=103.0, sma50=100.0, rsi14=55.0, atr14=2.5,
        adx14=15.0,  # Below gate of 25
        bb_upper=108.0, bb_mid=103.0, bb_lower=98.0, bb_width_pct=9.7,
        vol_ratio=1.3, range_ratio=1.1, high_20=107.0, low_20=99.0,
        high_40=110.0, low_40=95.0, momentum_5=2.0, momentum_10=4.0,
        atr_ratio=0.9, close_in_range=0.7,
    )
    assert model.should_enter(80.0, low_adx) is False


def test_contrarian_skip_high_adx() -> None:
    """CONTRARIAN_EDGE skips ADX > 30."""
    model = build_contrarian_edge()
    high_adx = Indicators(
        close=105.0, sma20=103.0, sma50=100.0, rsi14=55.0, atr14=2.5,
        adx14=35.0,  # Above skip threshold of 30
        bb_upper=108.0, bb_mid=103.0, bb_lower=98.0, bb_width_pct=9.7,
        vol_ratio=1.3, range_ratio=1.1, high_20=107.0, low_20=99.0,
        high_40=110.0, low_40=95.0, momentum_5=2.0, momentum_10=4.0,
        atr_ratio=0.9, close_in_range=0.7,
    )
    assert model.should_enter(80.0, high_adx) is False


# ── Performance Helper Tests ───────────────────────────────────────


def test_compute_max_drawdown_empty() -> None:
    assert _compute_max_drawdown([]) == 0.0


def test_compute_max_drawdown_no_drawdown() -> None:
    values = [("d1", 100.0), ("d2", 110.0), ("d3", 120.0)]
    assert _compute_max_drawdown(values) == 0.0


def test_compute_max_drawdown_basic() -> None:
    values = [("d1", 100.0), ("d2", 80.0), ("d3", 90.0)]
    dd = _compute_max_drawdown(values)
    assert dd == 20.0  # 20% drawdown from 100 to 80


def test_compute_sharpe_short_series() -> None:
    assert _compute_sharpe([("d1", 100.0)]) == 0.0


# ── Tournament Engine Tests ────────────────────────────────────────


def test_tournament_empty_bars() -> None:
    result = run_tournament_on_bars({})
    assert result.run_id != ""
    assert result.model_results == {}


def test_tournament_synthetic_data() -> None:
    """Run a full tournament on synthetic data."""
    bars = _make_bars(120, trend=0.003, vol=0.015)
    result = run_tournament_on_bars(
        {"SYNTH": bars},
        starting_capital=10_000.0,
        max_positions=3,
    )
    assert isinstance(result, TournamentResult)
    assert result.tickers == ["SYNTH"]
    # Should have results for all 5 models
    assert len(result.model_results) == 5
    for mn in [
        ModelName.EDGE_CORE,
        ModelName.MOMENTUM_ALPHA,
        ModelName.FLOW_HUNTER,
        ModelName.CONTRARIAN_EDGE,
        ModelName.REGIME_CHAMELEON,
    ]:
        assert mn.value in result.model_results
    # Ensemble should exist
    assert result.ensemble_result.model_name == ModelName.ENSEMBLE.value
    # Rankings should have 5 entries
    assert len(result.ranking_by_return) == 5


def test_tournament_multiple_tickers() -> None:
    """Tournament should handle multiple tickers."""
    bars_a = _make_bars(100, start=100.0, trend=0.003)
    bars_b = _make_bars(100, start=50.0, trend=-0.002)
    result = run_tournament_on_bars(
        {"TICK_A": bars_a, "TICK_B": bars_b},
        starting_capital=10_000.0,
    )
    assert len(result.tickers) == 2
    assert len(result.model_results) == 5


def test_tournament_consensus_tracking() -> None:
    """Consensus entries should be tracked."""
    bars = _make_bars(120, trend=0.004, vol=0.02)
    result = run_tournament_on_bars({"TEST": bars}, starting_capital=10_000.0)
    # Consensus entries list should exist (may be empty if no signals triggered)
    assert isinstance(result.consensus_entries, list)
    for ce in result.consensus_entries:
        assert 0 <= ce.consensus_level <= 5
        assert ce.ticker == "TEST"


def test_model_name_enum() -> None:
    assert ModelName.EDGE_CORE.value == "edge_core"
    assert ModelName.ENSEMBLE.value == "ensemble"
    assert len(ModelName) == 6
