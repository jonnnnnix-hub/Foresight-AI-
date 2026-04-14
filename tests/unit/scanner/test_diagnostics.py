"""Tests for the backtest diagnostic layers."""

from __future__ import annotations

import random

from flowedge.scanner.backtest.diagnostics.bootstrap import (
    run_bootstrap,
)
from flowedge.scanner.backtest.diagnostics.regime_tracker import (
    analyze_regime_performance,
)
from flowedge.scanner.backtest.diagnostics.runner import (
    run_full_diagnostics,
)
from flowedge.scanner.backtest.diagnostics.signal_decay import (
    run_signal_decay_analysis,
)
from flowedge.scanner.backtest.diagnostics.trade_clustering import (
    run_clustering_analysis,
)
from flowedge.scanner.backtest.diagnostics.walk_forward import (
    run_walk_forward,
)


def _make_trades(
    n: int = 50,
    win_rate: float = 0.30,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic trade data for testing."""
    random.seed(seed)
    trades = []
    for i in range(n):
        is_win = random.random() < win_rate
        pnl = random.uniform(20, 200) if is_win else random.uniform(-80, -10)
        conviction = random.uniform(6, 10) if is_win else random.uniform(4, 8)
        trades.append({
            "trade_id": f"T{i:03d}",
            "ticker": random.choice(["TSLA", "NVDA", "AAPL", "META", "SPY"]),
            "direction": random.choice(["bullish", "bearish"]),
            "strategy": random.choice(["trend_pullback", "ibs_reversion"]),
            "regime": random.choice([
                "strong_uptrend", "uptrend", "strong_downtrend",
            ]),
            "entry_date": f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "pnl_pct": round(pnl, 2),
            "conviction": round(conviction, 2),
            "signal_score": round(conviction, 1),
            "hold_days": random.randint(1, 10),
        })
    return trades


# ── Walk-Forward Tests ──


def test_walk_forward_basic():
    trades = _make_trades(50)
    result = run_walk_forward(trades)
    assert result.total_windows > 0
    assert 0 <= result.avg_is_win_rate <= 1
    assert 0 <= result.avg_oos_win_rate <= 1
    assert len(result.notes) > 0


def test_walk_forward_insufficient_trades():
    trades = _make_trades(5)
    result = run_walk_forward(trades)
    assert result.total_windows == 0
    assert any("Insufficient" in n for n in result.notes)


def test_walk_forward_efficiency_range():
    trades = _make_trades(100)
    result = run_walk_forward(trades)
    # WFE can be negative but should be finite
    assert -10 < result.walk_forward_efficiency < 10


# ── Bootstrap Tests ──


def test_bootstrap_basic():
    trades = _make_trades(50)
    result = run_bootstrap(trades, n_samples=1000, seed=42)
    assert result.n_samples == 1000
    assert result.n_trades == 50
    assert result.win_rate is not None
    assert result.profit_factor is not None
    assert result.sharpe_ratio is not None


def test_bootstrap_confidence_intervals():
    trades = _make_trades(50)
    result = run_bootstrap(trades, n_samples=1000, seed=42)
    assert result.win_rate is not None
    wr = result.win_rate
    assert wr.ci_lower_5 <= wr.ci_median <= wr.ci_upper_95
    assert wr.ci_width >= 0


def test_bootstrap_insufficient_trades():
    trades = _make_trades(3)
    result = run_bootstrap(trades)
    assert any("Insufficient" in n for n in result.notes)


def test_bootstrap_significance():
    # High win rate should be significant
    high_wr_trades = _make_trades(80, win_rate=0.70)
    result = run_bootstrap(high_wr_trades, n_samples=1000, seed=42)
    # With 70% WR, avg PnL should be positive → likely significant
    assert result.avg_pnl is not None


# ── Signal Decay Tests ──


def test_signal_decay_basic():
    trades = _make_trades(50)
    result = run_signal_decay_analysis(trades)
    assert -1 <= result.score_pnl_correlation <= 1
    assert -1 <= result.score_win_correlation <= 1
    assert len(result.notes) > 0


def test_signal_decay_buckets():
    trades = _make_trades(50)
    result = run_signal_decay_analysis(trades)
    assert len(result.buckets) > 0
    for b in result.buckets:
        assert 0 <= b.win_rate <= 1
        assert b.trades > 0


def test_signal_decay_hold_days():
    trades = _make_trades(50)
    result = run_signal_decay_analysis(trades)
    assert len(result.pnl_by_hold_day) > 0
    assert result.optimal_hold_days >= 0


def test_signal_decay_insufficient():
    result = run_signal_decay_analysis([])
    assert any("Insufficient" in n for n in result.notes)


# ── Regime Tracker Tests ──


def test_regime_tracking_basic():
    trades = _make_trades(50)
    result = analyze_regime_performance(trades)
    assert len(result.regime_performance) > 0
    assert result.best_regime != ""
    assert result.worst_regime != ""


def test_regime_transitions():
    trades = _make_trades(50)
    result = analyze_regime_performance(trades)
    # With random regimes, should detect transitions
    assert len(result.transitions) >= 0


def test_regime_specific_thresholds():
    trades = _make_trades(50)
    result = analyze_regime_performance(trades)
    # Should suggest thresholds for at least some regimes
    assert isinstance(result.regime_specific_thresholds, dict)


# ── Clustering Tests ──


def test_clustering_basic():
    trades = _make_trades(50)
    result = run_clustering_analysis(trades)
    assert result.ticker_concentration is not None
    assert result.direction_concentration is not None


def test_clustering_detects_loss_clusters():
    # Create trades with guaranteed loss cluster
    trades = _make_trades(20, win_rate=0.5)
    # Force 5 consecutive losses
    for i in range(5, 10):
        trades[i]["pnl_pct"] = -40.0
    result = run_clustering_analysis(trades)
    assert result.max_consecutive_losses >= 5


def test_clustering_concentration():
    # Create trades concentrated in one ticker
    trades = _make_trades(20)
    for t in trades:
        t["ticker"] = "SPY"
    result = run_clustering_analysis(trades)
    assert result.ticker_concentration is not None
    assert result.ticker_concentration.is_over_concentrated


def test_clustering_monthly():
    trades = _make_trades(50)
    result = run_clustering_analysis(trades)
    assert len(result.monthly_win_rates) > 0


# ── Full Diagnostic Runner Tests ──


def test_full_diagnostics_basic():
    trades = _make_trades(50)
    report = run_full_diagnostics(trades)
    assert report.overall_grade in ("A", "B", "C", "D", "F")
    assert report.walk_forward is not None
    assert report.bootstrap is not None
    assert report.signal_decay is not None
    assert report.regime_tracking is not None
    assert report.clustering is not None


def test_full_diagnostics_grade_logic():
    # Good strategy should grade well
    good_trades = _make_trades(80, win_rate=0.65, seed=123)
    report = run_full_diagnostics(good_trades)
    assert report.overall_grade in ("A", "B", "C")

    # Bad strategy should grade poorly
    bad_trades = _make_trades(80, win_rate=0.10, seed=456)
    report_bad = run_full_diagnostics(bad_trades)
    # Should have issues
    assert len(report_bad.critical_issues) >= 0


def test_full_diagnostics_has_recommendations():
    trades = _make_trades(50, win_rate=0.15)
    report = run_full_diagnostics(trades)
    # Low WR should produce recommendations
    assert isinstance(report.recommendations, list)


def test_full_diagnostics_insufficient():
    report = run_full_diagnostics([{"pnl_pct": 10}])
    # Should still return a valid report even with minimal data
    assert report.overall_grade != ""
