"""Tests for walk-forward validation and regime detection."""

from __future__ import annotations

from datetime import date, timedelta

from flowedge.scanner.learning.schemas import AdaptiveWeights
from flowedge.scanner.learning.walk_forward import (
    WalkForwardResult,
    detect_regime,
    find_optimal_weights,
    regime_adjusted_weights,
    run_walk_forward,
)
from flowedge.scanner.performance.schemas import SimulatedTrade, TradeResult


def _make_trades(
    n: int,
    win_rate: float = 0.3,
    start: date = date(2026, 1, 1),
) -> list[SimulatedTrade]:
    """Generate a list of simulated trades."""
    trades: list[SimulatedTrade] = []
    for i in range(n):
        is_win = (i % int(1 / win_rate)) == 0 if win_rate > 0 else False
        trades.append(SimulatedTrade(
            trade_id=f"T{i:04d}",
            ticker="TSLA" if i % 2 == 0 else "NVDA",
            entry_date=start + timedelta(days=i * 3),
            exit_date=start + timedelta(days=i * 3 + 5),
            option_type="call" if i % 3 != 0 else "put",
            entry_underlying=100.0,
            exit_underlying=105.0 if is_win else 95.0,
            entry_premium=2.0,
            cost_basis=200.0,
            nexus_score=min(100, 60 + i) if is_win else min(100, 30 + i),
            pnl_pct=50.0 if is_win else -40.0,
            pnl_dollars=100.0 if is_win else -80.0,
            result=TradeResult.WIN if is_win else TradeResult.LOSS,
            hold_days=5,
        ))
    return trades


# ── Walk-Forward Tests ──

def test_walk_forward_basic():
    trades = _make_trades(40, win_rate=0.25)
    result = run_walk_forward(trades, n_windows=3)
    assert isinstance(result, WalkForwardResult)
    assert len(result.windows) >= 1


def test_walk_forward_insufficient_data():
    trades = _make_trades(5)
    result = run_walk_forward(trades)
    assert len(result.windows) == 0
    assert not result.is_overfit


def test_walk_forward_efficiency_computed():
    trades = _make_trades(50, win_rate=0.3)
    result = run_walk_forward(trades, n_windows=4)
    result.compute()
    # WFE should be between 0 and some value
    assert result.walkforward_efficiency >= 0.0


# ── Regime Detection Tests ──

def test_detect_regime_bullish():
    """When most wins are calls, should detect bullish."""
    trades = []
    for i in range(20):
        trades.append(SimulatedTrade(
            trade_id=f"T{i}",
            ticker="TSLA",
            entry_date=date(2026, 1, 1) + timedelta(days=i),
            option_type="call",
            pnl_pct=50.0 if i % 2 == 0 else -30.0,
            pnl_dollars=50.0 if i % 2 == 0 else -30.0,
            result=TradeResult.WIN if i % 2 == 0 else TradeResult.LOSS,
        ))
    regime = detect_regime(trades)
    # Should detect bullish since all wins are calls
    assert regime.label in ("bullish", "mixed")


def test_detect_regime_insufficient_data():
    trades = _make_trades(2)
    regime = detect_regime(trades)
    assert regime.label == "unknown"


def test_detect_regime_volatile():
    """Wide P&L swings should detect volatile."""
    trades = []
    for i in range(20):
        pnl = 200.0 if i % 2 == 0 else -150.0
        trades.append(SimulatedTrade(
            trade_id=f"T{i}",
            ticker="TSLA",
            entry_date=date(2026, 1, 1) + timedelta(days=i),
            option_type="call",
            pnl_pct=pnl,
            pnl_dollars=pnl,
            result=TradeResult.WIN if pnl > 0 else TradeResult.LOSS,
        ))
    regime = detect_regime(trades)
    assert regime.label == "volatile"


# ── Regime-Adaptive Weight Tests ──

def test_regime_weights_bullish():
    w = AdaptiveWeights(uoa_weight=0.33)
    adj = regime_adjusted_weights(
        w, type("R", (), {"label": "bullish", "confidence": 0.8})()
    )
    # Should suggest increasing UOA weight
    uoa_adj = [a for a in adj if a.parameter == "uoa_weight"]
    assert len(uoa_adj) == 1
    assert uoa_adj[0].suggested_value > w.uoa_weight


def test_regime_weights_volatile():
    w = AdaptiveWeights(iv_weight=0.30)
    adj = regime_adjusted_weights(
        w, type("R", (), {"label": "volatile", "confidence": 0.7})()
    )
    iv_adj = [a for a in adj if a.parameter == "iv_weight"]
    assert len(iv_adj) == 1
    assert iv_adj[0].suggested_value > w.iv_weight


def test_regime_weights_unknown_no_changes():
    w = AdaptiveWeights()
    adj = regime_adjusted_weights(
        w, type("R", (), {"label": "unknown", "confidence": 0.1})()
    )
    assert len(adj) == 0


# ── Optimal Weight Finder Tests ──

def test_find_optimal_weights_basic():
    trades = _make_trades(30, win_rate=0.3)
    result = find_optimal_weights(trades)
    assert "uoa" in result
    assert "iv" in result
    assert "catalyst" in result
    # Weights should sum to ~1.0
    total = result["uoa"] + result["iv"] + result["catalyst"]
    assert abs(total - 1.0) < 0.02


def test_find_optimal_weights_insufficient():
    trades = _make_trades(5)
    result = find_optimal_weights(trades)
    assert result["uoa"] == 0.35  # Defaults
