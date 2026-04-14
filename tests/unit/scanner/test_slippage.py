"""Tests for the options slippage and bid-ask spread model."""

from __future__ import annotations

from flowedge.scanner.backtest.slippage import (
    SlippageModel,
    apply_entry_slippage,
    apply_exit_slippage,
    compute_roundtrip_cost,
    estimate_half_spread,
    estimate_portfolio_slippage,
)

# ── Half-Spread Estimation ──


def test_half_spread_basic():
    spread = estimate_half_spread(2.00, 0.02, "SPY")
    assert spread > 0
    assert spread < 1.0  # Should be reasonable


def test_half_spread_spy_tighter_than_amd():
    spy = estimate_half_spread(2.00, 0.02, "SPY")
    amd = estimate_half_spread(2.00, 0.02, "AMD")
    assert spy < amd  # SPY is more liquid


def test_half_spread_otm_wider():
    atm = estimate_half_spread(2.00, 0.005, "AAPL")  # Near ATM
    otm = estimate_half_spread(2.00, 0.05, "AAPL")  # 5% OTM
    assert otm > atm  # OTM should have wider spread


def test_half_spread_cheap_option_floor():
    spread = estimate_half_spread(0.05, 0.01, "SPY")
    assert spread >= 0.02  # Calibrated floor


def test_half_spread_disabled():
    model = SlippageModel(enabled=False)
    spread = estimate_half_spread(2.00, 0.02, "SPY", model)
    assert spread == 0.0


# ── Entry/Exit Fills ──


def test_entry_slippage_above_mid():
    mid = 2.00
    fill = apply_entry_slippage(mid, 0.02, "SPY")
    assert fill > mid  # Buy at ask (above mid)


def test_exit_slippage_below_mid():
    mid = 2.00
    fill = apply_exit_slippage(mid, 0.02, "SPY")
    assert fill < mid  # Sell at bid (below mid)


def test_exit_slippage_floor():
    fill = apply_exit_slippage(0.02, 0.05, "AMD")
    assert fill >= 0.01  # Never below $0.01


# ── Round-Trip Cost ──


def test_roundtrip_cost_positive():
    rt = compute_roundtrip_cost(2.00, 2.50, 0.02, "SPY", contracts=5)
    assert rt["total_slippage"] > 0
    assert rt["entry_fill"] > 2.00
    assert rt["exit_fill"] < 2.50
    assert rt["slippage_pct"] > 0


def test_roundtrip_cost_spy_cheaper_than_tsla():
    spy_rt = compute_roundtrip_cost(2.00, 2.50, 0.02, "SPY", contracts=5)
    tsla_rt = compute_roundtrip_cost(2.00, 2.50, 0.02, "TSLA", contracts=5)
    assert spy_rt["total_slippage"] < tsla_rt["total_slippage"]


def test_roundtrip_disabled():
    model = SlippageModel(enabled=False)
    rt = compute_roundtrip_cost(2.00, 2.50, 0.02, "SPY", model=model)
    assert rt["total_slippage"] == 0.0
    assert rt["entry_fill"] == 2.00
    assert rt["exit_fill"] == 2.50


# ── Portfolio Slippage ──


def test_portfolio_slippage_basic():
    trades = [
        {"entry_price": 2.0, "exit_price": 3.0, "ticker": "SPY",
         "contracts": 5, "otm_pct": 0.02},
        {"entry_price": 1.5, "exit_price": 0.5, "ticker": "TSLA",
         "contracts": 3, "otm_pct": 0.03},
    ]
    result = estimate_portfolio_slippage(trades)
    assert result["total_slippage"] > 0
    assert result["avg_per_trade"] > 0
    assert result["pnl_impact_pct"] > 0
    assert result["trade_count"] == 2.0


def test_portfolio_slippage_empty():
    result = estimate_portfolio_slippage([])
    assert result["total_slippage"] == 0.0


def test_portfolio_slippage_realistic_magnitude():
    """For a SPY option trade, slippage should be reasonable (1-5%)."""
    trades = [
        {"entry_price": 3.0, "exit_price": 4.0, "ticker": "SPY",
         "contracts": 2, "otm_pct": 0.015},
    ]
    result = estimate_portfolio_slippage(trades)
    # SPY round-trip slippage includes entry + exit spreads + market impact
    # For a $3 premium with 1.5% OTM, expect ~5-12% total round-trip cost
    assert 0.5 < result["pnl_impact_pct"] < 15.0
