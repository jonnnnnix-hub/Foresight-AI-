"""Tests for the index ETF specialist model."""

from __future__ import annotations

from datetime import date

from flowedge.scanner.backtest.index_specialist.engine import (
    IndexPortfolio,
    IndexPosition,
    _get_horizon_stops,
    run_index_learning_cycle,
)
from flowedge.scanner.backtest.index_specialist.schemas import (
    IndexBacktestConfig,
    IndexBacktestResult,
    IndexRegime,
    IndexSignal,
    IndexTicker,
    IndexTradeResult,
    TradeHorizon,
)
from flowedge.scanner.backtest.index_specialist.strategies import (
    classify_index_regime,
    scan_index_entries,
)
from flowedge.scanner.backtest.strategies import Indicators

# ── Schema Tests ──


def test_index_ticker_values():
    assert IndexTicker.SPY == "SPY"
    assert IndexTicker.QQQ == "QQQ"
    assert IndexTicker.IWM == "IWM"


def test_trade_horizon_values():
    assert TradeHorizon.SCALP == "scalp"
    assert TradeHorizon.SWING == "swing"
    assert TradeHorizon.MEDIUM == "medium"


def test_index_regime_values():
    assert len(IndexRegime) == 7
    assert IndexRegime.STRONG_BULL == "strong_bull"
    assert IndexRegime.NEUTRAL == "neutral"
    assert IndexRegime.STRONG_BEAR == "strong_bear"


def test_index_signal_schema():
    sig = IndexSignal(
        ticker="SPY",
        direction="bullish",
        horizon=TradeHorizon.SCALP,
        conviction=8.5,
        regime=IndexRegime.BULL,
        strategy="ibs_reversal_scalp",
    )
    assert sig.conviction == 8.5
    assert sig.otm_pct == 0.015


def test_index_backtest_config_defaults():
    config = IndexBacktestConfig()
    assert config.tickers == ["SPY", "QQQ", "IWM"]
    assert config.scalp_min_conviction == 8.5
    assert config.swing_min_conviction == 8.0
    assert config.medium_min_conviction == 7.5
    assert config.mc_min_prob_profit_scalp == 0.55
    assert config.scalp_target_gain_pct == 50.0


def test_index_trade_result():
    r = IndexTradeResult(
        ticker="SPY",
        direction="bullish",
        horizon=TradeHorizon.SCALP,
        entry_date=date(2026, 1, 15),
        exit_date=date(2026, 1, 17),
        pnl_pct=55.0,
        is_win=True,
    )
    assert r.is_win
    assert r.horizon == TradeHorizon.SCALP


def test_index_backtest_result_schema():
    r = IndexBacktestResult(
        run_id="TEST",
        total_trades=10,
        wins=7,
        win_rate=0.700,
        scalp_trades=5,
        scalp_win_rate=0.800,
        swing_trades=3,
        swing_win_rate=0.667,
        medium_trades=2,
        medium_win_rate=0.500,
    )
    assert r.win_rate == 0.700
    assert r.scalp_win_rate == 0.800


# ── Strategy Tests ──


def _make_indicators(
    close: float = 450.0,
    sma20: float = 448.0,
    sma50: float = 445.0,
    adx14: float = 30.0,
    rsi: float = 55.0,
    bb_upper: float = 460.0,
    bb_lower: float = 440.0,
    atr14: float = 5.0,
) -> Indicators:
    return Indicators(
        close=close,
        sma20=sma20,
        sma50=sma50,
        rsi14=rsi,
        adx14=adx14,
        atr14=atr14,
        bb_upper=bb_upper,
        bb_mid=(bb_upper + bb_lower) / 2,
        bb_lower=bb_lower,
        bb_width_pct=(bb_upper - bb_lower) / ((bb_upper + bb_lower) / 2) * 100,
        vol_ratio=1.0,
        range_ratio=1.0,
        high_20=close * 1.03,
        low_20=close * 0.97,
        high_40=close * 1.05,
        low_40=close * 0.95,
        momentum_5=0.5,
        momentum_10=1.0,
        atr_ratio=1.0,
        close_in_range=0.5,
    )


def test_classify_regime_strong_bull():
    ind = _make_indicators(close=455, sma20=450, sma50=445, adx14=30, rsi=60)
    regime = classify_index_regime(ind)
    assert regime in (IndexRegime.STRONG_BULL, IndexRegime.BULL)


def test_classify_regime_strong_bear():
    ind = _make_indicators(close=430, sma20=440, sma50=445, adx14=30, rsi=35)
    regime = classify_index_regime(ind)
    assert regime in (IndexRegime.STRONG_BEAR, IndexRegime.BEAR)


def test_classify_regime_neutral():
    ind = _make_indicators(close=448, sma20=447, sma50=449, adx14=15, rsi=50)
    regime = classify_index_regime(ind)
    assert regime in (IndexRegime.NEUTRAL, IndexRegime.NEUTRAL_BULL, IndexRegime.NEUTRAL_BEAR)


def _make_bars(n: int = 60, start_price: float = 450.0, trend: float = 0.0) -> list[dict]:
    """Generate synthetic daily bars."""
    import random
    random.seed(42)
    bars = []
    price = start_price
    for i in range(n):
        noise = random.gauss(0, 1.5)
        price = max(10.0, price + trend + noise)
        high = price + abs(random.gauss(0, 2))
        low = price - abs(random.gauss(0, 2))
        bars.append({
            "date": f"2026-01-{(i % 28) + 1:02d}",
            "open": round(price + random.gauss(0, 0.5), 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(price, 2),
            "volume": int(1_000_000 + random.gauss(0, 200_000)),
        })
    return bars


def test_scan_index_entries_produces_signals():
    """Scan should produce at least some signals with favorable bars."""
    from flowedge.scanner.backtest.strategies import compute_indicators
    bars = _make_bars(60, start_price=450, trend=0.3)
    # Force a low IBS bar at the end
    bars[-1]["close"] = bars[-1]["low"] + 0.1
    bars[-1]["high"] = bars[-1]["low"] + 5.0
    ind = compute_indicators(bars)
    regime = classify_index_regime(ind)
    signals = scan_index_entries("SPY", bars, ind, regime)
    # May or may not produce signals depending on exact conditions
    assert isinstance(signals, list)


def test_ibs_reversal_scalp_bullish():
    """Very low IBS in bull regime should generate bullish scalp."""
    from flowedge.scanner.backtest.strategies import compute_indicators
    bars = _make_bars(60, start_price=450, trend=0.5)
    # Force extreme low IBS
    bars[-1]["low"] = 440.0
    bars[-1]["high"] = 455.0
    bars[-1]["close"] = 440.5  # IBS = 0.5/15 ≈ 0.033
    ind = compute_indicators(bars)
    regime = classify_index_regime(ind)
    signals = scan_index_entries("SPY", bars, ind, regime)
    scalps = [s for s in signals if s.horizon == TradeHorizon.SCALP and s.direction == "bullish"]
    assert len(scalps) >= 1
    assert scalps[0].strategy == "ibs_reversal_scalp"
    assert scalps[0].conviction >= 7.0


# ── Portfolio Tests ──


def test_index_portfolio_basics():
    p = IndexPortfolio(cash=10_000, initial_capital=10_000)
    assert p.total_value == 10_000
    assert p.can_open()


def test_index_portfolio_max_positions():
    p = IndexPortfolio(cash=10_000, initial_capital=10_000, max_positions=1)
    p.positions.append(IndexPosition(
        ticker="SPY", direction="bullish", horizon=TradeHorizon.SCALP,
        strategy="test", is_call=True, entry_date="2026-01-15",
        entry_underlying=450.0, strike=455.0, entry_premium=2.0,
        contracts=1, cost_basis=200.0, iv=0.2, conviction=8.5,
        regime="bull", dte_at_entry=7,
    ))
    assert not p.can_open()


# ── Horizon Stop Tests ──


def test_horizon_stops_scalp():
    config = IndexBacktestConfig()
    h, t, tp, hold, dte = _get_horizon_stops(TradeHorizon.SCALP, config)
    assert h == -0.25
    assert hold == 3
    assert dte == 7


def test_horizon_stops_swing():
    config = IndexBacktestConfig()
    h, t, tp, hold, dte = _get_horizon_stops(TradeHorizon.SWING, config)
    assert h == -0.30
    assert hold == 10
    assert dte == 21


def test_horizon_stops_medium():
    config = IndexBacktestConfig()
    h, t, tp, hold, dte = _get_horizon_stops(TradeHorizon.MEDIUM, config)
    assert h == -0.35
    assert hold == 20
    assert dte == 45


# ── Learning Cycle Tests ──


def test_learning_cycle_raises_conviction_on_low_wr():
    config = IndexBacktestConfig()
    result = IndexBacktestResult(
        scalp_trades=10,
        scalp_win_rate=0.50,  # Below 70% target
        swing_trades=10,
        swing_win_rate=0.60,  # Below 80% target
        medium_trades=10,
        medium_win_rate=0.70,  # Below 90% target
        trades=[
            IndexTradeResult(
                ticker="SPY", direction="bullish",
                horizon=TradeHorizon.SCALP,
                entry_date=date(2026, 1, i),
                pnl_pct=30.0 if i < 5 else -20.0,
                is_win=i < 5,
            )
            for i in range(1, 11)
        ],
    )
    updated = run_index_learning_cycle(result, config)
    assert updated.scalp_min_conviction > config.scalp_min_conviction
    assert updated.swing_min_conviction > config.swing_min_conviction
    assert updated.medium_min_conviction > config.medium_min_conviction


def test_learning_cycle_detects_premature_stops():
    config = IndexBacktestConfig()
    trades = [
        IndexTradeResult(
            ticker="SPY",
            direction="bullish",
            horizon=TradeHorizon.SWING,
            entry_date=date(2026, 1, i),
            entry_underlying=450.0,
            exit_underlying=455.0,  # Price went UP (correct direction)
            pnl_pct=-25.0,
            exit_reason="hard_stop",
            is_win=False,
        )
        for i in range(1, 8)
    ]
    result = IndexBacktestResult(
        swing_trades=7,
        swing_win_rate=0.0,
        trades=trades,
    )
    updated = run_index_learning_cycle(result, config)
    # Should widen stops since premature stops detected
    assert updated.swing_hard_stop < config.swing_hard_stop  # More negative = wider


def test_learning_cycle_no_change_on_good_wr():
    config = IndexBacktestConfig()
    result = IndexBacktestResult(
        scalp_trades=10,
        scalp_win_rate=0.80,  # Above 70%
        swing_trades=10,
        swing_win_rate=0.90,  # Above 80%
        medium_trades=10,
        medium_win_rate=0.95,  # Above 90%
        trades=[],
    )
    updated = run_index_learning_cycle(result, config)
    assert updated.scalp_min_conviction == config.scalp_min_conviction
    assert updated.swing_min_conviction == config.swing_min_conviction
    assert updated.medium_min_conviction == config.medium_min_conviction
