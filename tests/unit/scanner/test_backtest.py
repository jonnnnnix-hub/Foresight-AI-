"""Tests for backtest engine v2 — pricing, strategies, and portfolio."""

from flowedge.scanner.backtest.pricing import (
    bs_price,
    estimate_iv_from_atr,
    estimate_premium,
)
from flowedge.scanner.backtest.schemas import BacktestResult, BacktestTrade
from flowedge.scanner.backtest.strategies import (
    EntrySignal,
    MarketRegime,
    compute_indicators,
    detect_regime,
    scan_for_entries,
)

# ── Black-Scholes Pricing Tests ──────────────────────────────────────


def test_bs_call_atm() -> None:
    """ATM call with 30% vol should have meaningful price."""
    price = bs_price(100.0, 100.0, 30 / 252, 0.05, 0.30, is_call=True)
    assert 2.0 < price < 6.0


def test_bs_put_atm() -> None:
    """ATM put should also have meaningful price."""
    price = bs_price(100.0, 100.0, 30 / 252, 0.05, 0.30, is_call=False)
    assert 1.5 < price < 5.5


def test_bs_deep_otm_cheap() -> None:
    """Deep OTM call (10% away) should be very cheap."""
    price = bs_price(100.0, 110.0, 10 / 252, 0.05, 0.30, is_call=True)
    assert price < 0.50


def test_bs_put_call_parity() -> None:
    """Put-call parity: C - P = S - K*exp(-rT)."""
    from math import exp

    s, k, t, r, sigma = 100.0, 100.0, 30 / 252, 0.05, 0.30
    call = bs_price(s, k, t, r, sigma, True)
    put = bs_price(s, k, t, r, sigma, False)
    expected = s - k * exp(-r * t)
    assert abs((call - put) - expected) < 0.01


def test_bs_expired_intrinsic() -> None:
    """At expiration, option = intrinsic value."""
    assert bs_price(110.0, 100.0, 0, 0.05, 0.30, True) == 10.0
    assert bs_price(90.0, 100.0, 0, 0.05, 0.30, True) == 0.0
    assert bs_price(90.0, 100.0, 0, 0.05, 0.30, False) == 10.0


def test_iv_from_atr() -> None:
    """IV estimate should be reasonable for typical ATR."""
    iv = estimate_iv_from_atr(3.0, 100.0)  # 3% daily ATR
    assert 0.30 < iv < 0.80


def test_estimate_premium_call() -> None:
    """3% OTM call on $100 stock should be cheap but nonzero."""
    premium = estimate_premium(100.0, 0.03, 15, 0.30, is_call=True)
    assert 0.10 < premium < 3.0


# ── Strategy Tests ───────────────────────────────────────────────────


def _make_bars(
    n: int = 60,
    start: float = 100.0,
    trend: float = 0.002,
    vol: float = 0.01,
) -> list[dict]:
    """Generate synthetic bars for strategy testing."""
    import random

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


def test_compute_indicators() -> None:
    """Indicators should compute without errors."""
    bars = _make_bars(60)
    ind = compute_indicators(bars)
    assert ind.close > 0
    assert 0 < ind.rsi14 < 100
    assert ind.atr14 > 0
    assert ind.sma20 > 0
    assert ind.sma50 > 0


def test_regime_detection_uptrend() -> None:
    """Bars with positive trend should detect uptrend."""
    bars = _make_bars(60, trend=0.005)
    ind = compute_indicators(bars)
    regime = detect_regime(ind)
    assert regime in (MarketRegime.UPTREND, MarketRegime.STRONG_UPTREND)


def test_regime_detection_downtrend() -> None:
    """Bars with negative trend should detect downtrend."""
    bars = _make_bars(60, trend=-0.005)
    ind = compute_indicators(bars)
    regime = detect_regime(ind)
    assert regime in (MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND)


def test_scan_entries_returns_list() -> None:
    """scan_for_entries should return a list."""
    bars = _make_bars(60)
    ind = compute_indicators(bars)
    regime = detect_regime(ind)
    signals = scan_for_entries("TSLA", bars, ind, regime)
    assert isinstance(signals, list)
    for s in signals:
        assert isinstance(s, EntrySignal)
        assert 0.0 <= s.conviction <= 10.0


# ── Schema Tests ─────────────────────────────────────────────────────


def test_backtest_trade_schema() -> None:
    """New schema fields should serialize correctly."""
    from datetime import date

    trade = BacktestTrade(
        ticker="TSLA",
        entry_date=date(2024, 1, 1),
        entry_price=2.50,
        strategy="trend_pullback",
        regime="uptrend",
        conviction=7.5,
        exit_reason="trailing_stop",
    )
    assert trade.strategy == "trend_pullback"
    dumped = trade.model_dump_json()
    rebuilt = BacktestTrade.model_validate_json(dumped)
    assert rebuilt.conviction == 7.5


def test_backtest_result_schema() -> None:
    result = BacktestResult(
        run_id="test",
        tickers=["AAPL"],
        total_trades=10,
        wins=3,
        losses=7,
        win_rate=0.3,
        starting_capital=10_000,
        ending_value=11_500,
        portfolio_return_pct=15.0,
        sharpe_ratio=1.2,
    )
    assert result.win_rate == 0.3
    assert result.sharpe_ratio == 1.2
    json_str = result.model_dump_json()
    rebuilt = BacktestResult.model_validate_json(json_str)
    assert rebuilt.run_id == "test"
    assert rebuilt.portfolio_return_pct == 15.0
