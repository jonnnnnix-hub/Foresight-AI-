"""Tests for backtest engine logic."""

from flowedge.scanner.backtest.engine import _simulate_option_trade
from flowedge.scanner.backtest.schemas import BacktestResult, TradeOutcome


def test_simulate_winning_trade() -> None:
    """Underlying rallies through strike → option wins."""
    bars = [
        {"date": "2024-01-02", "close": 205.0},
        {"date": "2024-01-03", "close": 210.0},
        {"date": "2024-01-04", "close": 220.0},
        {"date": "2024-01-05", "close": 230.0},
        {"date": "2024-01-06", "close": 240.0},
    ]
    trade = _simulate_option_trade(
        entry_price=2.0,
        underlying_entry=200.0,
        underlying_bars=bars,
        strike=205.0,
        is_call=True,
        max_hold_days=5,
        take_profit_pct=100.0,
    )
    assert trade.pnl_pct > 0
    assert trade.outcome == TradeOutcome.WIN


def test_simulate_losing_trade() -> None:
    """Underlying drops → OTM call loses."""
    bars = [
        {"date": "2024-01-02", "close": 195.0},
        {"date": "2024-01-03", "close": 190.0},
        {"date": "2024-01-04", "close": 185.0},
    ]
    trade = _simulate_option_trade(
        entry_price=2.0,
        underlying_entry=200.0,
        underlying_bars=bars,
        strike=210.0,
        is_call=True,
        max_hold_days=3,
    )
    assert trade.pnl_pct < 0


def test_simulate_empty_bars() -> None:
    trade = _simulate_option_trade(
        entry_price=2.0,
        underlying_entry=200.0,
        underlying_bars=[],
        strike=210.0,
        is_call=True,
    )
    assert trade.outcome == TradeOutcome.EXPIRED


def test_backtest_result_schema() -> None:
    result = BacktestResult(
        run_id="test",
        tickers=["AAPL"],
        total_trades=10,
        wins=3,
        losses=7,
        win_rate=0.3,
    )
    assert result.win_rate == 0.3
    json_str = result.model_dump_json()
    rebuilt = BacktestResult.model_validate_json(json_str)
    assert rebuilt.run_id == "test"
