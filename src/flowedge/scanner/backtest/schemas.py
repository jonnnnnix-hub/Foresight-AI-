"""Backtest result schemas."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TradeOutcome(StrEnum):
    """How a paper trade resolved."""

    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    EXPIRED = "expired"


class BacktestTrade(BaseModel):
    """A single simulated trade in the backtest."""

    ticker: str
    entry_date: date
    exit_date: date | None = None
    option_type: str = "call"
    strike: float = Field(ge=0.0, default=0.0)
    expiration: date = Field(default_factory=date.today)
    entry_price: float = Field(ge=0.0, description="Premium paid per contract")
    exit_price: float = Field(ge=0.0, default=0.0)
    underlying_entry: float = Field(ge=0.0, default=0.0)
    underlying_exit: float = Field(ge=0.0, default=0.0)
    underlying_move_pct: float = Field(default=0.0)
    pnl_per_contract: float = Field(default=0.0)
    pnl_pct: float = Field(default=0.0, description="Return on premium")
    outcome: TradeOutcome = TradeOutcome.LOSS
    signal_score: float = Field(ge=0.0, le=10.0, default=0.0)
    signal_type: str = ""
    hold_days: int = Field(ge=0, default=0)
    strategy: str = ""  # trend_pullback, breakout, mean_reversion, vol_squeeze
    regime: str = ""  # market regime at entry
    conviction: float = Field(ge=0.0, le=10.0, default=0.0)
    exit_reason: str = ""  # hard_stop, trailing_stop, take_profit, time_exit, etc.
    contracts: int = Field(ge=0, default=1)
    cost_basis: float = Field(ge=0.0, default=0.0)
    exit_value: float = Field(ge=0.0, default=0.0)


class BacktestMonthly(BaseModel):
    """Monthly backtest breakdown."""

    month: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    total_pnl_pct: float = 0.0


class BacktestResult(BaseModel):
    """Aggregate results from a backtest run."""

    run_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now())
    tickers: list[str] = Field(default_factory=list)
    lookback_days: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    expired_worthless: int = 0
    win_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    avg_win_pct: float = Field(default=0.0)
    avg_loss_pct: float = Field(default=0.0)
    best_trade_pct: float = Field(default=0.0)
    worst_trade_pct: float = Field(default=0.0)
    total_pnl_pct: float = Field(default=0.0, description="Sum of all trade returns")
    profit_factor: float = Field(
        default=0.0, description="Gross profits / gross losses"
    )
    avg_hold_days: float = Field(default=0.0)
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    expectancy_pct: float = Field(default=0.0, description="Avg return per trade")
    trades: list[BacktestTrade] = Field(default_factory=list)
    by_score_bucket: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Performance grouped by signal score ranges",
    )
    by_ticker: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Performance grouped by ticker",
    )
    monthly: list[BacktestMonthly] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # Portfolio-level metrics (v2)
    starting_capital: float = Field(default=10_000.0)
    ending_value: float = Field(default=0.0)
    portfolio_return_pct: float = Field(default=0.0)
    max_drawdown_pct: float = Field(default=0.0)
    sharpe_ratio: float = Field(default=0.0)
    by_strategy: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Performance grouped by strategy type",
    )
    by_regime: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Performance grouped by market regime",
    )
