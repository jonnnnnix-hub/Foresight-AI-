"""Performance tracking schemas."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TradeResult(StrEnum):
    WIN = "win"
    LOSS = "loss"
    OPEN = "open"
    EXPIRED = "expired"


class SimulatedTrade(BaseModel):
    """A single trade the bot would have taken."""

    trade_id: str = ""
    ticker: str
    direction: str = "bullish"
    entry_date: date = Field(default_factory=date.today)
    exit_date: date | None = None
    option_type: str = "call"
    strike: float = Field(ge=0.0, default=0.0)
    expiration: date | None = None
    entry_underlying: float = Field(ge=0.0, default=0.0)
    exit_underlying: float | None = None
    entry_premium: float = Field(ge=0.0, default=0.0)
    exit_premium: float | None = None
    contracts: int = Field(ge=1, default=1)
    cost_basis: float = Field(ge=0.0, default=0.0)
    exit_value: float | None = None
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0
    result: TradeResult = TradeResult.OPEN
    nexus_score: int = Field(ge=0, le=100, default=0)
    hold_days: int = 0
    exit_reason: str = ""


class DailySnapshot(BaseModel):
    """Portfolio value at end of each trading day."""

    date: date
    portfolio_value: float = Field(ge=0.0)
    cash: float = Field(ge=0.0)
    open_positions_value: float = Field(ge=0.0, default=0.0)
    daily_pnl: float = 0.0
    daily_return_pct: float = 0.0
    cumulative_return_pct: float = 0.0
    trades_opened: int = 0
    trades_closed: int = 0


class MonthlyReturn(BaseModel):
    """Monthly P&L breakdown."""

    month: str  # YYYY-MM format
    starting_value: float = 0.0
    ending_value: float = 0.0
    return_pct: float = 0.0
    return_dollars: float = 0.0
    trades_opened: int = 0
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0


class TickerPerformance(BaseModel):
    """Per-ticker accuracy and P&L breakdown."""

    ticker: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_dollars: float = 0.0
    avg_pnl_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0


class ModelAccuracy(BaseModel):
    """Model prediction accuracy metrics."""

    total_predictions: int = 0
    correct_direction: int = 0
    direction_accuracy: float = 0.0
    avg_score_winners: float = 0.0
    avg_score_losers: float = 0.0
    score_separation: float = 0.0  # gap between winner/loser avg scores
    high_score_win_rate: float = 0.0  # win rate for score >= 60
    low_score_win_rate: float = 0.0  # win rate for score < 40
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    expectancy: float = 0.0  # avg dollars per trade
    consecutive_wins_max: int = 0
    consecutive_losses_max: int = 0


class PerformanceReport(BaseModel):
    """Full performance report for the simulation."""

    start_date: date
    end_date: date
    starting_capital: float
    ending_value: float
    total_return_dollars: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    open_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_hold_days: float = 0.0
    daily_snapshots: list[DailySnapshot] = Field(default_factory=list)
    trades: list[SimulatedTrade] = Field(default_factory=list)
    by_score_bucket: dict[str, dict[str, float]] = Field(default_factory=dict)
    monthly_returns: list[MonthlyReturn] = Field(default_factory=list)
    by_ticker: list[TickerPerformance] = Field(default_factory=list)
    model_accuracy: ModelAccuracy = Field(default_factory=ModelAccuracy)
    generated_at: datetime = Field(default_factory=lambda: datetime.now())
