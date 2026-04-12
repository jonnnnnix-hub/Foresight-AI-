"""Schemas for the index ETF specialist model."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class IndexTicker(StrEnum):
    """Supported index ETFs."""

    SPY = "SPY"
    QQQ = "QQQ"
    IWM = "IWM"


class TradeHorizon(StrEnum):
    """Trade duration classification."""

    SCALP = "scalp"          # 1-3 days, target 50%+ gains
    SWING = "swing"          # 4-10 days, target 30-80% gains
    MEDIUM = "medium"        # 10-20 days, target 40-120% gains


class IndexRegime(StrEnum):
    """Index-specific regime classification using multiple timeframes."""

    STRONG_BULL = "strong_bull"      # All MAs aligned up, breadth expanding
    BULL = "bull"                    # SMA20 > SMA50, price above both
    NEUTRAL_BULL = "neutral_bull"    # Price above SMA20 but momentum fading
    NEUTRAL = "neutral"              # Consolidation / range-bound
    NEUTRAL_BEAR = "neutral_bear"    # Price below SMA20, SMA50 still rising
    BEAR = "bear"                    # SMA20 < SMA50, price below both
    STRONG_BEAR = "strong_bear"      # All MAs aligned down, breadth collapsing


class IndexSignal(BaseModel):
    """A trade signal from the index specialist model."""

    ticker: str
    direction: str  # "bullish" or "bearish"
    horizon: TradeHorizon
    conviction: float = Field(ge=0.0, le=10.0)
    regime: IndexRegime
    strategy: str
    otm_pct: float = 0.015  # Tighter OTM for indices
    entry_reason: str = ""

    # Multi-factor scores
    momentum_score: float = 0.0
    volume_score: float = 0.0
    mc_prob_profit: float = 0.0
    breadth_score: float = 0.0
    vix_regime_score: float = 0.0


class IndexBacktestConfig(BaseModel):
    """Configuration for index specialist backtesting."""

    tickers: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    lookback_days: int = 730
    starting_capital: float = 10_000.0
    max_positions: int = 3
    max_risk_per_trade: float = 0.10

    # Scalp mode settings
    scalp_min_conviction: float = 8.5
    scalp_max_hold: int = 3
    scalp_hard_stop: float = -0.25
    scalp_trailing_stop: float = 0.20
    scalp_take_profit: float = 1.50
    scalp_target_gain_pct: float = 50.0  # 50%+ target
    scalp_dte: int = 7

    # Swing mode settings
    swing_min_conviction: float = 8.0
    swing_max_hold: int = 10
    swing_hard_stop: float = -0.30
    swing_trailing_stop: float = 0.30
    swing_take_profit: float = 2.50
    swing_dte: int = 21

    # Medium term settings
    medium_min_conviction: float = 7.5
    medium_max_hold: int = 20
    medium_hard_stop: float = -0.35
    medium_trailing_stop: float = 0.35
    medium_take_profit: float = 3.00
    medium_dte: int = 45

    # MC simulation settings
    mc_simulations: int = 50_000
    mc_min_prob_profit_scalp: float = 0.55
    mc_min_prob_profit_swing: float = 0.50


class IndexTradeResult(BaseModel):
    """Result of a single index trade."""

    ticker: str
    direction: str
    horizon: TradeHorizon
    entry_date: date
    exit_date: date | None = None
    entry_underlying: float = 0.0
    exit_underlying: float = 0.0
    strike: float = 0.0
    entry_premium: float = 0.0
    exit_premium: float = 0.0
    pnl_pct: float = 0.0
    pnl_dollars: float = 0.0
    hold_days: int = 0
    exit_reason: str = ""
    conviction: float = 0.0
    regime: str = ""
    strategy: str = ""
    mc_prob_profit: float = 0.0
    is_win: bool = False


class IndexBacktestResult(BaseModel):
    """Full backtest results for the index specialist."""

    run_id: str = ""
    tickers: list[str] = Field(default_factory=list)
    lookback_days: int = 0
    starting_capital: float = 10_000.0
    ending_value: float = 10_000.0
    portfolio_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy_pct: float = 0.0

    # By horizon
    scalp_trades: int = 0
    scalp_win_rate: float = 0.0
    scalp_avg_pnl: float = 0.0
    swing_trades: int = 0
    swing_win_rate: float = 0.0
    swing_avg_pnl: float = 0.0
    medium_trades: int = 0
    medium_win_rate: float = 0.0
    medium_avg_pnl: float = 0.0

    # By ticker
    by_ticker: dict[str, dict[str, float]] = Field(default_factory=dict)

    trades: list[IndexTradeResult] = Field(default_factory=list)
