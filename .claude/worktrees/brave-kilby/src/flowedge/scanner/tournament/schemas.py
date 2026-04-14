"""Pydantic schemas for the tournament engine."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ModelName(StrEnum):
    """The five competing tournament models."""

    EDGE_CORE = "edge_core"
    MOMENTUM_ALPHA = "momentum_alpha"
    FLOW_HUNTER = "flow_hunter"
    CONTRARIAN_EDGE = "contrarian_edge"
    REGIME_CHAMELEON = "regime_chameleon"
    ENSEMBLE = "ensemble"


class ConsensusEntry(BaseModel):
    """Consensus-level tracking for a single trade opportunity."""

    ticker: str
    date: str
    models_agreeing: list[str] = Field(default_factory=list)
    models_disagreeing: list[str] = Field(default_factory=list)
    consensus_score: float = Field(
        default=0.0,
        description="Average score across agreeing models (0-100)",
    )
    consensus_level: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Number of models that agree to enter (0-5)",
    )
    direction: str = ""


class ModelPerformance(BaseModel):
    """Per-model results from a tournament run."""

    model_name: str = ""
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    profit_factor: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_hold_days: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    starting_capital: float = 10_000.0
    ending_value: float = 0.0


class TournamentResult(BaseModel):
    """Aggregate results from a full tournament run."""

    run_id: str = ""
    started_at: datetime = Field(default_factory=datetime.now)
    tickers: list[str] = Field(default_factory=list)
    lookback_days: int = 0
    starting_capital: float = 10_000.0

    # Per-model performance
    model_results: dict[str, ModelPerformance] = Field(
        default_factory=dict,
        description="Performance keyed by ModelName value",
    )

    # Ensemble performance
    ensemble_result: ModelPerformance = Field(default_factory=lambda: ModelPerformance())

    # Consensus analysis
    consensus_entries: list[ConsensusEntry] = Field(default_factory=list)
    avg_consensus_level: float = 0.0
    high_consensus_win_rate: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Win rate when 4+ models agree",
    )

    # Rankings
    ranking_by_return: list[str] = Field(default_factory=list)
    ranking_by_sharpe: list[str] = Field(default_factory=list)
    ranking_by_win_rate: list[str] = Field(default_factory=list)
