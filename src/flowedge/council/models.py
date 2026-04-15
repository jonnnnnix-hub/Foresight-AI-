"""Pydantic models for the Council review system.

Every specialist produces a SpecialistReview.  The CouncilEngine
aggregates them into a DailyReview, which the dashboard renders.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ── Enums ────────────────────────────────────────────────────────

class Severity(StrEnum):
    """How urgent is a finding?"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ReviewStatus(StrEnum):
    """Overall health of the model for a given day."""
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class RecommendationType(StrEnum):
    """Category of a recommendation."""
    PARAMETER_CHANGE = "parameter_change"
    TICKER_ADD = "ticker_add"
    TICKER_REMOVE = "ticker_remove"
    RISK_ADJUSTMENT = "risk_adjustment"
    FILTER_CHANGE = "filter_change"
    REGIME_ADAPTATION = "regime_adaptation"
    EXECUTION_IMPROVEMENT = "execution_improvement"
    NO_ACTION = "no_action"


# ── Finding / Recommendation ────────────────────────────────────

class Finding(BaseModel):
    """A single observation from a specialist."""
    title: str = Field(description="Short headline")
    detail: str = Field(description="Full explanation with evidence")
    severity: Severity = Severity.INFO
    metric_name: str = Field(default="", description="Key metric referenced")
    metric_value: float | str = Field(default="", description="Current value")
    threshold: float | str = Field(default="", description="Threshold that triggered this")
    evidence: list[str] = Field(default_factory=list, description="Supporting data points")


class Recommendation(BaseModel):
    """An actionable suggestion from a specialist."""
    title: str
    rationale: str = Field(description="Why this change is recommended")
    rec_type: RecommendationType = RecommendationType.NO_ACTION
    priority: int = Field(default=3, ge=1, le=5, description="1=highest, 5=lowest")
    confidence: float = Field(default=0.5, ge=0, le=1, description="Specialist confidence 0-1")
    current_value: str = Field(default="", description="Current parameter/state")
    suggested_value: str = Field(default="", description="Proposed parameter/state")
    expected_impact: str = Field(default="", description="Expected effect on performance")
    evidence: list[str] = Field(default_factory=list)


# ── Specialist Review ────────────────────────────────────────────

class SpecialistReview(BaseModel):
    """Output from one specialist's analysis of the model."""
    specialist_name: str = Field(description="e.g. 'Signal Quality Analyst'")
    specialist_id: str = Field(description="e.g. 'signal_analyst'")
    review_date: date
    health_score: float = Field(ge=0, le=100, description="0-100 health rating")
    severity: Severity = Severity.INFO
    summary: str = Field(description="2-3 sentence executive summary")
    findings: list[Finding] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    metrics: dict[str, float | str] = Field(
        default_factory=dict,
        description="Key metrics computed by this specialist",
    )
    computation_time_ms: float = Field(default=0, description="How long the analysis took")


# ── Per-Ticker Scorecard ─────────────────────────────────────────

class TickerScorecard(BaseModel):
    """Performance summary for a single ticker."""
    ticker: str
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    avg_hold_bars: float = 0.0
    signal_match_rate: float = Field(
        default=0.0,
        description="Fraction of signals that found an option contract",
    )
    recommendation: str = Field(default="hold", description="keep / watch / remove / add")


# ── Daily Review (aggregated) ────────────────────────────────────

class DailyReview(BaseModel):
    """Full council review for one trading day (run post-market)."""
    review_id: str
    review_date: date
    generated_at: datetime = Field(default_factory=datetime.now)
    status: ReviewStatus = ReviewStatus.HEALTHY
    overall_health: float = Field(ge=0, le=100, description="Weighted average of specialist scores")
    config_used: dict[str, Any] = Field(default_factory=dict, description="ScalpConfig snapshot")

    # Performance snapshot
    trades_today: int = 0
    wins_today: int = 0
    pnl_today: float = 0.0
    cumulative_trades: int = 0
    cumulative_wr: float = 0.0
    cumulative_pnl: float = 0.0

    # Specialist reviews
    specialist_reviews: list[SpecialistReview] = Field(default_factory=list)

    # Aggregated recommendations (merged from all specialists, sorted by priority)
    top_recommendations: list[Recommendation] = Field(default_factory=list)

    # Ticker scorecards
    ticker_scorecards: list[TickerScorecard] = Field(default_factory=list)

    # Council consensus
    consensus_summary: str = Field(
        default="",
        description="Synthesized view from all specialists",
    )
    dissenting_views: list[str] = Field(
        default_factory=list,
        description="Where specialists disagree",
    )

    # Metadata
    computation_time_ms: float = 0
    notes: list[str] = Field(default_factory=list)


# ── Historical Trend ─────────────────────────────────────────────

class ReviewTrend(BaseModel):
    """Summary of a DailyReview for historical trend charts."""
    review_date: date
    status: ReviewStatus
    overall_health: float
    trades: int
    win_rate: float
    pnl: float
    top_finding: str = ""
