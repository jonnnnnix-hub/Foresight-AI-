"""Schemas for the learning and refinement system."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class FailureCategory(StrEnum):
    """Why a trade failed — root cause classification."""

    FLOW_MISLEADING = "flow_misleading"  # UOA was noise, not signal
    IV_CRUSH = "iv_crush"  # Premium destroyed by volatility collapse
    BAD_TIMING = "bad_timing"  # Right direction, wrong entry window
    CATALYST_MISS = "catalyst_miss"  # Catalyst didn't trigger expected move
    WRONG_DIRECTION = "wrong_direction"  # Momentum read was backwards
    THETA_DECAY = "theta_decay"  # Time decay ate the position
    MARKET_REGIME = "market_regime"  # Broad market moved against thesis
    OVERSCORED = "overscored"  # Model gave too much conviction
    LOW_LIQUIDITY = "low_liquidity"  # Thin volume, bad fills
    UNKNOWN = "unknown"


class SpecialistVerdict(BaseModel):
    """One specialist's analysis of a trade failure."""

    specialist: str  # e.g. "flow_analyst", "vol_analyst"
    diagnosis: str  # What went wrong from this specialist's perspective
    root_cause: FailureCategory
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    recommendation: str = ""  # What should change


class TradePostMortem(BaseModel):
    """Deep analysis of why a specific trade failed."""

    trade_id: str
    ticker: str
    entry_date: str
    exit_date: str = ""
    pnl_pct: float = 0.0
    nexus_score: int = 0
    direction: str = ""
    option_type: str = ""
    strike: float = 0.0

    # Committee verdicts
    verdicts: list[SpecialistVerdict] = Field(default_factory=list)
    consensus_cause: FailureCategory = FailureCategory.UNKNOWN
    consensus_confidence: float = 0.0

    # Synthesized learnings
    thesis_at_entry: str = ""
    what_actually_happened: str = ""
    key_lesson: str = ""
    should_have_been_filtered: bool = False
    suggested_filter: str = ""


class LearningInsight(BaseModel):
    """A pattern extracted from analyzing multiple trades."""

    insight_id: str = ""
    category: FailureCategory
    pattern: str  # Description of the pattern
    frequency: int = 0  # How many trades showed this pattern
    avg_loss_when_present: float = 0.0
    suggested_action: str = ""  # What to change in the model
    evidence_trade_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class WeightAdjustment(BaseModel):
    """A specific model weight or threshold change."""

    parameter: str  # e.g. "uoa_weight", "iv_rank_threshold"
    current_value: float
    suggested_value: float
    reason: str
    expected_impact: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class ScoringRule(BaseModel):
    """A new scoring rule derived from learning."""

    rule_id: str = ""
    name: str
    condition: str  # Human-readable condition
    score_adjustment: float = 0.0  # How much to add/subtract
    applies_to: str = "composite"  # Which score component
    reason: str = ""
    win_rate_improvement: float = 0.0  # Expected improvement


class ModelRefinement(BaseModel):
    """Complete set of refinements for one learning cycle."""

    cycle_id: str = ""
    generated_at: datetime = Field(default_factory=datetime.now)
    trades_analyzed: int = 0
    losses_analyzed: int = 0
    wins_analyzed: int = 0

    # Failure analysis
    post_mortems: list[TradePostMortem] = Field(default_factory=list)
    failure_distribution: dict[str, int] = Field(default_factory=dict)

    # Extracted insights
    insights: list[LearningInsight] = Field(default_factory=list)

    # Concrete refinements
    weight_adjustments: list[WeightAdjustment] = Field(default_factory=list)
    new_rules: list[ScoringRule] = Field(default_factory=list)
    filters_to_add: list[str] = Field(default_factory=list)

    # Impact assessment
    expected_win_rate_change: float = 0.0
    expected_profit_factor_change: float = 0.0
    rationale: str = ""


class AdaptiveWeights(BaseModel):
    """Dynamic scoring weights that evolve from learning cycles."""

    # Base weights (start at defaults from settings)
    uoa_weight: float = 0.35
    iv_weight: float = 0.30
    catalyst_weight: float = 0.35

    # Score thresholds
    min_entry_score: float = 3.0  # Don't trade below this
    high_conviction_threshold: float = 7.0

    # Signal-specific adjustments
    uoa_min_premium: float = 10_000  # Min premium for UOA to count
    iv_rank_sweet_spot_low: float = 10.0
    iv_rank_sweet_spot_high: float = 35.0
    catalyst_max_days: int = 14
    catalyst_min_days: int = 1

    # Penalty rules (learned from losses)
    penalty_rules: list[ScoringRule] = Field(default_factory=list)
    bonus_rules: list[ScoringRule] = Field(default_factory=list)

    # Metadata
    version: int = 1
    last_updated: datetime = Field(default_factory=datetime.now)
    cycles_applied: int = 0
    learning_history: list[str] = Field(default_factory=list)
