"""Typed output schemas for each specialist agent."""

from pydantic import BaseModel, Field


class CartographerOutput(BaseModel):
    """Repo Cartographer: structural map of the repository."""

    repo_summary: str
    primary_languages: list[str] = Field(default_factory=list)
    key_directories: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    build_system: str = ""
    dependency_risks: list[str] = Field(default_factory=list)
    extension_points: list[str] = Field(default_factory=list)
    architecture_observations: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class ResearchAnalystOutput(BaseModel):
    """Research Engine Analyst: research and backtesting strength."""

    research_strengths: list[str] = Field(default_factory=list)
    research_weaknesses: list[str] = Field(default_factory=list)
    scanner_suitability_notes: list[str] = Field(default_factory=list)
    scalp_trading_relevance: list[str] = Field(default_factory=list)
    backtesting_type: str = Field(
        default="", description="vectorized | event-driven | hybrid | none"
    )
    parameter_sweep_support: bool = False
    walk_forward_support: bool = False
    multi_symbol_support: bool = False
    score: float = Field(ge=0.0, le=10.0, default=0.0)
    score_rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class ExecutionAnalystOutput(BaseModel):
    """Execution Realism Analyst: live trading suitability."""

    execution_strengths: list[str] = Field(default_factory=list)
    execution_weaknesses: list[str] = Field(default_factory=list)
    realism_gaps: list[str] = Field(default_factory=list)
    scalp_execution_fit: list[str] = Field(default_factory=list)
    slippage_modeling: bool = False
    order_type_support: list[str] = Field(default_factory=list)
    commission_handling: bool = False
    live_trading_support: bool = False
    broker_integrations: list[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=10.0, default=0.0)
    score_rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class MLAnalystOutput(BaseModel):
    """ML Pipeline Analyst: ML workflow readiness."""

    ml_strengths: list[str] = Field(default_factory=list)
    ml_weaknesses: list[str] = Field(default_factory=list)
    leakage_concerns: list[str] = Field(default_factory=list)
    scanner_model_suitability: list[str] = Field(default_factory=list)
    feature_pipeline_exists: bool = False
    train_val_test_split: bool = False
    walk_forward_training: bool = False
    online_inference_ready: bool = False
    labeling_support: bool = False
    score: float = Field(ge=0.0, le=10.0, default=0.0)
    score_rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class ProductArchitectOutput(BaseModel):
    """Product Architecture Analyst: productization value."""

    productization_strengths: list[str] = Field(default_factory=list)
    productization_weaknesses: list[str] = Field(default_factory=list)
    reusable_components: list[str] = Field(default_factory=list)
    likely_rewrite_zones: list[str] = Field(default_factory=list)
    modularity_assessment: str = ""
    config_system_quality: str = ""
    api_layering: str = ""
    observability_hooks: bool = False
    score: float = Field(ge=0.0, le=10.0, default=0.0)
    score_rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class RiskAnalystOutput(BaseModel):
    """Risk and Reliability Analyst: maintenance and operational risk."""

    reliability_strengths: list[str] = Field(default_factory=list)
    reliability_weaknesses: list[str] = Field(default_factory=list)
    operational_risks: list[str] = Field(default_factory=list)
    maintenance_risks: list[str] = Field(default_factory=list)
    test_coverage_assessment: str = ""
    dependency_health: str = ""
    bus_factor_estimate: str = ""
    release_health: str = ""
    score: float = Field(ge=0.0, le=10.0, default=0.0)
    score_rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class SkepticOutput(BaseModel):
    """Skeptic: challenges and counterarguments."""

    overclaim_flags: list[str] = Field(default_factory=list)
    unsupported_capabilities: list[str] = Field(default_factory=list)
    hidden_assumptions: list[str] = Field(default_factory=list)
    likely_failure_modes: list[str] = Field(default_factory=list)
    strongest_counterarguments: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class JudgeOutput(BaseModel):
    """Judge: final synthesis across all specialists."""

    repo_ranking: list[str] = Field(default_factory=list, description="Repos ordered best to worst")
    top_repo_by_category: dict[str, str] = Field(
        default_factory=dict, description="category -> repo_name"
    )
    borrow_avoid_replace: dict[str, dict[str, list[str]]] = Field(
        default_factory=dict,
        description="repo_name -> {borrow: [...], avoid: [...], replace: [...]}",
    )
    merged_architecture: dict[str, str] = Field(
        default_factory=dict, description="component -> source repo"
    )
    mvp_build_order: list[str] = Field(default_factory=list)
    risks_and_unknowns: list[str] = Field(default_factory=list)
    do_not_build_wrong: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
