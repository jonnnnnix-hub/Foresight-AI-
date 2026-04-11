"""Scoring schemas for repo evaluation."""

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    dimension: str
    score: float = Field(ge=0.0, le=10.0)
    weight: float = Field(ge=0.0, le=1.0, default=1.0)
    rationale: str = Field(description="Why this score was given")
    evidence_refs: list[str] = Field(
        default_factory=list, description="References to supporting evidence"
    )


class RepoScorecard(BaseModel):
    """Complete scorecard for one repository."""

    repo_name: str
    repo_url: str
    dimensions: list[DimensionScore] = Field(default_factory=list)

    @property
    def weighted_total(self) -> float:
        """Compute weighted total score."""
        if not self.dimensions:
            return 0.0
        total_weight = sum(d.weight for d in self.dimensions)
        if total_weight == 0:
            return 0.0
        return sum(d.score * d.weight for d in self.dimensions) / total_weight


DEFAULT_DIMENSIONS = [
    ("research_power", 0.15),
    ("execution_realism", 0.20),
    ("ml_readiness", 0.15),
    ("productization_value", 0.15),
    ("reliability", 0.15),
    ("scalp_fit", 0.20),
]
