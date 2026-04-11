"""Report schemas for final synthesis output."""

from datetime import datetime

from pydantic import BaseModel, Field


class BorrowAvoidReplace(BaseModel):
    """What to borrow, avoid, or replace from a repo."""

    repo_name: str
    borrow: list[str] = Field(default_factory=list, description="Components worth adopting")
    avoid: list[str] = Field(default_factory=list, description="Components to skip")
    replace: list[str] = Field(
        default_factory=list, description="Components to replace with custom"
    )


class RankingEntry(BaseModel):
    """A repo's position in the final ranking."""

    rank: int
    repo_name: str
    weighted_score: float
    best_for: list[str] = Field(default_factory=list)
    weakest_at: list[str] = Field(default_factory=list)


class SynthesisReport(BaseModel):
    """Final output report for FlowEdge."""

    run_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    executive_summary: str
    rankings: list[RankingEntry] = Field(default_factory=list)
    borrow_avoid_replace: list[BorrowAvoidReplace] = Field(default_factory=list)
    debate_highlights: list[str] = Field(default_factory=list)
    real_vs_hype: dict[str, str] = Field(default_factory=dict)
    merged_architecture: dict[str, str] = Field(
        default_factory=dict,
        description="Proposed FlowEdge architecture component sources",
    )
    mvp_recommendation: str = Field(default="")
    do_not_build_wrong: str = Field(default="")
    risks_and_unknowns: list[str] = Field(default_factory=list)
    evidence_appendix: list[str] = Field(default_factory=list)
