"""Typed state for the LangGraph orchestration."""

from typing import Any

from pydantic import BaseModel, Field

from flowedge.schemas.debate import DebateRecord
from flowedge.schemas.evidence import EvidencePack
from flowedge.schemas.report import SynthesisReport
from flowedge.schemas.scoring import RepoScorecard


class GraphState(BaseModel):
    """Top-level state passed through the analysis graph."""

    run_id: str = ""
    repo_urls: list[str] = Field(default_factory=list)
    evidence_packs: list[EvidencePack] = Field(default_factory=list)
    specialist_outputs: list[dict[str, Any]] = Field(default_factory=list)
    scorecards: list[RepoScorecard] = Field(default_factory=list)
    debate_record: DebateRecord | None = None
    report: SynthesisReport | None = None
    errors: list[str] = Field(default_factory=list)
    status: str = "pending"
