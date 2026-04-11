"""Debate schemas for adversarial review rounds."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DebateStance(StrEnum):
    """Position an agent takes on a claim."""

    SUPPORT = "support"
    CHALLENGE = "challenge"
    NEUTRAL = "neutral"


class DebateEntry(BaseModel):
    """A single debate contribution from an agent."""

    agent_name: str
    stance: DebateStance
    claim: str
    argument: str = Field(max_length=2000)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class DebateRound(BaseModel):
    """One round of adversarial debate."""

    round_number: int
    topic: str
    entries: list[DebateEntry] = Field(default_factory=list)
    resolution: str = Field(default="", description="Judge's resolution of this round")


class DebateRecord(BaseModel):
    """Complete debate record for a repo comparison."""

    repo_names: list[str]
    rounds: list[DebateRound] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    concluded_at: datetime | None = None
