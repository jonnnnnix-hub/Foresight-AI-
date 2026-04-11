"""Base agent interface for specialist analysts."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from flowedge.schemas.evidence import EvidencePack
from flowedge.schemas.scoring import DimensionScore


class AgentOutput(BaseModel):
    """Standard output from a specialist agent."""

    agent_name: str
    summary: str
    findings: list[str]
    scores: list[DimensionScore]
    evidence_refs: list[str]
    raw: dict[str, Any] = {}


class BaseAnalyst(ABC):
    """Abstract base class for specialist analysis agents."""

    name: str = "base"

    @abstractmethod
    async def analyze(self, evidence: EvidencePack) -> AgentOutput:
        """Run analysis on the evidence pack and return structured output."""
        ...
