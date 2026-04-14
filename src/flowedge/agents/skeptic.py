"""Skeptic — challenges optimistic conclusions and finds overclaims."""

from flowedge.agents.base import AgentOutput, BaseAnalyst
from flowedge.agents.cartographer import _build_evidence_summary
from flowedge.agents.llm import call_agent_llm
from flowedge.agents.prompts import SKEPTIC_PROMPT
from flowedge.schemas.agents import SkepticOutput
from flowedge.schemas.evidence import EvidencePack


class Skeptic(BaseAnalyst):
    """Challenges weak reasoning, hype, and unsupported claims."""

    name = "skeptic"

    async def analyze(self, evidence: EvidencePack) -> AgentOutput:
        summary = _build_evidence_summary(evidence)
        result = await call_agent_llm(
            system_prompt=SKEPTIC_PROMPT,
            user_content=f"Challenge the claims in this repo:\n\n{summary}",
            output_type=SkepticOutput,
        )
        return AgentOutput(
            agent_name=self.name,
            summary="; ".join(result.overclaim_flags[:3]) if result.overclaim_flags else "No flags",
            findings=result.overclaim_flags + result.unsupported_capabilities,
            scores=[],  # Skeptic does not score directly
            evidence_refs=result.evidence_refs,
            raw=result.model_dump(),
        )
