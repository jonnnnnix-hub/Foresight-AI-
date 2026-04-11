"""Product Architecture Analyst — evaluates productization value."""

from flowedge.agents.base import AgentOutput, BaseAnalyst
from flowedge.agents.cartographer import _build_evidence_summary
from flowedge.agents.llm import call_agent_llm
from flowedge.agents.prompts import PRODUCT_ARCHITECT_PROMPT
from flowedge.schemas.agents import ProductArchitectOutput
from flowedge.schemas.evidence import EvidencePack
from flowedge.schemas.scoring import DimensionScore


class ProductArchitect(BaseAnalyst):
    """Evaluates productization: modularity, config, API fitness, extensibility."""

    name = "product_architect"

    async def analyze(self, evidence: EvidencePack) -> AgentOutput:
        summary = _build_evidence_summary(evidence)
        result = await call_agent_llm(
            system_prompt=PRODUCT_ARCHITECT_PROMPT,
            user_content=f"Evaluate productization value:\n\n{summary}",
            output_type=ProductArchitectOutput,
        )
        return AgentOutput(
            agent_name=self.name,
            summary=result.score_rationale,
            findings=result.productization_strengths + result.productization_weaknesses,
            scores=[
                DimensionScore(
                    dimension="productization_value",
                    score=result.score,
                    weight=0.15,
                    rationale=result.score_rationale,
                    evidence_refs=result.evidence_refs,
                )
            ],
            evidence_refs=result.evidence_refs,
            raw=result.model_dump(),
        )
