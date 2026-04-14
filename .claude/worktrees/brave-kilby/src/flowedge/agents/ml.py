"""ML Pipeline Analyst — evaluates ML workflow readiness."""

from flowedge.agents.base import AgentOutput, BaseAnalyst
from flowedge.agents.cartographer import _build_evidence_summary
from flowedge.agents.llm import call_agent_llm
from flowedge.agents.prompts import ML_ANALYST_PROMPT
from flowedge.schemas.agents import MLAnalystOutput
from flowedge.schemas.evidence import EvidencePack
from flowedge.schemas.scoring import DimensionScore


class MLAnalyst(BaseAnalyst):
    """Evaluates ML readiness: feature pipelines, leakage, inference design."""

    name = "ml_analyst"

    async def analyze(self, evidence: EvidencePack) -> AgentOutput:
        summary = _build_evidence_summary(evidence)
        result = await call_agent_llm(
            system_prompt=ML_ANALYST_PROMPT,
            user_content=f"Evaluate ML pipeline readiness:\n\n{summary}",
            output_type=MLAnalystOutput,
        )
        return AgentOutput(
            agent_name=self.name,
            summary=result.score_rationale,
            findings=result.ml_strengths + result.ml_weaknesses,
            scores=[
                DimensionScore(
                    dimension="ml_readiness",
                    score=result.score,
                    weight=0.15,
                    rationale=result.score_rationale,
                    evidence_refs=result.evidence_refs,
                )
            ],
            evidence_refs=result.evidence_refs,
            raw=result.model_dump(),
        )
