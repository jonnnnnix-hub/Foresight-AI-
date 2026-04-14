"""Risk and Reliability Analyst — evaluates maintenance and operational risk."""

from flowedge.agents.base import AgentOutput, BaseAnalyst
from flowedge.agents.cartographer import _build_evidence_summary
from flowedge.agents.llm import call_agent_llm
from flowedge.agents.prompts import RISK_ANALYST_PROMPT
from flowedge.schemas.agents import RiskAnalystOutput
from flowedge.schemas.evidence import EvidencePack
from flowedge.schemas.scoring import DimensionScore


class RiskAnalyst(BaseAnalyst):
    """Evaluates reliability: test maturity, dependency health, failure modes."""

    name = "risk_analyst"

    async def analyze(self, evidence: EvidencePack) -> AgentOutput:
        summary = _build_evidence_summary(evidence)
        result = await call_agent_llm(
            system_prompt=RISK_ANALYST_PROMPT,
            user_content=f"Evaluate risk and reliability:\n\n{summary}",
            output_type=RiskAnalystOutput,
        )
        return AgentOutput(
            agent_name=self.name,
            summary=result.score_rationale,
            findings=result.reliability_strengths + result.reliability_weaknesses,
            scores=[
                DimensionScore(
                    dimension="reliability",
                    score=result.score,
                    weight=0.15,
                    rationale=result.score_rationale,
                    evidence_refs=result.evidence_refs,
                )
            ],
            evidence_refs=result.evidence_refs,
            raw=result.model_dump(),
        )
