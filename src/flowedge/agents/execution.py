"""Execution Realism Analyst — evaluates live trading suitability."""

from flowedge.agents.base import AgentOutput, BaseAnalyst
from flowedge.agents.cartographer import _build_evidence_summary
from flowedge.agents.llm import call_agent_llm
from flowedge.agents.prompts import EXECUTION_ANALYST_PROMPT
from flowedge.schemas.agents import ExecutionAnalystOutput
from flowedge.schemas.evidence import EvidencePack
from flowedge.schemas.scoring import DimensionScore


class ExecutionAnalyst(BaseAnalyst):
    """Evaluates execution realism: slippage, fills, broker integration."""

    name = "execution_analyst"

    async def analyze(self, evidence: EvidencePack) -> AgentOutput:
        summary = _build_evidence_summary(evidence)
        result = await call_agent_llm(
            system_prompt=EXECUTION_ANALYST_PROMPT,
            user_content=f"Evaluate execution realism:\n\n{summary}",
            output_type=ExecutionAnalystOutput,
        )
        return AgentOutput(
            agent_name=self.name,
            summary=result.score_rationale,
            findings=result.execution_strengths + result.execution_weaknesses,
            scores=[
                DimensionScore(
                    dimension="execution_realism",
                    score=result.score,
                    weight=0.20,
                    rationale=result.score_rationale,
                    evidence_refs=result.evidence_refs,
                )
            ],
            evidence_refs=result.evidence_refs,
            raw=result.model_dump(),
        )
