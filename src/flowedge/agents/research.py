"""Research Engine Analyst — evaluates backtesting and research iteration strength."""

from flowedge.agents.base import AgentOutput, BaseAnalyst
from flowedge.agents.cartographer import _build_evidence_summary
from flowedge.agents.llm import call_agent_llm
from flowedge.agents.prompts import RESEARCH_ANALYST_PROMPT
from flowedge.schemas.agents import ResearchAnalystOutput
from flowedge.schemas.evidence import EvidencePack
from flowedge.schemas.scoring import DimensionScore


class ResearchAnalyst(BaseAnalyst):
    """Evaluates research power: backtesting, parameter sweeps, scanner fit."""

    name = "research_analyst"

    async def analyze(self, evidence: EvidencePack) -> AgentOutput:
        summary = _build_evidence_summary(evidence)
        result = await call_agent_llm(
            system_prompt=RESEARCH_ANALYST_PROMPT,
            user_content=f"Evaluate research strength:\n\n{summary}",
            output_type=ResearchAnalystOutput,
        )
        return AgentOutput(
            agent_name=self.name,
            summary=result.score_rationale,
            findings=result.research_strengths + result.research_weaknesses,
            scores=[
                DimensionScore(
                    dimension="research_power",
                    score=result.score,
                    weight=0.15,
                    rationale=result.score_rationale,
                    evidence_refs=result.evidence_refs,
                )
            ],
            evidence_refs=result.evidence_refs,
            raw=result.model_dump(),
        )
