"""Debate engine — adversarial review where agents challenge each other."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from flowedge.agents.base import AgentOutput
from flowedge.agents.llm import call_agent_llm
from flowedge.schemas.debate import DebateEntry, DebateRecord, DebateRound, DebateStance

logger = structlog.get_logger()

DEBATE_TOPICS = [
    "execution_realism",
    "research_power_vs_hype",
    "scalp_trading_fitness",
    "ml_readiness_vs_notebook_fantasy",
    "productization_viability",
]


class ContradictionCandidate:
    """A potential disagreement between agents worth debating."""

    def __init__(self, topic: str, repo_name: str, agent_a: str, claim_a: str,
                 agent_b: str, claim_b: str) -> None:
        self.topic = topic
        self.repo_name = repo_name
        self.agent_a = agent_a
        self.claim_a = claim_a
        self.agent_b = agent_b
        self.claim_b = claim_b


def detect_contradictions(
    repo_outputs: dict[str, list[AgentOutput]],
) -> list[ContradictionCandidate]:
    """Find disagreements between agents on the same repo."""
    candidates: list[ContradictionCandidate] = []

    for repo_name, outputs in repo_outputs.items():
        # Compare each pair of agents
        for i, out_a in enumerate(outputs):
            for out_b in outputs[i + 1 :]:
                # Look for score disagreements (>3 point gap on same dimension)
                for score_a in out_a.scores:
                    for score_b in out_b.scores:
                        if score_a.dimension == score_b.dimension:
                            gap = abs(score_a.score - score_b.score)
                            if gap >= 3.0:
                                candidates.append(
                                    ContradictionCandidate(
                                        topic=score_a.dimension,
                                        repo_name=repo_name,
                                        agent_a=out_a.agent_name,
                                        claim_a=score_a.rationale,
                                        agent_b=out_b.agent_name,
                                        claim_b=score_b.rationale,
                                    )
                                )

                # Look for finding contradictions
                if out_a.findings and out_b.findings:
                    # Simple heuristic: if one agent is very positive and another very negative
                    a_positive = sum(
                        1 for f in out_a.findings
                        if "strong" in f.lower() or "good" in f.lower()
                    )
                    b_negative = sum(
                        1 for f in out_b.findings
                        if "weak" in f.lower()
                        or "missing" in f.lower()
                        or "lack" in f.lower()
                    )
                    if a_positive > 0 and b_negative > 0:
                        candidates.append(
                            ContradictionCandidate(
                                topic="agent_disagreement",
                                repo_name=repo_name,
                                agent_a=out_a.agent_name,
                                claim_a=out_a.summary,
                                agent_b=out_b.agent_name,
                                claim_b=out_b.summary,
                            )
                        )

    return candidates


async def _resolve_debate_round(
    contradiction: ContradictionCandidate,
    round_number: int,
) -> DebateRound:
    """Use LLM to resolve a contradiction between agents."""
    from pydantic import BaseModel, Field

    class DebateResolution(BaseModel):
        resolution: str = Field(description="How to resolve this disagreement")
        winner: str = Field(description="Which agent's position is better supported")
        confidence: float = Field(ge=0.0, le=1.0)

    prompt = (
        "You are a debate moderator. Two specialist agents disagree. "
        "Evaluate their positions based on evidence quality and resolve the dispute. "
        "Be direct and evidence-focused."
    )
    content = (
        f"Topic: {contradiction.topic}\n"
        f"Repo: {contradiction.repo_name}\n\n"
        f"Agent A ({contradiction.agent_a}): {contradiction.claim_a}\n\n"
        f"Agent B ({contradiction.agent_b}): {contradiction.claim_b}\n\n"
        "Which position is better supported? Why?"
    )

    try:
        result = await call_agent_llm(
            system_prompt=prompt,
            user_content=content,
            output_type=DebateResolution,
        )
        return DebateRound(
            round_number=round_number,
            topic=f"{contradiction.topic} ({contradiction.repo_name})",
            entries=[
                DebateEntry(
                    agent_name=contradiction.agent_a,
                    stance=DebateStance.SUPPORT,
                    claim=contradiction.claim_a[:500],
                    argument=contradiction.claim_a,
                    confidence=0.7,
                ),
                DebateEntry(
                    agent_name=contradiction.agent_b,
                    stance=DebateStance.CHALLENGE,
                    claim=contradiction.claim_b[:500],
                    argument=contradiction.claim_b,
                    confidence=0.7,
                ),
            ],
            resolution=f"{result.winner}: {result.resolution}",
        )
    except Exception as e:
        logger.warning("debate_resolution_failed", error=str(e))
        return DebateRound(
            round_number=round_number,
            topic=f"{contradiction.topic} ({contradiction.repo_name})",
            entries=[
                DebateEntry(
                    agent_name=contradiction.agent_a,
                    stance=DebateStance.SUPPORT,
                    claim=contradiction.claim_a[:500],
                    argument=contradiction.claim_a,
                    confidence=0.5,
                ),
                DebateEntry(
                    agent_name=contradiction.agent_b,
                    stance=DebateStance.CHALLENGE,
                    claim=contradiction.claim_b[:500],
                    argument=contradiction.claim_b,
                    confidence=0.5,
                ),
            ],
            resolution="Unresolved — insufficient evidence to determine winner",
        )


async def run_debate(
    repo_outputs: dict[str, list[AgentOutput]],
    max_rounds: int = 10,
) -> DebateRecord:
    """Run adversarial debate across all specialist outputs."""
    repo_names = list(repo_outputs.keys())
    contradictions = detect_contradictions(repo_outputs)
    logger.info("contradictions_detected", count=len(contradictions))

    rounds: list[DebateRound] = []
    for i, contradiction in enumerate(contradictions[:max_rounds]):
        debate_round = await _resolve_debate_round(contradiction, round_number=i + 1)
        rounds.append(debate_round)
        logger.info(
            "debate_round_complete",
            round=i + 1,
            topic=contradiction.topic,
            repo=contradiction.repo_name,
        )

    return DebateRecord(
        repo_names=repo_names,
        rounds=rounds,
        started_at=datetime.now(UTC),
        concluded_at=datetime.now(UTC),
    )
