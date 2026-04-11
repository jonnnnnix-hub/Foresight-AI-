"""Tests for the debate engine."""

from flowedge.agents.base import AgentOutput
from flowedge.debate.engine import ContradictionCandidate, detect_contradictions
from flowedge.schemas.scoring import DimensionScore


def _make_output(
    agent_name: str, dimension: str, score: float, summary: str = "test"
) -> AgentOutput:
    return AgentOutput(
        agent_name=agent_name,
        summary=summary,
        findings=["finding"],
        scores=[
            DimensionScore(
                dimension=dimension,
                score=score,
                weight=0.15,
                rationale=f"Score of {score}",
            )
        ],
        evidence_refs=[],
    )


def test_detect_score_contradictions() -> None:
    """Agents with >3 point gap on same dimension should trigger contradiction."""
    outputs = {
        "repo": [
            _make_output("agent_a", "execution_realism", 9.0),
            _make_output("agent_b", "execution_realism", 3.0),
        ]
    }
    contradictions = detect_contradictions(outputs)
    assert len(contradictions) >= 1
    c = contradictions[0]
    assert c.topic == "execution_realism"
    assert c.repo_name == "repo"


def test_no_contradiction_when_close() -> None:
    """Agents within 3 points should not trigger contradiction."""
    outputs = {
        "repo": [
            _make_output("agent_a", "execution_realism", 7.0),
            _make_output("agent_b", "execution_realism", 5.0),
        ]
    }
    contradictions = detect_contradictions(outputs)
    score_contradictions = [c for c in contradictions if c.topic == "execution_realism"]
    assert len(score_contradictions) == 0


def test_contradiction_candidate_fields() -> None:
    c = ContradictionCandidate(
        topic="test_topic",
        repo_name="test_repo",
        agent_a="alice",
        claim_a="It's great",
        agent_b="bob",
        claim_b="It's terrible",
    )
    assert c.topic == "test_topic"
    assert c.agent_a == "alice"
    assert c.agent_b == "bob"
