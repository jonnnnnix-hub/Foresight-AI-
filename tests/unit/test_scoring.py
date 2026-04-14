"""Tests for the scoring engine."""

from flowedge.agents.base import AgentOutput
from flowedge.schemas.scoring import DimensionScore
from flowedge.scoring.engine import compute_scorecards


def _make_output(agent_name: str, dimension: str, score: float) -> AgentOutput:
    return AgentOutput(
        agent_name=agent_name,
        summary="test",
        findings=["finding1"],
        scores=[
            DimensionScore(
                dimension=dimension,
                score=score,
                weight=0.15,
                rationale="test rationale",
            )
        ],
        evidence_refs=["file.py"],
    )


def test_compute_scorecards_single_repo() -> None:
    outputs = {
        "test_repo": [
            _make_output("research", "research_power", 8.0),
            _make_output("execution", "execution_realism", 6.0),
        ]
    }
    cards = compute_scorecards(outputs)
    assert len(cards) == 1
    card = cards[0]
    assert card.repo_name == "test_repo"
    assert card.weighted_total > 0


def test_compute_scorecards_ranking_order() -> None:
    outputs = {
        "high_repo": [
            _make_output("research", "research_power", 9.0),
            _make_output("execution", "execution_realism", 9.0),
        ],
        "low_repo": [
            _make_output("research", "research_power", 2.0),
            _make_output("execution", "execution_realism", 2.0),
        ],
    }
    cards = compute_scorecards(outputs)
    assert cards[0].repo_name == "high_repo"
    assert cards[1].repo_name == "low_repo"
    assert cards[0].weighted_total > cards[1].weighted_total


def test_missing_dimensions_get_zero() -> None:
    outputs = {
        "sparse_repo": [
            _make_output("research", "research_power", 7.0),
        ]
    }
    cards = compute_scorecards(outputs)
    card = cards[0]
    # Should have all 6 default dimensions
    assert len(card.dimensions) == 6
    # The ones not scored should be 0
    zero_dims = [d for d in card.dimensions if d.score == 0.0]
    assert len(zero_dims) == 5  # 6 total - 1 scored = 5 zeros


def test_weights_from_rubric() -> None:
    outputs = {
        "repo": [
            _make_output("execution", "execution_realism", 10.0),
        ]
    }
    cards = compute_scorecards(outputs)
    exec_dim = next(d for d in cards[0].dimensions if d.dimension == "execution_realism")
    assert exec_dim.weight == 0.20  # From DEFAULT_DIMENSIONS, not from agent
