"""Tests for Pydantic schemas."""

from flowedge.schemas.evidence import EvidenceItem, EvidencePack, EvidenceSource
from flowedge.schemas.scoring import DimensionScore, RepoScorecard


def test_evidence_item_creation() -> None:
    item = EvidenceItem(
        source=EvidenceSource.README,
        file_path="README.md",
        content_snippet="# Example project",
        claim="Project has a README",
        confidence=1.0,
    )
    assert item.source == EvidenceSource.README
    assert item.confidence == 1.0


def test_evidence_pack_defaults() -> None:
    pack = EvidencePack(repo_url="https://github.com/test/repo", repo_name="repo")
    assert pack.items == []
    assert pack.total_files == 0


def test_dimension_score_bounds() -> None:
    score = DimensionScore(
        dimension="research_power",
        score=7.5,
        weight=0.15,
        rationale="Strong backtesting framework",
    )
    assert 0 <= score.score <= 10
    assert 0 <= score.weight <= 1


def test_scorecard_weighted_total() -> None:
    card = RepoScorecard(
        repo_name="test",
        repo_url="https://github.com/test/repo",
        dimensions=[
            DimensionScore(
                dimension="research_power",
                score=8.0,
                weight=0.5,
                rationale="Good",
            ),
            DimensionScore(
                dimension="execution_realism",
                score=6.0,
                weight=0.5,
                rationale="OK",
            ),
        ],
    )
    assert card.weighted_total == 7.0


def test_scorecard_empty() -> None:
    card = RepoScorecard(
        repo_name="empty",
        repo_url="https://github.com/test/empty",
    )
    assert card.weighted_total == 0.0
