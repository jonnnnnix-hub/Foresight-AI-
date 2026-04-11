"""Tests for the synthesis engine and export."""

import json
from pathlib import Path

from flowedge.schemas.agents import JudgeOutput
from flowedge.schemas.debate import DebateRecord, DebateRound
from flowedge.schemas.report import SynthesisReport
from flowedge.schemas.scoring import DimensionScore, RepoScorecard
from flowedge.synthesis.engine import build_synthesis_report
from flowedge.synthesis.export import export_json, export_markdown


def _make_scorecard(name: str, score: float) -> RepoScorecard:
    return RepoScorecard(
        repo_name=name,
        repo_url=f"https://github.com/test/{name}",
        dimensions=[
            DimensionScore(
                dimension="research_power", score=score, weight=0.15, rationale="test"
            ),
            DimensionScore(
                dimension="execution_realism", score=score - 1, weight=0.20, rationale="test"
            ),
        ],
    )


def _make_judge_output() -> JudgeOutput:
    return JudgeOutput(
        repo_ranking=["repoA", "repoB"],
        top_repo_by_category={"research": "repoA", "execution": "repoB"},
        borrow_avoid_replace={
            "repoA": {"borrow": ["backtester"], "avoid": [], "replace": ["UI"]},
            "repoB": {"borrow": ["execution_engine"], "avoid": ["docs"], "replace": []},
        },
        merged_architecture={"data": "repoA", "execution": "repoB"},
        mvp_build_order=["Step 1: Data pipeline", "Step 2: Execution"],
        do_not_build_wrong=["Do not skip execution realism"],
        risks_and_unknowns=["License risk on repoA"],
    )


def test_build_synthesis_report() -> None:
    scorecards = [_make_scorecard("repoA", 8.0), _make_scorecard("repoB", 5.0)]
    judge = _make_judge_output()
    debate = DebateRecord(repo_names=["repoA", "repoB"], rounds=[])

    report = build_synthesis_report(
        run_id="test-run",
        scorecards=scorecards,
        debate_record=debate,
        judge_output=judge,
    )

    assert report.run_id == "test-run"
    assert len(report.rankings) == 2
    assert report.rankings[0].repo_name == "repoA"
    assert report.rankings[0].rank == 1
    assert report.executive_summary != ""
    assert len(report.borrow_avoid_replace) == 2


def test_report_json_roundtrip() -> None:
    scorecards = [_make_scorecard("repo", 7.0)]
    judge = _make_judge_output()
    report = build_synthesis_report("rt-test", scorecards, None, judge)

    json_str = report.model_dump_json()
    rebuilt = SynthesisReport.model_validate_json(json_str)
    assert rebuilt.run_id == report.run_id
    assert len(rebuilt.rankings) == len(report.rankings)


def test_export_json(tmp_path: Path) -> None:
    scorecards = [_make_scorecard("repo", 7.0)]
    judge = _make_judge_output()
    report = build_synthesis_report("json-test", scorecards, None, judge)

    out_path = export_json(report, tmp_path / "report.json")
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["run_id"] == "json-test"
    assert "rankings" in data


def test_export_markdown(tmp_path: Path) -> None:
    scorecards = [_make_scorecard("repo", 7.0)]
    judge = _make_judge_output()
    report = build_synthesis_report("md-test", scorecards, None, judge)

    out_path = export_markdown(report, tmp_path / "report.md")
    assert out_path.exists()
    content = out_path.read_text()
    assert "# FlowEdge Analysis Report" in content
    assert "md-test" in content
    assert "Repo Ranking" in content
    assert "Borrow / Avoid / Replace" in content


def test_report_has_required_sections() -> None:
    """CLAUDE.md requires specific sections in every report."""
    scorecards = [_make_scorecard("repoA", 8.0), _make_scorecard("repoB", 5.0)]
    judge = _make_judge_output()
    debate = DebateRecord(
        repo_names=["repoA", "repoB"],
        rounds=[
            DebateRound(round_number=1, topic="test", resolution="Resolved: A wins")
        ],
    )
    report = build_synthesis_report("sections-test", scorecards, debate, judge)

    # Verify all required fields are populated
    assert report.executive_summary
    assert len(report.rankings) > 0
    assert len(report.borrow_avoid_replace) > 0
    assert len(report.debate_highlights) > 0
    assert report.merged_architecture
    assert report.mvp_recommendation
    assert report.do_not_build_wrong
    assert len(report.risks_and_unknowns) > 0
