"""Synthesis engine — builds final report from scores, debate, and judge output."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from flowedge.schemas.agents import JudgeOutput
from flowedge.schemas.debate import DebateRecord
from flowedge.schemas.report import BorrowAvoidReplace, RankingEntry, SynthesisReport
from flowedge.schemas.scoring import RepoScorecard

logger = structlog.get_logger()


def build_synthesis_report(
    run_id: str,
    scorecards: list[RepoScorecard],
    debate_record: DebateRecord | None,
    judge_output: JudgeOutput,
) -> SynthesisReport:
    """Assemble a final synthesis report from all pipeline outputs."""

    # Build rankings from scorecards
    rankings: list[RankingEntry] = []
    for rank, card in enumerate(scorecards, 1):
        best_dims = sorted(card.dimensions, key=lambda d: d.score, reverse=True)
        worst_dims = sorted(card.dimensions, key=lambda d: d.score)
        rankings.append(
            RankingEntry(
                rank=rank,
                repo_name=card.repo_name,
                weighted_score=round(card.weighted_total, 2),
                best_for=[d.dimension for d in best_dims[:2]],
                weakest_at=[d.dimension for d in worst_dims[:2]],
            )
        )

    # Build borrow/avoid/replace from judge output
    bar_list: list[BorrowAvoidReplace] = []
    for repo_name, bar_dict in judge_output.borrow_avoid_replace.items():
        bar_list.append(
            BorrowAvoidReplace(
                repo_name=repo_name,
                borrow=bar_dict.get("borrow", []),
                avoid=bar_dict.get("avoid", []),
                replace=bar_dict.get("replace", []),
            )
        )

    # Extract debate highlights
    debate_highlights: list[str] = []
    if debate_record:
        for rd in debate_record.rounds:
            if rd.resolution:
                debate_highlights.append(f"[{rd.topic}] {rd.resolution}")

    # Build executive summary
    if rankings:
        top = rankings[0]
        exec_summary = (
            f"Analysis of {len(rankings)} repositories complete. "
            f"Top-ranked: {top.repo_name} (score: {top.weighted_score}). "
            f"Best dimensions: {', '.join(top.best_for)}."
        )
    else:
        exec_summary = "No repositories were successfully analyzed."

    # Determine best repo per category from judge output
    real_vs_hype: dict[str, str] = {}
    for repo_name, bar in judge_output.borrow_avoid_replace.items():
        borrow_count = len(bar.get("borrow", []))
        avoid_count = len(bar.get("avoid", []))
        if borrow_count > avoid_count:
            real_vs_hype[repo_name] = "More real than hype"
        elif avoid_count > borrow_count:
            real_vs_hype[repo_name] = "More hype than real"
        else:
            real_vs_hype[repo_name] = "Mixed"

    report = SynthesisReport(
        run_id=run_id,
        generated_at=datetime.now(UTC),
        executive_summary=exec_summary,
        rankings=rankings,
        borrow_avoid_replace=bar_list,
        debate_highlights=debate_highlights,
        real_vs_hype=real_vs_hype,
        merged_architecture=judge_output.merged_architecture,
        mvp_recommendation=(
            "\n".join(judge_output.mvp_build_order)
            if judge_output.mvp_build_order
            else ""
        ),
        do_not_build_wrong=(
            "\n".join(judge_output.do_not_build_wrong)
            if judge_output.do_not_build_wrong
            else ""
        ),
        risks_and_unknowns=judge_output.risks_and_unknowns,
        evidence_appendix=judge_output.evidence_refs,
    )

    logger.info("report_built", run_id=run_id, ranking_count=len(rankings))
    return report
