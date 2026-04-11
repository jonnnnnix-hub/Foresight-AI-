"""Export synthesis reports as JSON and Markdown."""

from __future__ import annotations

import json
from pathlib import Path

from flowedge.schemas.report import SynthesisReport


def export_json(report: SynthesisReport, output_path: Path) -> Path:
    """Export report as JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = report.model_dump(mode="json")
    output_path.write_text(json.dumps(data, indent=2, default=str))
    return output_path


def export_markdown(report: SynthesisReport, output_path: Path) -> Path:
    """Export report as Markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append(f"# FlowEdge Analysis Report — {report.run_id}")
    lines.append(f"\n*Generated: {report.generated_at}*\n")

    # Executive Summary
    lines.append("## Executive Summary\n")
    lines.append(report.executive_summary)
    lines.append("")

    # Repo Ranking Table
    lines.append("## Repo Ranking\n")
    lines.append("| Rank | Repository | Score | Best For | Weakest At |")
    lines.append("|------|-----------|-------|----------|------------|")
    for entry in report.rankings:
        best = ", ".join(entry.best_for)
        worst = ", ".join(entry.weakest_at)
        lines.append(
            f"| {entry.rank} | {entry.repo_name} "
            f"| {entry.weighted_score} | {best} | {worst} |"
        )
    lines.append("")

    # Borrow / Avoid / Replace
    lines.append("## Borrow / Avoid / Replace Matrix\n")
    for bar in report.borrow_avoid_replace:
        lines.append(f"### {bar.repo_name}\n")
        if bar.borrow:
            lines.append("**Borrow:**")
            for item in bar.borrow:
                lines.append(f"- {item}")
        if bar.avoid:
            lines.append("\n**Avoid:**")
            for item in bar.avoid:
                lines.append(f"- {item}")
        if bar.replace:
            lines.append("\n**Replace:**")
            for item in bar.replace:
                lines.append(f"- {item}")
        lines.append("")

    # Debate Highlights
    if report.debate_highlights:
        lines.append("## Debate Highlights\n")
        for highlight in report.debate_highlights:
            lines.append(f"- {highlight}")
        lines.append("")

    # Real vs Hype
    if report.real_vs_hype:
        lines.append("## What Is Real vs Hype\n")
        for repo, assessment in report.real_vs_hype.items():
            lines.append(f"- **{repo}**: {assessment}")
        lines.append("")

    # Merged Architecture
    if report.merged_architecture:
        lines.append("## Recommended FlowEdge Merged Architecture\n")
        lines.append("| Component | Source |")
        lines.append("|-----------|--------|")
        for component, source in report.merged_architecture.items():
            lines.append(f"| {component} | {source} |")
        lines.append("")

    # MVP Recommendation
    if report.mvp_recommendation:
        lines.append("## Suggested Build Order for MVP\n")
        lines.append(report.mvp_recommendation)
        lines.append("")

    # Do Not Build This Wrong
    if report.do_not_build_wrong:
        lines.append("## Do Not Build This Wrong\n")
        lines.append(report.do_not_build_wrong)
        lines.append("")

    # Risks and Unknowns
    if report.risks_and_unknowns:
        lines.append("## Risks and Unknowns\n")
        for risk in report.risks_and_unknowns:
            lines.append(f"- {risk}")
        lines.append("")

    # Evidence Appendix
    if report.evidence_appendix:
        lines.append("## Appendix: Evidence References\n")
        for ref in report.evidence_appendix:
            lines.append(f"- {ref}")
        lines.append("")

    output_path.write_text("\n".join(lines))
    return output_path
