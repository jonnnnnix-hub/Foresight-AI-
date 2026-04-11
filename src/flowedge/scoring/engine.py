"""Scoring engine — computes weighted scorecards from specialist outputs."""

from __future__ import annotations

import structlog

from flowedge.agents.base import AgentOutput
from flowedge.schemas.scoring import DEFAULT_DIMENSIONS, DimensionScore, RepoScorecard

logger = structlog.get_logger()


def compute_scorecards(
    repo_outputs: dict[str, list[AgentOutput]],
) -> list[RepoScorecard]:
    """Compute weighted scorecards for each repo from specialist scores.

    Collects all DimensionScore entries from specialist outputs,
    fills missing dimensions with 0, and computes weighted totals.
    """
    scorecards: list[RepoScorecard] = []

    for repo_name, outputs in repo_outputs.items():
        # Collect all scores from all agents
        dimension_scores: dict[str, DimensionScore] = {}
        for output in outputs:
            for score in output.scores:
                # If multiple agents score the same dimension, keep the latest
                # (should not happen with current design, but be safe)
                if score.dimension not in dimension_scores:
                    dimension_scores[score.dimension] = score

        # Ensure all default dimensions are present
        final_dimensions: list[DimensionScore] = []
        for dim_name, dim_weight in DEFAULT_DIMENSIONS:
            if dim_name in dimension_scores:
                existing = dimension_scores[dim_name]
                # Ensure correct weight from rubric
                final_dimensions.append(
                    DimensionScore(
                        dimension=existing.dimension,
                        score=existing.score,
                        weight=dim_weight,
                        rationale=existing.rationale,
                        evidence_refs=existing.evidence_refs,
                    )
                )
            else:
                # Dimension not scored by any agent — note the gap
                final_dimensions.append(
                    DimensionScore(
                        dimension=dim_name,
                        score=0.0,
                        weight=dim_weight,
                        rationale="Not scored by any specialist agent",
                    )
                )
                logger.warning(
                    "dimension_not_scored",
                    repo=repo_name,
                    dimension=dim_name,
                )

        # Find repo_url from any evidence pack reference
        repo_url = ""
        for output in outputs:
            raw = output.raw
            if isinstance(raw, dict) and "repo_url" in raw:
                repo_url = str(raw["repo_url"])
                break

        scorecard = RepoScorecard(
            repo_name=repo_name,
            repo_url=repo_url,
            dimensions=final_dimensions,
        )
        scorecards.append(scorecard)
        logger.info(
            "scorecard_computed",
            repo=repo_name,
            weighted_total=round(scorecard.weighted_total, 2),
        )

    # Sort by weighted total descending
    scorecards.sort(key=lambda s: s.weighted_total, reverse=True)
    return scorecards
