"""CouncilEngine — orchestrates all specialists and produces a DailyReview.

Usage:
    engine = CouncilEngine()
    review = engine.run_review(result, config)
"""

from __future__ import annotations

import time
import uuid
from datetime import date
from typing import Any

import structlog

from flowedge.council.models import (
    DailyReview,
    Recommendation,
    ReviewStatus,
    Severity,
    SpecialistReview,
    TickerScorecard,
)
from flowedge.council.specialists import ALL_SPECIALISTS
from flowedge.council.specialists.base import BaseSpecialist
from flowedge.scanner.backtest.scalp_config import ScalpConfig
from flowedge.scanner.backtest.schemas import BacktestResult

logger = structlog.get_logger()


class CouncilEngine:
    """Runs all specialists and aggregates into a DailyReview."""

    def __init__(self, specialists: list[type[BaseSpecialist]] | None = None):
        self.specialist_classes = specialists or ALL_SPECIALISTS

    def run_review(
        self,
        result: BacktestResult,
        config: ScalpConfig,
        review_date: date | None = None,
        history: list[BacktestResult] | None = None,
    ) -> DailyReview:
        """Execute all specialists and produce a DailyReview."""
        t0 = time.perf_counter()
        review_date = review_date or date.today()
        history = history or []

        # ── Run each specialist ──────────────────────────────────
        specialist_reviews = []
        for cls in self.specialist_classes:
            specialist = cls()
            try:
                review = specialist.run(result, config, review_date, history)
                specialist_reviews.append(review)
                logger.info(
                    "council_specialist_done",
                    specialist=specialist.specialist_id,
                    health=review.health_score,
                    findings=len(review.findings),
                    recs=len(review.recommendations),
                    ms=review.computation_time_ms,
                )
            except Exception as exc:
                logger.error(
                    "council_specialist_error",
                    specialist=specialist.specialist_id,
                    error=str(exc),
                )

        # ── Aggregate health score ───────────────────────────────
        if specialist_reviews:
            overall_health = round(
                sum(r.health_score for r in specialist_reviews) / len(specialist_reviews),
                1,
            )
        else:
            overall_health = 0.0

        # ── Determine status ─────────────────────────────────────
        status = self._compute_status(overall_health, specialist_reviews)

        # ── Merge and rank recommendations ───────────────────────
        all_recs: list[Recommendation] = []
        for sr in specialist_reviews:
            all_recs.extend(sr.recommendations)
        # Sort by priority (1=highest) then confidence (desc)
        all_recs.sort(key=lambda r: (r.priority, -r.confidence))
        top_recs = all_recs[:10]  # Top 10 most important

        # ── Build ticker scorecards ──────────────────────────────
        ticker_scorecards = self._build_scorecards(result)

        # ── Today's performance snapshot ─────────────────────────
        today_trades = [
            t for t in result.trades
            if t.entry_date == review_date
        ]
        trades_today = len(today_trades)
        wins_today = sum(1 for t in today_trades if t.outcome.value == "win")
        pnl_today = sum(t.exit_value - t.cost_basis for t in today_trades)

        # ── Consensus summary ────────────────────────────────────
        consensus, dissents = self._build_consensus(specialist_reviews)

        elapsed = round((time.perf_counter() - t0) * 1000, 1)

        daily_review = DailyReview(
            review_id=f"council-{uuid.uuid4().hex[:8]}",
            review_date=review_date,
            status=status,
            overall_health=overall_health,
            config_used=config.model_dump(),
            trades_today=trades_today,
            wins_today=wins_today,
            pnl_today=round(pnl_today, 2),
            cumulative_trades=result.total_trades,
            cumulative_wr=result.win_rate,
            cumulative_pnl=round(result.ending_value - result.starting_capital, 2),
            specialist_reviews=specialist_reviews,
            top_recommendations=top_recs,
            ticker_scorecards=ticker_scorecards,
            consensus_summary=consensus,
            dissenting_views=dissents,
            computation_time_ms=elapsed,
        )

        logger.info(
            "council_review_complete",
            review_id=daily_review.review_id,
            status=status.value,
            health=overall_health,
            specialists=len(specialist_reviews),
            recommendations=len(top_recs),
            ms=elapsed,
        )

        return daily_review

    @staticmethod
    def _compute_status(
        health: float,
        reviews: list[SpecialistReview],
    ) -> ReviewStatus:
        """Derive status from health score and specialist severities."""
        # Any critical finding overrides to at least DEGRADED
        has_critical = any(
            r.severity == Severity.CRITICAL for r in reviews
        )
        has_warning = any(
            r.severity == Severity.WARNING for r in reviews
        )

        if has_critical or health < 40:
            return ReviewStatus.CRITICAL
        if has_warning or health < 60:
            return ReviewStatus.DEGRADED
        if health < 75:
            return ReviewStatus.WATCH
        return ReviewStatus.HEALTHY

    @staticmethod
    def _build_scorecards(result: BacktestResult) -> list[TickerScorecard]:
        """Build per-ticker scorecards from trade data."""
        from collections import defaultdict

        ticker_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnls": [], "holds": []}
        )
        for t in result.trades:
            d = ticker_data[t.ticker]
            d["trades"] += 1
            if t.outcome.value == "win":
                d["wins"] += 1
            d["pnls"].append(t.pnl_pct)
            d["holds"].append(t.hold_days)

        cards = []
        for ticker, d in sorted(ticker_data.items()):
            pnls = d["pnls"]
            wr = d["wins"] / d["trades"] if d["trades"] > 0 else 0
            cards.append(
                TickerScorecard(
                    ticker=ticker,
                    trades=d["trades"],
                    wins=d["wins"],
                    win_rate=round(wr, 3),
                    total_pnl=round(sum(pnls), 2),
                    avg_pnl_pct=round(sum(pnls) / len(pnls), 2) if pnls else 0,
                    best_trade_pct=round(max(pnls), 2) if pnls else 0,
                    worst_trade_pct=round(min(pnls), 2) if pnls else 0,
                    avg_hold_bars=round(
                        sum(d["holds"]) / len(d["holds"]), 1
                    ) if d["holds"] else 0,
                    recommendation="keep" if wr >= 0.6 else ("watch" if wr >= 0.4 else "remove"),
                )
            )
        return cards

    @staticmethod
    def _build_consensus(specialist_reviews: list[SpecialistReview]) -> tuple[str, list[str]]:
        """Synthesize consensus view and identify dissenting opinions."""
        if not specialist_reviews:
            return "No specialist reviews available.", []

        # Group by health tier
        healthy = [r for r in specialist_reviews if r.health_score >= 70]
        concerning = [r for r in specialist_reviews if 40 <= r.health_score < 70]
        critical = [r for r in specialist_reviews if r.health_score < 40]

        parts = []
        if healthy:
            names = ", ".join(r.specialist_name for r in healthy)
            parts.append(f"{len(healthy)} specialist(s) report healthy status ({names})")
        if concerning:
            names = ", ".join(r.specialist_name for r in concerning)
            parts.append(f"{len(concerning)} specialist(s) flag concerns ({names})")
        if critical:
            names = ", ".join(r.specialist_name for r in critical)
            parts.append(f"{len(critical)} specialist(s) report critical issues ({names})")

        consensus = ". ".join(parts) + "."

        # Find dissenting views (specialists with >20pt health gap)
        dissents = []
        avg_health = sum(r.health_score for r in specialist_reviews) / len(specialist_reviews)
        for r in specialist_reviews:
            if abs(r.health_score - avg_health) > 20:
                direction = "more optimistic" if r.health_score > avg_health else "more pessimistic"
                dissents.append(
                    f"{r.specialist_name} is {direction} "
                    f"(score {r.health_score:.0f} vs avg {avg_health:.0f}): "
                    f"{r.summary[:120]}"
                )

        return consensus, dissents
