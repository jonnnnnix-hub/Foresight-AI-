"""FLUX Order Flow Analyst — evaluates tape confirmation quality.

Analyzes backtest trades to determine whether FLUX (order flow)
data at entry time correlated with trade outcomes. Produces findings
on FLUX confirmation rates, win-rate lift from FLUX-confirmed entries,
and whether FLUX vetoes are filtering losers effectively.

Trades must have flux_strength and flux_bias fields (recorded at entry
by the live scanner) for this specialist to produce meaningful analysis.
"""

from __future__ import annotations

import logging
from datetime import date

from flowedge.council.models import (
    Finding,
    Recommendation,
    RecommendationType,
    Severity,
    SpecialistReview,
)
from flowedge.council.specialists.base import BaseSpecialist
from flowedge.scanner.backtest.scalp_config import ScalpConfig
from flowedge.scanner.backtest.schemas import BacktestResult, TradeOutcome

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────

_FLUX_CONFIRMED_THRESHOLD = 5.0   # FLUX score >= this = "confirmed"
_FLUX_STRONG_THRESHOLD = 7.0      # FLUX score >= this = "strong confirmation"
_MIN_TRADES_FOR_ANALYSIS = 10     # Need at least this many FLUX-tagged trades
_EXPECTED_WR_LIFT = 0.05          # FLUX should improve WR by at least 5pp


class FluxAnalyst(BaseSpecialist):
    """Specialist that evaluates FLUX order flow confirmation quality."""

    name: str = "FLUX Order Flow Analyst"
    specialist_id: str = "flux_analyst"

    def analyze(
        self,
        result: BacktestResult,
        config: ScalpConfig,
        review_date: date,
        history: list[BacktestResult],
    ) -> SpecialistReview:
        findings: list[Finding] = []
        recommendations: list[Recommendation] = []
        metrics: dict[str, float | str] = {}

        # Extract FLUX data from trade records
        flux_trades = []
        no_flux_trades = []
        for trade in result.trades:
            # Trade notes may contain flux data as JSON or in extra fields
            flux_strength = _extract_flux_field(trade, "flux_strength")
            flux_bias = _extract_flux_field(trade, "flux_bias")

            if flux_strength is not None and flux_strength > 0:
                flux_trades.append({
                    "trade": trade,
                    "strength": flux_strength,
                    "bias": str(flux_bias or ""),
                    "is_win": trade.outcome == TradeOutcome.WIN,
                })
            else:
                no_flux_trades.append(trade)

        metrics["total_trades"] = result.total_trades
        metrics["flux_tagged_trades"] = len(flux_trades)
        metrics["no_flux_trades"] = len(no_flux_trades)

        # If insufficient FLUX data, report and exit early
        if len(flux_trades) < _MIN_TRADES_FOR_ANALYSIS:
            findings.append(Finding(
                title="Insufficient FLUX data for analysis",
                detail=(
                    f"Only {len(flux_trades)} trades have FLUX data recorded "
                    f"at entry (need {_MIN_TRADES_FOR_ANALYSIS}+). "
                    f"FLUX analysis will become meaningful as more live "
                    f"trades accumulate with FLUX telemetry."
                ),
                severity=Severity.INFO,
                metric_name="flux_tagged_trades",
                metric_value=len(flux_trades),
                threshold=_MIN_TRADES_FOR_ANALYSIS,
            ))

            return SpecialistReview(
                specialist_name=self.name,
                specialist_id=self.specialist_id,
                review_date=review_date,
                health_score=50.0,
                severity=Severity.INFO,
                summary=(
                    f"FLUX data available on {len(flux_trades)}/{result.total_trades} "
                    f"trades. Need {_MIN_TRADES_FOR_ANALYSIS}+ for meaningful analysis."
                ),
                findings=findings,
                recommendations=recommendations,
                metrics=metrics,
            )

        # ── Core Analysis ────────────────────────────────────────

        # 1. FLUX confirmation rate
        confirmed = [t for t in flux_trades if t["strength"] >= _FLUX_CONFIRMED_THRESHOLD]
        strong = [t for t in flux_trades if t["strength"] >= _FLUX_STRONG_THRESHOLD]
        weak = [t for t in flux_trades if t["strength"] < _FLUX_CONFIRMED_THRESHOLD]

        metrics["flux_confirmed_count"] = len(confirmed)
        metrics["flux_strong_count"] = len(strong)
        metrics["flux_weak_count"] = len(weak)

        confirmation_rate = len(confirmed) / len(flux_trades) if flux_trades else 0
        metrics["flux_confirmation_rate"] = round(confirmation_rate, 4)

        # 2. Win rate comparison: FLUX-confirmed vs weak/no FLUX
        confirmed_wins = sum(1 for t in confirmed if t["is_win"])
        confirmed_wr = confirmed_wins / len(confirmed) if confirmed else 0

        weak_wins = sum(1 for t in weak if t["is_win"])
        weak_wr = weak_wins / len(weak) if weak else 0

        strong_wins = sum(1 for t in strong if t["is_win"])
        strong_wr = strong_wins / len(strong) if strong else 0

        baseline_wr = result.win_rate

        metrics["confirmed_wr"] = round(confirmed_wr, 4)
        metrics["weak_wr"] = round(weak_wr, 4)
        metrics["strong_wr"] = round(strong_wr, 4)
        metrics["wr_lift"] = round(confirmed_wr - baseline_wr, 4)

        # 3. Generate findings
        findings.extend(
            self._assess_confirmation_quality(
                confirmed_wr, weak_wr, strong_wr, baseline_wr,
                len(confirmed), len(weak), len(strong),
            )
        )
        findings.extend(
            self._assess_bias_accuracy(flux_trades)
        )

        # 4. Generate recommendations
        recommendations.extend(
            self._recommend_flux_adjustments(
                confirmed_wr, weak_wr, baseline_wr,
                confirmation_rate, len(flux_trades),
            )
        )

        # 5. Health score
        health = self._compute_health(
            confirmed_wr, weak_wr, baseline_wr,
            confirmation_rate, len(flux_trades),
        )
        metrics["health_score"] = health

        severity = Severity.INFO
        if any(f.severity == Severity.CRITICAL for f in findings):
            severity = Severity.CRITICAL
        elif any(f.severity == Severity.WARNING for f in findings):
            severity = Severity.WARNING

        summary = (
            f"FLUX confirmation rate: {confirmation_rate:.0%}. "
            f"Confirmed WR: {confirmed_wr:.1%} vs baseline {baseline_wr:.1%} "
            f"(lift: {confirmed_wr - baseline_wr:+.1%}). "
            f"Strong FLUX WR: {strong_wr:.1%} ({len(strong)} trades)."
        )

        return SpecialistReview(
            specialist_name=self.name,
            specialist_id=self.specialist_id,
            review_date=review_date,
            health_score=health,
            severity=severity,
            summary=summary,
            findings=findings,
            recommendations=recommendations,
            metrics=metrics,
        )

    # ── Finding generators ────────────────────────────────────────

    @staticmethod
    def _assess_confirmation_quality(
        confirmed_wr: float,
        weak_wr: float,
        strong_wr: float,
        baseline_wr: float,
        n_confirmed: int,
        n_weak: int,
        n_strong: int,
    ) -> list[Finding]:
        findings: list[Finding] = []
        lift = confirmed_wr - baseline_wr

        if lift >= _EXPECTED_WR_LIFT:
            findings.append(Finding(
                title="FLUX confirmation adds predictive value",
                detail=(
                    f"FLUX-confirmed trades (score >= {_FLUX_CONFIRMED_THRESHOLD}) "
                    f"win at {confirmed_wr:.1%} vs {baseline_wr:.1%} baseline "
                    f"(+{lift:.1%} lift). FLUX is correctly identifying higher-"
                    f"probability setups."
                ),
                severity=Severity.INFO,
                metric_name="flux_wr_lift",
                metric_value=round(lift, 4),
                threshold=_EXPECTED_WR_LIFT,
                evidence=[
                    f"confirmed_wr={confirmed_wr:.4f} (n={n_confirmed})",
                    f"baseline_wr={baseline_wr:.4f}",
                    f"strong_wr={strong_wr:.4f} (n={n_strong})",
                ],
            ))
        elif lift < 0:
            findings.append(Finding(
                title="FLUX confirmation does NOT improve win rate",
                detail=(
                    f"FLUX-confirmed trades win at {confirmed_wr:.1%}, "
                    f"which is LOWER than baseline {baseline_wr:.1%} "
                    f"({lift:+.1%}). FLUX may be adding noise rather "
                    f"than signal. Consider reducing flux_weight."
                ),
                severity=Severity.WARNING,
                metric_name="flux_wr_lift",
                metric_value=round(lift, 4),
                threshold=0.0,
                evidence=[
                    f"confirmed_wr={confirmed_wr:.4f} (n={n_confirmed})",
                    f"weak_wr={weak_wr:.4f} (n={n_weak})",
                    f"baseline_wr={baseline_wr:.4f}",
                ],
            ))

        # Check if weak FLUX trades are indeed worse
        if n_weak >= 5 and n_confirmed >= 5:
            spread = confirmed_wr - weak_wr
            if spread > 0.10:
                findings.append(Finding(
                    title="FLUX separates winners from losers effectively",
                    detail=(
                        f"Confirmed FLUX WR ({confirmed_wr:.1%}) exceeds "
                        f"weak FLUX WR ({weak_wr:.1%}) by {spread:.1%}. "
                        f"The FLUX signal has discriminative power."
                    ),
                    severity=Severity.INFO,
                    metric_name="flux_spread",
                    metric_value=round(spread, 4),
                    evidence=[
                        f"confirmed_wr={confirmed_wr:.4f} (n={n_confirmed})",
                        f"weak_wr={weak_wr:.4f} (n={n_weak})",
                    ],
                ))

        return findings

    @staticmethod
    def _assess_bias_accuracy(
        flux_trades: list[dict],
    ) -> list[Finding]:
        """Check if FLUX buy/sell bias correctly predicts direction."""
        findings: list[Finding] = []

        buy_trades = [t for t in flux_trades if "buy" in t["bias"]]
        sell_trades = [t for t in flux_trades if "sell" in t["bias"]]

        if len(buy_trades) >= 5:
            buy_wr = sum(1 for t in buy_trades if t["is_win"]) / len(buy_trades)
            findings.append(Finding(
                title=f"FLUX buy-bias accuracy: {buy_wr:.0%}",
                detail=(
                    f"{len(buy_trades)} trades entered with FLUX buy bias: "
                    f"{buy_wr:.1%} win rate."
                ),
                severity=Severity.INFO,
                metric_name="flux_buy_accuracy",
                metric_value=round(buy_wr, 4),
                evidence=[f"buy_trades={len(buy_trades)}", f"buy_wr={buy_wr:.4f}"],
            ))

        if len(sell_trades) >= 5:
            sell_wr = sum(1 for t in sell_trades if t["is_win"]) / len(sell_trades)
            findings.append(Finding(
                title=f"FLUX sell-bias accuracy: {sell_wr:.0%}",
                detail=(
                    f"{len(sell_trades)} trades entered with FLUX sell bias: "
                    f"{sell_wr:.1%} win rate. (Bearish FLUX on bullish entries "
                    f"should have low WR — if high, FLUX veto is not working.)"
                ),
                severity=(
                    Severity.WARNING if sell_wr > 0.60 else Severity.INFO
                ),
                metric_name="flux_sell_accuracy",
                metric_value=round(sell_wr, 4),
                evidence=[
                    f"sell_trades={len(sell_trades)}",
                    f"sell_wr={sell_wr:.4f}",
                ],
            ))

        return findings

    # ── Recommendation generators ─────────────────────────────────

    @staticmethod
    def _recommend_flux_adjustments(
        confirmed_wr: float,
        weak_wr: float,
        baseline_wr: float,
        confirmation_rate: float,
        n_flux: int,
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []
        lift = confirmed_wr - baseline_wr

        if lift < 0 and n_flux >= 20:
            recs.append(Recommendation(
                title="Reduce FLUX weight in composite scoring",
                rationale=(
                    f"FLUX-confirmed entries underperform baseline by "
                    f"{abs(lift):.1%}. Reducing flux_weight from 0.20 to "
                    f"0.10 will limit FLUX's negative contribution."
                ),
                rec_type=RecommendationType.PARAMETER_CHANGE,
                priority=2,
                confidence=0.6,
                current_value="flux_weight=0.20",
                suggested_value="flux_weight=0.10",
                expected_impact="Reduce FLUX's negative drag on composite score.",
                evidence=[
                    f"confirmed_wr={confirmed_wr:.4f}",
                    f"baseline_wr={baseline_wr:.4f}",
                    f"lift={lift:+.4f}",
                ],
            ))
        elif lift >= 0.10 and n_flux >= 20:
            recs.append(Recommendation(
                title="Consider increasing FLUX weight",
                rationale=(
                    f"FLUX-confirmed entries outperform baseline by "
                    f"{lift:.1%}. Increasing flux_weight from 0.20 to "
                    f"0.25 could improve overall signal quality."
                ),
                rec_type=RecommendationType.PARAMETER_CHANGE,
                priority=3,
                confidence=0.5,
                current_value="flux_weight=0.20",
                suggested_value="flux_weight=0.25",
                expected_impact="Boost composite scores for FLUX-confirmed setups.",
                evidence=[
                    f"confirmed_wr={confirmed_wr:.4f}",
                    f"baseline_wr={baseline_wr:.4f}",
                    f"lift={lift:+.4f}",
                ],
            ))

        if not recs:
            recs.append(Recommendation(
                title="No FLUX weight changes recommended",
                rationale="FLUX contribution is within acceptable bounds.",
                rec_type=RecommendationType.NO_ACTION,
                priority=5,
                confidence=0.7,
                evidence=[f"lift={lift:+.4f}", f"n_flux={n_flux}"],
            ))

        return recs

    # ── Health score ──────────────────────────────────────────────

    @staticmethod
    def _compute_health(
        confirmed_wr: float,
        weak_wr: float,
        baseline_wr: float,
        confirmation_rate: float,
        n_flux: int,
    ) -> float:
        """0-100 health score for FLUX signal quality.

        Components:
        - WR lift (40%): how much FLUX confirmation improves WR
        - Separation (30%): confirmed WR vs weak WR spread
        - Data coverage (30%): what fraction of trades have FLUX data
        """
        lift = confirmed_wr - baseline_wr
        lift_score = min(100, max(0, (lift + 0.10) / 0.20 * 100))

        spread = confirmed_wr - weak_wr
        sep_score = min(100, max(0, (spread + 0.05) / 0.25 * 100))

        coverage_score = min(100, n_flux / 30 * 100)

        health = 0.40 * lift_score + 0.30 * sep_score + 0.30 * coverage_score
        return round(max(0, min(100, health)), 1)


def _extract_flux_field(trade: object, field: str) -> float | str | None:
    """Extract a FLUX field from a trade object.

    Trade may be a Pydantic model, dict, or have data in notes.
    """
    # Direct attribute
    if hasattr(trade, field):
        val = getattr(trade, field)
        if val is not None:
            return val  # type: ignore[no-any-return]

    # Dict-like access
    if hasattr(trade, "get"):
        val = trade.get(field)  # type: ignore[union-attr]
        if val is not None:
            return val  # type: ignore[no-any-return]

    # Check notes for JSON
    notes = getattr(trade, "notes", []) or []
    import json
    for note in notes:
        try:
            parsed = json.loads(note) if isinstance(note, str) and note.startswith("{") else {}
            if field in parsed:
                return parsed[field]  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError):
            continue

    return None
