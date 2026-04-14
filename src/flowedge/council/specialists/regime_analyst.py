"""Market Regime Analyst — council specialist for regime and time-period analysis.

Analyzes backtest performance across market regimes, calendar periods (year,
quarter, month), and temporal trends to detect drift, seasonal patterns, and
regime sensitivity.
"""

from __future__ import annotations

import ast
import statistics
from collections import defaultdict
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
from flowedge.scanner.backtest.schemas import BacktestResult, BacktestTrade


def _quarter_key(d: date) -> str:
    """Return 'YYYY-Q#' string for a date."""
    return f"{d.year}-Q{(d.month - 1) // 3 + 1}"


def _month_key(d: date) -> str:
    """Return 'YYYY-MM' string for a date."""
    return f"{d.year}-{d.month:02d}"


def _parse_by_year(notes: list[str]) -> dict[str, dict[str, float]]:
    """Extract by_year dict from notes list.

    Notes may contain entries like:
        by_year={"2022": {"trades": 2, "wr": 0.5, "pnl": 514}, ...}
    """
    for note in notes:
        if note.strip().startswith("by_year="):
            raw = note.strip()[len("by_year="):]
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                continue
    return {}


def _safe_mean(values: list[float]) -> float:
    """Mean that returns 0.0 for empty lists."""
    return statistics.mean(values) if values else 0.0


def _safe_stdev(values: list[float]) -> float:
    """Stdev that returns 0.0 for fewer than 2 data points."""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


# ---------------------------------------------------------------------------
# Specialist
# ---------------------------------------------------------------------------


class RegimeAnalyst(BaseSpecialist):
    """Analyzes performance across market regimes and time periods.

    Detects drift, seasonal patterns, regime sensitivity, and walk-forward
    divergence to assess model robustness over changing market conditions.
    """

    name: str = "Market Regime Analyst"
    specialist_id: str = "regime_analyst"

    # ── thresholds --------------------------------------------------------
    RECENT_MONTHS = 3
    DEGRADATION_THRESHOLD = 0.15  # 15 pp win-rate drop = significant
    MIN_TRADES_PER_BUCKET = 3  # ignore periods with fewer trades

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

        # ── 1. Parse by_year from notes ──────────────────────────
        by_year = _parse_by_year(result.notes)
        if by_year:
            metrics["years_covered"] = len(by_year)
            year_findings, year_recs = self._analyze_yearly(by_year)
            findings.extend(year_findings)
            recommendations.extend(year_recs)

        # ── 2-3. Group trades by quarter and month ───────────────
        trades = result.trades
        quarterly = self._group_trades_by(trades, _quarter_key)
        monthly = self._group_trades_by(trades, _month_key)

        if quarterly:
            q_findings, q_recs = self._analyze_periodic(
                quarterly, "quarter", metrics,
            )
            findings.extend(q_findings)
            recommendations.extend(q_recs)

        if monthly:
            m_findings, m_recs = self._analyze_periodic(
                monthly, "month", metrics,
            )
            findings.extend(m_findings)
            recommendations.extend(m_recs)

        # ── 4. Regime-specific performance ───────────────────────
        if result.by_regime:
            regime_findings, regime_recs = self._analyze_regimes(result.by_regime)
            findings.extend(regime_findings)
            recommendations.extend(regime_recs)
            metrics["regimes_analyzed"] = len(result.by_regime)
        else:
            # Try grouping trades by their regime field
            regime_from_trades = self._group_trades_by_regime(trades)
            if regime_from_trades:
                regime_findings, regime_recs = self._analyze_regimes(
                    regime_from_trades,
                )
                findings.extend(regime_findings)
                recommendations.extend(regime_recs)
                metrics["regimes_analyzed"] = len(regime_from_trades)

        # ── 5. Walk-forward drift ────────────────────────────────
        if by_year:
            wf_findings = self._check_walk_forward_drift(by_year, metrics)
            findings.extend(wf_findings)

        # ── 6-7. Best/worst periods and recent degradation ───────
        if monthly:
            period_findings = self._best_worst_and_recent(
                monthly, metrics, review_date,
            )
            findings.extend(period_findings)

        # ── 8. Health score ──────────────────────────────────────
        consistency = self._score_consistency(monthly, quarterly)
        trend = self._score_trend(monthly)
        seasonal = self._score_seasonal_stability(monthly)

        health = round(
            consistency * 0.40 + trend * 0.30 + seasonal * 0.30, 1,
        )
        metrics["consistency_score"] = round(consistency, 1)
        metrics["trend_score"] = round(trend, 1)
        metrics["seasonal_stability_score"] = round(seasonal, 1)

        # ── determine severity ───────────────────────────────────
        if health >= 70:
            severity = Severity.INFO
        elif health >= 45:
            severity = Severity.WARNING
        else:
            severity = Severity.CRITICAL

        # ── 9-10. Additional recommendations ─────────────────────
        if health < 50:
            recommendations.append(
                Recommendation(
                    title="Consider regime-adaptive parameter sets",
                    rationale=(
                        f"Overall regime health score is {health}/100. "
                        "Performance varies significantly across time periods "
                        "or market regimes, suggesting a single parameter set "
                        "may not be robust."
                    ),
                    rec_type=RecommendationType.REGIME_ADAPTATION,
                    priority=2,
                    confidence=0.7,
                    expected_impact="Improved consistency across market conditions",
                    evidence=[f"health_score={health}"],
                ),
            )

        # ── build summary ────────────────────────────────────────
        n_findings = len(findings)
        critical_count = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        summary_parts = [f"Regime analysis produced {n_findings} findings (health {health}/100)."]
        if critical_count:
            summary_parts.append(f"{critical_count} critical issue(s) detected.")
        if not trades:
            summary_parts.append("No trade data available for temporal analysis.")

        return SpecialistReview(
            specialist_name=self.name,
            specialist_id=self.specialist_id,
            review_date=review_date,
            health_score=health,
            severity=severity,
            summary=" ".join(summary_parts),
            findings=findings,
            recommendations=recommendations,
            metrics=metrics,
        )

    # =====================================================================
    # Internal helpers
    # =====================================================================

    @staticmethod
    def _group_trades_by(
        trades: list[BacktestTrade],
        key_fn: callable,  # type: ignore[valid-type]
    ) -> dict[str, list[BacktestTrade]]:
        """Group trades into buckets using *key_fn(trade.entry_date)*."""
        buckets: dict[str, list[BacktestTrade]] = defaultdict(list)
        for t in trades:
            buckets[key_fn(t.entry_date)].append(t)
        return dict(sorted(buckets.items()))

    @staticmethod
    def _group_trades_by_regime(
        trades: list[BacktestTrade],
    ) -> dict[str, dict[str, float]]:
        """Build regime stats from individual trade regime tags."""
        regime_trades: dict[str, list[BacktestTrade]] = defaultdict(list)
        for t in trades:
            regime = t.regime.strip() if t.regime else ""
            if regime:
                regime_trades[regime].append(t)
        if not regime_trades:
            return {}
        result: dict[str, dict[str, float]] = {}
        for regime, rtrades in regime_trades.items():
            wins = sum(1 for t in rtrades if t.outcome.value == "win")
            total = len(rtrades)
            wr = wins / total if total else 0.0
            avg_pnl = _safe_mean([t.pnl_pct for t in rtrades])
            result[regime] = {
                "trades": float(total),
                "wr": round(wr, 4),
                "avg_pnl_pct": round(avg_pnl, 4),
            }
        return result

    @staticmethod
    def _bucket_stats(
        trades: list[BacktestTrade],
    ) -> dict[str, float]:
        """Compute summary stats for a group of trades."""
        total = len(trades)
        wins = sum(1 for t in trades if t.outcome.value == "win")
        wr = wins / total if total else 0.0
        avg_pnl = _safe_mean([t.pnl_pct for t in trades])
        total_pnl = sum(t.pnl_pct for t in trades)
        return {
            "trades": float(total),
            "wins": float(wins),
            "wr": round(wr, 4),
            "avg_pnl_pct": round(avg_pnl, 4),
            "total_pnl_pct": round(total_pnl, 4),
        }

    # ── yearly analysis ---------------------------------------------------

    def _analyze_yearly(
        self,
        by_year: dict[str, dict[str, float]],
    ) -> tuple[list[Finding], list[Recommendation]]:
        findings: list[Finding] = []
        recommendations: list[Recommendation] = []

        years_sorted = sorted(by_year.keys())
        if len(years_sorted) < 2:
            return findings, recommendations

        win_rates = []
        for yr in years_sorted:
            data = by_year[yr]
            # Support both "wr" and "win_rate" key formats
            wr = data.get("wr", data.get("win_rate", 0.0))
            trades = data.get("trades", 0)
            win_rates.append((yr, wr, trades))

        # Check for year-over-year degradation
        first_wr = win_rates[0][1]
        last_wr = win_rates[-1][1]
        wr_delta = last_wr - first_wr

        evidence = [f"{yr}: WR={wr:.1%}, trades={int(t)}" for yr, wr, t in win_rates]

        if wr_delta < -self.DEGRADATION_THRESHOLD:
            findings.append(Finding(
                title="Year-over-year win rate decline",
                detail=(
                    f"Win rate declined from {first_wr:.1%} ({years_sorted[0]}) "
                    f"to {last_wr:.1%} ({years_sorted[-1]}), a drop of "
                    f"{abs(wr_delta):.1%}. This may indicate model decay or "
                    "changing market conditions."
                ),
                severity=Severity.WARNING,
                metric_name="yearly_wr_delta",
                metric_value=round(wr_delta, 4),
                threshold=-self.DEGRADATION_THRESHOLD,
                evidence=evidence,
            ))
            recommendations.append(Recommendation(
                title="Investigate year-over-year performance decay",
                rationale=(
                    f"Win rate dropped {abs(wr_delta):.1%} from {years_sorted[0]} "
                    f"to {years_sorted[-1]}. Re-optimize parameters on recent data "
                    "or add regime-conditional logic."
                ),
                rec_type=RecommendationType.REGIME_ADAPTATION,
                priority=2,
                confidence=0.65,
                current_value=f"WR {first_wr:.1%} -> {last_wr:.1%}",
                expected_impact="Restore performance consistency across years",
                evidence=evidence,
            ))
        elif wr_delta > self.DEGRADATION_THRESHOLD:
            findings.append(Finding(
                title="Year-over-year win rate improvement",
                detail=(
                    f"Win rate improved from {first_wr:.1%} ({years_sorted[0]}) "
                    f"to {last_wr:.1%} ({years_sorted[-1]}), a gain of "
                    f"{wr_delta:.1%}. Model may be well-adapted to recent conditions."
                ),
                severity=Severity.INFO,
                metric_name="yearly_wr_delta",
                metric_value=round(wr_delta, 4),
                evidence=evidence,
            ))

        # Check for years with very few trades
        low_count_years = [
            (yr, int(t)) for yr, _wr, t in win_rates
            if t < self.MIN_TRADES_PER_BUCKET
        ]
        if low_count_years:
            findings.append(Finding(
                title="Low trade count in some years",
                detail=(
                    f"{len(low_count_years)} year(s) have fewer than "
                    f"{self.MIN_TRADES_PER_BUCKET} trades, making statistics "
                    "unreliable for those periods."
                ),
                severity=Severity.INFO,
                metric_name="low_trade_years",
                metric_value=len(low_count_years),
                threshold=self.MIN_TRADES_PER_BUCKET,
                evidence=[f"{yr}: {cnt} trades" for yr, cnt in low_count_years],
            ))

        return findings, recommendations

    # ── periodic (quarter / month) analysis -------------------------------

    def _analyze_periodic(
        self,
        buckets: dict[str, list[BacktestTrade]],
        period_label: str,
        metrics: dict[str, float | str],
    ) -> tuple[list[Finding], list[Recommendation]]:
        findings: list[Finding] = []
        recommendations: list[Recommendation] = []

        stats_by_period: dict[str, dict[str, float]] = {}
        for period, trades in buckets.items():
            stats_by_period[period] = self._bucket_stats(trades)

        # Win rates across periods (only those with enough trades)
        wr_values = [
            s["wr"]
            for s in stats_by_period.values()
            if s["trades"] >= self.MIN_TRADES_PER_BUCKET
        ]
        if not wr_values:
            return findings, recommendations

        wr_mean = _safe_mean(wr_values)
        wr_std = _safe_stdev(wr_values)
        metrics[f"{period_label}_wr_mean"] = round(wr_mean, 4)
        metrics[f"{period_label}_wr_std"] = round(wr_std, 4)

        # Flag high variance
        if wr_std > 0.20:
            evidence = [
                f"{p}: WR={s['wr']:.1%} ({int(s['trades'])} trades)"
                for p, s in stats_by_period.items()
                if s["trades"] >= self.MIN_TRADES_PER_BUCKET
            ]
            findings.append(Finding(
                title=f"High win-rate variance across {period_label}s",
                detail=(
                    f"Win rate standard deviation across {period_label}s is "
                    f"{wr_std:.1%} (mean {wr_mean:.1%}). Performance is "
                    "inconsistent across time periods."
                ),
                severity=Severity.WARNING,
                metric_name=f"{period_label}_wr_std",
                metric_value=round(wr_std, 4),
                threshold=0.20,
                evidence=evidence,
            ))

        return findings, recommendations

    # ── regime analysis ---------------------------------------------------

    def _analyze_regimes(
        self,
        by_regime: dict[str, dict[str, float]],
    ) -> tuple[list[Finding], list[Recommendation]]:
        findings: list[Finding] = []
        recommendations: list[Recommendation] = []

        if not by_regime:
            return findings, recommendations

        evidence = []
        regime_wrs: dict[str, float] = {}
        for regime, data in by_regime.items():
            wr = data.get("wr", 0.0)
            trades = data.get("trades", 0)
            regime_wrs[regime] = wr
            evidence.append(
                f"{regime}: WR={wr:.1%}, trades={int(trades)}"
            )

        if not regime_wrs:
            return findings, recommendations

        best_regime = max(regime_wrs, key=regime_wrs.get)  # type: ignore[arg-type]
        worst_regime = min(regime_wrs, key=regime_wrs.get)  # type: ignore[arg-type]
        spread = regime_wrs[best_regime] - regime_wrs[worst_regime]

        findings.append(Finding(
            title="Regime performance spread",
            detail=(
                f"Best regime: {best_regime} (WR {regime_wrs[best_regime]:.1%}). "
                f"Worst regime: {worst_regime} (WR {regime_wrs[worst_regime]:.1%}). "
                f"Spread: {spread:.1%}."
            ),
            severity=Severity.WARNING if spread > 0.25 else Severity.INFO,
            metric_name="regime_wr_spread",
            metric_value=round(spread, 4),
            threshold=0.25,
            evidence=evidence,
        ))

        if spread > 0.25:
            recommendations.append(Recommendation(
                title=f"Add regime filter to avoid {worst_regime} conditions",
                rationale=(
                    f"The model underperforms significantly in {worst_regime} "
                    f"regime (WR {regime_wrs[worst_regime]:.1%} vs "
                    f"{regime_wrs[best_regime]:.1%} in {best_regime}). "
                    "Consider reducing position size or skipping signals "
                    "during unfavorable regimes."
                ),
                rec_type=RecommendationType.REGIME_ADAPTATION,
                priority=2,
                confidence=0.7,
                current_value=(
                    f"No regime filter (worst={worst_regime}"
                    f" WR={regime_wrs[worst_regime]:.1%})"
                ),
                suggested_value=f"Filter or reduce sizing in {worst_regime}",
                expected_impact="Higher consistency and reduced drawdowns",
                evidence=evidence,
            ))

        # Warn if any regime has very low WR
        for regime, wr in regime_wrs.items():
            trades_in_regime = by_regime[regime].get("trades", 0)
            if wr < 0.40 and trades_in_regime >= self.MIN_TRADES_PER_BUCKET:
                findings.append(Finding(
                    title=f"Losing regime: {regime}",
                    detail=(
                        f"Win rate in {regime} regime is {wr:.1%} across "
                        f"{int(trades_in_regime)} trades, well below breakeven."
                    ),
                    severity=Severity.CRITICAL,
                    metric_name=f"regime_wr_{regime}",
                    metric_value=round(wr, 4),
                    threshold=0.40,
                    evidence=[f"{regime}: WR={wr:.1%}, trades={int(trades_in_regime)}"],
                ))

        return findings, recommendations

    # ── walk-forward drift ------------------------------------------------

    def _check_walk_forward_drift(
        self,
        by_year: dict[str, dict[str, float]],
        metrics: dict[str, float | str],
    ) -> list[Finding]:
        findings: list[Finding] = []
        years_sorted = sorted(by_year.keys())
        if len(years_sorted) < 3:
            return findings

        # Split into first ~2/3 (train) and last ~1/3 (validation)
        split = max(1, len(years_sorted) * 2 // 3)
        train_years = years_sorted[:split]
        val_years = years_sorted[split:]

        train_wrs = [
            by_year[y].get("wr", by_year[y].get("win_rate", 0.0))
            for y in train_years
            if by_year[y].get("trades", 0) >= self.MIN_TRADES_PER_BUCKET
        ]
        val_wrs = [
            by_year[y].get("wr", by_year[y].get("win_rate", 0.0))
            for y in val_years
            if by_year[y].get("trades", 0) >= self.MIN_TRADES_PER_BUCKET
        ]

        if not train_wrs or not val_wrs:
            return findings

        train_mean = _safe_mean(train_wrs)
        val_mean = _safe_mean(val_wrs)
        drift = val_mean - train_mean
        metrics["wf_train_wr"] = round(train_mean, 4)
        metrics["wf_val_wr"] = round(val_mean, 4)
        metrics["wf_drift"] = round(drift, 4)

        evidence = [
            f"Train ({', '.join(train_years)}): avg WR={train_mean:.1%}",
            f"Validation ({', '.join(val_years)}): avg WR={val_mean:.1%}",
            f"Drift: {drift:+.1%}",
        ]

        if abs(drift) > self.DEGRADATION_THRESHOLD:
            direction = "degradation" if drift < 0 else "improvement"
            findings.append(Finding(
                title=f"Walk-forward {direction} detected",
                detail=(
                    f"Train period WR: {train_mean:.1%}, validation period WR: "
                    f"{val_mean:.1%}. Drift of {drift:+.1%} suggests "
                    f"{'overfitting or market shift' if drift < 0 else 'possible data-period bias'}."  # noqa: E501
                ),
                severity=Severity.WARNING if drift < 0 else Severity.INFO,
                metric_name="wf_drift",
                metric_value=round(drift, 4),
                threshold=self.DEGRADATION_THRESHOLD,
                evidence=evidence,
            ))

        return findings

    # ── best / worst periods and recent degradation -----------------------

    def _best_worst_and_recent(
        self,
        monthly: dict[str, list[BacktestTrade]],
        metrics: dict[str, float | str],
        review_date: date,
    ) -> list[Finding]:
        findings: list[Finding] = []
        stats_by_month: dict[str, dict[str, float]] = {}

        for period, trades in monthly.items():
            if len(trades) >= self.MIN_TRADES_PER_BUCKET:
                stats_by_month[period] = self._bucket_stats(trades)

        if not stats_by_month:
            return findings

        # Best and worst months by WR
        best_month = max(stats_by_month, key=lambda p: stats_by_month[p]["wr"])
        worst_month = min(stats_by_month, key=lambda p: stats_by_month[p]["wr"])

        metrics["best_month"] = best_month
        metrics["best_month_wr"] = stats_by_month[best_month]["wr"]
        metrics["worst_month"] = worst_month
        metrics["worst_month_wr"] = stats_by_month[worst_month]["wr"]

        findings.append(Finding(
            title="Best and worst performing months",
            detail=(
                f"Best: {best_month} (WR {stats_by_month[best_month]['wr']:.1%}, "
                f"{int(stats_by_month[best_month]['trades'])} trades). "
                f"Worst: {worst_month} (WR {stats_by_month[worst_month]['wr']:.1%}, "
                f"{int(stats_by_month[worst_month]['trades'])} trades)."
            ),
            severity=Severity.INFO,
            metric_name="month_wr_range",
            metric_value=round(
                stats_by_month[best_month]["wr"] - stats_by_month[worst_month]["wr"], 4,
            ),
            evidence=[
                f"{p}: WR={s['wr']:.1%} ({int(s['trades'])} trades)"
                for p, s in sorted(stats_by_month.items())
            ],
        ))

        # Recent performance vs historical (last RECENT_MONTHS)
        sorted_months = sorted(stats_by_month.keys())
        if len(sorted_months) <= self.RECENT_MONTHS:
            return findings

        recent_months = sorted_months[-self.RECENT_MONTHS:]
        historical_months = sorted_months[:-self.RECENT_MONTHS]

        recent_wrs = [stats_by_month[m]["wr"] for m in recent_months]
        hist_wrs = [stats_by_month[m]["wr"] for m in historical_months]

        recent_mean = _safe_mean(recent_wrs)
        hist_mean = _safe_mean(hist_wrs)
        delta = recent_mean - hist_mean
        metrics["recent_wr"] = round(recent_mean, 4)
        metrics["historical_wr"] = round(hist_mean, 4)
        metrics["recent_vs_hist_delta"] = round(delta, 4)

        evidence = [
            f"Recent ({', '.join(recent_months)}): avg WR={recent_mean:.1%}",
            f"Historical ({len(historical_months)} months): avg WR={hist_mean:.1%}",
            f"Delta: {delta:+.1%}",
        ]

        if delta < -self.DEGRADATION_THRESHOLD:
            findings.append(Finding(
                title="Recent performance degradation",
                detail=(
                    f"Last {self.RECENT_MONTHS} months avg WR ({recent_mean:.1%}) "
                    f"is {abs(delta):.1%} below historical avg ({hist_mean:.1%}). "
                    "Model may be losing edge in current conditions."
                ),
                severity=Severity.CRITICAL,
                metric_name="recent_vs_hist_delta",
                metric_value=round(delta, 4),
                threshold=-self.DEGRADATION_THRESHOLD,
                evidence=evidence,
            ))
        elif delta < -0.05:
            findings.append(Finding(
                title="Slight recent underperformance",
                detail=(
                    f"Last {self.RECENT_MONTHS} months avg WR ({recent_mean:.1%}) "
                    f"is {abs(delta):.1%} below historical ({hist_mean:.1%}). "
                    "Worth monitoring."
                ),
                severity=Severity.WARNING,
                metric_name="recent_vs_hist_delta",
                metric_value=round(delta, 4),
                threshold=-0.05,
                evidence=evidence,
            ))

        return findings

    # ── scoring helpers ---------------------------------------------------

    def _score_consistency(
        self,
        monthly: dict[str, list[BacktestTrade]],
        quarterly: dict[str, list[BacktestTrade]],
    ) -> float:
        """Score 0-100 for performance consistency across periods.

        Low variance in win rate across months/quarters = higher score.
        """
        if not monthly and not quarterly:
            return 50.0  # neutral when no data

        scores: list[float] = []

        for buckets, _label in [(monthly, "month"), (quarterly, "quarter")]:
            wr_values = [
                self._bucket_stats(trades)["wr"]
                for trades in buckets.values()
                if len(trades) >= self.MIN_TRADES_PER_BUCKET
            ]
            if len(wr_values) >= 2:
                std = _safe_stdev(wr_values)
                # std of 0 => perfect consistency (100), std >= 0.30 => 0
                score = max(0.0, 100.0 * (1.0 - std / 0.30))
                scores.append(score)

        return _safe_mean(scores) if scores else 50.0

    def _score_trend(
        self,
        monthly: dict[str, list[BacktestTrade]],
    ) -> float:
        """Score 0-100 for the direction of performance trend.

        Improving trend = high score, declining = low score.
        """
        sorted_months = sorted(monthly.keys())
        wr_series = []
        for m in sorted_months:
            trades = monthly[m]
            if len(trades) >= self.MIN_TRADES_PER_BUCKET:
                stats = self._bucket_stats(trades)
                wr_series.append(stats["wr"])

        if len(wr_series) < 3:
            return 50.0  # not enough data

        # Simple linear regression slope via least-squares
        n = len(wr_series)
        x_mean = (n - 1) / 2.0
        y_mean = _safe_mean(wr_series)
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(wr_series))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 50.0

        slope = numerator / denominator

        # Map slope to 0-100: slope of +0.05/period => 100, -0.05 => 0
        score = 50.0 + (slope / 0.05) * 50.0
        return max(0.0, min(100.0, score))

    def _score_seasonal_stability(
        self,
        monthly: dict[str, list[BacktestTrade]],
    ) -> float:
        """Score 0-100 for stability across calendar months (Jan-Dec).

        Groups trades by calendar month across all years. Low variance
        across calendar months = higher score.
        """
        calendar_month_trades: dict[int, list[BacktestTrade]] = defaultdict(list)
        for _key, trades in monthly.items():
            for t in trades:
                calendar_month_trades[t.entry_date.month].append(t)

        wr_by_cal_month: list[float] = []
        for month_num in sorted(calendar_month_trades.keys()):
            trades = calendar_month_trades[month_num]
            if len(trades) >= self.MIN_TRADES_PER_BUCKET:
                stats = self._bucket_stats(trades)
                wr_by_cal_month.append(stats["wr"])

        if len(wr_by_cal_month) < 2:
            return 50.0

        std = _safe_stdev(wr_by_cal_month)
        # std of 0 => 100, std >= 0.25 => 0
        return max(0.0, min(100.0, 100.0 * (1.0 - std / 0.25)))
