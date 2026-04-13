"""Risk Manager specialist — analyzes risk exposures and drawdown health.

Evaluates drawdown severity, risk/reward ratio, position sizing adequacy,
consecutive loss streaks, tail risk, portfolio concentration, and trailing
stop effectiveness.  Produces a 0-100 health score and actionable
recommendations for risk parameter adjustments.
"""

from __future__ import annotations

from collections import Counter
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
from flowedge.scanner.backtest.schemas import BacktestResult


class RiskManager(BaseSpecialist):
    """Specialist that evaluates risk exposures and drawdown health."""

    name: str = "Risk Manager"
    specialist_id: str = "risk_manager"

    # ── thresholds ────────────────────────────────────────────────
    _DD_CRITICAL: float = 10.0   # max drawdown % that triggers critical
    _DD_WARNING: float = 5.0     # max drawdown % that triggers warning
    _CONSEC_LOSS_CRITICAL: int = 5
    _CONSEC_LOSS_WARNING: int = 3
    _RR_GOOD: float = 1.5       # risk/reward ratio considered healthy
    _RR_MINIMUM: float = 1.0
    _CONCENTRATION_WARNING: float = 0.40  # >40% trades in one ticker

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

        # ── Edge case: no trades ──────────────────────────────────
        if result.total_trades == 0:
            return SpecialistReview(
                specialist_name=self.name,
                specialist_id=self.specialist_id,
                review_date=review_date,
                health_score=50.0,
                severity=Severity.INFO,
                summary="No trades to evaluate risk on. Insufficient data for risk analysis.",
                findings=[
                    Finding(
                        title="No trades in backtest",
                        detail="The backtest produced zero trades, so risk metrics cannot be computed.",
                        severity=Severity.INFO,
                        metric_name="total_trades",
                        metric_value=0,
                        evidence=["total_trades=0"],
                    )
                ],
                recommendations=[],
                metrics={"total_trades": 0},
            )

        # ── 1. Drawdown severity ──────────────────────────────────
        dd_pct = abs(result.max_drawdown_pct)
        metrics["max_drawdown_pct"] = dd_pct

        if dd_pct >= self._DD_CRITICAL:
            dd_score = max(0.0, 100.0 - (dd_pct - self._DD_CRITICAL) * 10)
            findings.append(Finding(
                title="Critical drawdown",
                detail=(
                    f"Max drawdown of {dd_pct:.1f}% exceeds critical threshold "
                    f"of {self._DD_CRITICAL}%. Capital preservation is at risk."
                ),
                severity=Severity.CRITICAL,
                metric_name="max_drawdown_pct",
                metric_value=dd_pct,
                threshold=self._DD_CRITICAL,
                evidence=[
                    f"max_drawdown_pct={dd_pct:.2f}%",
                    f"starting_capital=${result.starting_capital:,.0f}",
                    f"ending_value=${result.ending_value:,.0f}",
                ],
            ))
            recommendations.append(Recommendation(
                title="Reduce risk per trade to limit drawdown",
                rationale=(
                    f"Drawdown of {dd_pct:.1f}% suggests position sizing is too aggressive. "
                    f"Current risk_per_trade={config.risk_per_trade:.1%}."
                ),
                rec_type=RecommendationType.RISK_ADJUSTMENT,
                priority=1,
                confidence=0.85,
                current_value=f"{config.risk_per_trade:.1%}",
                suggested_value=f"{config.risk_per_trade * 0.6:.1%}",
                expected_impact="Reduce max drawdown by ~40%",
                evidence=[f"max_drawdown_pct={dd_pct:.2f}%"],
            ))
        elif dd_pct >= self._DD_WARNING:
            dd_score = max(20.0, 100.0 - (dd_pct - self._DD_WARNING) * 8)
            findings.append(Finding(
                title="Elevated drawdown",
                detail=(
                    f"Max drawdown of {dd_pct:.1f}% exceeds warning threshold "
                    f"of {self._DD_WARNING}%. Monitor closely."
                ),
                severity=Severity.WARNING,
                metric_name="max_drawdown_pct",
                metric_value=dd_pct,
                threshold=self._DD_WARNING,
                evidence=[
                    f"max_drawdown_pct={dd_pct:.2f}%",
                    f"portfolio_return_pct={result.portfolio_return_pct:.2f}%",
                ],
            ))
        else:
            dd_score = min(100.0, 100.0 - dd_pct * 4)

        # ── 2. Risk/reward ratio ──────────────────────────────────
        avg_win = abs(result.avg_win_pct) if result.avg_win_pct else 0.0
        avg_loss = abs(result.avg_loss_pct) if result.avg_loss_pct else 0.0
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        metrics["risk_reward_ratio"] = round(rr_ratio, 3)

        if rr_ratio == 0.0 and avg_loss == 0.0 and avg_win > 0:
            # All winners, no losses — best case
            rr_score = 100.0
        elif rr_ratio >= self._RR_GOOD:
            rr_score = 100.0
        elif rr_ratio >= self._RR_MINIMUM:
            rr_score = 60.0 + (rr_ratio - self._RR_MINIMUM) / (self._RR_GOOD - self._RR_MINIMUM) * 40.0
        elif rr_ratio > 0:
            rr_score = rr_ratio / self._RR_MINIMUM * 60.0
        else:
            rr_score = 0.0

        if rr_ratio < self._RR_MINIMUM and avg_loss > 0:
            severity = Severity.CRITICAL if rr_ratio < 0.5 else Severity.WARNING
            findings.append(Finding(
                title="Poor risk/reward ratio",
                detail=(
                    f"Risk/reward ratio of {rr_ratio:.2f} is below minimum "
                    f"threshold of {self._RR_MINIMUM:.1f}. Average win "
                    f"({avg_win:.2f}%) does not adequately compensate for "
                    f"average loss ({avg_loss:.2f}%)."
                ),
                severity=severity,
                metric_name="risk_reward_ratio",
                metric_value=round(rr_ratio, 3),
                threshold=self._RR_MINIMUM,
                evidence=[
                    f"avg_win_pct={result.avg_win_pct:.2f}%",
                    f"avg_loss_pct={result.avg_loss_pct:.2f}%",
                    f"risk_reward_ratio={rr_ratio:.3f}",
                ],
            ))
            recommendations.append(Recommendation(
                title="Tighten trailing stop or widen take-profit",
                rationale=(
                    f"R:R of {rr_ratio:.2f} means losses eat winners. "
                    f"Consider tighter trail_pct or higher tp_underlying."
                ),
                rec_type=RecommendationType.PARAMETER_CHANGE,
                priority=2,
                confidence=0.70,
                current_value=f"trail_pct={config.trail_pct}, avg_loss={avg_loss:.2f}%",
                suggested_value=f"trail_pct={config.trail_pct * 0.7:.3f}",
                expected_impact="Improve R:R toward 1.0+",
                evidence=[f"risk_reward_ratio={rr_ratio:.3f}"],
            ))

        # ── 3. Position sizing adequacy ───────────────────────────
        if result.trades:
            loss_pcts = [abs(t.pnl_pct) for t in result.trades if t.pnl_pct < 0]
            if loss_pcts:
                max_actual_loss = max(loss_pcts)
                avg_actual_loss = sum(loss_pcts) / len(loss_pcts)
                risk_budget = config.risk_per_trade * 100  # convert to %
                metrics["max_actual_loss_pct"] = round(max_actual_loss, 3)
                metrics["avg_actual_loss_pct"] = round(avg_actual_loss, 3)
                metrics["risk_budget_pct"] = round(risk_budget, 3)

                # Score: how well does risk_per_trade contain losses?
                overshoot = max_actual_loss / risk_budget if risk_budget > 0 else 0.0
                if overshoot <= 1.0:
                    ps_score = 100.0
                elif overshoot <= 2.0:
                    ps_score = max(30.0, 100.0 - (overshoot - 1.0) * 70.0)
                else:
                    ps_score = max(0.0, 30.0 - (overshoot - 2.0) * 15.0)

                if overshoot > 1.5:
                    findings.append(Finding(
                        title="Position sizing breach",
                        detail=(
                            f"Worst single-trade loss of {max_actual_loss:.2f}% "
                            f"exceeds risk budget of {risk_budget:.1f}% by "
                            f"{(overshoot - 1.0) * 100:.0f}%. "
                            f"Average loss {avg_actual_loss:.2f}% "
                            f"({'within' if avg_actual_loss <= risk_budget else 'exceeds'} budget)."
                        ),
                        severity=Severity.WARNING if overshoot < 2.5 else Severity.CRITICAL,
                        metric_name="position_sizing_overshoot",
                        metric_value=round(overshoot, 2),
                        threshold=1.5,
                        evidence=[
                            f"max_actual_loss={max_actual_loss:.2f}%",
                            f"avg_actual_loss={avg_actual_loss:.2f}%",
                            f"risk_per_trade={config.risk_per_trade:.1%}",
                            f"loss_count={len(loss_pcts)}",
                        ],
                    ))
            else:
                # No losing trades
                ps_score = 100.0
        else:
            ps_score = 50.0  # no data

        # ── 4. Consecutive loss streaks ───────────────────────────
        max_consec = result.max_consecutive_losses
        metrics["max_consecutive_losses"] = max_consec

        if max_consec >= self._CONSEC_LOSS_CRITICAL:
            findings.append(Finding(
                title="Severe consecutive loss streak",
                detail=(
                    f"Max consecutive losses of {max_consec} exceeds critical "
                    f"threshold of {self._CONSEC_LOSS_CRITICAL}. Indicates "
                    f"possible regime mismatch or systematic edge degradation."
                ),
                severity=Severity.CRITICAL,
                metric_name="max_consecutive_losses",
                metric_value=max_consec,
                threshold=self._CONSEC_LOSS_CRITICAL,
                evidence=[
                    f"max_consecutive_losses={max_consec}",
                    f"win_rate={result.win_rate:.1%}",
                    f"total_trades={result.total_trades}",
                ],
            ))
            recommendations.append(Recommendation(
                title="Add circuit-breaker for consecutive losses",
                rationale=(
                    f"{max_consec} consecutive losses suggests the model should "
                    f"pause trading after {self._CONSEC_LOSS_CRITICAL} straight losses "
                    f"and require a regime-check before resuming."
                ),
                rec_type=RecommendationType.RISK_ADJUSTMENT,
                priority=1,
                confidence=0.80,
                current_value=f"no circuit breaker, max_consec={max_consec}",
                suggested_value=f"pause after {self._CONSEC_LOSS_CRITICAL} consecutive losses",
                expected_impact="Prevent runaway drawdowns during adverse regimes",
                evidence=[f"max_consecutive_losses={max_consec}"],
            ))
        elif max_consec >= self._CONSEC_LOSS_WARNING:
            findings.append(Finding(
                title="Notable consecutive loss streak",
                detail=(
                    f"Max consecutive losses of {max_consec} exceeds warning "
                    f"threshold of {self._CONSEC_LOSS_WARNING}. Worth monitoring."
                ),
                severity=Severity.WARNING,
                metric_name="max_consecutive_losses",
                metric_value=max_consec,
                threshold=self._CONSEC_LOSS_WARNING,
                evidence=[
                    f"max_consecutive_losses={max_consec}",
                    f"win_rate={result.win_rate:.1%}",
                ],
            ))

        # ── 5. Tail risk: worst 10% of trades ────────────────────
        if result.trades:
            all_pnl = sorted(t.pnl_pct for t in result.trades)
            tail_n = max(1, len(all_pnl) // 10)
            tail_trades = all_pnl[:tail_n]
            tail_avg = sum(tail_trades) / len(tail_trades)
            tail_worst = tail_trades[0]
            metrics["tail_avg_pnl_pct"] = round(tail_avg, 3)
            metrics["tail_worst_pnl_pct"] = round(tail_worst, 3)
            metrics["tail_count"] = tail_n

            # Score based on how bad the tail is relative to avg loss
            if avg_loss > 0:
                tail_severity = abs(tail_avg) / avg_loss
            else:
                tail_severity = 0.0

            if tail_severity <= 1.5:
                tail_score = 100.0
            elif tail_severity <= 3.0:
                tail_score = max(20.0, 100.0 - (tail_severity - 1.5) * 40.0)
            else:
                tail_score = max(0.0, 20.0 - (tail_severity - 3.0) * 10.0)

            if abs(tail_avg) > avg_loss * 2 and avg_loss > 0:
                findings.append(Finding(
                    title="Fat tail risk detected",
                    detail=(
                        f"Worst 10% of trades ({tail_n} trades) average "
                        f"{tail_avg:.2f}% vs overall avg loss of "
                        f"{result.avg_loss_pct:.2f}%. Tail losses are "
                        f"{abs(tail_avg) / avg_loss:.1f}x the average loss, "
                        f"indicating fat-tail exposure."
                    ),
                    severity=Severity.WARNING if tail_severity < 3.0 else Severity.CRITICAL,
                    metric_name="tail_avg_pnl_pct",
                    metric_value=round(tail_avg, 3),
                    threshold=f"{result.avg_loss_pct * 2:.3f}",
                    evidence=[
                        f"tail_avg_pnl={tail_avg:.2f}%",
                        f"tail_worst={tail_worst:.2f}%",
                        f"avg_loss_pct={result.avg_loss_pct:.2f}%",
                        f"tail_count={tail_n} of {result.total_trades}",
                    ],
                ))
        else:
            tail_score = 50.0

        # ── 6. Portfolio concentration ────────────────────────────
        if result.trades:
            ticker_counts: Counter[str] = Counter(t.ticker for t in result.trades)
            total = result.total_trades
            most_common_ticker, most_common_count = ticker_counts.most_common(1)[0]
            concentration = most_common_count / total if total > 0 else 0.0
            metrics["top_ticker"] = most_common_ticker
            metrics["top_ticker_concentration"] = round(concentration, 3)
            metrics["unique_tickers"] = len(ticker_counts)

            if concentration > self._CONCENTRATION_WARNING:
                findings.append(Finding(
                    title="High ticker concentration",
                    detail=(
                        f"{most_common_ticker} accounts for "
                        f"{concentration:.0%} of all trades "
                        f"({most_common_count}/{total}). Over-concentration "
                        f"in a single name amplifies idiosyncratic risk."
                    ),
                    severity=Severity.WARNING,
                    metric_name="ticker_concentration",
                    metric_value=round(concentration, 3),
                    threshold=self._CONCENTRATION_WARNING,
                    evidence=[
                        f"top_ticker={most_common_ticker} ({most_common_count}/{total})",
                        f"unique_tickers={len(ticker_counts)}",
                        *[
                            f"{tick}={cnt} trades"
                            for tick, cnt in ticker_counts.most_common(5)
                        ],
                    ],
                ))
                recommendations.append(Recommendation(
                    title=f"Diversify away from {most_common_ticker}",
                    rationale=(
                        f"{most_common_ticker} represents {concentration:.0%} of trades. "
                        f"Reduce to <{self._CONCENTRATION_WARNING:.0%} per ticker."
                    ),
                    rec_type=RecommendationType.RISK_ADJUSTMENT,
                    priority=3,
                    confidence=0.65,
                    current_value=f"{most_common_ticker}={concentration:.0%}",
                    suggested_value=f"<{self._CONCENTRATION_WARNING:.0%} per ticker",
                    expected_impact="Reduce single-name risk exposure",
                    evidence=[f"concentration={concentration:.0%}"],
                ))

        # ── 7. Trailing stop effectiveness ────────────────────────
        if result.trades:
            trail_exits = [
                t for t in result.trades if t.exit_reason == "trailing_stop"
            ]
            time_exits = [
                t for t in result.trades if t.exit_reason == "time_exit"
            ]
            total_exits = len(result.trades)
            trail_count = len(trail_exits)
            time_count = len(time_exits)

            metrics["trail_exit_count"] = trail_count
            metrics["time_exit_count"] = time_count

            if trail_count > 0:
                trail_avg_pnl = sum(t.pnl_pct for t in trail_exits) / trail_count
                metrics["trail_exit_avg_pnl_pct"] = round(trail_avg_pnl, 3)
            if time_count > 0:
                time_avg_pnl = sum(t.pnl_pct for t in time_exits) / time_count
                metrics["time_exit_avg_pnl_pct"] = round(time_avg_pnl, 3)

            # If most exits are time-based and trail_pct is rarely triggered,
            # the trailing stop may be set too tight or too loose.
            if total_exits > 5:
                trail_ratio = trail_count / total_exits
                time_ratio = time_count / total_exits

                if trail_ratio > 0.5 and trail_count > 0:
                    trail_avg = sum(t.pnl_pct for t in trail_exits) / trail_count
                    if trail_avg < 0:
                        findings.append(Finding(
                            title="Trailing stop exits are net negative",
                            detail=(
                                f"{trail_count} trailing-stop exits "
                                f"({trail_ratio:.0%} of trades) average "
                                f"{trail_avg:.2f}% P&L. The trail_pct of "
                                f"{config.trail_pct:.1%} may be too tight, "
                                f"cutting winners prematurely."
                            ),
                            severity=Severity.WARNING,
                            metric_name="trail_exit_avg_pnl_pct",
                            metric_value=round(trail_avg, 3),
                            threshold=0.0,
                            evidence=[
                                f"trail_exits={trail_count}/{total_exits}",
                                f"trail_avg_pnl={trail_avg:.2f}%",
                                f"trail_pct={config.trail_pct:.1%}",
                            ],
                        ))
                        recommendations.append(Recommendation(
                            title="Widen trailing stop",
                            rationale=(
                                f"Trailing stop exits average {trail_avg:.2f}% "
                                f"(negative). Current trail_pct={config.trail_pct:.1%} "
                                f"may be too tight."
                            ),
                            rec_type=RecommendationType.PARAMETER_CHANGE,
                            priority=2,
                            confidence=0.65,
                            current_value=f"trail_pct={config.trail_pct}",
                            suggested_value=f"trail_pct={config.trail_pct * 1.5:.3f}",
                            expected_impact="Reduce premature exits, improve avg winner",
                            evidence=[
                                f"trail_exit_avg_pnl={trail_avg:.2f}%",
                                f"trail_exits={trail_count}",
                            ],
                        ))

                if time_ratio > 0.6 and trail_count < 3:
                    findings.append(Finding(
                        title="Trailing stop rarely triggered",
                        detail=(
                            f"Only {trail_count} trailing-stop exits vs "
                            f"{time_count} time exits ({time_ratio:.0%}). "
                            f"The trail_pct of {config.trail_pct:.1%} may be "
                            f"too wide — most trades expire on max_hold_bars "
                            f"instead of being protected by the trail."
                        ),
                        severity=Severity.INFO,
                        metric_name="trail_exit_ratio",
                        metric_value=round(trail_ratio, 3),
                        threshold=0.1,
                        evidence=[
                            f"trail_exits={trail_count}",
                            f"time_exits={time_count}",
                            f"trail_pct={config.trail_pct:.1%}",
                            f"max_hold_bars={config.max_hold_bars}",
                        ],
                    ))

        # ── 8. Health score ───────────────────────────────────────
        # Weights: risk/reward 30%, drawdown 30%, position sizing 20%, tail risk 20%
        health_score = (
            rr_score * 0.30
            + dd_score * 0.30
            + ps_score * 0.20
            + tail_score * 0.20
        )
        health_score = round(max(0.0, min(100.0, health_score)), 1)

        metrics["rr_sub_score"] = round(rr_score, 1)
        metrics["dd_sub_score"] = round(dd_score, 1)
        metrics["ps_sub_score"] = round(ps_score, 1)
        metrics["tail_sub_score"] = round(tail_score, 1)
        metrics["profit_factor"] = round(result.profit_factor, 3)
        metrics["sharpe_ratio"] = round(result.sharpe_ratio, 3)

        # ── Overall severity ──────────────────────────────────────
        if any(f.severity == Severity.CRITICAL for f in findings):
            overall_severity = Severity.CRITICAL
        elif any(f.severity == Severity.WARNING for f in findings):
            overall_severity = Severity.WARNING
        else:
            overall_severity = Severity.INFO

        # ── Summary ───────────────────────────────────────────────
        summary_parts: list[str] = []
        summary_parts.append(
            f"Risk health score: {health_score}/100 "
            f"(R:R={rr_ratio:.2f}, DD={dd_pct:.1f}%)."
        )
        if findings:
            crit_count = sum(1 for f in findings if f.severity == Severity.CRITICAL)
            warn_count = sum(1 for f in findings if f.severity == Severity.WARNING)
            if crit_count:
                summary_parts.append(f"{crit_count} critical finding(s).")
            if warn_count:
                summary_parts.append(f"{warn_count} warning(s).")
        else:
            summary_parts.append("No risk issues detected.")

        # If no actionable issues, add a no-action recommendation
        if not recommendations:
            recommendations.append(Recommendation(
                title="Risk parameters are adequate",
                rationale=(
                    f"Drawdown at {dd_pct:.1f}%, R:R at {rr_ratio:.2f}, "
                    f"no position sizing breaches detected. No changes needed."
                ),
                rec_type=RecommendationType.NO_ACTION,
                priority=5,
                confidence=0.75,
                evidence=[
                    f"max_drawdown_pct={dd_pct:.2f}%",
                    f"risk_reward_ratio={rr_ratio:.3f}",
                ],
            ))

        return SpecialistReview(
            specialist_name=self.name,
            specialist_id=self.specialist_id,
            review_date=review_date,
            health_score=health_score,
            severity=overall_severity,
            summary=" ".join(summary_parts),
            findings=findings,
            recommendations=recommendations,
            metrics=metrics,
        )
