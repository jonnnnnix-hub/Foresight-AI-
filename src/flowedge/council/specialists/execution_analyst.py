"""Execution Analyst — evaluates trade execution quality.

Examines exit-reason distribution, hold-period patterns, take-profit
and trailing-stop effectiveness, commission drag, and underlying-move
capture to quantify how well the backtest engine converts signals into
realised P&L.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from statistics import mean, median

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


class ExecutionAnalyst(BaseSpecialist):
    """Specialist that analyses execution quality — timing, exit efficiency,
    hold periods, commission drag, and parameter balance."""

    name: str = "Execution Analyst"
    specialist_id: str = "execution_analyst"

    # ── thresholds ────────────────────────────────────────────────
    TIME_EXIT_WARNING_THRESHOLD = 0.30  # >30% time exits is concerning
    TIME_EXIT_CRITICAL_THRESHOLD = 0.50
    TP_TIGHT_THRESHOLD = 0.60  # >60% TP hits + low avg win → too tight
    TP_LOOSE_THRESHOLD = 0.10  # <10% TP hits → too loose / never reached
    COMMISSION_DRAG_WARNING = 0.15  # commissions > 15% of gross profit
    COMMISSION_DRAG_CRITICAL = 0.30
    TRAILING_STOP_GIVEBACK_THRESHOLD = 0.40  # trail exits give back >40% of peak

    def analyze(
        self,
        result: BacktestResult,
        config: ScalpConfig,
        review_date: date,
        history: list[BacktestResult],
    ) -> SpecialistReview:
        trades = result.trades
        findings: list[Finding] = []
        recommendations: list[Recommendation] = []
        metrics: dict[str, float | str] = {}

        # ── guard: no trades ──────────────────────────────────────
        if not trades:
            return SpecialistReview(
                specialist_name=self.name,
                specialist_id=self.specialist_id,
                review_date=review_date,
                health_score=50.0,
                severity=Severity.WARNING,
                summary="No trades to evaluate execution quality.",
                findings=[
                    Finding(
                        title="No trades in result",
                        detail="Backtest produced zero trades; execution analysis not possible.",
                        severity=Severity.WARNING,
                        evidence=["total_trades=0"],
                    )
                ],
                metrics={"total_trades": 0},
            )

        # ── 1. Exit reason distribution ───────────────────────────
        exit_counts = Counter(t.exit_reason for t in trades)
        total = len(trades)
        exit_pcts = {reason: count / total for reason, count in exit_counts.items()}

        tp_pct = exit_pcts.get("take_profit", 0.0)
        trail_pct = exit_pcts.get("trailing_stop", 0.0)
        time_pct = exit_pcts.get("time_exit", 0.0)
        hard_stop_pct = exit_pcts.get("hard_stop", 0.0)

        metrics["exit_take_profit_pct"] = round(tp_pct * 100, 1)
        metrics["exit_trailing_stop_pct"] = round(trail_pct * 100, 1)
        metrics["exit_time_exit_pct"] = round(time_pct * 100, 1)
        metrics["exit_hard_stop_pct"] = round(hard_stop_pct * 100, 1)

        # ── 2. Time exit analysis ─────────────────────────────────
        time_exit_trades = [t for t in trades if t.exit_reason == "time_exit"]
        time_exit_severity = Severity.INFO
        if time_pct > self.TIME_EXIT_CRITICAL_THRESHOLD:
            time_exit_severity = Severity.CRITICAL
        elif time_pct > self.TIME_EXIT_WARNING_THRESHOLD:
            time_exit_severity = Severity.WARNING

        if time_pct > self.TIME_EXIT_WARNING_THRESHOLD:
            time_exit_wr = _win_rate(time_exit_trades) if time_exit_trades else 0.0
            findings.append(
                Finding(
                    title="High time-exit rate",
                    detail=(
                        f"{time_pct:.0%} of trades expire via time exit "
                        f"(threshold: {self.TIME_EXIT_WARNING_THRESHOLD:.0%}). "
                        f"Time-exit win rate is {time_exit_wr:.0%}. "
                        "Trades hitting max_hold_bars without TP or stop typically "
                        "indicate the move has stalled or the entry was late."
                    ),
                    severity=time_exit_severity,
                    metric_name="time_exit_pct",
                    metric_value=round(time_pct * 100, 1),
                    threshold=round(self.TIME_EXIT_WARNING_THRESHOLD * 100, 1),
                    evidence=[
                        f"time_exit_trades={len(time_exit_trades)}/{total}",
                        f"time_exit_win_rate={time_exit_wr:.1%}",
                        f"max_hold_bars={config.max_hold_bars}",
                    ],
                )
            )
            recommendations.append(
                Recommendation(
                    title="Reduce time-exit rate",
                    rationale=(
                        "High time-exit fraction suggests max_hold_bars may be too "
                        "short for the current TP target, or entries are occurring "
                        "too late in the move."
                    ),
                    rec_type=RecommendationType.PARAMETER_CHANGE,
                    priority=2,
                    confidence=0.65,
                    current_value=f"max_hold_bars={config.max_hold_bars}",
                    suggested_value=f"max_hold_bars={config.max_hold_bars + 4}",
                    expected_impact="Fewer time exits; more trades reach TP or trail",
                    evidence=[f"time_exit_pct={time_pct:.1%}"],
                )
            )

        # ── 3. Hold period analysis (winners vs losers) ───────────
        winners = [t for t in trades if t.outcome.value == "win"]
        losers = [t for t in trades if t.outcome.value == "loss"]

        avg_hold_win = mean(t.hold_days for t in winners) if winners else 0.0
        avg_hold_loss = mean(t.hold_days for t in losers) if losers else 0.0
        median_hold = median(t.hold_days for t in trades)

        metrics["avg_hold_days_winners"] = round(avg_hold_win, 2)
        metrics["avg_hold_days_losers"] = round(avg_hold_loss, 2)
        metrics["median_hold_days"] = round(median_hold, 2)

        if winners and losers and avg_hold_loss > 0 and avg_hold_win > avg_hold_loss * 1.5:
            findings.append(
                Finding(
                    title="Winners held significantly longer than losers",
                    detail=(
                        f"Avg winner hold: {avg_hold_win:.2f} days vs "
                        f"avg loser hold: {avg_hold_loss:.2f} days. "
                        "This may indicate losers are stopped out quickly (good) or "
                        "that winners need extended time to develop (latency risk)."
                    ),
                    severity=Severity.INFO,
                    metric_name="hold_ratio_win_loss",
                    metric_value=round(avg_hold_win / max(avg_hold_loss, 0.001), 2),
                    evidence=[
                        f"avg_hold_win={avg_hold_win:.2f}d",
                        f"avg_hold_loss={avg_hold_loss:.2f}d",
                    ],
                )
            )

        # ── 4. Take-profit analysis ──────────────────────────────
        tp_trades = [t for t in trades if t.exit_reason == "take_profit"]
        avg_win_pct = mean(t.pnl_pct for t in winners) if winners else 0.0

        if tp_pct > self.TP_TIGHT_THRESHOLD and avg_win_pct < 5.0:
            # Many small TP wins — might be leaving money on the table
            tp_underlying_moves = [
                t.underlying_move_pct for t in tp_trades if t.underlying_move_pct > 0
            ]
            avg_underlying_at_tp = mean(tp_underlying_moves) if tp_underlying_moves else 0.0
            findings.append(
                Finding(
                    title="Take-profit may be too tight",
                    detail=(
                        f"{tp_pct:.0%} of trades hit TP with avg win of "
                        f"{avg_win_pct:.1f}%. Average underlying move at TP exit "
                        f"is {avg_underlying_at_tp:.3%}. The move often continues "
                        "past the TP threshold — consider widening."
                    ),
                    severity=Severity.WARNING,
                    metric_name="tp_hit_rate",
                    metric_value=round(tp_pct * 100, 1),
                    threshold=round(self.TP_TIGHT_THRESHOLD * 100, 1),
                    evidence=[
                        f"tp_hit_pct={tp_pct:.0%}",
                        f"avg_win_pct={avg_win_pct:.1f}%",
                        f"avg_underlying_move_at_tp={avg_underlying_at_tp:.4f}",
                        f"tp_underlying={config.tp_underlying}",
                    ],
                )
            )
            recommendations.append(
                Recommendation(
                    title="Widen take-profit threshold",
                    rationale=(
                        "TP is hit very frequently with small average gains. "
                        "Widening the threshold could capture larger moves."
                    ),
                    rec_type=RecommendationType.PARAMETER_CHANGE,
                    priority=2,
                    confidence=0.6,
                    current_value=f"tp_underlying={config.tp_underlying}",
                    suggested_value=f"tp_underlying={config.tp_underlying * 1.5:.4f}",
                    expected_impact="Larger average win; fewer but more profitable TP exits",
                    evidence=[
                        f"current_avg_win={avg_win_pct:.1f}%",
                        f"tp_hit_rate={tp_pct:.0%}",
                    ],
                )
            )

        if tp_pct < self.TP_LOOSE_THRESHOLD and total >= 20:
            findings.append(
                Finding(
                    title="Take-profit rarely reached",
                    detail=(
                        f"Only {tp_pct:.0%} of trades hit TP "
                        f"(threshold: {self.TP_LOOSE_THRESHOLD:.0%}). "
                        "The TP target may be unrealistically far for the "
                        "typical intraday move size."
                    ),
                    severity=Severity.WARNING,
                    metric_name="tp_hit_rate",
                    metric_value=round(tp_pct * 100, 1),
                    threshold=round(self.TP_LOOSE_THRESHOLD * 100, 1),
                    evidence=[
                        f"tp_trades={len(tp_trades)}/{total}",
                        f"tp_underlying={config.tp_underlying}",
                    ],
                )
            )
            recommendations.append(
                Recommendation(
                    title="Tighten take-profit threshold",
                    rationale=(
                        "TP is almost never hit. Most profits come from trailing "
                        "stops or time exits. A tighter TP would lock in gains."
                    ),
                    rec_type=RecommendationType.PARAMETER_CHANGE,
                    priority=2,
                    confidence=0.55,
                    current_value=f"tp_underlying={config.tp_underlying}",
                    suggested_value=f"tp_underlying={config.tp_underlying * 0.7:.4f}",
                    expected_impact="More frequent TP exits; reduced reliance on trail/time exits",
                    evidence=[f"tp_hit_rate={tp_pct:.0%}"],
                )
            )

        # ── 5. Trailing stop effectiveness ────────────────────────
        trail_trades = [t for t in trades if t.exit_reason == "trailing_stop"]
        trail_win_rate = _win_rate(trail_trades) if trail_trades else 0.0
        metrics["trailing_stop_win_rate"] = round(trail_win_rate * 100, 1)

        # Estimate giveback: for trail-exit winners, compare exit pnl to
        # an ideal exit (underlying_move_pct as proxy for available move)
        trail_winners = [t for t in trail_trades if t.outcome.value == "win"]
        if trail_winners:
            givebacks = []
            for t in trail_winners:
                if t.underlying_move_pct > 0 and t.pnl_pct > 0:
                    # rough proxy: what fraction of the available underlying
                    # move did we NOT capture?
                    captured = t.pnl_pct
                    # Use underlying_move_pct as a ceiling estimate.  The
                    # option's leverage means pnl_pct can exceed
                    # underlying_move_pct, so clamp giveback at 0.
                    potential = t.underlying_move_pct * 100  # convert to pct
                    if potential > 0:
                        giveback = max(0, 1.0 - (captured / potential))
                        givebacks.append(giveback)
            avg_giveback = mean(givebacks) if givebacks else 0.0
            metrics["trail_avg_giveback_pct"] = round(avg_giveback * 100, 1)

            if avg_giveback > self.TRAILING_STOP_GIVEBACK_THRESHOLD:
                findings.append(
                    Finding(
                        title="Trailing stop gives back too much profit",
                        detail=(
                            f"Trailing-stop winners give back ~{avg_giveback:.0%} "
                            f"of the underlying move on average. Trail pct "
                            f"({config.trail_pct:.1%}) may be too wide, "
                            "allowing gains to erode before the stop triggers."
                        ),
                        severity=Severity.WARNING,
                        metric_name="trail_giveback_pct",
                        metric_value=round(avg_giveback * 100, 1),
                        threshold=round(self.TRAILING_STOP_GIVEBACK_THRESHOLD * 100, 1),
                        evidence=[
                            f"trail_winners={len(trail_winners)}",
                            f"avg_giveback={avg_giveback:.1%}",
                            f"trail_pct={config.trail_pct}",
                        ],
                    )
                )
                recommendations.append(
                    Recommendation(
                        title="Tighten trailing stop",
                        rationale=(
                            "Trailing stop exits are giving back a large "
                            "portion of available profit before triggering."
                        ),
                        rec_type=RecommendationType.PARAMETER_CHANGE,
                        priority=3,
                        confidence=0.55,
                        current_value=f"trail_pct={config.trail_pct}",
                        suggested_value=f"trail_pct={config.trail_pct * 0.7:.4f}",
                        expected_impact="Captures more of the peak move before reversal",
                        evidence=[f"avg_giveback={avg_giveback:.1%}"],
                    )
                )

        if trail_trades and trail_win_rate < 0.40:
            findings.append(
                Finding(
                    title="Trailing stop has low win rate",
                    detail=(
                        f"Trailing-stop exits win only {trail_win_rate:.0%} of the "
                        f"time ({len(trail_trades)} trades). The trail may be too "
                        "tight, getting triggered by normal noise before the move "
                        "develops."
                    ),
                    severity=Severity.WARNING,
                    metric_name="trailing_stop_win_rate",
                    metric_value=round(trail_win_rate * 100, 1),
                    threshold=40.0,
                    evidence=[
                        f"trail_wins={len([t for t in trail_trades if t.outcome.value == 'win'])}/"
                        f"{len(trail_trades)}",
                        f"trail_pct={config.trail_pct}",
                    ],
                )
            )

        # ── 6. Commission drag ────────────────────────────────────
        total_commissions = sum(
            t.contracts * config.commission_per_contract * 2  # entry + exit
            for t in trades
        )
        gross_profits = sum(
            (t.exit_value - t.cost_basis) for t in trades if t.exit_value > t.cost_basis
        )
        commission_drag = (
            total_commissions / gross_profits
            if gross_profits > 0
            else 1.0 if total_commissions > 0 else 0.0
        )
        metrics["total_commissions"] = round(total_commissions, 2)
        metrics["gross_profits"] = round(gross_profits, 2)
        metrics["commission_drag_pct"] = round(commission_drag * 100, 1)

        if commission_drag > self.COMMISSION_DRAG_CRITICAL:
            findings.append(
                Finding(
                    title="Commission drag is critically high",
                    detail=(
                        f"Commissions (${total_commissions:,.2f}) consume "
                        f"{commission_drag:.0%} of gross profits "
                        f"(${gross_profits:,.2f}). Net profitability is "
                        "severely impacted by transaction costs."
                    ),
                    severity=Severity.CRITICAL,
                    metric_name="commission_drag_pct",
                    metric_value=round(commission_drag * 100, 1),
                    threshold=round(self.COMMISSION_DRAG_CRITICAL * 100, 1),
                    evidence=[
                        f"total_commissions=${total_commissions:,.2f}",
                        f"gross_profits=${gross_profits:,.2f}",
                        f"commission_per_contract=${config.commission_per_contract}",
                        f"avg_contracts_per_trade={mean(t.contracts for t in trades):.1f}",
                    ],
                )
            )
        elif commission_drag > self.COMMISSION_DRAG_WARNING:
            findings.append(
                Finding(
                    title="Commission drag is elevated",
                    detail=(
                        f"Commissions (${total_commissions:,.2f}) consume "
                        f"{commission_drag:.0%} of gross profits "
                        f"(${gross_profits:,.2f}). Consider reducing trade "
                        "frequency or increasing position size to amortise costs."
                    ),
                    severity=Severity.WARNING,
                    metric_name="commission_drag_pct",
                    metric_value=round(commission_drag * 100, 1),
                    threshold=round(self.COMMISSION_DRAG_WARNING * 100, 1),
                    evidence=[
                        f"total_commissions=${total_commissions:,.2f}",
                        f"gross_profits=${gross_profits:,.2f}",
                    ],
                )
            )

        # ── 7. Underlying move capture ────────────────────────────
        moves = [t.underlying_move_pct for t in trades if t.underlying_move_pct != 0]
        avg_underlying_move = mean(moves) if moves else 0.0
        metrics["avg_underlying_move_pct"] = round(avg_underlying_move * 100, 3)

        winner_moves = [t.underlying_move_pct for t in winners if t.underlying_move_pct > 0]
        if winner_moves:
            avg_winner_underlying = mean(winner_moves)
            metrics["avg_winner_underlying_move_pct"] = round(
                avg_winner_underlying * 100, 3
            )
            # Compare TP target to realised underlying moves
            moves_above_tp = [m for m in winner_moves if m > config.tp_underlying * 1.5]
            if len(moves_above_tp) > len(winner_moves) * 0.3 and len(winner_moves) >= 10:
                findings.append(
                    Finding(
                        title="Leaving money on the table",
                        detail=(
                            f"{len(moves_above_tp)}/{len(winner_moves)} winning trades "
                            f"({len(moves_above_tp)/len(winner_moves):.0%}) saw underlying "
                            f"moves >1.5x the TP target ({config.tp_underlying * 1.5:.4f}). "
                            "A wider TP or more aggressive trail could capture more upside."
                        ),
                        severity=Severity.INFO,
                        metric_name="moves_above_1.5x_tp_pct",
                        metric_value=round(len(moves_above_tp) / len(winner_moves) * 100, 1),
                        threshold=30.0,
                        evidence=[
                            f"moves_above_1.5x_tp={len(moves_above_tp)}/{len(winner_moves)}",
                            f"tp_underlying={config.tp_underlying}",
                            f"avg_winner_underlying={avg_winner_underlying:.4f}",
                        ],
                    )
                )

        # ── 8. Health score calculation ───────────────────────────
        exit_efficiency_score = self._exit_efficiency_score(
            tp_pct, trail_pct, time_pct, hard_stop_pct, trail_win_rate, result.win_rate
        )
        timing_score = self._timing_score(
            avg_hold_win, avg_hold_loss, time_pct, config.max_hold_bars
        )
        commission_score = self._commission_score(commission_drag)
        tp_stop_balance_score = self._tp_stop_balance_score(
            tp_pct, trail_pct, time_pct, hard_stop_pct, avg_win_pct
        )

        health_score = (
            exit_efficiency_score * 0.35
            + timing_score * 0.25
            + commission_score * 0.20
            + tp_stop_balance_score * 0.20
        )
        health_score = round(min(100.0, max(0.0, health_score)), 1)

        metrics["exit_efficiency_score"] = round(exit_efficiency_score, 1)
        metrics["timing_score"] = round(timing_score, 1)
        metrics["commission_score"] = round(commission_score, 1)
        metrics["tp_stop_balance_score"] = round(tp_stop_balance_score, 1)

        # ── overall severity ──────────────────────────────────────
        if health_score < 40:
            overall_severity = Severity.CRITICAL
        elif health_score < 65:
            overall_severity = Severity.WARNING
        else:
            overall_severity = Severity.INFO

        # ── if no issues found, add a positive finding ────────────
        if not findings:
            findings.append(
                Finding(
                    title="Execution quality is healthy",
                    detail=(
                        f"Exit distribution is balanced (TP {tp_pct:.0%}, "
                        f"trail {trail_pct:.0%}, time {time_pct:.0%}, "
                        f"stop {hard_stop_pct:.0%}). Commission drag is "
                        f"{commission_drag:.0%}."
                    ),
                    severity=Severity.INFO,
                    evidence=[
                        f"health_score={health_score}",
                        f"total_trades={total}",
                    ],
                )
            )

        if not recommendations:
            recommendations.append(
                Recommendation(
                    title="No execution changes needed",
                    rationale="Execution parameters are performing within acceptable bounds.",
                    rec_type=RecommendationType.NO_ACTION,
                    priority=5,
                    confidence=0.7,
                )
            )

        # ── summary ───────────────────────────────────────────────
        summary = (
            f"Execution health {health_score}/100. "
            f"Exit mix: TP {tp_pct:.0%} / trail {trail_pct:.0%} / "
            f"time {time_pct:.0%} / stop {hard_stop_pct:.0%}. "
            f"Commission drag {commission_drag:.0%} of gross profits."
        )

        return SpecialistReview(
            specialist_name=self.name,
            specialist_id=self.specialist_id,
            review_date=review_date,
            health_score=health_score,
            severity=overall_severity,
            summary=summary,
            findings=findings,
            recommendations=recommendations,
            metrics=metrics,
        )

    # ── sub-score helpers ─────────────────────────────────────────

    @staticmethod
    def _exit_efficiency_score(
        tp_pct: float,
        trail_pct: float,
        time_pct: float,
        hard_stop_pct: float,
        trail_wr: float,
        overall_wr: float,
    ) -> float:
        """Score 0-100: higher when profitable exits dominate.

        Ideal: most exits via TP or winning trail; few time/hard stops.
        """
        # Profitable exit fraction: TP + trail weighted by their win rate
        profitable_fraction = tp_pct + trail_pct * trail_wr
        # Penalise high time-exit and hard-stop rates
        penalty = time_pct * 30 + hard_stop_pct * 20
        score = profitable_fraction * 100 + overall_wr * 20 - penalty
        return min(100.0, max(0.0, score))

    @staticmethod
    def _timing_score(
        avg_hold_win: float,
        avg_hold_loss: float,
        time_exit_pct: float,
        max_hold_bars: int,
    ) -> float:
        """Score 0-100: higher when losers are cut quickly and time exits are low."""
        score = 70.0  # baseline

        # Reward: losers held shorter than winners (fast cut)
        if avg_hold_win > 0 and avg_hold_loss > 0:
            if avg_hold_loss < avg_hold_win:
                score += 15.0  # good: losers cut early
            elif avg_hold_loss > avg_hold_win * 1.5:
                score -= 15.0  # bad: losers held too long

        # Penalise time exits
        score -= time_exit_pct * 50

        return min(100.0, max(0.0, score))

    @staticmethod
    def _commission_score(commission_drag: float) -> float:
        """Score 0-100: lower commission drag = higher score."""
        if commission_drag <= 0.05:
            return 100.0
        if commission_drag >= 0.50:
            return 0.0
        # Linear interpolation between 5% and 50% drag
        return max(0.0, 100.0 - (commission_drag - 0.05) / 0.45 * 100.0)

    @staticmethod
    def _tp_stop_balance_score(
        tp_pct: float,
        trail_pct: float,
        time_pct: float,
        hard_stop_pct: float,
        avg_win_pct: float,
    ) -> float:
        """Score 0-100: rewards a balanced exit mix with meaningful TP wins.

        A healthy scalp model has:
        - TP between 20-60% (not too tight, not too loose)
        - trail contributing meaningfully
        - time exits < 30%
        - hard stops < 25%
        """
        score = 50.0  # baseline

        # TP in sweet spot
        if 0.20 <= tp_pct <= 0.60:
            score += 20.0
        elif tp_pct > 0.60:
            score += 10.0  # too tight but still hitting
        else:
            score -= 10.0  # rarely hitting TP

        # Trail contributing
        if trail_pct > 0.10:
            score += 15.0
        elif trail_pct > 0.05:
            score += 5.0

        # Penalise heavy time exits
        if time_pct > 0.40:
            score -= 20.0
        elif time_pct > 0.30:
            score -= 10.0

        # Penalise heavy hard stops
        if hard_stop_pct > 0.30:
            score -= 15.0

        # Reward meaningful win size
        if avg_win_pct > 10.0:
            score += 10.0
        elif avg_win_pct > 5.0:
            score += 5.0

        return min(100.0, max(0.0, score))


# ── helpers ───────────────────────────────────────────────────────

def _win_rate(trades: list[BacktestTrade]) -> float:
    """Compute win rate for a list of trades.  Returns 0 if empty."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.outcome.value == "win")
    return wins / len(trades)
