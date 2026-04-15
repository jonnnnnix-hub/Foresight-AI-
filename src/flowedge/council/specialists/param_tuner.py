"""Parameter Tuner specialist — analyzes config sensitivity and suggests tuning.

Compares the current ScalpConfig against known sweep-optimal ranges
(25,600 combos on 4-year OPRA data), evaluates trade volume adequacy,
and checks exit efficiency.  Only recommends changes at >60 % confidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

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

# ── Known sweep-optimal ranges ───────────────────────────────────
# Derived from 25,600 parameter combinations tested on 4-year OPRA data
# (2022-2026) with walk-forward confirmation (train 83% -> validate 93%).


@dataclass(frozen=True)
class _ParamRange:
    """Optimal range for a single parameter."""

    low: float
    high: float
    sweet_spot: float  # single best value observed


_OPTIMAL_RANGES: dict[str, _ParamRange] = {
    "ibs_threshold": _ParamRange(low=0.07, high=0.12, sweet_spot=0.10),
    "rsi3_threshold": _ParamRange(low=12.0, high=18.0, sweet_spot=15.0),
    "vol_spike": _ParamRange(low=2.0, high=3.0, sweet_spot=2.5),
    "intraday_drop": _ParamRange(low=-0.004, high=-0.001, sweet_spot=-0.002),
    "max_hold_bars": _ParamRange(low=8, high=15, sweet_spot=12),
    "trail_pct": _ParamRange(low=0.02, high=0.04, sweet_spot=0.03),
    "tp_underlying": _ParamRange(low=0.0015, high=0.003, sweet_spot=0.002),
    "min_premium": _ParamRange(low=0.15, high=0.50, sweet_spot=0.30),
}

# Minimum trades for statistically meaningful analysis
_MIN_TRADES_MEANINGFUL = 20
# Target trades per month for adequate volume
_TARGET_TRADES_PER_MONTH = 15
# Confidence threshold -- only emit recommendations above this
_CONFIDENCE_THRESHOLD = 0.60


class ParamTuner(BaseSpecialist):
    """Analyzes parameter sensitivity and suggests tuning adjustments."""

    name: str = "Parameter Tuner"
    specialist_id: str = "param_tuner"

    # ── Public interface ──────────────────────────────────────────

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

        # Guard: not enough data for meaningful analysis
        if result.total_trades < 1:
            return SpecialistReview(
                specialist_name=self.name,
                specialist_id=self.specialist_id,
                review_date=review_date,
                health_score=50.0,
                severity=Severity.WARNING,
                summary=(
                    "No trades in the backtest result. "
                    "Cannot assess parameter sensitivity without trade data."
                ),
                findings=[
                    Finding(
                        title="No trade data available",
                        detail=(
                            "The backtest produced zero trades. "
                            "Entry filters may be too strict or data may be missing."
                        ),
                        severity=Severity.WARNING,
                        metric_name="total_trades",
                        metric_value=0,
                        threshold=_MIN_TRADES_MEANINGFUL,
                        evidence=["total_trades=0"],
                    )
                ],
                recommendations=_suggest_loosen_all(config),
                metrics={"total_trades": 0, "param_optimality": 0.0},
            )

        # ── 1. Parameter optimality vs sweep ranges ───────────
        optimality_score, param_findings, param_recs = _assess_param_optimality(config)
        findings.extend(param_findings)
        recommendations.extend(param_recs)
        metrics["param_optimality_pct"] = round(optimality_score * 100, 1)

        # ── 2. Trade volume adequacy ──────────────────────────
        volume_score, vol_findings, vol_recs = _assess_trade_volume(result, config, history)
        findings.extend(vol_findings)
        recommendations.extend(vol_recs)
        metrics["trade_volume_score_pct"] = round(volume_score * 100, 1)
        metrics["total_trades"] = result.total_trades

        # ── 3. Exit efficiency ────────────────────────────────
        exit_score, exit_findings, exit_recs = _assess_exit_efficiency(result, config)
        findings.extend(exit_findings)
        recommendations.extend(exit_recs)
        metrics["exit_efficiency_pct"] = round(exit_score * 100, 1)

        # ── 4. TP capture efficiency ──────────────────────────
        tp_findings, tp_recs, tp_metrics = _assess_tp_capture(result, config)
        findings.extend(tp_findings)
        recommendations.extend(tp_recs)
        metrics.update(tp_metrics)

        # ── 5. Premium filter analysis ────────────────────────
        prem_findings, prem_recs = _assess_premium_filter(result, config)
        findings.extend(prem_findings)
        recommendations.extend(prem_recs)

        # ── Filter to >60 % confidence only ───────────────────
        recommendations = [r for r in recommendations if r.confidence > _CONFIDENCE_THRESHOLD]

        # ── Health score: optimality 50%, volume 25%, exit 25% ─
        health = (optimality_score * 50.0) + (volume_score * 25.0) + (exit_score * 25.0)
        health = max(0.0, min(100.0, health))
        metrics["health_score"] = round(health, 1)

        # Determine severity from health score
        if health >= 80:
            severity = Severity.INFO
        elif health >= 55:
            severity = Severity.WARNING
        else:
            severity = Severity.CRITICAL

        summary = _build_summary(
            health, optimality_score, volume_score,
            exit_score, result, len(recommendations),
        )

        return SpecialistReview(
            specialist_name=self.name,
            specialist_id=self.specialist_id,
            review_date=review_date,
            health_score=round(health, 1),
            severity=severity,
            summary=summary,
            findings=findings,
            recommendations=recommendations,
            metrics=metrics,
        )


# ── Internal helpers ────────────────────────────────────────────────


def _pct_in_range(value: float, rng: _ParamRange) -> float:
    """Return 1.0 if *value* is inside [low, high], tapering to 0 as it diverges."""
    if rng.low <= value <= rng.high:
        return 1.0
    if value < rng.low:
        dist = rng.low - value
        span = rng.high - rng.low if rng.high != rng.low else 1.0
        return max(0.0, 1.0 - dist / span)
    # value > rng.high
    dist = value - rng.high
    span = rng.high - rng.low if rng.high != rng.low else 1.0
    return max(0.0, 1.0 - dist / span)


def _assess_param_optimality(
    config: ScalpConfig,
) -> tuple[float, list[Finding], list[Recommendation]]:
    """Compare every tunable param against sweep-optimal ranges."""
    findings: list[Finding] = []
    recs: list[Recommendation] = []
    scores: list[float] = []

    param_values: dict[str, float] = {
        "ibs_threshold": config.ibs_threshold,
        "rsi3_threshold": config.rsi3_threshold,
        "vol_spike": config.vol_spike,
        "intraday_drop": config.intraday_drop,
        "max_hold_bars": float(config.max_hold_bars),
        "trail_pct": config.trail_pct,
        "tp_underlying": config.tp_underlying,
        "min_premium": config.min_premium,
    }

    for param_name, current_val in param_values.items():
        rng = _OPTIMAL_RANGES[param_name]
        score = _pct_in_range(current_val, rng)
        scores.append(score)

        if score < 1.0:
            distance_label = "below" if current_val < rng.low else "above"
            bound = rng.low if current_val < rng.low else rng.high
            deviation = abs(current_val - bound)
            sev = Severity.WARNING if score < 0.7 else Severity.INFO
            findings.append(
                Finding(
                    title=f"{param_name} is {distance_label} optimal range",
                    detail=(
                        f"Current {param_name}={current_val} is outside the sweep-optimal "
                        f"range [{rng.low}, {rng.high}] (sweet spot={rng.sweet_spot}). "
                        f"Optimality score: {score:.0%}."
                    ),
                    severity=sev,
                    metric_name=param_name,
                    metric_value=current_val,
                    threshold=f"[{rng.low}, {rng.high}]",
                    evidence=[
                        f"25,600-combo sweep optimal range: [{rng.low}, {rng.high}]",
                        f"Sweet spot: {rng.sweet_spot}",
                        f"Current value: {current_val}",
                        f"Optimality score: {score:.0%}",
                    ],
                )
            )

            # Confidence is higher the further from optimal the param is
            confidence = min(0.95, 0.60 + (1.0 - score) * 0.35)
            recs.append(
                Recommendation(
                    title=f"Adjust {param_name} toward sweep-optimal range",
                    rationale=(
                        f"{param_name}={current_val} scores {score:.0%} optimality. "
                        f"Moving toward {rng.sweet_spot} aligns with validated sweep results."
                    ),
                    rec_type=RecommendationType.PARAMETER_CHANGE,
                    priority=2 if score < 0.5 else 3,
                    confidence=round(confidence, 2),
                    current_value=str(current_val),
                    suggested_value=str(rng.sweet_spot),
                    expected_impact=(
                        f"Align with 25,600-combo sweep results. "
                        f"Configs in [{rng.low}, {rng.high}] achieved 90%+ WR."
                    ),
                    evidence=[
                        f"Sweep-optimal range: [{rng.low}, {rng.high}]",
                        f"Current deviation: {distance_label} range"
                        f" by {deviation:.4f}",
                    ],
                )
            )
        else:
            findings.append(
                Finding(
                    title=f"{param_name} within optimal range",
                    detail=f"{param_name}={current_val} is within [{rng.low}, {rng.high}].",
                    severity=Severity.INFO,
                    metric_name=param_name,
                    metric_value=current_val,
                    threshold=f"[{rng.low}, {rng.high}]",
                    evidence=[f"Value {current_val} in range [{rng.low}, {rng.high}]"],
                )
            )

    avg_score = sum(scores) / len(scores) if scores else 0.5
    return avg_score, findings, recs


def _assess_trade_volume(
    result: BacktestResult,
    config: ScalpConfig,
    history: list[BacktestResult],
) -> tuple[float, list[Finding], list[Recommendation]]:
    """Evaluate whether the model is generating enough trades."""
    findings: list[Finding] = []
    recs: list[Recommendation] = []

    total = result.total_trades

    # Estimate months from lookback_days (fallback to trade date range)
    months = max(1.0, result.lookback_days / 30.0) if result.lookback_days > 0 else 1.0
    trades_per_month = total / months

    # Score: 1.0 at target, tapering below
    if trades_per_month >= _TARGET_TRADES_PER_MONTH:
        score = 1.0
    elif trades_per_month <= 0:
        score = 0.0
    else:
        score = trades_per_month / _TARGET_TRADES_PER_MONTH

    if total < _MIN_TRADES_MEANINGFUL:
        findings.append(
            Finding(
                title="Trade count below statistical significance",
                detail=(
                    f"Only {total} trades across {months:.0f} months "
                    f"({trades_per_month:.1f}/month). "
                    f"Need >= {_MIN_TRADES_MEANINGFUL} for meaningful stats."
                ),
                severity=Severity.WARNING,
                metric_name="total_trades",
                metric_value=total,
                threshold=_MIN_TRADES_MEANINGFUL,
                evidence=[
                    f"total_trades={total}",
                    f"lookback_days={result.lookback_days}",
                    f"trades_per_month={trades_per_month:.1f}",
                ],
            )
        )
        # Suggest loosening the most restrictive filter
        recs.extend(_suggest_loosen_for_volume(config, trades_per_month))
    elif trades_per_month < _TARGET_TRADES_PER_MONTH:
        findings.append(
            Finding(
                title="Trade volume below target",
                detail=(
                    f"{trades_per_month:.1f} trades/month vs target of {_TARGET_TRADES_PER_MONTH}. "
                    f"Consider slightly loosening entry filters."
                ),
                severity=Severity.INFO,
                metric_name="trades_per_month",
                metric_value=round(trades_per_month, 1),
                threshold=_TARGET_TRADES_PER_MONTH,
                evidence=[
                    f"total_trades={total}",
                    f"months={months:.1f}",
                    f"trades_per_month={trades_per_month:.1f}",
                ],
            )
        )
        recs.extend(_suggest_loosen_for_volume(config, trades_per_month))
    else:
        findings.append(
            Finding(
                title="Adequate trade volume",
                detail=(
                    f"{trades_per_month:.1f} trades/month meets "
                    f"the {_TARGET_TRADES_PER_MONTH}/month target."
                ),
                severity=Severity.INFO,
                metric_name="trades_per_month",
                metric_value=round(trades_per_month, 1),
                threshold=_TARGET_TRADES_PER_MONTH,
                evidence=[f"total_trades={total}", f"months={months:.1f}"],
            )
        )

    # Win rate too low -> suggest tightening filters
    if total >= _MIN_TRADES_MEANINGFUL and result.win_rate < 0.55:
        findings.append(
            Finding(
                title="Win rate below acceptable threshold",
                detail=(
                    f"Win rate {result.win_rate:.1%} is below 55%. "
                    f"Consider tightening entry filters to improve signal quality."
                ),
                severity=Severity.WARNING,
                metric_name="win_rate",
                metric_value=round(result.win_rate, 4),
                threshold=0.55,
                evidence=[
                    f"win_rate={result.win_rate:.1%}",
                    f"wins={result.wins}, losses={result.losses}",
                    f"total_trades={total}",
                ],
            )
        )
        recs.extend(_suggest_tighten_for_wr(config, result.win_rate))

    return score, findings, recs


def _assess_exit_efficiency(
    result: BacktestResult,
    config: ScalpConfig,
) -> tuple[float, list[Finding], list[Recommendation]]:
    """Check if exit parameters (max_hold, trail, TP) are optimal."""
    findings: list[Finding] = []
    recs: list[Recommendation] = []

    trades = result.trades
    if not trades:
        return 0.5, findings, recs  # neutral when no trade data

    total = len(trades)

    # ── Time-exit analysis ────────────────────────────────────
    time_exits = [t for t in trades if t.exit_reason == "time_exit"]
    time_exit_rate = len(time_exits) / total if total else 0.0
    time_exit_losses = [t for t in time_exits if t.pnl_pct < 0]
    time_exit_loss_rate = len(time_exit_losses) / len(time_exits) if time_exits else 0.0

    time_exit_score = 1.0
    if time_exit_rate > 0.30:
        time_exit_score = max(0.0, 1.0 - (time_exit_rate - 0.30) / 0.40)
        sev = Severity.WARNING if time_exit_rate > 0.40 else Severity.INFO
        findings.append(
            Finding(
                title="High time-exit rate",
                detail=(
                    f"{time_exit_rate:.1%} of trades exit via time limit "
                    f"(max_hold_bars={config.max_hold_bars}). "
                    f"Of those, {time_exit_loss_rate:.1%} are losses. "
                    f"The TP or hold period may need adjustment."
                ),
                severity=sev,
                metric_name="time_exit_rate",
                metric_value=round(time_exit_rate, 4),
                threshold=0.30,
                evidence=[
                    f"time_exits={len(time_exits)}/{total} ({time_exit_rate:.1%})",
                    f"time_exit_losses={len(time_exit_losses)}"
                    f"/{len(time_exits)} ({time_exit_loss_rate:.1%})",
                    f"max_hold_bars={config.max_hold_bars}",
                ],
            )
        )
        # If many time exits are losses, extend hold period
        if time_exit_loss_rate > 0.50 and config.max_hold_bars <= 15:
            suggested_hold = min(20, config.max_hold_bars + 4)
            recs.append(
                Recommendation(
                    title="Increase max_hold_bars to reduce time-exit losses",
                    rationale=(
                        f"{time_exit_loss_rate:.0%} of time-exits are losses, "
                        f"suggesting trades need more time to reach TP."
                    ),
                    rec_type=RecommendationType.PARAMETER_CHANGE,
                    priority=2,
                    confidence=round(min(0.85, 0.60 + time_exit_loss_rate * 0.20), 2),
                    current_value=str(config.max_hold_bars),
                    suggested_value=str(suggested_hold),
                    expected_impact=(
                        f"Allow trades {(suggested_hold - config.max_hold_bars) * 5} more minutes "
                        f"to reach TP, potentially converting time-exit losses to wins."
                    ),
                    evidence=[
                        f"time_exit_loss_rate={time_exit_loss_rate:.1%}",
                        f"current max_hold_bars={config.max_hold_bars}",
                    ],
                )
            )

    # ── Trailing stop analysis ────────────────────────────────
    trail_stops = [t for t in trades if t.exit_reason == "trailing_stop"]
    trail_stop_rate = len(trail_stops) / total if total else 0.0

    trail_score = 1.0
    if trail_stops:
        # Check if trail is stopping out winners too early
        trail_wins = [t for t in trail_stops if t.pnl_pct > 0]
        trail_win_avg_pnl = (
            sum(t.pnl_pct for t in trail_wins) / len(trail_wins) if trail_wins else 0.0
        )
        # Compare trail-stopped winners vs overall avg win
        if result.avg_win_pct > 0 and trail_wins:
            capture_ratio = trail_win_avg_pnl / result.avg_win_pct if result.avg_win_pct else 0.0
            if capture_ratio < 0.60:
                trail_score = max(0.3, capture_ratio)
                findings.append(
                    Finding(
                        title="Trailing stop may be too tight",
                        detail=(
                            f"Trail-stopped winners average {trail_win_avg_pnl:.2%} vs "
                            f"overall avg win of {result.avg_win_pct:.2%} "
                            f"(capture ratio: {capture_ratio:.1%}). "
                            f"Trail at {config.trail_pct:.1%} may be cutting winners short."
                        ),
                        severity=Severity.WARNING,
                        metric_name="trail_capture_ratio",
                        metric_value=round(capture_ratio, 4),
                        threshold=0.60,
                        evidence=[
                            f"trail_stopped_winners={len(trail_wins)}",
                            f"trail_win_avg_pnl={trail_win_avg_pnl:.2%}",
                            f"overall_avg_win={result.avg_win_pct:.2%}",
                            f"trail_pct={config.trail_pct}",
                        ],
                    )
                )
                # Only suggest widening if within a reasonable range
                suggested_trail = min(0.06, config.trail_pct * 1.5)
                recs.append(
                    Recommendation(
                        title="Widen trailing stop to capture more upside",
                        rationale=(
                            f"Trail-stopped winners capture only {capture_ratio:.0%} of "
                            f"avg win PnL. Wider trail may let winners run further."
                        ),
                        rec_type=RecommendationType.PARAMETER_CHANGE,
                        priority=3,
                        confidence=round(min(0.80, 0.55 + (0.60 - capture_ratio) * 0.50), 2),
                        current_value=str(config.trail_pct),
                        suggested_value=str(round(suggested_trail, 4)),
                        expected_impact=(
                            "Increase average win size by letting "
                            "profitable trades run longer."
                        ),
                        evidence=[
                            f"capture_ratio={capture_ratio:.1%}",
                            f"trail_win_avg_pnl={trail_win_avg_pnl:.2%}",
                        ],
                    )
                )

        # Check if trail is triggering too many premature exits (losses)
        trail_losses = [t for t in trail_stops if t.pnl_pct < 0]
        if len(trail_losses) > 0.5 * len(trail_stops) and trail_stop_rate > 0.15:
            findings.append(
                Finding(
                    title="Trailing stop producing excessive losses",
                    detail=(
                        f"{len(trail_losses)}/{len(trail_stops)} trail-stop exits are losses. "
                        f"Trail at {config.trail_pct:.1%} may be too tight for current volatility."
                    ),
                    severity=Severity.WARNING,
                    metric_name="trail_loss_fraction",
                    metric_value=round(len(trail_losses) / len(trail_stops), 4),
                    threshold=0.50,
                    evidence=[
                        f"trail_stop_exits={len(trail_stops)}",
                        f"trail_stop_losses={len(trail_losses)}",
                        f"trail_pct={config.trail_pct}",
                    ],
                )
            )

    # Combine exit score: average of time-exit and trail sub-scores
    exit_score = (time_exit_score + trail_score) / 2.0
    return exit_score, findings, recs


def _assess_tp_capture(
    result: BacktestResult,
    config: ScalpConfig,
) -> tuple[list[Finding], list[Recommendation], dict[str, float | str]]:
    """Analyze relationship between underlying_move_pct and pnl_pct.

    Checks whether TP is efficiently capturing underlying moves.
    """
    findings: list[Finding] = []
    recs: list[Recommendation] = []
    metrics: dict[str, float | str] = {}

    trades = result.trades
    if not trades:
        return findings, recs, metrics

    # Only consider trades where we have underlying move data
    trades_with_move = [t for t in trades if t.underlying_move_pct != 0.0]
    if len(trades_with_move) < 5:
        return findings, recs, metrics

    # TP exits specifically
    tp_trades = [t for t in trades if t.exit_reason == "take_profit"]

    if tp_trades:
        avg_tp_underlying_move = sum(t.underlying_move_pct for t in tp_trades) / len(tp_trades)
        avg_tp_pnl = sum(t.pnl_pct for t in tp_trades) / len(tp_trades)
        metrics["avg_tp_underlying_move_pct"] = round(avg_tp_underlying_move * 100, 4)
        metrics["avg_tp_pnl_pct"] = round(avg_tp_pnl * 100, 4)
        metrics["tp_trade_count"] = len(tp_trades)

        # How much of the underlying move is the option capturing?
        if avg_tp_underlying_move > 0:
            leverage_ratio = avg_tp_pnl / avg_tp_underlying_move
            metrics["tp_leverage_ratio"] = round(leverage_ratio, 2)

    # Check for "left money on the table": underlying moved significantly more than TP
    winning_trades = [t for t in trades if t.pnl_pct > 0]
    if winning_trades:
        moves_beyond_tp = [
            t for t in winning_trades
            if t.underlying_move_pct > config.tp_underlying * 2.0
        ]
        beyond_tp_rate = len(moves_beyond_tp) / len(winning_trades)
        metrics["moves_beyond_2x_tp_rate"] = round(beyond_tp_rate, 4)

        if beyond_tp_rate > 0.30:
            avg_excess = sum(
                t.underlying_move_pct - config.tp_underlying for t in moves_beyond_tp
            ) / len(moves_beyond_tp)
            findings.append(
                Finding(
                    title="TP may be leaving money on the table",
                    detail=(
                        f"{beyond_tp_rate:.0%} of winners had underlying moves > 2x the TP target "
                        f"({config.tp_underlying:.3%}). Average excess move: {avg_excess:.3%}. "
                        f"A wider TP or partial-exit strategy could capture more upside."
                    ),
                    severity=Severity.INFO,
                    metric_name="moves_beyond_2x_tp_rate",
                    metric_value=round(beyond_tp_rate, 4),
                    threshold=0.30,
                    evidence=[
                        f"winners_beyond_2x_tp={len(moves_beyond_tp)}/{len(winning_trades)}",
                        f"avg_excess_underlying_move={avg_excess:.4%}",
                        f"tp_underlying={config.tp_underlying}",
                    ],
                )
            )
            suggested_tp = round(config.tp_underlying * 1.5, 5)
            # Only recommend if it stays in the optimal range
            tp_range = _OPTIMAL_RANGES["tp_underlying"]
            if suggested_tp <= tp_range.high:
                recs.append(
                    Recommendation(
                        title="Widen take-profit target to capture larger moves",
                        rationale=(
                            f"{beyond_tp_rate:.0%} of winning trades had underlying moves "
                            f"exceeding 2x TP. Widening TP could increase average win size."
                        ),
                        rec_type=RecommendationType.PARAMETER_CHANGE,
                        priority=3,
                        confidence=round(min(0.80, 0.55 + beyond_tp_rate * 0.30), 2),
                        current_value=str(config.tp_underlying),
                        suggested_value=str(suggested_tp),
                        expected_impact=(
                            f"Capture avg {avg_excess:.3%} more underlying "
                            f"move per winning trade."
                        ),
                        evidence=[
                            f"beyond_2x_tp_rate={beyond_tp_rate:.0%}",
                            f"avg_excess_move={avg_excess:.4%}",
                        ],
                    )
                )

    # Check for TP too aggressive (rarely hit)
    if tp_trades and trades:
        tp_hit_rate = len(tp_trades) / len(trades)
        metrics["tp_hit_rate"] = round(tp_hit_rate, 4)
        if tp_hit_rate < 0.10 and len(trades) >= _MIN_TRADES_MEANINGFUL:
            findings.append(
                Finding(
                    title="Take-profit rarely hit",
                    detail=(
                        f"Only {tp_hit_rate:.1%} of trades exit via TP. "
                        f"Current TP target ({config.tp_underlying:.3%} underlying move) "
                        f"may be too aggressive for typical price action."
                    ),
                    severity=Severity.WARNING,
                    metric_name="tp_hit_rate",
                    metric_value=round(tp_hit_rate, 4),
                    threshold=0.10,
                    evidence=[
                        f"tp_exits={len(tp_trades)}/{len(trades)} ({tp_hit_rate:.1%})",
                        f"tp_underlying={config.tp_underlying}",
                    ],
                )
            )
            suggested_tp_tighter = round(config.tp_underlying * 0.75, 5)
            if suggested_tp_tighter >= _OPTIMAL_RANGES["tp_underlying"].low:
                recs.append(
                    Recommendation(
                        title="Tighten take-profit to increase hit rate",
                        rationale=(
                            f"TP is hit only {tp_hit_rate:.1%} of the time. "
                            f"A tighter target may capture more frequent smaller wins."
                        ),
                        rec_type=RecommendationType.PARAMETER_CHANGE,
                        priority=2,
                        confidence=round(min(0.85, 0.65 + (0.10 - tp_hit_rate) * 2.0), 2),
                        current_value=str(config.tp_underlying),
                        suggested_value=str(suggested_tp_tighter),
                        expected_impact=(
                            "Increase TP hit rate by targeting more "
                            "achievable underlying moves."
                        ),
                        evidence=[
                            f"current_tp_hit_rate={tp_hit_rate:.1%}",
                            f"tp_underlying={config.tp_underlying}",
                        ],
                    )
                )

    return findings, recs, metrics


def _assess_premium_filter(
    result: BacktestResult,
    config: ScalpConfig,
) -> tuple[list[Finding], list[Recommendation]]:
    """Check if the min_premium filter is too restrictive.

    Looks for filter_stats in notes for pass/reject ratios.
    """
    findings: list[Finding] = []
    recs: list[Recommendation] = []

    # Try to extract filter_stats from notes
    filter_stats: dict[str, Any] | None = None
    for note in result.notes:
        try:
            parsed = json.loads(note)
            if isinstance(parsed, dict) and (
                "filter_stats" in parsed
                or "premium_rejection_rate" in parsed
            ):
                filter_stats = parsed
                break
        except (json.JSONDecodeError, TypeError):
            continue

    if filter_stats is not None:
        rejection_rate = filter_stats.get("premium_rejection_rate", 0.0)
        if isinstance(rejection_rate, (int, float)) and rejection_rate > 0.40:
            findings.append(
                Finding(
                    title="min_premium filter rejecting many signals",
                    detail=(
                        f"{rejection_rate:.0%} of signals are rejected by the min_premium "
                        f"filter (${config.min_premium:.2f}). This may be excluding viable trades."
                    ),
                    severity=Severity.WARNING,
                    metric_name="premium_rejection_rate",
                    metric_value=round(rejection_rate, 4),
                    threshold=0.40,
                    evidence=[
                        f"premium_rejection_rate={rejection_rate:.0%}",
                        f"min_premium=${config.min_premium:.2f}",
                        "source=filter_stats in backtest notes",
                    ],
                )
            )
            suggested_premium = round(max(0.15, config.min_premium * 0.75), 2)
            recs.append(
                Recommendation(
                    title="Lower min_premium to admit more trades",
                    rationale=(
                        f"{rejection_rate:.0%} rejection rate suggests the premium floor "
                        f"is excluding tradeable setups."
                    ),
                    rec_type=RecommendationType.FILTER_CHANGE,
                    priority=3,
                    confidence=round(min(0.80, 0.55 + rejection_rate * 0.25), 2),
                    current_value=f"${config.min_premium:.2f}",
                    suggested_value=f"${suggested_premium:.2f}",
                    expected_impact=(
                        f"Potentially admit ~{rejection_rate * 100 * 0.5:.0f}% "
                        f"more trades."
                    ),
                    evidence=[
                        f"premium_rejection_rate={rejection_rate:.0%}",
                        f"current min_premium=${config.min_premium:.2f}",
                    ],
                )
            )
    else:
        # No filter stats available, evaluate from trade data heuristics
        if result.trades:
            premiums = [t.entry_price for t in result.trades if t.entry_price > 0]
            if premiums:
                near_floor = [p for p in premiums if p < config.min_premium * 1.20]
                near_floor_rate = len(near_floor) / len(premiums)
                if near_floor_rate > 0.40:
                    findings.append(
                        Finding(
                            title="Many trades near min_premium floor",
                            detail=(
                                f"{near_floor_rate:.0%} of trades have "
                                f"entry premiums within 20% of the "
                                f"min_premium floor "
                                f"(${config.min_premium:.2f}). "
                                f"The filter may be excluding "
                                f"similar setups just below "
                                f"the threshold."
                            ),
                            severity=Severity.INFO,
                            metric_name="near_floor_rate",
                            metric_value=round(near_floor_rate, 4),
                            threshold=0.40,
                            evidence=[
                                f"trades_near_floor="
                                f"{len(near_floor)}/{len(premiums)}"
                                f" ({near_floor_rate:.0%})",
                                f"min_premium=${config.min_premium:.2f}",
                                f"median_premium="
                                f"${sorted(premiums)[len(premiums)//2]:.2f}",
                            ],
                        )
                    )

    return findings, recs


# ── Recommendation builders ────────────────────────────────────────


def _suggest_loosen_all(config: ScalpConfig) -> list[Recommendation]:
    """When there are zero trades, suggest loosening all major filters."""
    recs: list[Recommendation] = []
    # IBS
    if config.ibs_threshold < _OPTIMAL_RANGES["ibs_threshold"].high:
        recs.append(
            Recommendation(
                title="Loosen ibs_threshold (zero trades)",
                rationale="No trades generated. Raising IBS threshold admits more signals.",
                rec_type=RecommendationType.FILTER_CHANGE,
                priority=1,
                confidence=0.70,
                current_value=str(config.ibs_threshold),
                suggested_value=str(_OPTIMAL_RANGES["ibs_threshold"].high),
                expected_impact="Admit more oversold signals that pass the IBS gate.",
                evidence=["total_trades=0"],
            )
        )
    # RSI3
    if config.rsi3_threshold < _OPTIMAL_RANGES["rsi3_threshold"].high:
        recs.append(
            Recommendation(
                title="Loosen rsi3_threshold (zero trades)",
                rationale="No trades generated. Raising RSI3 threshold admits more signals.",
                rec_type=RecommendationType.FILTER_CHANGE,
                priority=1,
                confidence=0.70,
                current_value=str(config.rsi3_threshold),
                suggested_value=str(_OPTIMAL_RANGES["rsi3_threshold"].high),
                expected_impact="Admit more oversold signals that pass the RSI3 gate.",
                evidence=["total_trades=0"],
            )
        )
    # Min premium
    if config.min_premium > _OPTIMAL_RANGES["min_premium"].low:
        recs.append(
            Recommendation(
                title="Lower min_premium (zero trades)",
                rationale="No trades generated. Lowering min_premium admits cheaper options.",
                rec_type=RecommendationType.FILTER_CHANGE,
                priority=2,
                confidence=0.65,
                current_value=f"${config.min_premium:.2f}",
                suggested_value=f"${_OPTIMAL_RANGES['min_premium'].low:.2f}",
                expected_impact="Admit lower-premium options that were previously excluded.",
                evidence=["total_trades=0"],
            )
        )
    return recs


def _suggest_loosen_for_volume(
    config: ScalpConfig,
    trades_per_month: float,
) -> list[Recommendation]:
    """Suggest loosening the tightest filter to increase trade volume."""
    recs: list[Recommendation] = []

    # Identify the most constrained params (furthest below their range ceiling)
    # IBS: lower = tighter, raise to loosen
    ibs_headroom = _OPTIMAL_RANGES["ibs_threshold"].high - config.ibs_threshold
    # RSI3: lower = tighter, raise to loosen
    rsi_headroom = _OPTIMAL_RANGES["rsi3_threshold"].high - config.rsi3_threshold

    if ibs_headroom > 0.01:
        step = min(ibs_headroom, 0.03)
        suggested = round(config.ibs_threshold + step, 3)
        confidence = min(0.80, 0.55 + (1.0 - trades_per_month / _TARGET_TRADES_PER_MONTH) * 0.30)
        recs.append(
            Recommendation(
                title="Loosen ibs_threshold to increase trade volume",
                rationale=(
                    f"Only {trades_per_month:.1f} trades/month. "
                    f"Raising IBS from {config.ibs_threshold} to {suggested} "
                    f"(still within optimal [{_OPTIMAL_RANGES['ibs_threshold'].low}, "
                    f"{_OPTIMAL_RANGES['ibs_threshold'].high}]) should admit more signals."
                ),
                rec_type=RecommendationType.FILTER_CHANGE,
                priority=2,
                confidence=round(max(0.61, confidence), 2),
                current_value=str(config.ibs_threshold),
                suggested_value=str(suggested),
                expected_impact=(
                    f"Increase trade frequency toward "
                    f"{_TARGET_TRADES_PER_MONTH}/month target."
                ),
                evidence=[
                    f"trades_per_month={trades_per_month:.1f}",
                    f"target={_TARGET_TRADES_PER_MONTH}",
                    f"ibs_headroom={ibs_headroom:.3f}",
                ],
            )
        )

    if rsi_headroom > 1.0:
        step = min(rsi_headroom, 3.0)
        suggested_rsi = round(config.rsi3_threshold + step, 1)
        confidence = min(0.75, 0.55 + (1.0 - trades_per_month / _TARGET_TRADES_PER_MONTH) * 0.25)
        recs.append(
            Recommendation(
                title="Loosen rsi3_threshold to increase trade volume",
                rationale=(
                    f"Only {trades_per_month:.1f} trades/month. "
                    f"Raising RSI3 from {config.rsi3_threshold} to {suggested_rsi} "
                    f"should admit more signals."
                ),
                rec_type=RecommendationType.FILTER_CHANGE,
                priority=3,
                confidence=round(max(0.61, confidence), 2),
                current_value=str(config.rsi3_threshold),
                suggested_value=str(suggested_rsi),
                expected_impact="Increase signal count by widening RSI3 gate.",
                evidence=[
                    f"trades_per_month={trades_per_month:.1f}",
                    f"rsi3_headroom={rsi_headroom:.1f}",
                ],
            )
        )

    return recs


def _suggest_tighten_for_wr(
    config: ScalpConfig,
    win_rate: float,
) -> list[Recommendation]:
    """Suggest tightening filters when win rate is too low."""
    recs: list[Recommendation] = []

    # Lower IBS = stricter oversold requirement
    ibs_range = _OPTIMAL_RANGES["ibs_threshold"]
    if config.ibs_threshold > ibs_range.low:
        suggested = round(max(ibs_range.low, config.ibs_threshold - 0.02), 3)
        confidence = min(0.80, 0.55 + (0.55 - win_rate) * 1.5)
        if confidence > _CONFIDENCE_THRESHOLD:
            recs.append(
                Recommendation(
                    title="Tighten ibs_threshold to improve win rate",
                    rationale=(
                        f"Win rate is {win_rate:.1%}. Lowering IBS threshold from "
                        f"{config.ibs_threshold} to {suggested} requires deeper oversold "
                        f"conditions, filtering out weaker signals."
                    ),
                    rec_type=RecommendationType.FILTER_CHANGE,
                    priority=2,
                    confidence=round(confidence, 2),
                    current_value=str(config.ibs_threshold),
                    suggested_value=str(suggested),
                    expected_impact=(
                        "Improve win rate by requiring deeper "
                        "oversold conditions at entry."
                    ),
                    evidence=[
                        f"win_rate={win_rate:.1%}",
                        "target_wr>=55%",
                    ],
                )
            )

    # Lower RSI3 = stricter
    rsi_range = _OPTIMAL_RANGES["rsi3_threshold"]
    if config.rsi3_threshold > rsi_range.low:
        suggested_rsi = round(max(rsi_range.low, config.rsi3_threshold - 2.0), 1)
        confidence = min(0.75, 0.50 + (0.55 - win_rate) * 1.2)
        if confidence > _CONFIDENCE_THRESHOLD:
            recs.append(
                Recommendation(
                    title="Tighten rsi3_threshold to improve win rate",
                    rationale=(
                        f"Win rate is {win_rate:.1%}. Lowering RSI3 from "
                        f"{config.rsi3_threshold} to {suggested_rsi} should improve signal quality."
                    ),
                    rec_type=RecommendationType.FILTER_CHANGE,
                    priority=3,
                    confidence=round(confidence, 2),
                    current_value=str(config.rsi3_threshold),
                    suggested_value=str(suggested_rsi),
                    expected_impact="Improve win rate by requiring stronger RSI oversold signals.",
                    evidence=[
                        f"win_rate={win_rate:.1%}",
                        "target_wr>=55%",
                    ],
                )
            )

    return recs


def _build_summary(
    health: float,
    optimality: float,
    volume: float,
    exit_eff: float,
    result: BacktestResult,
    rec_count: int,
) -> str:
    """Build 2-3 sentence executive summary."""
    parts: list[str] = []

    parts.append(
        f"Parameter health score: {health:.0f}/100 "
        f"(optimality {optimality:.0%}, volume {volume:.0%}, exit efficiency {exit_eff:.0%})."
    )

    if result.total_trades < _MIN_TRADES_MEANINGFUL:
        parts.append(
            f"Only {result.total_trades} trades — insufficient for reliable parameter tuning."
        )
    elif health >= 80:
        parts.append("Parameters are well-aligned with sweep-validated optimal ranges.")
    elif health >= 55:
        parts.append(
            "Some parameters are outside optimal ranges; "
            "targeted adjustments recommended."
        )
    else:
        parts.append("Multiple parameters deviate from sweep-optimal ranges; review recommended.")

    if rec_count > 0:
        suffix = "s" if rec_count != 1 else ""
        parts.append(
            f"{rec_count} high-confidence "
            f"recommendation{suffix} generated."
        )
    else:
        parts.append("No high-confidence changes recommended at this time.")

    return " ".join(parts)
