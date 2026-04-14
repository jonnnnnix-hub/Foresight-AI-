"""Signal decay analysis.

Measures how conviction score correlates with actual trade outcome.
If a scoring model has predictive power, high-conviction trades should
win more often than low-conviction trades. If correlation is ~0,
the model has no edge and we're trading noise.

Also measures "signal freshness" — do signals lose edge over time?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any


@dataclass
class ConvictionBucket:
    """Performance by conviction score range."""

    bucket_name: str
    min_conviction: float
    max_conviction: float
    trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0


@dataclass
class SignalDecayResult:
    """Signal quality and decay analysis."""

    # Score-outcome correlation
    score_pnl_correlation: float = 0.0  # Pearson r between conviction and PnL
    score_win_correlation: float = 0.0  # Point-biserial r between score and win/loss

    # Conviction buckets
    buckets: list[ConvictionBucket] = field(default_factory=list)

    # Monotonicity: does WR increase with conviction?
    is_monotonically_increasing: bool = False
    monotonicity_violations: int = 0

    # Hold-day analysis: P&L by day held
    pnl_by_hold_day: dict[int, float] = field(default_factory=dict)
    optimal_hold_days: int = 0

    # Signal has predictive power?
    has_predictive_power: bool = False
    notes: list[str] = field(default_factory=list)


def _pearson_r(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3 or len(y) != n:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov_xy / denom


def run_signal_decay_analysis(
    trades: list[dict[str, Any]],
) -> SignalDecayResult:
    """Analyze signal quality and decay patterns.

    Args:
        trades: List of trade dicts with conviction, pnl_pct, hold_days.

    Returns:
        SignalDecayResult with correlation, bucket, and decay metrics.
    """
    if len(trades) < 10:
        return SignalDecayResult(
            notes=["Insufficient trades for signal decay analysis"],
        )

    # Extract scores and outcomes
    scores = [float(t.get("conviction", t.get("signal_score", 5))) for t in trades]
    pnls = [float(t.get("pnl_pct", 0)) for t in trades]
    wins = [1.0 if p > 10 else 0.0 for p in pnls]
    hold_days_list = [int(t.get("hold_days", 0)) for t in trades]

    # Score-outcome correlation
    score_pnl_r = round(_pearson_r(scores, pnls), 3)
    score_win_r = round(_pearson_r(scores, wins), 3)

    # Conviction buckets
    bucket_ranges = [
        ("0-5", 0, 5),
        ("5-6", 5, 6),
        ("6-7", 6, 7),
        ("7-8", 7, 8),
        ("8-9", 8, 9),
        ("9-10", 9, 10.1),
    ]

    buckets: list[ConvictionBucket] = []
    for name, lo, hi in bucket_ranges:
        bt = [
            (s, p) for s, p in zip(scores, pnls, strict=True)
            if lo <= s < hi
        ]
        if not bt:
            continue
        n = len(bt)
        w = sum(1 for _, p in bt if p > 10)
        avg = sum(p for _, p in bt) / n
        total = sum(p for _, p in bt)
        buckets.append(ConvictionBucket(
            bucket_name=name,
            min_conviction=lo,
            max_conviction=hi,
            trades=n,
            win_rate=round(w / n, 3),
            avg_pnl=round(avg, 2),
            total_pnl=round(total, 2),
        ))

    # Monotonicity check
    violations = 0
    for i in range(1, len(buckets)):
        if buckets[i].win_rate < buckets[i - 1].win_rate:
            violations += 1
    is_monotonic = violations == 0 and len(buckets) > 1

    # Hold-day analysis
    day_pnls: dict[int, list[float]] = {}
    for hd, pnl in zip(hold_days_list, pnls, strict=True):
        day_pnls.setdefault(hd, []).append(pnl)

    pnl_by_day: dict[int, float] = {}
    for day, ps in sorted(day_pnls.items()):
        pnl_by_day[day] = round(sum(ps) / len(ps), 2) if ps else 0.0

    optimal_day = max(pnl_by_day, key=lambda d: pnl_by_day[d], default=0) if pnl_by_day else 0

    # Determine if signal has predictive power
    has_power = abs(score_pnl_r) > 0.10 or abs(score_win_r) > 0.10
    notes: list[str] = []

    if score_pnl_r > 0.15:
        notes.append(
            f"GOOD: Score positively correlates with PnL (r={score_pnl_r:.3f})"
        )
    elif score_pnl_r < -0.10:
        notes.append(
            f"BAD: Score negatively correlates with PnL (r={score_pnl_r:.3f}) "
            "— higher scores produce WORSE outcomes"
        )
    else:
        notes.append(
            f"WEAK: Low score-PnL correlation (r={score_pnl_r:.3f}) "
            "— scoring model has minimal predictive value"
        )

    if not is_monotonic and len(buckets) > 2:
        notes.append(
            f"NON-MONOTONIC: {violations} violations — "
            "higher conviction doesn't always mean better outcomes"
        )

    if optimal_day > 0:
        notes.append(f"Optimal hold period: {optimal_day} days")

    return SignalDecayResult(
        score_pnl_correlation=score_pnl_r,
        score_win_correlation=score_win_r,
        buckets=buckets,
        is_monotonically_increasing=is_monotonic,
        monotonicity_violations=violations,
        pnl_by_hold_day=pnl_by_day,
        optimal_hold_days=optimal_day,
        has_predictive_power=has_power,
        notes=notes,
    )
