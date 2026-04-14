"""Regime-conditioned performance tracking.

Tracks win rate, PnL, and Sharpe separately per market regime.
Detects regime transitions and flags when model enters a regime
where it historically underperforms.

Also computes regime-specific conviction thresholds — the minimum
conviction needed to maintain positive expectancy in each regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegimePerformance:
    """Performance metrics for a specific market regime."""

    regime: str
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    min_conviction_for_positive: float = 0.0  # Minimum score to profit
    is_profitable: bool = False


@dataclass
class RegimeTransition:
    """A detected regime change event."""

    date: str
    from_regime: str
    to_regime: str
    performance_change: str = ""  # "favorable" or "unfavorable"


@dataclass
class RegimeTrackingResult:
    """Full regime-conditioned analysis."""

    regime_performance: dict[str, RegimePerformance] = field(default_factory=dict)
    transitions: list[RegimeTransition] = field(default_factory=list)
    best_regime: str = ""
    worst_regime: str = ""
    current_regime: str = ""
    regime_specific_thresholds: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def analyze_regime_performance(
    trades: list[dict[str, Any]],
) -> RegimeTrackingResult:
    """Analyze trade performance conditioned on market regime.

    Args:
        trades: List of trade dicts with 'regime', 'pnl_pct', 'conviction'.

    Returns:
        RegimeTrackingResult with per-regime metrics and thresholds.
    """
    if len(trades) < 10:
        return RegimeTrackingResult(
            notes=["Insufficient trades for regime analysis"],
        )

    # Group by regime
    regime_trades: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        regime = str(t.get("regime", "unknown"))
        regime_trades.setdefault(regime, []).append(t)

    regime_perf: dict[str, RegimePerformance] = {}
    best_regime = ""
    best_wr = -1.0
    worst_regime = ""
    worst_wr = 2.0

    for regime, rt in regime_trades.items():
        n = len(rt)
        wins = sum(1 for t in rt if float(t.get("pnl_pct", 0)) > 10)
        wr = wins / n if n > 0 else 0.0

        pnls = [float(t.get("pnl_pct", 0)) for t in rt]
        avg = sum(pnls) / n if n > 0 else 0.0
        total = sum(pnls)

        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        pf = gp / gl if gl > 0 else 0.0

        # Find minimum conviction threshold for positive expectancy
        min_conv_for_profit = 0.0
        if n >= 5:
            # Sort trades by conviction and find cutoff
            sorted_by_conv = sorted(
                rt,
                key=lambda x: float(x.get("conviction", x.get("signal_score", 0))),
                reverse=True,
            )
            cumulative_pnl = 0.0
            for i, t in enumerate(sorted_by_conv):
                cumulative_pnl += float(t.get("pnl_pct", 0))
                if cumulative_pnl <= 0 and i > 0:
                    # Crossed below zero — this is the threshold
                    min_conv_for_profit = float(
                        t.get("conviction", t.get("signal_score", 0))
                    )
                    break

        perf = RegimePerformance(
            regime=regime,
            trades=n,
            wins=wins,
            win_rate=round(wr, 3),
            avg_pnl=round(avg, 2),
            total_pnl=round(total, 2),
            profit_factor=round(pf, 2),
            best_trade=round(max(pnls, default=0), 2),
            worst_trade=round(min(pnls, default=0), 2),
            min_conviction_for_positive=round(min_conv_for_profit, 1),
            is_profitable=total > 0,
        )
        regime_perf[regime] = perf

        if wr > best_wr:
            best_wr = wr
            best_regime = regime
        if wr < worst_wr:
            worst_wr = wr
            worst_regime = regime

    # Detect regime transitions from trade sequence
    transitions: list[RegimeTransition] = []
    sorted_trades = sorted(trades, key=lambda t: str(t.get("entry_date", "")))
    prev_regime = ""
    for t in sorted_trades:
        regime = str(t.get("regime", "unknown"))
        if prev_regime and regime != prev_regime:
            # Determine if transition is favorable
            from_perf = regime_perf.get(prev_regime)
            to_perf = regime_perf.get(regime)
            if from_perf and to_perf:
                change = (
                    "favorable" if to_perf.win_rate > from_perf.win_rate
                    else "unfavorable"
                )
            else:
                change = "unknown"
            transitions.append(RegimeTransition(
                date=str(t.get("entry_date", "")),
                from_regime=prev_regime,
                to_regime=regime,
                performance_change=change,
            ))
        prev_regime = regime

    # Build regime-specific thresholds
    thresholds: dict[str, float] = {}
    for regime, perf in regime_perf.items():
        if perf.min_conviction_for_positive > 0:
            thresholds[regime] = perf.min_conviction_for_positive
        elif not perf.is_profitable:
            thresholds[regime] = 10.0  # Block trades in unprofitable regimes

    notes: list[str] = []
    notes.append(f"Best regime: {best_regime} (WR={best_wr:.1%})")
    notes.append(f"Worst regime: {worst_regime} (WR={worst_wr:.1%})")

    unprofitable_regimes = [
        r for r, p in regime_perf.items() if not p.is_profitable
    ]
    if unprofitable_regimes:
        notes.append(
            f"BLOCK these regimes (negative total PnL): "
            f"{', '.join(unprofitable_regimes)}"
        )

    if transitions:
        unfav = sum(1 for t in transitions if t.performance_change == "unfavorable")
        notes.append(
            f"{len(transitions)} regime transitions detected, "
            f"{unfav} unfavorable"
        )

    current = sorted_trades[-1].get("regime", "unknown") if sorted_trades else "unknown"

    return RegimeTrackingResult(
        regime_performance=regime_perf,
        transitions=transitions,
        best_regime=best_regime,
        worst_regime=worst_regime,
        current_regime=str(current),
        regime_specific_thresholds=thresholds,
        notes=notes,
    )
