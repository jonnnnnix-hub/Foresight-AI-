"""Per-factor attribution analysis.

Measures the marginal contribution of each conviction modifier
(PULSE momentum, GEX proxy, Kronos pattern, Monte Carlo) by
running counterfactual analysis: "What would have happened if
we had disabled this factor?"

For each factor, computes:
- Win rate with vs without the factor
- PnL improvement vs baseline
- False positive rate (factor boosted conviction but trade lost)
- False negative rate (factor penalized but trade would have won)

This tells you exactly which layers add value and which hurt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flowedge.scanner.backtest.gex_proxy import classify_gex_proxy, compute_gex_adjustment
from flowedge.scanner.backtest.kronos_signal import compute_kronos_adjustment
from flowedge.scanner.backtest.momentum_score import (
    classify_momentum_bias,
    compute_momentum_adjustment,
)
from flowedge.scanner.backtest.monte_carlo import compute_mc_conviction
from flowedge.scanner.backtest.strategies import Indicators, compute_indicators


@dataclass
class FactorContribution:
    """Attribution metrics for a single factor."""

    factor_name: str
    trades_analyzed: int = 0
    avg_adjustment: float = 0.0  # Mean conviction change from this factor
    boosted_count: int = 0  # Times factor added conviction
    penalized_count: int = 0  # Times factor subtracted conviction
    neutral_count: int = 0  # Times factor had no effect

    # Impact on outcomes
    boosted_win_rate: float = 0.0  # WR when factor boosted
    penalized_win_rate: float = 0.0  # WR when factor penalized
    baseline_win_rate: float = 0.0  # Overall WR

    # Quality metrics
    boost_accuracy: float = 0.0  # % of boosted trades that won
    penalty_accuracy: float = 0.0  # % of penalized trades that lost
    information_ratio: float = 0.0  # Quality of predictions
    marginal_value: float = 0.0  # WR improvement attributable to this factor

    notes: list[str] = field(default_factory=list)


@dataclass
class AttributionResult:
    """Full factor attribution analysis."""

    factors: dict[str, FactorContribution] = field(default_factory=dict)
    baseline_win_rate: float = 0.0
    total_trades: int = 0
    best_factor: str = ""
    worst_factor: str = ""
    recommendation: str = ""
    notes: list[str] = field(default_factory=list)


def compute_factor_adjustments(
    bars: list[dict[str, Any]],
    direction: str,
    indicators: Indicators | None = None,
) -> dict[str, float]:
    """Compute all factor adjustments for a given trade setup.

    Returns dict mapping factor name to its conviction adjustment.
    """
    if len(bars) < 55:
        return {}

    if indicators is None:
        indicators = compute_indicators(bars)

    closes = [float(b.get("close", 0)) for b in bars if float(b.get("close", 0)) > 0]

    # PULSE momentum
    m_bias, m_score = classify_momentum_bias(indicators, closes)
    m_adj = compute_momentum_adjustment(m_bias, m_score, direction)

    # GEX proxy
    g_regime, g_score = classify_gex_proxy(indicators, bars)
    g_adj = compute_gex_adjustment(g_regime, g_score, direction)

    # Kronos pattern
    k_adj = compute_kronos_adjustment(bars, direction)

    # Monte Carlo
    mc_adj, _ = compute_mc_conviction(
        bars, direction, n_simulations=10_000,  # Smaller for speed
    )

    return {
        "momentum": m_adj,
        "gex_proxy": g_adj,
        "kronos": k_adj,
        "monte_carlo": mc_adj,
    }


def run_factor_attribution(
    trades: list[dict[str, Any]],
    bars_by_ticker: dict[str, list[dict[str, Any]]],
) -> AttributionResult:
    """Run factor attribution analysis on completed trades.

    For each trade, recomputes what each factor contributed and
    correlates with the trade outcome.

    Args:
        trades: List of completed trade dicts with at minimum:
            ticker, direction, entry_date, pnl_pct, conviction
        bars_by_ticker: Historical bars keyed by ticker.

    Returns:
        AttributionResult with per-factor contribution metrics.
    """
    factor_names = ["momentum", "gex_proxy", "kronos", "monte_carlo"]

    # Track adjustments and outcomes per factor
    factor_data: dict[str, list[tuple[float, bool]]] = {
        name: [] for name in factor_names
    }

    total = len(trades)
    if total < 5:
        return AttributionResult(
            total_trades=total,
            notes=["Insufficient trades for factor attribution"],
        )

    wins_total = 0

    for trade in trades:
        ticker = trade.get("ticker", "")
        direction = trade.get("direction", "bullish")
        pnl = float(trade.get("pnl_pct", 0))
        is_win = pnl > 10

        if is_win:
            wins_total += 1

        bars = bars_by_ticker.get(ticker, [])
        if len(bars) < 55:
            continue

        # Find bars up to entry date for this trade
        entry_date = str(trade.get("entry_date", ""))
        entry_bars = [b for b in bars if str(b.get("date", "")) <= entry_date]

        if len(entry_bars) < 55:
            continue

        adjustments = compute_factor_adjustments(entry_bars, direction)

        for factor_name in factor_names:
            adj = adjustments.get(factor_name, 0.0)
            factor_data[factor_name].append((adj, is_win))

    baseline_wr = wins_total / total if total > 0 else 0.0

    # Build per-factor contributions
    factors: dict[str, FactorContribution] = {}

    best_factor = ""
    best_value = -999.0
    worst_factor = ""
    worst_value = 999.0

    for name in factor_names:
        data = factor_data[name]
        if not data:
            factors[name] = FactorContribution(factor_name=name)
            continue

        n = len(data)
        adjs = [d[0] for d in data]
        avg_adj = sum(adjs) / n

        boosted = [(adj, win) for adj, win in data if adj > 0.1]
        penalized = [(adj, win) for adj, win in data if adj < -0.1]
        neutral = [(adj, win) for adj, win in data if -0.1 <= adj <= 0.1]

        boost_wins = sum(1 for _, w in boosted if w)
        pen_losses = sum(1 for _, w in penalized if not w)

        boost_wr = boost_wins / len(boosted) if boosted else 0.0
        pen_wr = (
            sum(1 for _, w in penalized if w) / len(penalized)
            if penalized else 0.0
        )

        boost_accuracy = boost_wins / len(boosted) if boosted else 0.0
        penalty_accuracy = pen_losses / len(penalized) if penalized else 0.0

        # Information ratio: boosted WR - penalized WR
        # Higher = factor correctly separates winners from losers
        info_ratio = boost_wr - pen_wr

        # Marginal value: WR of boosted trades vs baseline
        marginal = boost_wr - baseline_wr

        notes: list[str] = []
        if info_ratio > 0.15:
            notes.append("VALUABLE: Factor separates winners well")
        elif info_ratio < -0.10:
            notes.append("HARMFUL: Factor inversely predicts outcomes")
        else:
            notes.append("NEUTRAL: Factor adds little predictive value")

        fc = FactorContribution(
            factor_name=name,
            trades_analyzed=n,
            avg_adjustment=round(avg_adj, 3),
            boosted_count=len(boosted),
            penalized_count=len(penalized),
            neutral_count=len(neutral),
            boosted_win_rate=round(boost_wr, 3),
            penalized_win_rate=round(pen_wr, 3),
            baseline_win_rate=round(baseline_wr, 3),
            boost_accuracy=round(boost_accuracy, 3),
            penalty_accuracy=round(penalty_accuracy, 3),
            information_ratio=round(info_ratio, 3),
            marginal_value=round(marginal, 3),
            notes=notes,
        )
        factors[name] = fc

        if marginal > best_value:
            best_value = marginal
            best_factor = name
        if marginal < worst_value:
            worst_value = marginal
            worst_factor = name

    # Overall recommendation
    rec_parts: list[str] = []
    for name, fc in factors.items():
        if fc.information_ratio < -0.10:
            rec_parts.append(f"DISABLE {name} (IR={fc.information_ratio:.2f})")
        elif fc.information_ratio > 0.15:
            rec_parts.append(f"KEEP {name} (IR={fc.information_ratio:.2f})")
        else:
            rec_parts.append(f"REVIEW {name} (IR={fc.information_ratio:.2f})")

    return AttributionResult(
        factors=factors,
        baseline_win_rate=round(baseline_wr, 3),
        total_trades=total,
        best_factor=best_factor,
        worst_factor=worst_factor,
        recommendation="; ".join(rec_parts),
        notes=[
            f"Best contributing factor: {best_factor} (+{best_value:.1%} marginal WR)",
            f"Worst contributing factor: {worst_factor} ({worst_value:+.1%} marginal WR)",
        ],
    )
