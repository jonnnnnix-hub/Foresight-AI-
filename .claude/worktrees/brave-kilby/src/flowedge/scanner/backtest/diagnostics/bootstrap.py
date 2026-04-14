"""Bootstrap confidence intervals for backtest metrics.

Resamples trade results with replacement to estimate the distribution
of key metrics (win rate, Sharpe, profit factor). This answers:
"Could this performance just be luck?"

If the 95% CI for Sharpe includes zero, the strategy may have no
real edge. If the CI for win rate spans from 15% to 45%, the model's
accuracy is unreliable.

Uses 10,000 bootstrap samples by default — no external dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import sqrt
from typing import Any


@dataclass
class ConfidenceInterval:
    """A metric with confidence interval."""

    metric_name: str
    point_estimate: float
    ci_lower_5: float  # 5th percentile
    ci_lower_25: float  # 25th percentile
    ci_median: float  # 50th percentile
    ci_upper_75: float  # 75th percentile
    ci_upper_95: float  # 95th percentile
    ci_width: float = 0.0  # 95th - 5th
    includes_zero: bool = False  # Does CI span zero?


@dataclass
class BootstrapResult:
    """Full bootstrap analysis output."""

    n_samples: int = 0
    n_trades: int = 0
    win_rate: ConfidenceInterval | None = None
    profit_factor: ConfidenceInterval | None = None
    sharpe_ratio: ConfidenceInterval | None = None
    avg_pnl: ConfidenceInterval | None = None
    expectancy: ConfidenceInterval | None = None
    max_drawdown: ConfidenceInterval | None = None
    is_statistically_significant: bool = False
    notes: list[str] = field(default_factory=list)


def _percentile(data: list[float], pct: float) -> float:
    """Compute percentile of a sorted list."""
    if not data:
        return 0.0
    idx = int(len(data) * pct / 100)
    idx = min(idx, len(data) - 1)
    return data[idx]


def _make_ci(
    name: str,
    samples: list[float],
    point_estimate: float,
) -> ConfidenceInterval:
    """Build a ConfidenceInterval from bootstrap samples."""
    samples.sort()
    p5 = _percentile(samples, 5)
    p25 = _percentile(samples, 25)
    p50 = _percentile(samples, 50)
    p75 = _percentile(samples, 75)
    p95 = _percentile(samples, 95)

    return ConfidenceInterval(
        metric_name=name,
        point_estimate=round(point_estimate, 4),
        ci_lower_5=round(p5, 4),
        ci_lower_25=round(p25, 4),
        ci_median=round(p50, 4),
        ci_upper_75=round(p75, 4),
        ci_upper_95=round(p95, 4),
        ci_width=round(p95 - p5, 4),
        includes_zero=p5 <= 0 <= p95,
    )


def _compute_metrics_from_sample(
    pnl_sample: list[float],
) -> tuple[float, float, float, float]:
    """Compute (win_rate, profit_factor, avg_pnl, sharpe) from a PnL sample."""
    n = len(pnl_sample)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    wins = sum(1 for p in pnl_sample if p > 10)
    wr = wins / n

    gross_profit = sum(p for p in pnl_sample if p > 0)
    gross_loss = abs(sum(p for p in pnl_sample if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    avg = sum(pnl_sample) / n

    # Trade-level Sharpe approximation
    if n > 1:
        mean_p = avg
        var_p = sum((p - mean_p) ** 2 for p in pnl_sample) / (n - 1)
        std_p = sqrt(var_p) if var_p > 0 else 0.01
        sharpe = mean_p / std_p
    else:
        sharpe = 0.0

    return wr, pf, avg, sharpe


def run_bootstrap(
    trades: list[dict[str, Any]],
    n_samples: int = 10_000,
    seed: int | None = 42,
) -> BootstrapResult:
    """Run bootstrap resampling on backtest trades.

    Resamples trade PnLs with replacement to estimate confidence
    intervals for win rate, profit factor, Sharpe, and expectancy.

    Args:
        trades: List of trade dicts with 'pnl_pct' field.
        n_samples: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with CIs for all key metrics.
    """
    if len(trades) < 10:
        return BootstrapResult(
            n_trades=len(trades),
            notes=["Insufficient trades for bootstrap (need 10+)"],
        )

    if seed is not None:
        random.seed(seed)

    pnls = [float(t.get("pnl_pct", 0)) for t in trades]
    n = len(pnls)

    # Point estimates
    pt_wr, pt_pf, pt_avg, pt_sh = _compute_metrics_from_sample(pnls)

    # Bootstrap resampling
    wr_samples: list[float] = []
    pf_samples: list[float] = []
    avg_samples: list[float] = []
    sh_samples: list[float] = []

    for _ in range(n_samples):
        # Resample with replacement
        sample = [random.choice(pnls) for _ in range(n)]
        wr, pf, avg, sh = _compute_metrics_from_sample(sample)
        wr_samples.append(wr)
        pf_samples.append(pf)
        avg_samples.append(avg)
        sh_samples.append(sh)

    win_rate_ci = _make_ci("win_rate", wr_samples, pt_wr)
    pf_ci = _make_ci("profit_factor", pf_samples, pt_pf)
    avg_ci = _make_ci("avg_pnl_pct", avg_samples, pt_avg)
    sharpe_ci = _make_ci("sharpe_ratio", sh_samples, pt_sh)

    notes: list[str] = []

    # Statistical significance check
    is_significant = not sharpe_ci.includes_zero and not avg_ci.includes_zero
    if is_significant:
        notes.append("Strategy has statistically significant edge (95% CI)")
    else:
        if sharpe_ci.includes_zero:
            notes.append(
                "WARNING: Sharpe CI includes zero — edge may not be real"
            )
        if avg_ci.includes_zero:
            notes.append(
                "WARNING: Avg PnL CI includes zero — profitability uncertain"
            )

    # Win rate precision
    if win_rate_ci.ci_width > 0.20:
        notes.append(
            f"Win rate CI very wide ({win_rate_ci.ci_width:.1%}) — "
            "need more trades for confidence"
        )

    # Profit factor reliability
    if pf_ci.ci_lower_5 < 1.0:
        notes.append(
            "Profit factor may be below 1.0 in adverse scenarios"
        )

    return BootstrapResult(
        n_samples=n_samples,
        n_trades=n,
        win_rate=win_rate_ci,
        profit_factor=pf_ci,
        sharpe_ratio=sharpe_ci,
        avg_pnl=avg_ci,
        expectancy=avg_ci,  # Same as avg_pnl for options
        is_statistically_significant=is_significant,
        notes=notes,
    )
