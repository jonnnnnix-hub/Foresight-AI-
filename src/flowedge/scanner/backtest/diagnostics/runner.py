"""Unified diagnostic runner.

Runs all diagnostic layers on a backtest result and produces
a comprehensive robustness report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flowedge.scanner.backtest.diagnostics.bootstrap import (
    BootstrapResult,
    run_bootstrap,
)
from flowedge.scanner.backtest.diagnostics.regime_tracker import (
    RegimeTrackingResult,
    analyze_regime_performance,
)
from flowedge.scanner.backtest.diagnostics.signal_decay import (
    SignalDecayResult,
    run_signal_decay_analysis,
)
from flowedge.scanner.backtest.diagnostics.trade_clustering import (
    ClusteringResult,
    run_clustering_analysis,
)
from flowedge.scanner.backtest.diagnostics.walk_forward import (
    WalkForwardResult,
    run_walk_forward,
)


@dataclass
class DiagnosticReport:
    """Full diagnostic report across all analysis layers."""

    walk_forward: WalkForwardResult | None = None
    bootstrap: BootstrapResult | None = None
    signal_decay: SignalDecayResult | None = None
    regime_tracking: RegimeTrackingResult | None = None
    clustering: ClusteringResult | None = None

    overall_grade: str = ""  # A/B/C/D/F
    is_robust: bool = False
    critical_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def run_full_diagnostics(
    trades: list[dict[str, Any]],
) -> DiagnosticReport:
    """Run all diagnostic layers and produce a unified report.

    Args:
        trades: List of trade dicts from backtest results.

    Returns:
        DiagnosticReport with all analysis layers populated.
    """
    # Run each layer
    wf = run_walk_forward(trades)
    bs = run_bootstrap(trades)
    sd = run_signal_decay_analysis(trades)
    rt = analyze_regime_performance(trades)
    cl = run_clustering_analysis(trades)

    # Score and grade
    issues: list[str] = []
    recs: list[str] = []
    score = 100  # Start at A, deduct for issues

    # Walk-forward checks
    if wf.is_overfit:
        issues.append("Model is overfit (walk-forward efficiency < 50%)")
        score -= 30
        recs.append("Reduce model complexity or add regularization")

    # Bootstrap checks
    if bs and not bs.is_statistically_significant:
        issues.append("Edge is NOT statistically significant")
        score -= 25
        recs.append("Need more trades or stronger signal")
    if bs and bs.win_rate and bs.win_rate.ci_width > 0.25:
        issues.append(f"Win rate CI very wide ({bs.win_rate.ci_width:.0%})")
        score -= 10

    # Signal decay checks
    if sd and not sd.has_predictive_power:
        issues.append("Scoring model has NO predictive power")
        score -= 25
        recs.append("Redesign conviction scoring — current model is noise")
    if sd and sd.monotonicity_violations > 1:
        issues.append(
            f"Score is non-monotonic ({sd.monotonicity_violations} violations)"
        )
        score -= 10

    # Regime checks
    if rt and rt.worst_regime:
        worst_perf = rt.regime_performance.get(rt.worst_regime)
        if worst_perf and not worst_perf.is_profitable:
            issues.append(f"Unprofitable in {rt.worst_regime} regime")
            score -= 10
            recs.append(f"Block or raise thresholds in {rt.worst_regime}")

    # Clustering checks
    if cl and cl.max_consecutive_losses > 5:
        issues.append(
            f"Severe loss clustering ({cl.max_consecutive_losses} consecutive)"
        )
        score -= 15
        recs.append("Add drawdown circuit breaker or diversification rules")
    if cl and cl.ticker_concentration and cl.ticker_concentration.is_over_concentrated:
        issues.append("Over-concentrated in specific tickers")
        score -= 10
        recs.append("Limit max trades per ticker")

    # Grade assignment
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 55:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    return DiagnosticReport(
        walk_forward=wf,
        bootstrap=bs,
        signal_decay=sd,
        regime_tracking=rt,
        clustering=cl,
        overall_grade=grade,
        is_robust=score >= 70,
        critical_issues=issues,
        recommendations=recs,
    )


def print_diagnostic_report(report: DiagnosticReport) -> None:
    """Print a formatted diagnostic report."""
    print("\n" + "=" * 65)
    print("DIAGNOSTIC ROBUSTNESS REPORT")
    print("=" * 65)
    print(f"\nOverall Grade: {report.overall_grade}")
    print(f"Is Robust:     {'Yes' if report.is_robust else 'NO'}")

    if report.critical_issues:
        print("\n--- CRITICAL ISSUES ---")
        for issue in report.critical_issues:
            print(f"  ! {issue}")

    if report.recommendations:
        print("\n--- RECOMMENDATIONS ---")
        for rec in report.recommendations:
            print(f"  > {rec}")

    # Walk-forward
    if report.walk_forward and report.walk_forward.total_windows > 0:
        wf = report.walk_forward
        print("\n--- WALK-FORWARD VALIDATION ---")
        print(f"  Windows:              {wf.total_windows}")
        print(f"  IS Win Rate:          {wf.avg_is_win_rate:.1%}")
        print(f"  OOS Win Rate:         {wf.avg_oos_win_rate:.1%}")
        print(f"  Degradation:          {wf.degradation_pct:.1f}%")
        print(f"  Walk-Forward Eff:     {wf.walk_forward_efficiency:.1%}")
        print(f"  Overfit Probability:  {wf.overfitting_probability:.1%}")
        for note in wf.notes:
            print(f"  > {note}")

    # Bootstrap
    if report.bootstrap and report.bootstrap.win_rate:
        bs = report.bootstrap
        wr_ci = bs.win_rate
        assert wr_ci is not None  # Already checked above
        print("\n--- BOOTSTRAP CONFIDENCE INTERVALS (10,000 samples) ---")
        print(f"  Win Rate:      {wr_ci.point_estimate:.1%} "
              f"[{wr_ci.ci_lower_5:.1%} - {wr_ci.ci_upper_95:.1%}]")
        if bs.profit_factor:
            pf_ci = bs.profit_factor
            print(f"  Profit Factor: {pf_ci.point_estimate:.2f} "
                  f"[{pf_ci.ci_lower_5:.2f} - {pf_ci.ci_upper_95:.2f}]")
        if bs.sharpe_ratio:
            sh_ci = bs.sharpe_ratio
            print(f"  Sharpe:        {sh_ci.point_estimate:.3f} "
                  f"[{sh_ci.ci_lower_5:.3f} - {sh_ci.ci_upper_95:.3f}]")
        print(f"  Significant:   {'Yes' if bs.is_statistically_significant else 'NO'}")
        for note in bs.notes:
            print(f"  > {note}")

    # Signal decay
    if report.signal_decay:
        sd = report.signal_decay
        print("\n--- SIGNAL QUALITY ---")
        print(f"  Score-PnL Correlation:  {sd.score_pnl_correlation:.3f}")
        print(f"  Score-Win Correlation:  {sd.score_win_correlation:.3f}")
        print(f"  Has Predictive Power:   {'Yes' if sd.has_predictive_power else 'NO'}")
        print(f"  Monotonic:              {'Yes' if sd.is_monotonically_increasing else 'No'}")
        if sd.optimal_hold_days > 0:
            print(f"  Optimal Hold Days:      {sd.optimal_hold_days}")
        if sd.buckets:
            print("  By Conviction Bucket:")
            for b in sd.buckets:
                print(f"    [{b.bucket_name:5s}]: {b.trades:3d}T  "
                      f"WR={b.win_rate:.1%}  avg={b.avg_pnl:+.1f}%")
        for note in sd.notes:
            print(f"  > {note}")

    # Regime
    if report.regime_tracking and report.regime_tracking.regime_performance:
        rt = report.regime_tracking
        print("\n--- REGIME-CONDITIONED PERFORMANCE ---")
        for regime, perf in sorted(
            rt.regime_performance.items(),
            key=lambda x: x[1].win_rate,
            reverse=True,
        ):
            profit_marker = "+" if perf.is_profitable else "-"
            print(f"  {regime:22s}: {perf.trades:3d}T  "
                  f"WR={perf.win_rate:.1%}  "
                  f"PnL={perf.total_pnl:+.0f}% [{profit_marker}]")
        if rt.regime_specific_thresholds:
            print("  Suggested Thresholds:")
            for regime, threshold in rt.regime_specific_thresholds.items():
                print(f"    {regime}: min_conviction >= {threshold:.1f}")
        for note in rt.notes:
            print(f"  > {note}")

    # Clustering
    if report.clustering:
        cl = report.clustering
        print("\n--- TRADE CLUSTERING ---")
        print(f"  Loss Clusters:         {len(cl.loss_clusters)}")
        print(f"  Max Consecutive Losses: {cl.max_consecutive_losses}")
        if cl.ticker_concentration:
            tc = cl.ticker_concentration
            print(f"  Ticker Concentration:  HHI={tc.herfindahl_index:.2f} "
                  f"(max: {tc.max_item} at {tc.max_concentration:.0%})")
        if cl.direction_concentration:
            dc = cl.direction_concentration
            print(f"  Direction Concentration: HHI={dc.herfindahl_index:.2f} "
                  f"(max: {dc.max_item} at {dc.max_concentration:.0%})")
        if cl.best_month:
            print(f"  Best Month:  {cl.best_month} "
                  f"(WR={cl.monthly_win_rates.get(cl.best_month, 0):.1%})")
        if cl.worst_month:
            print(f"  Worst Month: {cl.worst_month} "
                  f"(WR={cl.monthly_win_rates.get(cl.worst_month, 0):.1%})")
        for note in cl.notes:
            print(f"  > {note}")
