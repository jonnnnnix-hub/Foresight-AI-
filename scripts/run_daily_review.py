#!/usr/bin/env python3
"""Run the FlowEdge Council daily post-market review.

Usage:
    # Run with fresh backtest (default config):
    python scripts/run_daily_review.py

    # Run with a specific config:
    python scripts/run_daily_review.py --config configs/sweep_best_90wr.json

    # Run against the latest saved backtest (no new backtest):
    python scripts/run_daily_review.py --no-backtest

    # Specify review date:
    python scripts/run_daily_review.py --date 2026-04-11

    # Launch dashboard after review:
    python scripts/run_daily_review.py --dashboard
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlowEdge Council — Post-market specialist review",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to ScalpConfig JSON file",
    )
    parser.add_argument(
        "--result", "-r",
        type=str,
        default=None,
        help="Path to a specific BacktestResult JSON file",
    )
    parser.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip backtest, use latest saved result",
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        default=None,
        help="Review date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the dashboard after the review",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Dashboard port (default: 8050)",
    )
    args = parser.parse_args()

    review_date = date.fromisoformat(args.date) if args.date else date.today()
    run_backtest = not args.no_backtest

    from flowedge.council.daily_review import run_daily_review

    print("=" * 70)
    print("FLOWEDGE COUNCIL — Post-Market Review")
    print(f"Date: {review_date}  |  Backtest: {'yes' if run_backtest else 'no (using saved)'}")
    if args.config:
        print(f"Config: {args.config}")
    print("=" * 70)
    print()

    review = run_daily_review(
        config_path=args.config,
        result_path=args.result,
        run_backtest=run_backtest,
        review_date=review_date,
    )

    # ── Print summary ────────────────────────────────────────────
    status_colors = {
        "healthy": "\033[92m",
        "watch": "\033[93m",
        "degraded": "\033[33m",
        "critical": "\033[91m",
    }
    c = status_colors.get(review.status.value, "")
    reset = "\033[0m"

    print(f"\n{c}{'=' * 70}")
    print(f"  STATUS: {review.status.value.upper()}")
    print(f"  HEALTH: {review.overall_health:.0f}/100")
    print(f"{'=' * 70}{reset}\n")

    print(f"  Trades: {review.cumulative_trades}  |  "
          f"Win Rate: {review.cumulative_wr:.1%}  |  "
          f"P&L: ${review.cumulative_pnl:+,.0f}")
    print()

    # Specialist scores
    print("  SPECIALIST SCORES:")
    for sr in review.specialist_reviews:
        sc = status_colors.get(sr.severity.value, "")
        print(f"    {sc}{sr.specialist_name:<28} {sr.health_score:>5.0f}/100  "
              f"[{sr.severity.value}]  "
              f"({len(sr.findings)} findings, {len(sr.recommendations)} recs){reset}")
    print()

    # Consensus
    print(f"  CONSENSUS: {review.consensus_summary}")
    if review.dissenting_views:
        print()
        print("  DISSENTING VIEWS:")
        for d in review.dissenting_views:
            print(f"    - {d}")
    print()

    # Top recommendations
    if review.top_recommendations:
        print("  TOP RECOMMENDATIONS:")
        for i, rec in enumerate(review.top_recommendations[:5], 1):
            priority_color = "\033[91m" if rec.priority <= 2 else (
                "\033[93m" if rec.priority <= 3 else "\033[94m"
            )
            print(f"    {priority_color}{i}. [P{rec.priority}] {rec.title}{reset}")
            print(f"       {rec.rationale[:100]}")
            if rec.current_value and rec.suggested_value:
                print(f"       Change: {rec.current_value} -> {rec.suggested_value}")
            print()

    # Ticker scorecards
    if review.ticker_scorecards:
        print("  TICKER SCORECARDS:")
        print(f"    {'Ticker':<8} {'Trades':>6} {'WR':>6} {'P&L':>8} {'Status':>8}")
        print(f"    {'-'*38}")
        for tc in review.ticker_scorecards:
            wr_c = "\033[92m" if tc.win_rate >= 0.7 else (
                "\033[93m" if tc.win_rate >= 0.5 else "\033[91m"
            )
            print(f"    {tc.ticker:<8} {tc.trades:>6} "
                  f"{wr_c}{tc.win_rate:>5.0%}{reset} "
                  f"{'$' + f'{tc.total_pnl:>+.0f}':>8} "
                  f"{tc.recommendation:>8}")
        print()

    print(f"  Review saved: {review.review_id}")
    print(f"  Computation: {review.computation_time_ms:.0f}ms")
    print()

    # ── Optionally launch dashboard ──────────────────────────────
    if args.dashboard:
        print(f"Launching dashboard at http://localhost:{args.port}")
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "flowedge.dashboard.app:app",
                "--reload",
                "--port", str(args.port),
            ],
            cwd=str(ROOT),
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        )


if __name__ == "__main__":
    main()
