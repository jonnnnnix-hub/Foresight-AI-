#!/usr/bin/env python3
"""Run PHANTOM performance simulation — CI-friendly entry point.

Usage:
    python scripts/run_phantom.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from flowedge.scanner.performance.simulator import run_historical_simulation

    report = asyncio.run(run_historical_simulation())

    # Ensure output dir exists
    out = Path("data/performance/performance.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHANTOM PERFORMANCE SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Period:     {report.start_date} -> {report.end_date}")
    print(f"  Starting:   ${report.starting_value:,.2f}")
    print(f"  Ending:     ${report.ending_value:,.2f}")
    print(f"  Return:     {report.total_return_pct:+.1f}%")
    print(f"  Trades:     {report.total_trades} ({report.wins}W / {report.losses}L)")
    print(f"  Win Rate:   {report.win_rate:.0%}")
    print(f"  Profit Factor: {report.profit_factor:.2f}")
    print("=" * 70)

    # Write GitHub Actions summary if running in CI
    summary_path = Path(
        __import__("os").environ.get("GITHUB_STEP_SUMMARY", "/dev/null"),
    )
    if summary_path.name != "null":
        summary_path.write_text(
            f"## PHANTOM Nightly Report\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Period | {report.start_date} - {report.end_date} |\n"
            f"| Starting Capital | ${report.starting_value:,.2f} |\n"
            f"| Ending Value | ${report.ending_value:,.2f} |\n"
            f"| Total Return | {report.total_return_pct:+.1f}% |\n"
            f"| Trades | {report.total_trades} ({report.wins}W / {report.losses}L) |\n"
            f"| Win Rate | {report.win_rate:.0%} |\n"
            f"| Profit Factor | {report.profit_factor:.2f} |\n",
        )


if __name__ == "__main__":
    main()
