#!/usr/bin/env python3
"""Run PHANTOM v9 — SPY/QQQ near-ATM with slippage.

The first backtest that accounts for real-world bid-ask spreads.

Usage:
    .venv/bin/python scripts/run_backtest_v9.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.backtest.diagnostics.runner import (
    print_diagnostic_report,
    run_full_diagnostics,
)
from flowedge.scanner.backtest.v9_index_atm import run_v9_backtest


async def main() -> None:
    print("PHANTOM v9: SPY/QQQ only, 0.5% OTM, slippage-aware")
    print("First backtest with realistic bid-ask spread costs")
    print("Target: NET positive return after spreads\n")

    result = await run_v9_backtest(
        tickers=["SPY"],  # v9.2: SPY only — QQQ was -82% in v9.1
        lookback_days=730,
        starting_capital=10_000.0,
        max_positions=2,
        min_conviction=8.0,
        dte=21,
    )

    trades = result.trades

    print(f"\n{'=' * 65}")
    print("PHANTOM v9 — SPY/QQQ NEAR-ATM (SLIPPAGE INCLUDED)")
    print(f"{'=' * 65}")
    print(f"Run ID: {result.run_id}")
    print(f"Tickers: {result.tickers}")

    print("\n--- PORTFOLIO (NET OF SLIPPAGE) ---")
    print(f"Starting capital:   ${result.starting_capital:,.2f}")
    print(f"Ending value:       ${result.ending_value:,.2f}")
    print(f"Portfolio return:   {result.portfolio_return_pct:+.1f}%")
    print(f"Max drawdown:       {result.max_drawdown_pct:.1f}%")
    print(f"Sharpe ratio:       {result.sharpe_ratio:.3f}")

    print("\n--- TRADES ---")
    print(f"Total: {result.total_trades} | Wins: {result.wins} "
          f"| Losses: {result.losses}")
    print(f"Win rate:           {result.win_rate:.1%}")
    print(f"Avg win:            {result.avg_win_pct:+.1f}%")
    print(f"Avg loss:           {result.avg_loss_pct:+.1f}%")
    print(f"Profit factor:      {result.profit_factor:.2f}")
    print(f"Expectancy:         {result.expectancy_pct:+.1f}%")
    print(f"Avg hold:           {result.avg_hold_days:.1f} days")

    if result.by_strategy:
        print("\n--- BY STRATEGY ---")
        for name, stats in sorted(
            result.by_strategy.items(),
            key=lambda x: x[1].get("total_pnl_pct", 0),
            reverse=True,
        ):
            t = int(stats.get("trades", 0))
            wr = stats.get("win_rate", 0)
            avg = stats.get("avg_pnl_pct", 0)
            total = stats.get("total_pnl_pct", 0)
            print(f"  {name:20s}: {t:3d}T  WR={wr:5.1%}  "
                  f"avg={avg:+7.1f}%  total={total:+8.1f}%")

    if result.by_ticker:
        print("\n--- BY TICKER ---")
        for name, stats in sorted(result.by_ticker.items()):
            t = int(stats.get("trades", 0))
            wr = stats.get("win_rate", 0)
            avg = stats.get("avg_pnl_pct", 0)
            total = stats.get("total_pnl_pct", 0)
            print(f"  {name:5s}: {t:3d}T  WR={wr:5.1%}  "
                  f"avg={avg:+7.1f}%  total={total:+8.1f}%")

    if trades:
        reasons: dict[str, int] = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print("\n--- EXIT REASONS ---")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / len(trades) * 100
            print(f"  {reason:20s}: {count:3d} ({pct:.0f}%)")

    print("\n--- VERSION COMPARISON (all gross except v9) ---")
    print("  v7 (gross):  +129.4% |  80T | 33.8% WR | 0.830 Sharpe")
    print("  v7 (net):     ~-1.3% |  80T | (after slippage)")
    print(f"  v9 (net):    {result.portfolio_return_pct:+.1f}% | "
          f"{result.total_trades:3d}T | {result.win_rate:.1%} WR | "
          f"{result.sharpe_ratio:.3f} Sharpe (slippage included)")

    # Save
    out_dir = Path("data/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest_v9_2y.json"
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, default=str)
    )
    print(f"\nSaved to {out_path}")

    # Diagnostics
    trade_dicts = [t.model_dump(mode="json") for t in trades]
    if len(trade_dicts) >= 10:
        print("\n\nRunning diagnostic analysis...")
        diag = run_full_diagnostics(trade_dicts)
        print_diagnostic_report(diag)


if __name__ == "__main__":
    asyncio.run(main())
