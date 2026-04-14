#!/usr/bin/env python3
"""FlowEdge Active — Hybrid model backtest.

Usage:
    .venv/bin/python scripts/run_hybrid_backtest.py
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
from flowedge.scanner.backtest.hybrid_active import run_hybrid_backtest


async def main() -> None:
    print("FLOWEDGE ACTIVE — Hybrid Model Backtest")
    print("8 tickers | IBS + trend + squeeze | Calls + Puts | Slippage included")
    print("Target: 15-30 trades/year, 55-65% WR, $25K account\n")

    result = await run_hybrid_backtest(starting_capital=25_000.0)
    trades = result.trades

    print(f"\n{'=' * 65}")
    print("FLOWEDGE ACTIVE — HYBRID RESULTS (NET OF SLIPPAGE)")
    print(f"{'=' * 65}")

    print("\n--- PORTFOLIO ---")
    print(f"Starting:   ${result.starting_capital:,.2f}")
    print(f"Ending:     ${result.ending_value:,.2f}")
    print(f"Return:     {result.portfolio_return_pct:+.1f}%")
    print(f"Max DD:     {result.max_drawdown_pct:.1f}%")
    print(f"Sharpe:     {result.sharpe_ratio:.3f}")

    print("\n--- TRADES ---")
    print(f"Total:      {result.total_trades} ({result.total_trades/2:.0f}/year)")
    print(f"Wins:       {result.wins}  Losses: {result.losses}")
    wr_marker = "✓" if result.win_rate >= 0.55 else "✗"
    print(f"Win Rate:   {result.win_rate:.1%} {wr_marker} (target: 55-65%)")
    print(f"Avg Win:    {result.avg_win_pct:+.1f}%")
    print(f"Avg Loss:   {result.avg_loss_pct:+.1f}%")
    print(f"PF:         {result.profit_factor:.2f}")
    print(f"Expectancy: {result.expectancy_pct:+.1f}%")
    print(f"Avg Hold:   {result.avg_hold_days:.1f} days")

    if result.by_strategy:
        print("\n--- BY STRATEGY ---")
        for name, stats in sorted(
            result.by_strategy.items(),
            key=lambda x: x[1].get("win_rate", 0),
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
        for name, stats in sorted(
            result.by_ticker.items(),
            key=lambda x: x[1].get("total_pnl_pct", 0),
            reverse=True,
        ):
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
            rt = [t for t in trades if t.exit_reason == reason]
            rw = sum(1 for t in rt if t.pnl_pct > 5)
            rwr = rw / len(rt) * 100 if rt else 0
            print(f"  {reason:20s}: {count:2d} ({count/len(trades)*100:.0f}%)  "
                  f"WR={rwr:.0f}%")

    # Trade log
    print("\n--- TRADE LOG ---")
    for t in trades:
        m = "W" if t.pnl_pct > 5 else ("L" if t.pnl_pct < -5 else "B")
        print(f"  [{m}] {t.ticker:5s} {t.entry_date} → {t.exit_date}  "
              f"hold={t.hold_days}d  {t.strategy:20s}  "
              f"pnl={t.pnl_pct:+6.1f}%  exit={t.exit_reason}")

    print("\n--- MODEL COMPARISON ---")
    print("  v10.2 (precision): +27.1% |   5T | 80.0% WR | 0.708 Sharpe | SPY only")
    print(f"  Hybrid (active):   {result.portfolio_return_pct:+.1f}% | "
          f"{result.total_trades:3d}T | {result.win_rate:.1%} WR | "
          f"{result.sharpe_ratio:.3f} Sharpe | 8 tickers")

    out = Path("data/backtest/backtest_hybrid_2y.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    print(f"\nSaved to {out}")

    if len(trades) >= 5:
        trade_dicts = [t.model_dump(mode="json") for t in trades]
        print("\n")
        diag = run_full_diagnostics(trade_dicts)
        print_diagnostic_report(diag)


if __name__ == "__main__":
    asyncio.run(main())
