#!/usr/bin/env python3
"""PHANTOM v10 — Precision engine targeting 60%+ WR.

SPY only | IBS reversion only | No hard stops | Slippage included

Usage:
    .venv/bin/python scripts/run_backtest_v10.py
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
from flowedge.scanner.backtest.v10_precision import run_v10_backtest


async def main() -> None:
    print("PHANTOM v10: PRECISION ENGINE")
    print("SPY only | IBS reversion only | No hard stops | Slippage included")
    print("Target: 60%+ win rate, net positive after spreads\n")

    result = await run_v10_backtest(lookback_days=730)
    trades = result.trades

    print(f"\n{'=' * 65}")
    print("PHANTOM v10 — PRECISION RESULTS (NET OF SLIPPAGE)")
    print(f"{'=' * 65}")

    print("\n--- PORTFOLIO ---")
    print(f"Starting:   ${result.starting_capital:,.2f}")
    print(f"Ending:     ${result.ending_value:,.2f}")
    print(f"Return:     {result.portfolio_return_pct:+.1f}%")
    print(f"Max DD:     {result.max_drawdown_pct:.1f}%")
    print(f"Sharpe:     {result.sharpe_ratio:.3f}")

    print("\n--- TRADES ---")
    print(f"Total:      {result.total_trades}")
    print(f"Wins:       {result.wins}")
    print(f"Losses:     {result.losses}")
    wr_marker = "✓" if result.win_rate >= 0.60 else "✗"
    print(f"Win Rate:   {result.win_rate:.1%} {wr_marker} (target: 60%+)")
    print(f"Avg Win:    {result.avg_win_pct:+.1f}%")
    print(f"Avg Loss:   {result.avg_loss_pct:+.1f}%")
    print(f"PF:         {result.profit_factor:.2f}")
    print(f"Expectancy: {result.expectancy_pct:+.1f}%")
    print(f"Avg Hold:   {result.avg_hold_days:.1f} days")

    if trades:
        reasons: dict[str, int] = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print("\n--- EXIT REASONS ---")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            # Show WR per exit reason
            reason_trades = [t for t in trades if t.exit_reason == reason]
            reason_wins = sum(1 for t in reason_trades if t.pnl_pct > 5)
            rwr = reason_wins / len(reason_trades) * 100 if reason_trades else 0
            print(f"  {reason:20s}: {count:2d} ({count/len(trades)*100:.0f}%)  "
                  f"WR={rwr:.0f}%")

    print("\n--- TRADE LOG ---")
    for t in trades:
        marker = "W" if t.pnl_pct > 5 else ("L" if t.pnl_pct < -5 else "B")
        print(f"  [{marker}] {t.entry_date} → {t.exit_date}  "
              f"hold={t.hold_days}d  pnl={t.pnl_pct:+.1f}%  "
              f"exit={t.exit_reason}  conv={t.conviction:.1f}")

    print("\n--- VERSION EVOLUTION ---")
    print("  v7  (gross): +129.4% |  80T | 33.8% WR | 0.830 Sharpe | 10 tickers OTM")
    print("  v9.2 (net):   +10.6% |  21T | 38.1% WR | 0.189 Sharpe | SPY near-ATM")
    print(f"  v10  (net):  {result.portfolio_return_pct:+.1f}% | "
          f"{result.total_trades:3d}T | {result.win_rate:.1%} WR | "
          f"{result.sharpe_ratio:.3f} Sharpe | SPY IBS precision")

    out = Path("data/backtest/backtest_v10_2y.json")
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
