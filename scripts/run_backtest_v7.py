#!/usr/bin/env python3
"""Run PHANTOM v7 backtest — data-driven from v5 diagnostics.

v7 changes:
- Wider stops: hard_stop -50% (was -40%), take_profit 3.5x (was 3.0x)
- Longer holds: max_hold 12d (was 9d) — winners avg 7.1d hold
- Drawdown circuit breaker: pause after 4 consecutive losses
- Ticker performance gating: AMZN -2.0, NVDA -1.0, TSLA -0.5
- IBS reversion bonus: +1.0 conviction (50% WR vs 26% trend)
- MC removed from entry (added noise in v6)
- GEX weight reduced to 50%

Usage:
    .venv/bin/python scripts/run_backtest_v7.py
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
from flowedge.scanner.backtest.engine import run_backtest

TICKERS = [
    "TSLA", "NVDA", "AAPL", "META", "AMZN",
    "SPY", "QQQ", "AMD", "GOOGL", "MSFT",
]


async def main() -> None:
    print("PHANTOM v7: Data-driven from diagnostic analysis")
    print("Changes: wider stops, longer holds, circuit breaker,")
    print("         ticker penalties, IBS bonus, MC removed")
    print(f"Estimated: ~2 min/ticker = ~{len(TICKERS) * 2} minutes\n")

    result = await run_backtest(
        tickers=TICKERS,
        lookback_days=730,
        starting_capital=10_000.0,
        max_positions=5,
        max_risk_per_trade=0.08,
        max_hold_days=12,       # v7: extended from 9
        trailing_stop_pct=0.35,
        hard_stop_pct=-0.50,    # v7: wider from -0.35
        take_profit_pct=3.50,   # v7: wider from 2.50
        dte=15,
        min_conviction=7.0,
    )

    trades = result.trades

    print(f"\n{'=' * 65}")
    print("PHANTOM v7 — DATA-DRIVEN 2-YEAR BACKTEST")
    print(f"{'=' * 65}")
    print(f"Run ID: {result.run_id}")
    print(f"Period: {result.lookback_days} days | Tickers: {len(TICKERS)}")

    print("\n--- PORTFOLIO ---")
    print(f"Starting capital:   ${result.starting_capital:,.2f}")
    print(f"Ending value:       ${result.ending_value:,.2f}")
    print(f"Portfolio return:   {result.portfolio_return_pct:+.1f}%")
    print(f"Max drawdown:       {result.max_drawdown_pct:.1f}%")
    print(f"Sharpe ratio:       {result.sharpe_ratio:.3f}")

    print("\n--- TRADES ---")
    print(f"Total: {result.total_trades} | Wins: {result.wins} "
          f"| Losses: {result.losses} | Expired: {result.expired_worthless}")
    print(f"Win rate:           {result.win_rate:.1%}")
    print(f"Avg win:            {result.avg_win_pct:+.1f}%")
    print(f"Avg loss:           {result.avg_loss_pct:+.1f}%")
    print(f"Best:               {result.best_trade_pct:+.1f}%")
    print(f"Worst:              {result.worst_trade_pct:+.1f}%")
    print(f"Profit factor:      {result.profit_factor:.2f}")
    print(f"Expectancy:         {result.expectancy_pct:+.1f}%")
    print(f"Avg hold:           {result.avg_hold_days:.1f} days")
    print(f"Max consec wins:    {result.max_consecutive_wins}")
    print(f"Max consec losses:  {result.max_consecutive_losses}")

    # By strategy
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

    # By ticker
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

    # Exit reasons
    if trades:
        reasons: dict[str, int] = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print("\n--- EXIT REASONS ---")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / len(trades) * 100
            print(f"  {reason:20s}: {count:3d} ({pct:.0f}%)")

    # Version comparison
    print("\n--- VERSION COMPARISON ---")
    print("  v3 (tuned):       +57.2% | 165T | 29.1% WR | 1.15 PF | 0.683 Sharpe")
    print("  v4 (over-filter): +12.4% |  55T | 27.3% WR | 1.20 PF | 0.492 Sharpe")
    print("  v5 (balanced):    +44.2% |  77T | 31.2% WR | 1.31 PF | 0.600 Sharpe")
    print("  v6 (monte carlo): -50.5% |  65T | 29.2% WR | 1.06 PF | 0.065 Sharpe")
    print(f"  v7 (data-driven): {result.portfolio_return_pct:+.1f}% | "
          f"{result.total_trades:3d}T | {result.win_rate:.1%} WR | "
          f"{result.profit_factor:.2f} PF | {result.sharpe_ratio:.3f} Sharpe")

    # Save
    out_dir = Path("data/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest_v7_2y.json"
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, default=str)
    )
    print(f"\nSaved to {out_path}")

    # Run diagnostics
    trade_dicts = [t.model_dump(mode="json") for t in trades]
    print("\n\nRunning diagnostic analysis on v7 results...")
    diag = run_full_diagnostics(trade_dicts)
    print_diagnostic_report(diag)


if __name__ == "__main__":
    asyncio.run(main())
