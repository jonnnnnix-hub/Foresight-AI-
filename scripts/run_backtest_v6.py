#!/usr/bin/env python3
"""Run PHANTOM v6 backtest — with Monte Carlo conviction layer.

v6 changes from v5:
- Monte Carlo (50k GBM) simulation added as 4th conviction modifier
- MC P(profit) filters low-probability entries
- Combined adjustment cap stays at -2.0

Usage:
    .venv/bin/python scripts/run_backtest_v6.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.backtest.engine import run_backtest
from flowedge.scanner.backtest.schemas import BacktestResult

TICKERS = [
    "TSLA", "NVDA", "AAPL", "META", "AMZN",
    "SPY", "QQQ", "AMD", "GOOGL", "MSFT",
]


def print_report(result: BacktestResult) -> None:
    """Print formatted v6 report."""
    trades = result.trades

    print(f"\n{'=' * 65}")
    print("PHANTOM v6 — MONTE CARLO ENHANCED 2-YEAR BACKTEST")
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

    # By regime
    if result.by_regime:
        print("\n--- BY REGIME ---")
        for name, stats in sorted(
            result.by_regime.items(),
            key=lambda x: x[1].get("trades", 0),
            reverse=True,
        ):
            t = int(stats.get("trades", 0))
            wr = stats.get("win_rate", 0)
            avg = stats.get("avg_pnl_pct", 0)
            print(f"  {name:22s}: {t:3d}T  WR={wr:5.1%}  avg={avg:+7.1f}%")

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

    # S&P comparison
    print("\n--- S&P 500 COMPARISON ---")
    sp_return = 22.1  # Approximate 2-year SPY return
    alpha = result.portfolio_return_pct - sp_return
    print(f"  S&P avg (2.0yr): {sp_return:+.1f}%")
    print(f"  PHANTOM v6:       {result.portfolio_return_pct:+.1f}%")
    print(f"  Alpha:            {alpha:+.1f}%")

    # Version comparison
    print("\n--- VERSION COMPARISON ---")
    print("  v3 (tuned):       +57.2% | 165T | 29.1% WR | 1.15 PF | 0.683 Sharpe")
    print("  v4 (over-filter): +12.4% |  55T | 27.3% WR | 1.20 PF | 0.492 Sharpe")
    print("  v5 (balanced):    +44.2% |  77T | 31.2% WR | 1.31 PF | 0.600 Sharpe")
    print(f"  v6 (monte carlo): {result.portfolio_return_pct:+.1f}% | "
          f"{result.total_trades:3d}T | {result.win_rate:.1%} WR | "
          f"{result.profit_factor:.2f} PF | {result.sharpe_ratio:.3f} Sharpe")


async def main() -> None:
    print("PHANTOM v6: Monte Carlo enhanced multi-factor")
    print("Changes: +50k MC GBM simulations per trade entry")
    print(f"Estimated: ~2 min/ticker = ~{len(TICKERS) * 2} minutes")

    result = await run_backtest(
        tickers=TICKERS,
        lookback_days=730,
        starting_capital=10_000.0,
        max_positions=5,
        max_risk_per_trade=0.08,
        max_hold_days=9,
        trailing_stop_pct=0.35,
        hard_stop_pct=-0.35,
        take_profit_pct=2.50,
        dte=15,
        min_conviction=7.0,
    )

    print_report(result)

    # Save
    out_dir = Path("data/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest_v6_2y.json"
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, default=str)
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
