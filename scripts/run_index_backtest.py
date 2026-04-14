#!/usr/bin/env python3
"""Run the Index ETF Specialist backtest with self-learning loop.

Usage:
    .venv/bin/python scripts/run_index_backtest.py [--iterations N]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.backtest.index_specialist.engine import (
    IndexBacktestResult,
    run_index_backtest,
    run_index_learning_cycle,
    save_index_results,
)
from flowedge.scanner.backtest.index_specialist.schemas import (
    IndexBacktestConfig,
    TradeHorizon,
)


def print_results(result: IndexBacktestResult, iteration: int) -> None:
    """Print formatted backtest results."""
    print(f"\n{'=' * 65}")
    print(f"INDEX SPECIALIST — ITERATION {iteration}")
    print(f"{'=' * 65}")
    print(f"Run ID: {result.run_id}")
    print(f"Tickers: {result.tickers}")
    print(f"Period: {result.lookback_days} days")

    print("\n--- PORTFOLIO ---")
    print(f"Starting capital:   ${result.starting_capital:,.2f}")
    print(f"Ending value:       ${result.ending_value:,.2f}")
    print(f"Portfolio return:   {result.portfolio_return_pct:+.1f}%")
    print(f"Max drawdown:       {result.max_drawdown_pct:.1f}%")
    print(f"Sharpe ratio:       {result.sharpe_ratio:.3f}")

    print("\n--- TRADES ---")
    print(f"Total: {result.total_trades} | Wins: {result.wins} | Losses: {result.losses}")
    print(f"Win rate:           {result.win_rate:.1%}")
    print(f"Avg win:            {result.avg_win_pct:+.1f}%")
    print(f"Avg loss:           {result.avg_loss_pct:+.1f}%")
    print(f"Profit factor:      {result.profit_factor:.2f}")
    print(f"Expectancy:         {result.expectancy_pct:+.1f}%")

    print("\n--- BY HORIZON ---")
    print(f"  Scalp   : {result.scalp_trades:3d}T  "
          f"WR={result.scalp_win_rate:5.1%}  "
          f"avg={result.scalp_avg_pnl:+7.1f}%")
    print(f"  Swing   : {result.swing_trades:3d}T  "
          f"WR={result.swing_win_rate:5.1%}  "
          f"avg={result.swing_avg_pnl:+7.1f}%")
    print(f"  Medium  : {result.medium_trades:3d}T  "
          f"WR={result.medium_win_rate:5.1%}  "
          f"avg={result.medium_avg_pnl:+7.1f}%")

    print("\n--- BY TICKER ---")
    for ticker, stats in sorted(result.by_ticker.items()):
        trades = int(stats.get("trades", 0))
        wr = stats.get("win_rate", 0)
        avg = stats.get("avg_pnl_pct", 0)
        total = stats.get("total_pnl_pct", 0)
        print(f"  {ticker:5s}: {trades:3d}T  "
              f"WR={wr:5.1%}  avg={avg:+7.1f}%  total={total:+8.1f}%")

    # Exit reason distribution
    if result.trades:
        reasons: dict[str, int] = {}
        for t in result.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print("\n--- EXIT REASONS ---")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / result.total_trades * 100
            print(f"  {reason:20s}: {count:3d} ({pct:.0f}%)")

    # Win rate targets
    print("\n--- TARGET CHECK ---")
    scalp_target = "✓" if result.scalp_win_rate >= 0.70 else "✗"
    swing_target = "✓" if result.swing_win_rate >= 0.80 else "✗"
    medium_target = "✓" if result.medium_win_rate >= 0.90 else "✗"
    print(f"  Scalp  70%+ WR: {scalp_target} ({result.scalp_win_rate:.1%})")
    print(f"  Swing  80%+ WR: {swing_target} ({result.swing_win_rate:.1%})")
    print(f"  Medium 90%+ WR: {medium_target} ({result.medium_win_rate:.1%})")

    # Check scalp 50%+ gain target
    if result.trades:
        scalp_wins = [
            t for t in result.trades
            if t.horizon == TradeHorizon.SCALP and t.is_win
        ]
        if scalp_wins:
            avg_scalp_win = sum(t.pnl_pct for t in scalp_wins) / len(scalp_wins)
            gain_target = "✓" if avg_scalp_win >= 50.0 else "✗"
            print(f"  Scalp 50%+ gain: {gain_target} ({avg_scalp_win:+.1f}%)")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Index ETF Specialist Backtest")
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of self-learning iterations (default: 3)",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["SPY", "QQQ", "IWM"],
        help="Tickers to trade",
    )
    args = parser.parse_args()

    config = IndexBacktestConfig(tickers=args.tickers)

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'#' * 65}")
        print(f"# ITERATION {iteration}/{args.iterations}")
        print(f"# Config: scalp_min={config.scalp_min_conviction:.1f} "
              f"swing_min={config.swing_min_conviction:.1f} "
              f"medium_min={config.medium_min_conviction:.1f}")
        print(f"# MC prob: scalp={config.mc_min_prob_profit_scalp:.2f} "
              f"swing={config.mc_min_prob_profit_swing:.2f}")
        print(f"{'#' * 65}")

        result = await run_index_backtest(config=config)
        print_results(result, iteration)

        # Save results
        filepath = save_index_results(result, config)
        print(f"\nSaved to {filepath}")

        # Run learning cycle for next iteration
        if iteration < args.iterations:
            config = run_index_learning_cycle(result, config)
            print("\n--- LEARNING ADJUSTMENTS ---")
            print(f"  Next scalp_min_conviction:  {config.scalp_min_conviction:.1f}")
            print(f"  Next swing_min_conviction:  {config.swing_min_conviction:.1f}")
            print(f"  Next medium_min_conviction: {config.medium_min_conviction:.1f}")
            print(f"  Next MC prob scalp:         {config.mc_min_prob_profit_scalp:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
