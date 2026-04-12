#!/usr/bin/env python3
"""Run PHANTOM v8 backtest — adaptive scorer replaces noise-level stacking.

v8 changes from v7:
- Adaptive conviction scorer with data-driven feature weights
- Replaces momentum/GEX/Kronos/ticker penalty stacking (r=0.004)
- Features: ticker_wr, strategy_wr, regime_wr, RSI extreme, ADX,
  volume, momentum alignment, trend alignment, IBS extreme
- Blend: 30% original signal + 70% adaptive scorer
- UPTREND regime blocked (0% WR in v7 diagnostics)
- Circuit breaker + wider stops from v7 retained

Usage:
    .venv/bin/python scripts/run_backtest_v8.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.backtest.adaptive_scorer import (
    load_scorer_weights,
    save_scorer_weights,
    update_weights_from_trades,
)
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
    print("PHANTOM v8: Adaptive scorer — data-driven conviction")
    print("Changes: Replaces multi-factor stacking with learned weights")
    print("         Blend: 30% signal conviction + 70% adaptive scorer")
    print(f"Tickers: {TICKERS}")
    print()

    result = await run_backtest(
        tickers=TICKERS,
        lookback_days=730,
        starting_capital=10_000.0,
        max_positions=5,
        max_risk_per_trade=0.08,
        max_hold_days=12,
        trailing_stop_pct=0.35,
        hard_stop_pct=-0.50,
        take_profit_pct=3.50,
        dte=15,
        min_conviction=7.0,
    )

    trades = result.trades

    print(f"\n{'=' * 65}")
    print("PHANTOM v8 — ADAPTIVE SCORER 2-YEAR BACKTEST")
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
            pct = count / len(trades) * 100
            print(f"  {reason:20s}: {count:3d} ({pct:.0f}%)")

    print("\n--- VERSION COMPARISON ---")
    print("  v5 (balanced):    +44.2% |  77T | 31.2% WR | 1.31 PF | 0.600 Sharpe")
    print("  v7 (data-driven): +129.4%|  80T | 33.8% WR | 1.41 PF | 0.830 Sharpe")
    print(f"  v8 (adaptive):    {result.portfolio_return_pct:+.1f}% | "
          f"{result.total_trades:3d}T | {result.win_rate:.1%} WR | "
          f"{result.profit_factor:.2f} PF | {result.sharpe_ratio:.3f} Sharpe")

    # Save results
    out_dir = Path("data/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest_v8_2y.json"
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, default=str)
    )
    print(f"\nSaved to {out_path}")

    # Update scorer weights from this run's data
    trade_dicts = [t.model_dump(mode="json") for t in trades]
    if len(trade_dicts) >= 20:
        weights = load_scorer_weights()
        updated = update_weights_from_trades(trade_dicts, weights)
        save_scorer_weights(updated)
        print(f"\nScorer weights updated: v{updated.version} "
              f"(trained on {updated.trained_on_trades} trades, "
              f"WR={updated.training_win_rate:.1%})")

    # Run diagnostics
    print("\n\nRunning diagnostic analysis on v8 results...")
    diag = run_full_diagnostics(trade_dicts)
    print_diagnostic_report(diag)


if __name__ == "__main__":
    asyncio.run(main())
