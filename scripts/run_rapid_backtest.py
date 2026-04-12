#!/usr/bin/env python3
"""FlowEdge Rapid — High-frequency 70%+ WR backtest."""

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
from flowedge.scanner.backtest.rapid_model import run_rapid_backtest


async def main() -> None:
    print("FLOWEDGE RAPID — High Frequency Model")
    print("6 tickers | Gap-fill + IBS + RSI3 + Range | 1-3 day holds")
    print("Target: 8-10 trades/month, 70%+ WR, $25K\n")

    result = await run_rapid_backtest(starting_capital=25_000.0)
    trades = result.trades

    print(f"\n{'=' * 65}")
    print("FLOWEDGE RAPID — RESULTS (NET OF SLIPPAGE)")
    print(f"{'=' * 65}")

    print("\n--- PORTFOLIO ---")
    print(f"Starting:   ${result.starting_capital:,.2f}")
    print(f"Ending:     ${result.ending_value:,.2f}")
    print(f"Return:     {result.portfolio_return_pct:+.1f}%")
    print(f"Max DD:     {result.max_drawdown_pct:.1f}%")
    print(f"Sharpe:     {result.sharpe_ratio:.3f}")

    trades_per_month = result.total_trades / 24
    print("\n--- TRADES ---")
    print(f"Total:      {result.total_trades} ({trades_per_month:.1f}/month)")
    print(f"Wins:       {result.wins}  Losses: {result.losses}")
    freq_ok = "✓" if 6 <= trades_per_month <= 12 else "✗"
    wr_ok = "✓" if result.win_rate >= 0.70 else "✗"
    print(f"Frequency:  {trades_per_month:.1f}/mo {freq_ok} (target: 8-10)")
    print(f"Win Rate:   {result.win_rate:.1%} {wr_ok} (target: 70%+)")
    print(f"Avg Win:    {result.avg_win_pct:+.1f}%")
    print(f"Avg Loss:   {result.avg_loss_pct:+.1f}%")
    print(f"PF:         {result.profit_factor:.2f}")
    print(f"Expectancy: {result.expectancy_pct:+.1f}%")
    print(f"Avg Hold:   {result.avg_hold_days:.1f} days")

    if result.by_ticker:
        print("\n--- BY TICKER ---")
        for name, stats in sorted(
            result.by_ticker.items(),
            key=lambda x: x[1].get("win_rate", 0), reverse=True,
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
            print(f"  {reason:20s}: {count:3d} ({count / len(trades) * 100:.0f}%)  WR={rwr:.0f}%")

        # Signal type analysis
        sig_types: dict[str, list[float]] = {}
        for t in trades:
            st = t.signal_type
            sig_types.setdefault(st, []).append(t.pnl_pct)
        print("\n--- SIGNAL CONFLUENCE ---")
        for st, pnls in sorted(sig_types.items(), key=lambda x: -len(x[1])):
            w = sum(1 for p in pnls if p > 5)
            wr = w / len(pnls) * 100
            avg = sum(pnls) / len(pnls)
            print(f"  {st:40s}: {len(pnls):3d}T  WR={wr:4.0f}%  avg={avg:+.1f}%")

    print("\n--- MODEL SUITE ---")
    print("  Precision v10.2:  80% WR |   5T/2yr | SPY only, IBS extreme")
    print("  Hybrid v3:        64% WR |  22T/2yr | 6 tickers, IBS reversion")
    print(f"  Rapid:            {result.win_rate:.0%} WR | {result.total_trades:3d}T/2yr | "
          f"6 tickers, gap+IBS+RSI3+range")

    out = Path("data/backtest/backtest_rapid_2y.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    print(f"\nSaved to {out}")

    if len(trades) >= 10:
        trade_dicts = [t.model_dump(mode="json") for t in trades]
        print("\n")
        diag = run_full_diagnostics(trade_dicts)
        print_diagnostic_report(diag)


if __name__ == "__main__":
    asyncio.run(main())
