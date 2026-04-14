"""Run PHANTOM v4 backtest with multi-factor signal layers.

Changes from v3:
- PULSE momentum layer: conviction boost/penalty from momentum alignment
- GEX proxy layer: conviction boost/penalty from synthetic gamma exposure
- Kronos pattern layer: historical pattern matching for directional prediction
- UPTREND regime removed (16.7% WR, -18.2% avg in v3)
- Only STRONG_UPTREND and STRONG_DOWNTREND regimes allowed
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.backtest.engine import run_backtest  # noqa: E402


async def main() -> None:
    print("PHANTOM v4: 2-year backtest with multi-factor signal layers (10 tickers)")
    print("Changes: +PULSE momentum, +GEX proxy, +Kronos pattern, -UPTREND regime")
    print("Estimated: ~2 min/ticker = ~20 minutes\n")

    tickers = ["TSLA", "NVDA", "AAPL", "META", "AMZN", "SPY", "QQQ", "AMD", "GOOGL", "MSFT"]

    result = await run_backtest(
        tickers=tickers,
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

    # Save results
    out_dir = Path("data/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest_v4_2y.json"
    out_path.write_text(result.model_dump_json(indent=2))

    # Print summary
    trades = result.trades or []
    wins = sum(1 for t in trades if t.outcome.value == "win")
    losses = sum(1 for t in trades if t.outcome.value in ("loss", "expired"))
    expired = sum(1 for t in trades if t.outcome.value == "expired")

    win_pnls = [t.pnl_pct for t in trades if t.outcome.value == "win"]
    loss_pnls = [t.pnl_pct for t in trades if t.outcome.value != "win"]
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
    best = max((t.pnl_pct for t in trades), default=0)
    worst = min((t.pnl_pct for t in trades), default=0)

    total = len(trades)
    expectancy = sum(t.pnl_pct for t in trades) / total if total > 0 else 0
    avg_hold = sum(t.hold_days for t in trades) / total if total > 0 else 0

    print(f"\n{'=' * 65}")
    print("PHANTOM v4 — MULTI-FACTOR 2-YEAR BACKTEST")
    print(f"{'=' * 65}")
    print(f"Run ID: {result.run_id}")
    print(f"Period: {result.lookback_days} days | Tickers: {len(tickers)}")

    print(f"\n--- PORTFOLIO ---")
    print(f"Starting capital:   ${result.starting_capital:,.2f}")
    print(f"Ending value:       ${result.ending_value:,.2f}")
    print(f"Portfolio return:   {result.portfolio_return_pct:+.1f}%")
    print(f"Max drawdown:       {result.max_drawdown_pct:.1f}%")
    print(f"Sharpe ratio:       {result.sharpe_ratio:.3f}")

    print(f"\n--- TRADES ---")
    print(f"Total: {total} | Wins: {wins} | Losses: {losses} | Expired: {expired}")
    print(f"Win rate:           {result.win_rate * 100:.1f}%")
    print(f"Avg win:            +{avg_win:.1f}%")
    print(f"Avg loss:           {avg_loss:.1f}%")
    print(f"Best:               +{best:.1f}%")
    print(f"Worst:              {worst:.1f}%")
    print(f"Profit factor:      {result.profit_factor:.2f}")
    print(f"Expectancy:         {expectancy:+.1f}%")
    print(f"Avg hold:           {avg_hold:.1f} days")
    print(f"Max consec wins:    {result.max_consecutive_wins}")
    print(f"Max consec losses:  {result.max_consecutive_losses}")

    # By strategy
    if result.by_strategy:
        print(f"\n--- BY STRATEGY ---")
        for name, stats in sorted(result.by_strategy.items()):
            t_count = int(stats.get("trades", 0))
            wr = stats.get("win_rate", 0) * 100
            avg = stats.get("avg_pnl_pct", 0)
            total_pnl = stats.get("total_pnl_pct", 0)
            print(f"  {name:<20s}: {t_count:3d}T  WR={wr:5.1f}%  avg={avg:+7.1f}%  total={total_pnl:+8.1f}%")

    # By regime
    if result.by_regime:
        print(f"\n--- BY REGIME ---")
        for name, stats in sorted(result.by_regime.items()):
            t_count = int(stats.get("trades", 0))
            wr = stats.get("win_rate", 0) * 100
            avg = stats.get("avg_pnl_pct", 0)
            print(f"  {name:<22s}: {t_count:3d}T  WR={wr:5.1f}%  avg={avg:+7.1f}%")

    # By ticker
    print(f"\n--- BY TICKER ---")
    if result.by_ticker:
        for name, stats in sorted(
            result.by_ticker.items(),
            key=lambda x: x[1].get("total_pnl_pct", 0),
            reverse=True,
        ):
            t_count = int(stats.get("trades", 0))
            wr = stats.get("win_rate", 0) * 100
            avg = stats.get("avg_pnl_pct", 0)
            total_pnl = stats.get("total_pnl_pct", 0)
            print(f"  {name:<6s}: {t_count:3d}T  WR={wr:5.1f}%  avg={avg:+7.1f}%  total={total_pnl:+8.1f}%")

    # By score bucket
    if result.by_score_bucket:
        print(f"\n--- BY SCORE BUCKET ---")
        for name, stats in sorted(result.by_score_bucket.items()):
            t_count = int(stats.get("count", 0))
            wr = stats.get("win_rate", 0) * 100
            avg = stats.get("avg_pnl_pct", 0)
            print(f"  [{name:>4s}]: {t_count:3d}T  WR={wr:5.1f}%  avg={avg:+7.1f}%")

    # Monthly
    if result.monthly:
        print(f"\n--- MONTHLY ---")
        for m in result.monthly:
            print(
                f"  {m.month}:  {m.trades:3d}T  {m.wins}W/{m.losses}L  "
                f"WR={m.win_rate * 100:5.1f}%  avg={m.avg_pnl_pct:+7.1f}%  "
                f"total={m.total_pnl_pct:+8.1f}%"
            )

    # S&P comparison
    annualized_sp = 10.5  # ~10.5% historical average
    sp_2yr = (1 + annualized_sp / 100) ** 2 - 1
    print(f"\n--- S&P 500 COMPARISON ---")
    print(f"  S&P avg (2.0yr): +{sp_2yr * 100:.1f}%")
    print(f"  PHANTOM v4:       {result.portfolio_return_pct:+.1f}%")
    print(f"  Alpha:            {result.portfolio_return_pct - sp_2yr * 100:+.1f}%")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
