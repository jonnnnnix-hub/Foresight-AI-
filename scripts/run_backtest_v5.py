"""Run PHANTOM v5 backtest — balanced multi-factor with strategy-specific stops.

v5 changes from v4:
- Restored UPTREND regime (v4 over-filtered, 165→55 trades)
- Capped negative adjustment to -2.0 max (prevent over-filtering)
- Reduced momentum penalty magnitudes
- Strategy-specific stops (trend=wider, reversion=tighter)
- High-conviction trades get 15% wider stops
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.backtest.engine import run_backtest  # noqa: E402


async def main() -> None:
    print("PHANTOM v5: Balanced multi-factor + strategy-specific stops")
    print("Changes: +UPTREND restored, capped negative adj, strategy stops")
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

    out_dir = Path("data/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest_v5_2y.json"
    out_path.write_text(result.model_dump_json(indent=2))

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
    print("PHANTOM v5 — BALANCED MULTI-FACTOR 2-YEAR BACKTEST")
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

    if result.by_strategy:
        print(f"\n--- BY STRATEGY ---")
        for name, stats in sorted(result.by_strategy.items()):
            tc = int(stats.get("trades", 0))
            wr = stats.get("win_rate", 0) * 100
            avg = stats.get("avg_pnl_pct", 0)
            tot = stats.get("total_pnl_pct", 0)
            print(f"  {name:<20s}: {tc:3d}T  WR={wr:5.1f}%  avg={avg:+7.1f}%  total={tot:+8.1f}%")

    if result.by_regime:
        print(f"\n--- BY REGIME ---")
        for name, stats in sorted(result.by_regime.items()):
            tc = int(stats.get("trades", 0))
            wr = stats.get("win_rate", 0) * 100
            avg = stats.get("avg_pnl_pct", 0)
            print(f"  {name:<22s}: {tc:3d}T  WR={wr:5.1f}%  avg={avg:+7.1f}%")

    print(f"\n--- BY TICKER ---")
    if result.by_ticker:
        for name, stats in sorted(
            result.by_ticker.items(),
            key=lambda x: x[1].get("total_pnl_pct", 0),
            reverse=True,
        ):
            tc = int(stats.get("trades", 0))
            wr = stats.get("win_rate", 0) * 100
            avg = stats.get("avg_pnl_pct", 0)
            tot = stats.get("total_pnl_pct", 0)
            print(f"  {name:<6s}: {tc:3d}T  WR={wr:5.1f}%  avg={avg:+7.1f}%  total={tot:+8.1f}%")

    # Stop-loss exit analysis
    exit_reasons: dict[str, int] = {}
    for t in trades:
        er = getattr(t, "exit_reason", "unknown") or "unknown"
        base = er.split(" ")[0] if " " in er else er
        exit_reasons[base] = exit_reasons.get(base, 0) + 1
    if exit_reasons:
        print(f"\n--- EXIT REASONS ---")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            print(f"  {reason:<20s}: {count:3d} ({pct:.0f}%)")

    annualized_sp = 10.5
    sp_2yr = (1 + annualized_sp / 100) ** 2 - 1
    print(f"\n--- S&P 500 COMPARISON ---")
    print(f"  S&P avg (2.0yr): +{sp_2yr * 100:.1f}%")
    print(f"  PHANTOM v5:       {result.portfolio_return_pct:+.1f}%")
    print(f"  Alpha:            {result.portfolio_return_pct - sp_2yr * 100:+.1f}%")

    # v3 → v4 → v5 comparison
    print(f"\n--- VERSION COMPARISON ---")
    print(f"  v3 (tuned):       +57.2% | 165T | 29.1% WR | 1.15 PF | 0.683 Sharpe")
    print(f"  v4 (over-filter): +12.4% |  55T | 27.3% WR | 1.20 PF | 0.492 Sharpe")
    print(f"  v5 (balanced):    {result.portfolio_return_pct:+.1f}% | {total:3d}T | {result.win_rate*100:.1f}% WR | {result.profit_factor:.2f} PF | {result.sharpe_ratio:.3f} Sharpe")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
