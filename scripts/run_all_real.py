#!/usr/bin/env python3
"""Run ALL 4 models on REAL option contract prices.

No BS. No slippage model. No mock data. Actual market fills.
$0.18/contract rebate applied (Public.com).

Usage:
    .venv/bin/python scripts/run_all_real.py
"""

from __future__ import annotations

import json
import sys
import uuid
from collections import defaultdict
from datetime import date
from math import sqrt
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.backtest.real_options import (
    execute_trade,
    find_best_option,
    load_options_for_ticker,
    load_stock_daily,
)
from flowedge.scanner.backtest.schemas import BacktestResult, BacktestTrade, TradeOutcome


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses_ = 0.0, 0.0
    for i in range(-period, 0):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gains += d
        else:
            losses_ += abs(d)
    ag = gains / period
    al = losses_ / period
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


def _compile_result(
    name: str,
    trades: list[BacktestTrade],
    starting_capital: float,
    ending_cash: float,
    daily_values: list[tuple[str, float]],
    tickers: list[str],
) -> BacktestResult:
    total = len(trades)
    wins = sum(1 for t in trades if t.outcome == TradeOutcome.WIN)
    win_pnls = [t.pnl_pct for t in trades if t.outcome == TradeOutcome.WIN]
    loss_pnls = [t.pnl_pct for t in trades if t.outcome != TradeOutcome.WIN]
    gp = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gl = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))

    by_ticker: dict[str, dict[str, float]] = {}
    for tk in tickers:
        tt = [t for t in trades if t.ticker == tk]
        if tt:
            tw = sum(1 for t in tt if t.outcome == TradeOutcome.WIN)
            by_ticker[tk] = {
                "trades": float(len(tt)),
                "win_rate": round(tw / len(tt), 3),
                "avg_pnl_pct": round(sum(t.pnl_pct for t in tt) / len(tt), 2),
                "total_pnl_pct": round(sum(t.pnl_pct for t in tt), 2),
            }

    ret = (ending_cash - starting_capital) / starting_capital * 100

    peak = starting_capital
    max_dd = 0.0
    for _, v in daily_values:
        if v > peak:
            peak = v
        max_dd = max(max_dd, (peak - v) / peak * 100 if peak > 0 else 0)

    sharpe = 0.0
    if len(daily_values) > 10:
        rets = [
            (daily_values[i][1] - daily_values[i - 1][1]) / daily_values[i - 1][1]
            for i in range(1, len(daily_values)) if daily_values[i - 1][1] > 0
        ]
        if rets:
            m = sum(rets) / len(rets)
            var = sum((r - m) ** 2 for r in rets) / len(rets)
            s = sqrt(var) if var > 0 else 0.001
            sharpe = round((m * 252 - 0.05) / (s * sqrt(252)), 3)

    return BacktestResult(
        run_id=f"{name}-{uuid.uuid4().hex[:6]}",
        tickers=tickers, lookback_days=730,
        total_trades=total, wins=wins, losses=total - wins,
        win_rate=round(wins / total, 3) if total > 0 else 0,
        avg_win_pct=round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0,
        avg_loss_pct=round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0,
        best_trade_pct=round(max((t.pnl_pct for t in trades), default=0), 2),
        worst_trade_pct=round(min((t.pnl_pct for t in trades), default=0), 2),
        total_pnl_pct=round(sum(t.pnl_pct for t in trades), 2),
        profit_factor=round(gp / gl, 2) if gl > 0 else 0,
        avg_hold_days=round(sum(t.hold_days for t in trades) / total, 1) if total > 0 else 0,
        expectancy_pct=round(sum(t.pnl_pct for t in trades) / total, 2) if total > 0 else 0,
        trades=trades, by_ticker=by_ticker,
        starting_capital=starting_capital,
        ending_value=round(ending_cash, 2),
        portfolio_return_pct=round(ret, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=sharpe,
    )


def run_model_real(
    model_name: str,
    tickers: list[str],
    signal_fn: Any,
    starting_capital: float = 25_000.0,
    tp_pct: float = 0.50,
    max_hold: int = 12,
    min_hold: int = 3,
    trail_activation: float = 0.30,
    trail_pct: float = 0.25,
    risk_pct: float = 0.08,
    max_positions: int = 2,
    min_dte: int = 5,
    max_dte: int = 21,
    min_conviction: float = 9.0,
    cooldown: int = 3,
) -> BacktestResult:
    """Run any model using real option contract prices.

    signal_fn(ticker, daily_bars, idx) → dict with 'direction', 'conviction'
    or None if no signal.
    """
    print(f"\n{'─'*60}")
    print(f"  {model_name} — REAL OPTIONS")
    print(f"  Tickers: {tickers}")
    print(f"  TP={tp_pct*100:.0f}% | Hold={min_hold}-{max_hold}d | Rebate=$0.18/contract")
    print(f"{'─'*60}")

    # Load data
    all_daily: dict[str, list[dict[str, Any]]] = {}
    all_options: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for ticker in tickers:
        daily = load_stock_daily(ticker)
        options = load_options_for_ticker(ticker)
        if daily and options:
            all_daily[ticker] = daily
            all_options[ticker] = options
            overlap = len(set(b["date"] for b in daily) & set(options.keys()))
            print(f"  Loaded {ticker}: {len(daily)} stock days, {len(options)} option days, {overlap} overlap")

    if not all_daily:
        print("  No data loaded!")
        return BacktestResult(
            run_id=f"{model_name}-empty",
            tickers=tickers, starting_capital=starting_capital,
        )

    # Get all dates
    all_dates: set[str] = set()
    for bars in all_daily.values():
        for b in bars:
            all_dates.add(b["date"])
    sorted_dates = sorted(all_dates)

    cash = starting_capital
    closed: list[BacktestTrade] = []
    daily_values: list[tuple[str, float]] = []
    last_entry: dict[str, int] = {}
    open_positions: int = 0

    for day_idx, d in enumerate(sorted_dates):
        # Check if any positions closed on previous day
        # (simplified — we track net cash only)

        # Scan for signals
        if open_positions < max_positions:
            for ticker in all_daily:
                if day_idx - last_entry.get(ticker, -999) < cooldown:
                    continue

                tk_bars = all_daily[ticker]
                idx = next(
                    (i for i, b in enumerate(tk_bars) if b["date"] == d),
                    None,
                )
                if idx is None or idx < 55:
                    continue

                # Check signal
                signal = signal_fn(ticker, tk_bars, idx)
                if signal is None:
                    continue
                if signal.get("conviction", 0) < min_conviction:
                    continue

                # Get options for this date
                day_options = all_options.get(ticker, {}).get(d, [])
                if not day_options:
                    continue

                current_price = tk_bars[idx]["close"]
                direction = signal.get("direction", "bullish")

                option_info, contract_bars = find_best_option(
                    day_options, current_price, direction,
                    min_dte=min_dte, max_dte=max_dte,
                )
                if not option_info:
                    continue

                # Execute trade on real prices (pass all_options for multi-day holds)
                ticker_options = all_options.get(ticker, {})
                trade, pnl_net = execute_trade(
                    option_info, contract_bars,
                    entry_date=d, ticker=ticker,
                    direction=direction,
                    strategy=model_name,
                    conviction=signal["conviction"],
                    underlying_entry=current_price,
                    cash=cash,
                    risk_pct=risk_pct,
                    tp_pct=tp_pct,
                    max_hold_days=max_hold,
                    min_hold_days=min_hold,
                    trail_activation=trail_activation,
                    trail_pct=trail_pct,
                    all_options=ticker_options,
                )

                if trade:
                    cash += pnl_net
                    closed.append(trade)
                    last_entry[ticker] = day_idx

        daily_values.append((d, cash))

    result = _compile_result(
        model_name, closed, starting_capital, cash, daily_values, tickers,
    )

    # Print results
    wr_mark = "✓" if result.win_rate >= 0.60 else "✗"
    print(f"\n  Return:  {result.portfolio_return_pct:+.1f}%")
    print(f"  Trades:  {result.total_trades}")
    print(f"  WR:      {result.win_rate:.1%} {wr_mark}")
    print(f"  Avg Win: {result.avg_win_pct:+.1f}%  Avg Loss: {result.avg_loss_pct:+.1f}%")
    print(f"  PF:      {result.profit_factor:.2f}")
    print(f"  Sharpe:  {result.sharpe_ratio:.3f}")
    print(f"  Max DD:  {result.max_drawdown_pct:.1f}%")
    print(f"  Rebates: ${0.18 * sum(t.contracts for t in closed) * 2:.2f}")

    if result.by_ticker:
        for tk, s in sorted(result.by_ticker.items(), key=lambda x: -x[1].get("win_rate", 0)):
            print(f"    {tk:5s}: {int(s['trades']):3d}T  WR={s['win_rate']:.1%}  total={s['total_pnl_pct']:+.1f}%")

    reasons: dict[str, int] = {}
    for t in closed:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        rt = [t for t in closed if t.exit_reason == r]
        rw = sum(1 for t in rt if t.pnl_pct > 5)
        print(f"    {r:20s}: {c:3d} ({c / len(closed) * 100:.0f}%)  WR={rw / len(rt) * 100:.0f}%")

    return result


# ── SIGNAL FUNCTIONS FOR EACH MODEL ──


def precision_signal(ticker: str, bars: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    """Precision v10.2: IBS extreme + uptrend + prior-day down."""
    if ticker != "SPY":
        return None

    bar = bars[idx]
    closes = [bars[j]["close"] for j in range(max(0, idx - 49), idx + 1)]
    if len(closes) < 50:
        return None

    sma20 = sum(closes[-20:]) / 20
    sma50 = sum(closes[-50:]) / 50
    if sma20 <= sma50:
        return None

    rng = bar["high"] - bar["low"]
    if rng <= 0:
        return None
    ibs = (bar["close"] - bar["low"]) / rng
    if ibs >= 0.20:
        return None

    rsi = _rsi(closes)
    if rsi >= 45:
        return None

    if idx > 0 and bar["close"] >= bars[idx - 1]["close"]:
        return None

    conv = 7.0 + (0.20 - ibs) / 0.20 * 2.0
    if rsi < 30:
        conv += 0.5
    return {"direction": "bullish", "conviction": min(10, conv)}


def hybrid_signal(ticker: str, bars: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    """Hybrid v7.2: IBS reversion, 5 tickers, conviction 9.5+."""
    bar = bars[idx]
    closes = [bars[j]["close"] for j in range(max(0, idx - 49), idx + 1)]
    if len(closes) < 50:
        return None

    sma20 = sum(closes[-20:]) / 20
    sma50 = sum(closes[-50:]) / 50
    if sma20 <= sma50:
        return None

    rng = bar["high"] - bar["low"]
    if rng <= 0:
        return None
    ibs = (bar["close"] - bar["low"]) / rng

    # Ticker-specific IBS thresholds
    thresholds = {"SPY": 0.20, "QQQ": 0.18, "IWM": 0.15, "AAPL": 0.18, "META": 0.18}
    thresh = thresholds.get(ticker, 0.18)
    if ibs >= thresh:
        return None

    rsi = _rsi(closes)
    if rsi >= 45:
        return None

    if idx > 0 and bar["close"] >= bars[idx - 1]["close"]:
        return None

    conv = 7.0 + (thresh - ibs) / thresh * 2.0
    if rsi < 30:
        conv += 0.5
    if ticker in ("SPY", "META"):
        conv += 0.5  # Tier 1 bonus
    return {"direction": "bullish", "conviction": min(10, conv)}


def rapid_signal(ticker: str, bars: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    """Rapid v5.2: 4-signal confluence."""
    bar = bars[idx]
    closes = [bars[j]["close"] for j in range(max(0, idx - 20), idx + 1)]
    if len(closes) < 10 or idx < 25:
        return None

    sma20 = sum(closes[-min(20, len(closes)):]) / min(20, len(closes))
    sma50_data = [bars[j]["close"] for j in range(max(0, idx - 49), idx + 1)]
    sma50 = sum(sma50_data) / len(sma50_data)
    if sma20 <= sma50:
        return None

    rng = bar["high"] - bar["low"]
    if rng <= 0:
        return None
    ibs = (bar["close"] - bar["low"]) / rng
    if ibs >= 0.35:
        return None

    # RSI(3)
    if len(closes) < 4:
        return None
    g = [max(0, closes[-i] - closes[-i - 1]) for i in range(1, 4)]
    ls = [max(0, closes[-i - 1] - closes[-i]) for i in range(1, 4)]
    ag = sum(g) / 3
    al = sum(ls) / 3
    rsi3 = 100 - 100 / (1 + ag / al) if al > 0 else 100
    if rsi3 >= 30:
        return None

    # Range position
    recent_bars = bars[max(0, idx - 4):idx + 1]
    rh = max(b["high"] for b in recent_bars)
    rl = min(b["low"] for b in recent_bars)
    rr = rh - rl
    rpos = (bar["close"] - rl) / rr if rr > 0 else 0.5
    if rpos >= 0.20:
        return None

    # Volume
    avg_v = sum(bars[j]["volume"] for j in range(max(0, idx - 20), idx)) / max(1, min(20, idx))
    vr = bar["volume"] / avg_v if avg_v > 0 else 1
    if vr < 1.3:
        return None

    # Prior day down
    if idx > 0 and bar["close"] >= bars[idx - 1]["close"]:
        return None

    conv = 7.0 + (0.35 - ibs) + (1 if rsi3 < 15 else 0) + (0.5 if vr > 2 else 0)
    return {"direction": "bullish", "conviction": min(10, conv)}


def main() -> None:
    print("=" * 60)
    print("ALL MODELS — REAL OPTION CONTRACT PRICES")
    print("$0.18/contract rebate (Public.com) | NO BS | NO mock data")
    print("=" * 60)

    capital = 25_000.0

    # 1. Precision v10.2
    r1 = run_model_real(
        "precision", ["SPY"], precision_signal,
        starting_capital=capital,
        tp_pct=0.50, max_hold=12, min_hold=3,
        trail_activation=0.30, trail_pct=0.25,
        min_dte=5, max_dte=21, min_conviction=9.0,
        max_positions=1, cooldown=3, risk_pct=0.15,
    )

    # 2. Hybrid v7.2
    r2 = run_model_real(
        "hybrid", ["SPY", "QQQ", "IWM", "AAPL", "META"], hybrid_signal,
        starting_capital=capital,
        tp_pct=0.50, max_hold=7, min_hold=3,
        trail_activation=0.30, trail_pct=0.25,
        min_dte=5, max_dte=21, min_conviction=9.5,
        max_positions=2, cooldown=3, risk_pct=0.10,
    )

    # 3. Rapid v5.2
    r3 = run_model_real(
        "rapid", ["SPY", "QQQ", "XLK", "PLTR"], rapid_signal,
        starting_capital=capital,
        tp_pct=0.12, max_hold=3, min_hold=0,
        trail_activation=0.12, trail_pct=0.08,
        min_dte=1, max_dte=7, min_conviction=8.0,
        max_positions=3, cooldown=1, risk_pct=0.08,
    )

    # 4. Scalp (already uses real options — just run for comparison)
    from flowedge.scanner.backtest.scalp_real import run_scalp_real_backtest
    print(f"\n{'─'*60}")
    print(f"  scalp — REAL OPTIONS (existing)")
    print(f"{'─'*60}")
    r4 = run_scalp_real_backtest(starting_capital=capital)
    print(f"  Return: {r4.portfolio_return_pct:+.1f}% | {r4.total_trades}T | {r4.win_rate:.1%} WR")

    # Summary
    print(f"\n{'=' * 60}")
    print("FINAL MODEL SUITE — ALL REAL OPTION PRICES")
    print(f"{'=' * 60}")
    print(f"{'Model':>12s} {'WR':>6s} {'Return':>8s} {'PF':>6s} {'Sharpe':>7s} {'Trades':>7s}")
    print("-" * 50)
    for name, r in [("Precision", r1), ("Hybrid", r2), ("Rapid", r3), ("Scalp", r4)]:
        print(f"{name:>12s} {r.win_rate:5.1%} {r.portfolio_return_pct:+7.1f}% "
              f"{r.profit_factor:5.2f} {r.sharpe_ratio:+6.3f} {r.total_trades:6d}")

    # Save all
    out = Path("data/backtest")
    out.mkdir(parents=True, exist_ok=True)
    for name, r in [("precision", r1), ("hybrid", r2), ("rapid", r3), ("scalp", r4)]:
        path = out / f"REAL_{name}_backtest.json"
        path.write_text(json.dumps(r.model_dump(mode="json"), indent=2, default=str))
    print(f"\nAll results saved to data/backtest/REAL_*.json")


if __name__ == "__main__":
    main()
