#!/usr/bin/env python3
"""A/B comparison: original v1 (sweep_best_90wr) vs council v2 (wide TP + loose RSI3).

Runs both configs on the full 4-year dataset, then slices the last N trading
days for a head-to-head comparison.  Also shows the full-period comparison.

Usage:
    python scripts/compare_versions.py              # default 7-day window
    python scripts/compare_versions.py --days 14    # custom window
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flowedge.scanner.backtest.result_store import save_result  # noqa: E402
from flowedge.scanner.backtest.scalp_config import ScalpConfig  # noqa: E402
from flowedge.scanner.backtest.scalp_model_v2 import run_scalp_backtest_v2  # noqa: E402
from flowedge.scanner.backtest.schemas import BacktestResult, BacktestTrade  # noqa: E402


def _slice_trades(
    trades: list[BacktestTrade], last_n_days: int, end_date: date,
) -> list[BacktestTrade]:
    """Return trades whose entry_date falls within the last N trading days."""
    cutoff = end_date - timedelta(days=last_n_days * 2)  # rough calendar buffer
    recent = [t for t in trades if t.entry_date >= cutoff]
    # Now take only the last N unique trading days
    unique_days = sorted(set(t.entry_date for t in recent), reverse=True)
    target_days = set(unique_days[:last_n_days])
    return [t for t in recent if t.entry_date in target_days]


def _trade_stats(trades: list[BacktestTrade]) -> dict:
    """Compute stats for a slice of trades."""
    if not trades:
        return {
            "trades": 0, "wins": 0, "wr": 0, "pnl": 0,
            "avg_win": 0, "avg_loss": 0, "best": 0, "worst": 0,
            "exit_tp": 0, "exit_trail": 0, "exit_time": 0,
        }
    wins = [t for t in trades if t.outcome.value == "win"]
    losses = [t for t in trades if t.outcome.value == "loss"]
    win_pnls = [t.pnl_pct for t in wins]
    loss_pnls = [t.pnl_pct for t in losses]
    pnl_dollars = sum(t.exit_value - t.cost_basis for t in trades)

    exit_counts = defaultdict(int)
    for t in trades:
        exit_counts[t.exit_reason] += 1

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "wr": len(wins) / len(trades) if trades else 0,
        "pnl": round(pnl_dollars, 2),
        "avg_win": round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0,
        "avg_loss": round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0,
        "best": round(max(t.pnl_pct for t in trades), 2),
        "worst": round(min(t.pnl_pct for t in trades), 2),
        "exit_tp": exit_counts.get("take_profit", 0),
        "exit_trail": exit_counts.get("trailing_stop", 0),
        "exit_time": exit_counts.get("time_exit", 0),
        "tickers_hit": sorted(set(t.ticker for t in trades)),
    }


def _per_day_breakdown(trades: list[BacktestTrade]) -> list[dict]:
    """Break down trades by day."""
    by_day: dict[date, list[BacktestTrade]] = defaultdict(list)
    for t in trades:
        by_day[t.entry_date].append(t)
    rows = []
    for d in sorted(by_day.keys()):
        day_trades = by_day[d]
        wins = sum(1 for t in day_trades if t.outcome.value == "win")
        pnl = sum(t.exit_value - t.cost_basis for t in day_trades)
        tickers = [t.ticker for t in day_trades]
        rows.append({
            "date": d.isoformat(),
            "trades": len(day_trades),
            "wins": wins,
            "wr": wins / len(day_trades) if day_trades else 0,
            "pnl": round(pnl, 2),
            "tickers": tickers,
        })
    return rows


def run_version(label: str, config: ScalpConfig) -> BacktestResult:
    """Run a single backtest version."""
    print(f"\n  Running {label}...")
    t0 = time.time()
    result = run_scalp_backtest_v2(
        tickers=config.tickers,
        config=config,
        entry_mode="next_open",
        exit_mode="bar_close",
    )
    elapsed = time.time() - t0
    print(
        f"  {label}: {result.total_trades} trades, "
        f"WR {result.win_rate:.1%}, "
        f"P&L ${result.ending_value - result.starting_capital:+,.0f}, "
        f"PF {result.profit_factor:.2f}, "
        f"Sharpe {result.sharpe_ratio:+.2f}  "
        f"({elapsed:.1f}s)"
    )
    return result


def print_comparison(
    label_a: str, stats_a: dict,
    label_b: str, stats_b: dict,
    title: str,
) -> None:
    """Print side-by-side comparison table."""
    w = 70
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")
    print(f"  {'Metric':<22} {label_a:>20} {label_b:>20}")
    print(f"  {'-' * 62}")

    rows = [
        ("Trades", stats_a["trades"], stats_b["trades"]),
        ("Wins", stats_a["wins"], stats_b["wins"]),
        ("Win Rate", f"{stats_a['wr']:.1%}", f"{stats_b['wr']:.1%}"),
        ("P&L ($)", f"${stats_a['pnl']:+,.0f}", f"${stats_b['pnl']:+,.0f}"),
        ("Avg Win %", f"{stats_a['avg_win']:+.1f}%", f"{stats_b['avg_win']:+.1f}%"),
        ("Avg Loss %", f"{stats_a['avg_loss']:+.1f}%", f"{stats_b['avg_loss']:+.1f}%"),
        ("Best Trade %", f"{stats_a['best']:+.1f}%", f"{stats_b['best']:+.1f}%"),
        ("Worst Trade %", f"{stats_a['worst']:+.1f}%", f"{stats_b['worst']:+.1f}%"),
        ("Exit: TP", stats_a["exit_tp"], stats_b["exit_tp"]),
        ("Exit: Trail", stats_a["exit_trail"], stats_b["exit_trail"]),
        ("Exit: Time", stats_a["exit_time"], stats_b["exit_time"]),
    ]

    for metric, va, vb in rows:
        print(f"  {metric:<22} {str(va):>20} {str(vb):>20}")

    # Highlight winner
    wr_a, wr_b = stats_a["wr"], stats_b["wr"]
    pnl_a, pnl_b = stats_a["pnl"], stats_b["pnl"]

    print(f"\n  {'VERDICT':>22}", end="")
    if stats_a["trades"] == 0 and stats_b["trades"] == 0:
        print("     No trades in this window")
    else:
        wr_winner = label_a if wr_a > wr_b else (label_b if wr_b > wr_a else "TIE")
        pnl_winner = label_a if pnl_a > pnl_b else (label_b if pnl_b > pnl_a else "TIE")
        print(f"  WR: {wr_winner:<12}  P&L: {pnl_winner}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B version comparison")
    parser.add_argument("--days", type=int, default=7, help="Trading days to compare (default: 7)")
    args = parser.parse_args()
    compare_days = args.days

    print("=" * 70)
    print("  SCALP MODEL A/B COMPARISON")
    print("  v1 (sweep_best_90wr) vs v2 (council: wide TP + loose RSI3)")
    print(f"  Window: last {compare_days} trading days")
    print("=" * 70)

    # ── Load configs ─────────────────────────────────────────────
    cfg_v1 = ScalpConfig.from_json_file("configs/sweep_best_90wr.json")
    cfg_v2 = ScalpConfig.from_json_file("configs/council_v2_wide_tp.json")

    print(
        f"\n  v1: TP={cfg_v1.tp_underlying},"
        f" RSI3={cfg_v1.rsi3_threshold}, Trail={cfg_v1.trail_pct}"
    )
    print(
        f"  v2: TP={cfg_v2.tp_underlying},"
        f" RSI3={cfg_v2.rsi3_threshold}, Trail={cfg_v2.trail_pct}"
    )
    delta_params = []
    if cfg_v1.tp_underlying != cfg_v2.tp_underlying:
        delta_params.append(f"TP: {cfg_v1.tp_underlying} -> {cfg_v2.tp_underlying}")
    if cfg_v1.rsi3_threshold != cfg_v2.rsi3_threshold:
        delta_params.append(f"RSI3: {cfg_v1.rsi3_threshold} -> {cfg_v2.rsi3_threshold}")
    if delta_params:
        print(f"  Changes: {', '.join(delta_params)}")

    # ── Run both versions ────────────────────────────────────────
    result_v1 = run_version("v1 (original)", cfg_v1)
    result_v2 = run_version("v2 (council)", cfg_v2)

    save_result(result_v1, "compare-v1-original")
    save_result(result_v2, "compare-v2-council")

    # ── Full period comparison ───────────────────────────────────
    full_stats_v1 = _trade_stats(result_v1.trades)
    full_stats_v2 = _trade_stats(result_v2.trades)
    print_comparison("v1 (original)", full_stats_v1, "v2 (council)", full_stats_v2,
                     "FULL PERIOD COMPARISON (all 4 years)")

    # ── Last N trading days comparison ───────────────────────────
    all_dates = sorted(set(
        t.entry_date for t in result_v1.trades + result_v2.trades
    ))
    end_date = all_dates[-1] if all_dates else date.today()

    recent_v1 = _slice_trades(result_v1.trades, compare_days, end_date)
    recent_v2 = _slice_trades(result_v2.trades, compare_days, end_date)

    recent_stats_v1 = _trade_stats(recent_v1)
    recent_stats_v2 = _trade_stats(recent_v2)
    print_comparison(
        "v1 (original)", recent_stats_v1,
        "v2 (council)", recent_stats_v2,
        f"LAST {compare_days} TRADING DAYS (ending {end_date})",
    )

    # ── Day-by-day breakdown ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  DAY-BY-DAY BREAKDOWN (last {compare_days} trading days)")
    print(f"{'=' * 70}")
    print(
        f"  {'Date':<12} {'v1 Trades':>9} {'v1 WR':>7} {'v1 P&L':>9}"
        f" {'v2 Trades':>9} {'v2 WR':>7} {'v2 P&L':>9}"
    )
    print(f"  {'-' * 64}")

    days_v1 = {d["date"]: d for d in _per_day_breakdown(recent_v1)}
    days_v2 = {d["date"]: d for d in _per_day_breakdown(recent_v2)}
    all_recent_dates = sorted(set(list(days_v1.keys()) + list(days_v2.keys())))

    empty = {"trades": 0, "wins": 0, "wr": 0, "pnl": 0, "tickers": []}
    for d in all_recent_dates:
        d1 = days_v1.get(d, empty)
        d2 = days_v2.get(d, empty)

        def _c(wr):
            if wr >= 0.7:
                return "\033[92m"
            if wr >= 0.5:
                return "\033[93m"
            return "\033[91m" if wr > 0 else ""

        r = "\033[0m"
        print(
            f"  {d:<12} "
            f"{d1['trades']:>5}t   "
            f"{_c(d1['wr'])}{d1['wr']:>5.0%}{r}  "
            f"${d1['pnl']:>+7,.0f}  "
            f"{d2['trades']:>5}t   "
            f"{_c(d2['wr'])}{d2['wr']:>5.0%}{r}  "
            f"${d2['pnl']:>+7,.0f}"
        )

    # ── Per-ticker comparison ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  PER-TICKER COMPARISON (full period)")
    print(f"{'=' * 70}")
    print(
        f"  {'Ticker':<8} {'v1 T':>5} {'v1 WR':>7} {'v1 P&L':>9}"
        f" {'v2 T':>5} {'v2 WR':>7} {'v2 P&L':>9}  {'Winner':>8}"
    )
    print(f"  {'-' * 62}")

    def _ticker_stats(trades, ticker):
        tt = [t for t in trades if t.ticker == ticker]
        if not tt:
            return {"trades": 0, "wr": 0, "pnl": 0}
        wins = sum(1 for t in tt if t.outcome.value == "win")
        pnl = sum(t.exit_value - t.cost_basis for t in tt)
        return {"trades": len(tt), "wr": wins / len(tt), "pnl": round(pnl, 0)}

    all_tickers = sorted(set(
        t.ticker for t in result_v1.trades + result_v2.trades
    ))
    for ticker in all_tickers:
        s1 = _ticker_stats(result_v1.trades, ticker)
        s2 = _ticker_stats(result_v2.trades, ticker)
        winner = "v1" if s1["pnl"] > s2["pnl"] else ("v2" if s2["pnl"] > s1["pnl"] else "tie")
        print(
            f"  {ticker:<8} "
            f"{s1['trades']:>5} {s1['wr']:>6.0%} ${s1['pnl']:>+7,.0f}  "
            f"{s2['trades']:>5} {s2['wr']:>6.0%} ${s2['pnl']:>+7,.0f}  "
            f"{winner:>8}"
        )

    # ── Final verdict ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  FINAL VERDICT")
    print(f"{'=' * 70}")

    scores = {"v1": 0, "v2": 0}
    checks = [
        ("Full WR", full_stats_v1["wr"], full_stats_v2["wr"]),
        ("Full P&L", full_stats_v1["pnl"], full_stats_v2["pnl"]),
        ("Full Trades", full_stats_v1["trades"], full_stats_v2["trades"]),
        (f"{compare_days}d WR", recent_stats_v1["wr"], recent_stats_v2["wr"]),
        (f"{compare_days}d P&L", recent_stats_v1["pnl"], recent_stats_v2["pnl"]),
        ("Avg Win Size", full_stats_v1["avg_win"], full_stats_v2["avg_win"]),
    ]
    for metric, v1_val, v2_val in checks:
        if v1_val > v2_val:
            winner = "v1"
            scores["v1"] += 1
        elif v2_val > v1_val:
            winner = "v2"
            scores["v2"] += 1
        else:
            winner = "tie"
        print(f"  {metric:<18} v1={str(v1_val):>12} vs v2={str(v2_val):>12}  -> {winner}")

    overall = "v1 (original)" if scores["v1"] > scores["v2"] else (
        "v2 (council)" if scores["v2"] > scores["v1"] else "TIE"
    )
    print(f"\n  Score: v1={scores['v1']} vs v2={scores['v2']}  ->  {overall} WINS")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
