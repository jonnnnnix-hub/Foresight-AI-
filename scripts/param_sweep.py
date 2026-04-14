#!/usr/bin/env python3
"""Parameter sweep for scalp model v2 — find 70%+ WR configs.

Tests many parameter combinations in a single backtest run by
post-filtering trades from a base run, plus running new backtests
for parameters that change signal generation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flowedge.scanner.backtest.scalp_config import ScalpConfig  # noqa: E402
from flowedge.scanner.backtest.scalp_model_v2 import run_scalp_backtest_v2  # noqa: E402

# ── Ticker universes to test ─────────────────────────────────────
UNIVERSES = {
    "low_vol": [
        "IWM", "DIA", "SPY", "QQQ", "XLF", "XLK", "XLV", "XLE",
        "V", "WMT", "JPM", "BAC", "COST", "INTC",
    ],
    "winners_only": [
        "IWM", "COST", "V", "INTC", "PLTR", "CRM", "WMT", "NVDA",
    ],
    "etf_only": [
        "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLV", "XLE",
    ],
    "proven_8": [
        "IWM", "COST", "INTC", "PLTR", "CRM", "V", "WMT", "AMD",
    ],
    "top_wr": [
        "IWM", "COST", "V", "INTC", "PLTR", "CRM", "WMT", "NVDA",
        "AMD", "DIA",
    ],
}

# ── Parameter grid ────────────────────────────────────────────────
PARAM_GRID = {
    "ibs_threshold": [0.03, 0.05, 0.08],
    "rsi3_threshold": [12.0, 15.0, 20.0],
    "vol_spike": [2.0, 2.5, 3.0],
    "intraday_drop": [-0.002, -0.003, -0.005],
    "max_hold_bars": [6, 8, 10],
    "trail_pct": [0.03, 0.04, 0.05],
    "tp_underlying": [0.001, 0.0015, 0.002],
    "min_premium": [0.30, 0.50],
    "risk_per_trade": [0.03, 0.05],
}

# ── Phase 1: Universe sweep with current filter params ────────────
def sweep_universes() -> list[dict]:
    """Test each ticker universe with default + tightened params."""
    results = []

    configs = [
        ("default", {}),
        ("tight_exit", {"max_hold_bars": 8, "trail_pct": 0.03}),
        ("tight_entry", {"ibs_threshold": 0.05, "rsi3_threshold": 15.0, "vol_spike": 2.5}),
        ("tight_all", {
            "ibs_threshold": 0.05, "rsi3_threshold": 15.0, "vol_spike": 2.5,
            "max_hold_bars": 8, "trail_pct": 0.03,
        }),
        ("ultra_tight", {
            "ibs_threshold": 0.03, "rsi3_threshold": 12.0, "vol_spike": 3.0,
            "intraday_drop": -0.005, "max_hold_bars": 6, "trail_pct": 0.03,
            "tp_underlying": 0.001, "min_premium": 0.50,
        }),
        ("high_conviction", {
            "ibs_threshold": 0.03, "rsi3_threshold": 15.0, "vol_spike": 3.0,
            "intraday_drop": -0.005, "max_hold_bars": 8, "trail_pct": 0.03,
            "tp_underlying": 0.001,
        }),
        ("fast_scalp", {
            "ibs_threshold": 0.05, "rsi3_threshold": 15.0, "vol_spike": 2.5,
            "max_hold_bars": 6, "trail_pct": 0.03, "tp_underlying": 0.001,
        }),
        ("deep_dip", {
            "ibs_threshold": 0.03, "rsi3_threshold": 12.0, "vol_spike": 2.0,
            "intraday_drop": -0.005, "max_hold_bars": 10, "trail_pct": 0.04,
            "tp_underlying": 0.002, "min_premium": 0.30,
        }),
    ]

    total_runs = len(UNIVERSES) * len(configs)
    run_num = 0

    for univ_name, tickers in UNIVERSES.items():
        for cfg_name, overrides in configs:
            run_num += 1
            cfg = ScalpConfig(
                tickers=tickers,
                commission_per_contract=0.50,
                **overrides,
            )

            t0 = time.time()
            r = run_scalp_backtest_v2(
                tickers=tickers,
                config=cfg,
                entry_mode="next_open",
                exit_mode="bar_close",
            )
            elapsed = time.time() - t0

            rec = {
                "universe": univ_name,
                "config": cfg_name,
                "tickers": len(tickers),
                "trades": r.total_trades,
                "wr": r.win_rate,
                "return_pct": r.portfolio_return_pct,
                "pnl": round(r.ending_value - r.starting_capital, 0),
                "sharpe": r.sharpe_ratio,
                "pf": r.profit_factor,
                "max_dd": r.max_drawdown_pct,
                "avg_win": r.avg_win_pct,
                "avg_loss": r.avg_loss_pct,
                "expectancy": r.expectancy_pct,
                "elapsed": round(elapsed, 1),
            }
            results.append(rec)

            wr_color = "\033[92m" if r.win_rate >= 0.60 else (
                "\033[93m" if r.win_rate >= 0.50 else "\033[91m"
            )
            print(
                f"  [{run_num:>2}/{total_runs}] {univ_name:<14} {cfg_name:<16} "
                f"{r.total_trades:>3}t  {wr_color}WR {r.win_rate:.0%}\033[0m  "
                f"${r.ending_value - r.starting_capital:>+8,.0f}  "
                f"PF {r.profit_factor:.2f}  "
                f"Sharpe {r.sharpe_ratio:>+.2f}  "
                f"({elapsed:.0f}s)"
            )

    return results


def main() -> None:
    print("=" * 70)
    print("SCALP v2 PARAMETER SWEEP — Finding 70%+ WR")
    print("=" * 70)
    print()

    results = sweep_universes()

    # Sort by WR, then by number of trades (prefer more trades at same WR)
    results.sort(key=lambda x: (x["wr"], x["trades"]), reverse=True)

    print("\n" + "=" * 70)
    print("TOP 15 CONFIGURATIONS BY WIN RATE")
    print("=" * 70)
    for i, r in enumerate(results[:15]):
        wr_tag = (
            "***" if r["wr"] >= 0.70
            else ("**" if r["wr"] >= 0.60
                  else "*" if r["wr"] >= 0.50 else "")
        )
        print(
            f"  {i+1:>2}. {r['universe']:<14} {r['config']:<16} "
            f"{r['trades']:>3}t  WR {r['wr']:.1%} {wr_tag}  "
            f"${r['pnl']:>+8,.0f}  PF {r['pf']:.2f}  "
            f"Sharpe {r['sharpe']:>+.2f}  "
            f"AvgW {r['avg_win']:+.0f}% AvgL {r['avg_loss']:+.0f}%"
        )

    # Also show profitable configs
    profitable = [r for r in results if r["pnl"] > 0]
    if profitable:
        profitable.sort(key=lambda x: x["pnl"], reverse=True)
        print(f"\n{'=' * 70}")
        print(f"PROFITABLE CONFIGURATIONS ({len(profitable)} found)")
        print(f"{'=' * 70}")
        for i, r in enumerate(profitable[:10]):
            print(
                f"  {i+1:>2}. {r['universe']:<14} {r['config']:<16} "
                f"{r['trades']:>3}t  WR {r['wr']:.1%}  "
                f"${r['pnl']:>+8,.0f}  PF {r['pf']:.2f}  Sharpe {r['sharpe']:>+.2f}"
            )

    # Save full results
    out_path = Path("data/backtest_results/param_sweep_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
