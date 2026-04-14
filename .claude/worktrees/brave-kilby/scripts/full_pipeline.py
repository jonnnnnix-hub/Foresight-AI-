"""Full pipeline: download historical data → optimize → score → analyze → ensemble.

Run after download_historical.py finishes, or as a standalone script.
Uses the locally cached 7-year data (2018-2026) for all 78 tickers.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flowedge.config.logging import setup_logging

setup_logging("INFO")

OUTPUT_DIR = Path("data/optimizer")


def main() -> None:
    t0 = time.time()
    results: dict[str, dict] = {}  # type: ignore[type-arg]

    print("\n" + "=" * 80)
    print("FLOWEDGE FULL PIPELINE — 7 YEARS × 78 TICKERS")
    print("=" * 80)

    # ── Step 1: Verify data availability ──
    print("\n[1/6] Verifying data...")
    from flowedge.scanner.backtest.shares_engine import _load_daily

    cache = Path("data/flat_files_s3")
    tickers_with_data = []
    for d in sorted(cache.iterdir()):
        if d.is_dir() and (d / "1min").exists():
            daily = _load_daily(d.name)
            if len(daily) >= 200:  # Need 200+ days for meaningful optimization
                tickers_with_data.append((d.name, len(daily)))

    print(f"  {len(tickers_with_data)} tickers with 200+ days of data")
    if not tickers_with_data:
        print("  ERROR: No tickers with enough data. Run download_historical.py first.")
        return

    for t, n in tickers_with_data[:10]:
        print(f"    {t}: {n} days")
    if len(tickers_with_data) > 10:
        print(f"    ... and {len(tickers_with_data) - 10} more")

    ticker_names = [t for t, _ in tickers_with_data]
    results["data"] = {
        "tickers": len(tickers_with_data),
        "sample": tickers_with_data[:5],
    }

    # ── Step 2: Grid search optimization ──
    print(f"\n[2/6] Grid search on {len(ticker_names)} tickers...")
    try:
        from flowedge.scanner.backtest.optimizer import run_shares_grid_search

        grid = run_shares_grid_search(tickers=ticker_names)
        results["grid_search"] = {
            "tickers": len(grid.tickers),
            "trials": grid.total_trials,
            "elapsed": grid.elapsed_seconds,
            "top_5": [
                {"ticker": t.ticker, "wr": t.optimized_win_rate, "ret": t.optimized_return_pct}
                for t in grid.tickers[:5]
            ],
        }
        print(f"  Done: {len(grid.tickers)} tickers, {grid.total_trials} trials")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["grid_search"] = {"error": str(e)}

    # ── Step 3: Create specialists ──
    print("\n[3/6] Creating specialist bots...")
    try:
        from flowedge.scanner.backtest.specialist import (
            generate_specialists_from_optimizer,
            save_specialists,
        )

        specs = generate_specialists_from_optimizer(min_win_rate=0.55, min_trades=10)
        save_specialists(specs)
        results["specialists"] = {
            "count": len(specs),
            "tickers": [s.tickers[0] for s in specs[:10]],
        }
        print(f"  Created {len(specs)} specialist bots")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["specialists"] = {"error": str(e)}

    # ── Step 4: Train scorer ──
    print("\n[4/6] Training logistic regression scorer...")
    try:
        from flowedge.scanner.backtest.score_trainer import (
            extract_training_data,
            save_scorer_model,
            train_scorer,
            walk_forward_validation,
        )

        # Use FINAL results + any new optimizer results
        bt_files = sorted(Path("data/backtest").glob("*.json"))
        rows = extract_training_data(bt_files)
        if len(rows) >= 30:
            model, tr = train_scorer(rows)
            save_scorer_model(model)
            folds = walk_forward_validation(rows)
            avg_auc = sum(f.val_auc for f in folds) / len(folds) if folds else 0
            results["scorer"] = {
                "train_auc": tr.train_auc,
                "val_auc": tr.val_auc,
                "pnl_corr": tr.pnl_correlation,
                "walk_forward_auc": round(avg_auc, 4),
                "n_trades": len(rows),
            }
            print(f"  AUC={tr.val_auc:.4f}, PnL r={tr.pnl_correlation:.4f}, "
                  f"WF AUC={avg_auc:.4f}")
        else:
            print(f"  Skipped: {len(rows)} trades (need 30+)")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["scorer"] = {"error": str(e)}

    # ── Step 5: Entry/exit regression ──
    print("\n[5/6] Running entry/exit regression...")
    try:
        from flowedge.scanner.backtest.entry_exit_analysis import run_full_analysis

        analysis = run_full_analysis(bt_files)
        results["entry_exit"] = {
            "entry_r2": analysis.entry_r2,
            "exit_r2": analysis.exit_r2,
            "recommendations": analysis.recommendations[:5],
        }
        print(f"  Entry R²={analysis.entry_r2:.4f}, Exit R²={analysis.exit_r2:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["entry_exit"] = {"error": str(e)}

    # ── Step 6: Ensemble backtest ──
    print("\n[6/6] Running ensemble portfolio backtest...")
    try:
        from flowedge.scanner.backtest.ensemble import run_ensemble_backtest

        ensemble = run_ensemble_backtest()
        bt = ensemble.backtest
        results["ensemble"] = {
            "specialists": len(ensemble.specialists_used),
            "trades": bt.total_trades,
            "win_rate": bt.win_rate,
            "return_pct": bt.portfolio_return_pct,
            "profit_factor": bt.profit_factor,
            "sharpe": bt.sharpe_ratio,
        }
        print(f"  {bt.total_trades} trades, WR={bt.win_rate:.1%}, "
              f"Ret={bt.portfolio_return_pct:+.1f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["ensemble"] = {"error": str(e)}

    # ── Save summary ──
    elapsed = time.time() - t0
    results["elapsed_minutes"] = round(elapsed / 60, 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "full_pipeline_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE — {elapsed / 60:.1f} minutes")
    print("=" * 80)
    for step, data in results.items():
        if step == "elapsed_minutes":
            continue
        status = "ERROR" if isinstance(data, dict) and "error" in data else "OK"
        print(f"  {step:<20} {status}")
    print(f"\n  Summary: {summary_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
