"""Overnight optimization pipeline — runs everything unattended.

Pipeline order:
1. Grid search on existing 33 tickers (shares + scalp) with position sizing
2. Download 200+ new tickers from Massive S3
3. Quick screen all tickers → filter viable
4. Full grid search on viable tickers
5. Create specialist configs for winners
6. Train logistic regression scorer on all trade data
7. Run entry/exit regression analysis
8. Run ensemble backtest with all specialists
9. Save comprehensive results

This script is designed to run overnight (~2-4 hours total).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import structlog

logger = structlog.get_logger()

OUTPUT_DIR = Path("data/optimizer")


def run_overnight_pipeline(skip_download: bool = False) -> None:
    """Run the full overnight optimization pipeline."""
    from flowedge.config.logging import setup_logging
    setup_logging("INFO")

    results: dict[str, dict] = {}  # type: ignore[type-arg]
    t0 = time.time()

    print("\n" + "=" * 80)
    print("FLOWEDGE OVERNIGHT OPTIMIZATION PIPELINE")
    print("=" * 80)

    # ── Step 1: Grid search on existing 33 tickers ──
    print("\n[1/8] Grid search on existing 33 tickers (shares)...")
    try:
        from flowedge.scanner.backtest.optimizer import run_shares_grid_search
        shares_result = run_shares_grid_search()
        results["shares_grid"] = {
            "tickers": len(shares_result.tickers),
            "trials": shares_result.total_trials,
            "elapsed": shares_result.elapsed_seconds,
            "top_3": [
                {
                    "ticker": t.ticker,
                    "wr": t.optimized_win_rate,
                    "ret": t.optimized_return_pct,
                }
                for t in shares_result.tickers[:3]
            ],
        }
        print(f"  Done: {len(shares_result.tickers)} tickers, "
              f"{shares_result.total_trials} trials")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["shares_grid"] = {"error": str(e)}

    # ── Step 2: Scalp options grid search ──
    print("\n[2/8] Grid search on scalp options tickers...")
    try:
        from flowedge.scanner.backtest.optimizer import run_scalp_grid_search
        scalp_result = run_scalp_grid_search()
        results["scalp_grid"] = {
            "tickers": len(scalp_result.tickers),
            "trials": scalp_result.total_trials,
            "elapsed": scalp_result.elapsed_seconds,
        }
        print(f"  Done: {len(scalp_result.tickers)} tickers, "
              f"{scalp_result.total_trials} trials")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["scalp_grid"] = {"error": str(e)}

    # ── Step 3: Download expanded universe ──
    if not skip_download:
        print("\n[3/8] Downloading 200+ tickers from Massive S3...")
        try:
            from flowedge.scanner.backtest.universe_expansion import download_new_tickers
            downloaded = download_new_tickers()
            results["download"] = {
                "tickers": len(downloaded),
                "bars": sum(downloaded.values()),
            }
            print(f"  Done: {len(downloaded)} tickers downloaded")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results["download"] = {"error": str(e)}
    else:
        print("\n[3/8] Download skipped (--skip-download)")
        results["download"] = {"skipped": True}

    # ── Step 4: Screen + optimize expanded universe ──
    print("\n[4/8] Screen + optimize all available tickers...")
    try:
        from flowedge.scanner.backtest.universe_expansion import run_full_expansion_pipeline
        expansion = run_full_expansion_pipeline(skip_download=True)
        results["expansion"] = expansion
        print(f"  Done: {expansion.get('total_specialists', 0)} specialists created")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["expansion"] = {"error": str(e)}

    # ── Step 5: Train scorer ──
    print("\n[5/8] Training logistic regression scorer...")
    try:
        from flowedge.scanner.backtest.score_trainer import (
            extract_training_data,
            save_scorer_model,
            train_scorer,
            walk_forward_validation,
        )
        rows = extract_training_data()
        if len(rows) >= 30:
            model, training_result = train_scorer(rows)
            save_scorer_model(model)
            results["scorer"] = {
                "train_auc": training_result.train_auc,
                "val_auc": training_result.val_auc,
                "pnl_correlation": training_result.pnl_correlation,
                "n_train": training_result.n_train,
                "n_val": training_result.n_val,
                "feature_importance": training_result.feature_importance,
            }
            print(f"  Done: AUC={training_result.val_auc:.4f}, "
                  f"PnL r={training_result.pnl_correlation:.4f}")

            # Walk-forward validation
            folds = walk_forward_validation(rows)
            if folds:
                avg_auc = sum(f.val_auc for f in folds) / len(folds)
                avg_corr = sum(f.pnl_correlation for f in folds) / len(folds)
                results["walk_forward"] = {
                    "folds": len(folds),
                    "avg_auc": round(avg_auc, 4),
                    "avg_pnl_corr": round(avg_corr, 4),
                }
                print(f"  Walk-forward: {len(folds)} folds, "
                      f"avg AUC={avg_auc:.4f}, avg PnL r={avg_corr:.4f}")
        else:
            print(f"  Skipped: only {len(rows)} trades (need 30+)")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["scorer"] = {"error": str(e)}

    # ── Step 6: Entry/exit regression ──
    print("\n[6/8] Running entry/exit regression analysis...")
    try:
        from flowedge.scanner.backtest.entry_exit_analysis import run_full_analysis
        analysis = run_full_analysis()
        results["entry_exit"] = {
            "entry_r2": analysis.entry_r2,
            "exit_r2": analysis.exit_r2,
            "tp_calibration": analysis.tp_calibration,
            "recommendations": analysis.recommendations[:5],
        }
        print(f"  Done: Entry R²={analysis.entry_r2:.4f}, Exit R²={analysis.exit_r2:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["entry_exit"] = {"error": str(e)}

    # ── Step 7: Ensemble backtest ──
    print("\n[7/8] Running ensemble portfolio backtest...")
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
            "max_drawdown_pct": bt.max_drawdown_pct,
            "sharpe": bt.sharpe_ratio,
        }
        print(f"  Done: {bt.total_trades} trades, "
              f"WR={bt.win_rate:.1%}, Ret={bt.portfolio_return_pct:+.1f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["ensemble"] = {"error": str(e)}

    # ── Step 8: Save summary ──
    elapsed = time.time() - t0
    results["pipeline"] = {
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_human": f"{elapsed / 60:.0f} minutes",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "overnight_pipeline_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE — {elapsed / 60:.0f} minutes")
    print("=" * 80)
    for step, data in results.items():
        if step == "pipeline":
            continue
        status = "ERROR" if isinstance(data, dict) and "error" in data else "OK"
        print(f"  {step:<20} {status}")
    print(f"\n  Results saved to: {summary_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    skip_dl = "--skip-download" in sys.argv or "--local" in sys.argv
    run_overnight_pipeline(skip_download=skip_dl)
