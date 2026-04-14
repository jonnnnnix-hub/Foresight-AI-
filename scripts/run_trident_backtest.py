#!/usr/bin/env python3
"""Run Trident backtest and/or optimization.

Usage:
    # Quick single test with defaults:
    .venv/bin/python scripts/run_trident_backtest.py

    # Full optimization (all 5 phases):
    .venv/bin/python scripts/run_trident_backtest.py --optimize

    # Run specific optimization phases:
    .venv/bin/python scripts/run_trident_backtest.py --optimize --phases signals combos

    # Check data availability first:
    .venv/bin/python scripts/run_trident_backtest.py --check-data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trident")


def check_data() -> None:
    """Check OPRA data availability for Trident tickers."""
    from flowedge.scanner.backtest.trident.config import (
        CACHE_DIR,
        TRIDENT_TICKERS,
    )

    logger.info("Checking data availability for Trident tickers...")
    for ticker in TRIDENT_TICKERS:
        stock_dir = CACHE_DIR / ticker / "1min"
        opts_dir = CACHE_DIR / ticker / "options_1min"

        stock_files = (
            list(stock_dir.glob("*.json")) if stock_dir.exists() else []
        )
        opts_files = (
            list(opts_dir.glob("*.json")) if opts_dir.exists() else []
        )

        opt_dates = sorted(f.stem.split("_")[-1] for f in opts_files)
        first = opt_dates[0] if opt_dates else "none"
        last = opt_dates[-1] if opt_dates else "none"

        logger.info(
            "  %s: %d stock files, %d OPRA days [%s → %s]",
            ticker, len(stock_files), len(opts_files), first, last,
        )

    logger.info("")
    logger.info(
        "If OPRA data is missing, run: "
        ".venv/bin/python scripts/download_trident_opra.py"
    )
    logger.info(
        "NOTE: Trident uses REAL OPRA data only — no simulated prices."
    )


def run_quick_test() -> None:
    """Run a single backtest with default config."""
    from flowedge.scanner.backtest.trident.optimizer import run_single_test

    logger.info("=" * 60)
    logger.info("Trident Quick Test (default config)")
    logger.info("Using REAL OPRA option pricing only")
    logger.info("=" * 60)

    result = run_single_test()

    logger.info("")
    logger.info("=== RESULTS ===")
    logger.info("Trades: %d", result.total_trades)
    logger.info("Win Rate: %.1f%%", result.win_rate)
    logger.info("Total P&L: $%.2f", result.total_pnl)
    logger.info("Profit Factor: %.2f", result.profit_factor)
    logger.info("Sharpe Ratio: %.2f", result.sharpe_ratio)
    logger.info("Max Drawdown: %.1f%%", result.max_drawdown_pct)
    logger.info("Avg Hold: %.0f min", result.avg_hold_minutes)
    logger.info("Calls: %d (%.1f%% WR)", result.calls_taken, result.call_win_rate)
    logger.info("Puts: %d (%.1f%% WR)", result.puts_taken, result.put_win_rate)
    logger.info("OPRA days used: %d", result.days_with_opra)
    logger.info("Days skipped (no OPRA): %d", result.days_skipped_no_opra)

    if result.per_ticker:
        logger.info("\nPer-Ticker:")
        for ticker, stats in result.per_ticker.items():
            logger.info(
                "  %s: %d trades, %.1f%% WR, $%.2f P&L",
                ticker, stats["trades"], stats["win_rate"],
                stats["total_pnl"],
            )


def run_optimization(
    phases: list[str] | None = None,
    max_configs: int = 200,
) -> None:
    """Run the full Trident optimization pipeline."""
    from flowedge.scanner.backtest.trident.optimizer import run_optimization

    logger.info("=" * 60)
    logger.info("Trident Full Optimization")
    logger.info("Phases: %s", phases or "all")
    logger.info("Max configs per phase: %d", max_configs)
    logger.info("Using REAL OPRA option pricing only")
    logger.info("=" * 60)

    run = run_optimization(
        phases=phases,
        max_configs_per_phase=max_configs,
    )

    logger.info("")
    logger.info("=== OPTIMIZATION COMPLETE ===")
    logger.info("Total configs tested: %d", run.total_configs)
    logger.info("Best config: %s", run.best_config.get("name", "none"))
    logger.info("Results saved to: data/trident_backtest_results/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trident ETF 0DTE Options Scalper — Backtest & Optimize",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run full optimization (default: quick single test)",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["signals", "combos", "exits", "options", "time"],
        help="Which optimization phases to run (default: all)",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=200,
        help="Max configs per optimization phase (default: 200)",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check data availability and exit",
    )
    args = parser.parse_args()

    if args.check_data:
        check_data()
        return

    if args.optimize:
        run_optimization(phases=args.phases, max_configs=args.max_configs)
    else:
        run_quick_test()


if __name__ == "__main__":
    main()
