#!/usr/bin/env python3
"""Launch ZEUS — FlowEdge Specialist Ensemble paper trading scanner.

Runs 77 per-ticker optimized IBS reversion specialists on Alpaca paper Account 4.
All trades are tagged with model_name="zeus" for tracking.

Usage:
    .venv/bin/python scripts/run_specialist_scanner.py
    .venv/bin/python scripts/run_specialist_scanner.py --dry-run
    .venv/bin/python scripts/run_specialist_scanner.py --tickers PLTR,MRVL,SQ,META
    .venv/bin/python scripts/run_specialist_scanner.py --max-positions 6
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.live.specialist_scanner import main as scanner_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ZEUS — FlowEdge Specialist Ensemble — Alpaca Account 4",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run signal detection only — no orders placed",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated ticker filter (default: all 77 specialists)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=4,
        help="Maximum concurrent positions (default: 4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ticker_filter = [t.strip() for t in args.tickers.split(",") if t.strip()] or None
    asyncio.run(scanner_main(
        dry_run=args.dry_run,
        ticker_filter=ticker_filter,
        max_positions=args.max_positions,
    ))
