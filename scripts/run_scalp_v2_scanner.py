#!/usr/bin/env python3
"""Launch the FlowEdge Scalp v2 paper trading scanner.

Runs the sweep-validated 90% WR scalp model on Alpaca paper trading.
All trades are tagged with model_name="scalp_v2" for tracking.

Usage:
    .venv/bin/python scripts/run_scalp_v2_scanner.py
    .venv/bin/python scripts/run_scalp_v2_scanner.py --config configs/sweep_best_90wr.json
    .venv/bin/python scripts/run_scalp_v2_scanner.py --dry-run
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.live.scalp_v2_scanner import main as scanner_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlowEdge Scalp v2 — Live Paper Trader",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sweep_best_90wr.json",
        help="Path to ScalpConfig JSON (default: configs/sweep_best_90wr.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run signal detection only — no orders placed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(scanner_main(config_path=args.config, dry_run=args.dry_run))
