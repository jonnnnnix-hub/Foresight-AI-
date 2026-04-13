#!/usr/bin/env python3
"""Launch the Volume Scalper v1 high-frequency signal scanner.

Runs on the same Alpaca account as scalp v2 (from .env).
Fires 2-5 signals/day with a relaxed 3-of-5 confluence filter.
Every trade is tagged "vol_scalp_v1" in the client_order_id.

Usage:
    .venv/bin/python scripts/run_volume_scalper.py
    .venv/bin/python scripts/run_volume_scalper.py --dry-run
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.live.volume_scalper_v1_scanner import scanner_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlowEdge Volume Scalper v1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect signals but don't place orders",
    )
    args = parser.parse_args()
    asyncio.run(scanner_main(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
