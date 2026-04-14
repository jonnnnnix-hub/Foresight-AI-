#!/usr/bin/env python3
"""Launch the Trident live ETF 0DTE options scalper.

Runs on a dedicated Alpaca paper account (TRIDENT_ALPACA_KEY_ID).
Trades SPY, QQQ, IWM calls and puts with 0-90 min hold windows.
Every trade is tagged "trident_scalp" in the client_order_id.

Usage:
    .venv/bin/python scripts/run_trident.py
    .venv/bin/python scripts/run_trident.py --dry-run
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.live.trident_scanner import scanner_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlowEdge Trident — ETF 0DTE Options Scalper",
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
