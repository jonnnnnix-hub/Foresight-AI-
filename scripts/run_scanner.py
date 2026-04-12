#!/usr/bin/env python3
"""Launch the FlowEdge production scanner.

Runs all 3 models during market hours, executes on Alpaca paper.

Usage:
    .venv/bin/python scripts/run_scanner.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.live.scanner import main

if __name__ == "__main__":
    asyncio.run(main())
