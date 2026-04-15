#!/usr/bin/env python3
"""Entry point for the unified options scanner process.

Runs all 4 options scanners (scalp_v2, vol_scalp_v1, production, trident)
in a single async event loop with shared Polygon WebSocket + ORATS cache.

Usage:
    .venv/bin/python scripts/run_unified_options.py
    python -m scripts.run_unified_options
"""

import asyncio
import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.orchestrator import run_unified_options  # noqa: E402

if __name__ == "__main__":
    asyncio.run(run_unified_options())
