#!/usr/bin/env python3
"""Launch the FlowEdge Production Scanner (precision/hybrid/rapid).

Runs 3 legacy models on a SEPARATE Alpaca paper account from scalp_v2.
Loads credentials from .env.production instead of .env.

Models:
  - Precision (SPY only, 80% WR, conviction 9.5+)
  - Hybrid (SPY, QQQ, IWM, AAPL, META, 72.7% WR, conviction 9.5+)
  - Rapid (SPY, QQQ, XLK, PLTR, 64.1% WR, conviction 8.0+)

Usage:
    .venv/bin/python scripts/run_production_scanner.py
"""

import asyncio
import os
import sys
from pathlib import Path

# In Docker/Fly.io: ALPACA_PROD_KEY_ID and ALPACA_PROD_SECRET_KEY are set
# as environment variables. Locally: load from .env.production file.
# The production scanner uses Account 2 (separate from scalp v2).

# Check if prod keys are already in env (Docker/Fly.io)
prod_key = os.getenv("ALPACA_PROD_KEY_ID", "")
prod_secret = os.getenv("ALPACA_PROD_SECRET_KEY", "")

if prod_key and prod_secret:
    # Running in Docker/Fly.io — override the default Alpaca keys with Account 2
    os.environ["ALPACA_API_KEY_ID"] = prod_key
    os.environ["ALPACA_API_SECRET_KEY"] = prod_secret
    print(f"Using Account 2 from env: {prod_key[:8]}...{prod_key[-4:]}")
else:
    # Running locally — load from .env.production file
    from dotenv import load_dotenv
    env_file = Path(__file__).resolve().parent.parent / ".env.production"
    if not env_file.exists():
        env_file = Path(__file__).resolve().parent.parent.parent / ".env.production"
    load_dotenv(env_file, override=True)
    _key = os.getenv("ALPACA_API_KEY_ID", "")
    print(f"Loaded credentials from: {env_file}")
    print(f"Alpaca key: {_key[:8]}...{_key[-4:]}")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.live.scanner import main  # noqa: E402

if __name__ == "__main__":
    asyncio.run(main())
