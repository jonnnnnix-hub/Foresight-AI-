#!/usr/bin/env python3
"""Download OPRA options minute bars from Massive S3 for all tickers.

Downloads the full date range matching existing stock bar data,
filtered to near-ATM, 0-5 DTE, calls + puts.  Resume-safe: skips
dates already cached.

Usage:
    python scripts/download_opra.py
    python scripts/download_opra.py --tickers PLTR,NVDA,QQQ
    python scripts/download_opra.py --from-date 2025-01-01 --to-date 2025-06-30
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flowedge.scanner.data_feeds.options_s3 import (  # noqa: E402
    OptionsS3Downloader,
    get_underlying_closes_from_cache,
)

# All tickers with cached stock bars (the full universe)
ALL_TICKERS = [
    "AAPL", "AMD", "AMZN", "ARM", "AVGO", "BAC", "COIN", "COST", "CRM",
    "DIA", "GOOGL", "HOOD", "INTC", "IWM", "JPM", "META", "MSFT", "MSTR",
    "NFLX", "NVDA", "PLTR", "QQQ", "RDDT", "SMCI", "SOFI", "SPY", "TSLA",
    "V", "WMT", "XLE", "XLF", "XLK", "XLV",
]

# Default date range: full accessible OPRA range
DEFAULT_FROM = "2022-04-12"
DEFAULT_TO = "2026-04-10"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OPRA options data from Massive S3",
    )
    parser.add_argument(
        "--tickers",
        default=",".join(ALL_TICKERS),
        help="Comma-separated tickers (default: all 33)",
    )
    parser.add_argument(
        "--from-date", default=DEFAULT_FROM, help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--to-date", default=DEFAULT_TO, help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--max-dte", type=int, default=5, help="Max DTE (default: 5)",
    )
    parser.add_argument(
        "--strike-range", type=float, default=0.05,
        help="Strike range as fraction (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--env-file",
        default=str(ROOT / ".env"),
        help="Path to .env file",
    )
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    from_date = date.fromisoformat(args.from_date)
    to_date = date.fromisoformat(args.to_date)

    print("=" * 60)
    print("OPRA Options Data Download")
    print("=" * 60)
    print(f"  Tickers:      {len(tickers)} ({', '.join(tickers[:5])}...)")
    print(f"  Date range:   {from_date} -> {to_date}")
    print(f"  Max DTE:      {args.max_dte}")
    print(f"  Strike range: ±{args.strike_range * 100:.0f}%")
    print(f"  Env file:     {args.env_file}")
    print()

    # Load underlying closing prices from stock bar cache
    print("Loading underlying closing prices from stock cache...")
    t0 = time.time()
    prices_by_date = get_underlying_closes_from_cache(tickers)
    elapsed = time.time() - t0
    print(f"  Loaded {len(prices_by_date)} trading days in {elapsed:.1f}s")
    print()

    # Initialize downloader
    downloader = OptionsS3Downloader(env_path=args.env_file)

    print("Starting OPRA download...")
    print()
    t0 = time.time()

    totals = downloader.download_options_range(
        from_date=from_date,
        to_date=to_date,
        underlying_tickers=tickers,
        underlying_prices_by_date=prices_by_date,
        max_dte=args.max_dte,
        strike_range_pct=args.strike_range,
    )

    elapsed = time.time() - t0
    total_bars = sum(totals.values())

    print()
    print("=" * 60)
    print("Download Complete")
    print("=" * 60)
    print(f"  Time:       {elapsed / 60:.1f} minutes")
    print(f"  Total bars: {total_bars:,}")
    print()
    print("  Per ticker:")
    for tk in sorted(totals.keys()):
        if totals[tk] > 0:
            print(f"    {tk:<6} {totals[tk]:>10,} bars")

    # Check disk usage
    cache_dir = Path("data/flat_files_s3")
    total_size = 0
    for tk in tickers:
        opts_dir = cache_dir / tk / "options_1min"
        if opts_dir.exists():
            for f in opts_dir.glob("*.json"):
                total_size += f.stat().st_size
    print(f"\n  Disk usage: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
