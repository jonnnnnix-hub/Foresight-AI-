"""Download 10 years of 1-minute stock data from Massive S3.

Each S3 file contains ALL tickers for one day. We download day-by-day
and filter to our target tickers, caching locally as JSON.

Usage:
    python scripts/download_historical.py [--start 2016-01-01] [--end 2026-04-14]
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flowedge.scanner.data_feeds.massive_s3 import MassiveS3Downloader

CACHE_DIR = Path("data/flat_files_s3")

# Top tickers to download — focus on the ones that showed signal + high liquidity
TARGET_TICKERS = [
    # Original 33 (proven from prior backtesting)
    "AAPL", "AMD", "AMZN", "ARM", "AVGO", "BAC", "COIN", "COST", "CRM",
    "DIA", "GOOGL", "HOOD", "INTC", "IWM", "JPM", "META", "MSFT", "MSTR",
    "NFLX", "NVDA", "PLTR", "QQQ", "RDDT", "SMCI", "SOFI", "SPY", "TSLA",
    "V", "WMT", "XLE", "XLF", "XLK", "XLV",
    # Additional high-liquidity names
    "MRVL", "MU", "QCOM", "TXN", "LRCX", "AMAT",  # Semiconductors
    "ORCL", "ADBE", "NOW", "PANW", "CRWD", "SHOP", "SQ", "UBER", "PYPL",  # Tech
    "GS", "MS", "C", "WFC", "MA",  # Financials
    "UNH", "JNJ", "PFE", "LLY", "ABBV",  # Healthcare
    "XOM", "CVX", "COP",  # Energy
    "SBUX", "MCD", "NKE", "HD", "TGT",  # Consumer
    "CAT", "DE", "GE", "HON", "BA",  # Industrials
    "XLI", "XLY", "XLC", "XLP",  # Sector ETFs
    "GLD", "TLT", "SMH",  # Commodity/Bond/Sector ETFs
]


def download_historical(
    start: date = date(2016, 1, 4),
    end: date = date(2026, 4, 14),
    tickers: list[str] | None = None,
    batch_days: int = 30,
) -> None:
    """Download historical minute bars from S3, saving monthly chunks."""
    tickers = tickers or TARGET_TICKERS
    dl = MassiveS3Downloader(
        access_key=os.getenv("MASSIVE_ACCESS_KEY", "e275a83a-bdd4-402d-adea-2d2cb802095a"),
        secret_key=os.getenv("MASSIVE_SECRET_KEY", "q1stGSVMJ6SFQaFCLRjuqD1fcexJftT6"),
    )

    print(f"Downloading {len(tickers)} tickers from {start} to {end}")
    print(f"Target directory: {CACHE_DIR}")

    # Process month-by-month
    current = start
    total_bars = 0
    total_days = 0
    t0 = time.time()

    while current <= end:
        # Determine month boundary
        month_end = date(
            current.year + (1 if current.month == 12 else 0),
            (current.month % 12) + 1,
            1,
        ) - timedelta(days=1)
        month_end = min(month_end, end)

        month_label = f"{current.year}-{current.month:02d}"

        # Check if we already have this month cached for a representative ticker
        sample_ticker = "AAPL"
        sample_path = CACHE_DIR / sample_ticker / "1min" / f"{sample_ticker}_1min_{month_label}.json"
        if sample_path.exists():
            print(f"  {month_label}: already cached, skipping")
            current = month_end + timedelta(days=1)
            continue

        # Download the month
        print(f"  {month_label}: downloading...", end="", flush=True)
        month_bars: dict[str, list[dict]] = {t: [] for t in tickers}  # type: ignore[type-arg]
        days_in_month = 0

        d = current
        while d <= month_end:
            if d.weekday() < 5:  # Skip weekends
                try:
                    day_bars = dl.download_day(d, tickers)
                    for bar in day_bars:
                        tk = bar["ticker"]
                        if tk in month_bars:
                            month_bars[tk].append(bar)
                    days_in_month += 1
                    total_days += 1
                except Exception as e:
                    print(f"\n    Error on {d}: {e}", flush=True)
            d += timedelta(days=1)

        # Save monthly files per ticker
        month_total = 0
        for ticker, bars in month_bars.items():
            if not bars:
                continue
            cache_dir = CACHE_DIR / ticker / "1min"
            cache_dir.mkdir(parents=True, exist_ok=True)
            filepath = cache_dir / f"{ticker}_1min_{month_label}.json"
            filepath.write_text(json.dumps(bars, default=str))
            month_total += len(bars)

        total_bars += month_total
        elapsed = time.time() - t0
        rate = total_days / elapsed * 60 if elapsed > 0 else 0
        print(
            f" {month_total:,} bars, {days_in_month} days "
            f"({rate:.0f} days/min, total: {total_bars:,})"
        )

        current = month_end + timedelta(days=1)

    elapsed = time.time() - t0
    print(f"\nDone: {total_bars:,} bars across {total_days} days in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2026-04-14")
    args = parser.parse_args()

    download_historical(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )
