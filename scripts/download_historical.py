#!/usr/bin/env python3
"""Download historical stock + OPRA options data from Massive S3.

Expands the dataset backwards from the existing 2024-04-12 start date
to 2022-04-12, adding ~500 more trading days of both stock minute bars
and options minute bars for all 33 tickers.

Resume-safe: stocks are saved per-month, options per-day.
Skips files that already exist on disk.

Usage:
    python scripts/download_historical.py
    python scripts/download_historical.py --stocks-only
    python scripts/download_historical.py --options-only
    python scripts/download_historical.py --tickers SPY,QQQ,NVDA
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flowedge.scanner.data_feeds.massive_s3 import (  # noqa: E402
    S3_BUCKET,
    S3_ENDPOINT,
    STOCKS_MINUTE_PREFIX,
)
from flowedge.scanner.data_feeds.options_s3 import (  # noqa: E402
    OptionsS3Downloader,
    get_underlying_closes_from_cache,
)

ALL_TICKERS = [
    "AAPL", "AMD", "AMZN", "ARM", "AVGO", "BAC", "COIN", "COST", "CRM",
    "DIA", "GOOGL", "HOOD", "INTC", "IWM", "JPM", "META", "MSFT", "MSTR",
    "NFLX", "NVDA", "PLTR", "QQQ", "RDDT", "SMCI", "SOFI", "SPY", "TSLA",
    "V", "WMT", "XLE", "XLF", "XLK", "XLV",
]

# Historical expansion range (gap between S3 access start and current data)
HIST_FROM = "2022-04-12"
HIST_TO = "2024-04-11"

CACHE_DIR = Path("data/flat_files_s3")

MAX_RETRIES = 3


def _load_env() -> None:
    """Load .env into os.environ."""
    if os.getenv("MASSIVE_ACCESS_KEY"):
        return
    for p in [Path(".env"), ROOT / ".env"]:
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())
            return


def _get_s3_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ["MASSIVE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MASSIVE_SECRET_KEY"],
        config=Config(signature_version="s3v4"),
    )


# ── Stock Download ────────────────────────────────────────────


def _stock_cache_path(ticker: str, month_str: str) -> Path:
    """Cache path for one month of stock bars: {ticker}_1min_{YYYY-MM}.json"""
    return CACHE_DIR / ticker / "1min" / f"{ticker}_1min_{month_str}.json"


def download_stocks_month(
    s3,
    year: int,
    month: int,
    tickers: list[str],
) -> dict[str, int]:
    """Download one month of stock minute bars, save per-ticker.

    Returns dict of ticker -> bar count for newly downloaded data.
    """
    month_str = f"{year}-{month:02d}"
    ticker_set = set(tickers)

    # Check which tickers already have this month cached
    missing = [t for t in tickers if not _stock_cache_path(t, month_str).exists()]
    if not missing:
        return {}

    # Collect bars for the month
    bars_by_ticker: dict[str, list[dict]] = defaultdict(list)
    d = date(year, month, 1)

    while d.month == month:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue

        key = f"{STOCKS_MINUTE_PREFIX}/{d.year}/{d.month:02d}/{d.isoformat()}.csv.gz"
        raw = _s3_get_with_retry(s3, key)
        if raw is None:
            d += timedelta(days=1)
            continue

        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            text = gz.read().decode("utf-8")

        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            tk = row.get("ticker", "")
            if tk not in ticker_set:
                continue

            ts_ns = int(row.get("window_start", 0))
            ts = datetime.fromtimestamp(ts_ns / 1_000_000_000)

            bars_by_ticker[tk].append({
                "ticker": tk,
                "ts": str(ts_ns),
                "timestamp": ts.isoformat(),
                "date": d.isoformat(),
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": int(float(row.get("volume", 0))),
                "vwap": float(row.get("vwap", 0)),
                "transactions": int(float(row.get("transactions", 0))),
            })

        d += timedelta(days=1)

    # Save per ticker
    counts: dict[str, int] = {}
    for tk, bars in bars_by_ticker.items():
        if not bars:
            continue
        path = _stock_cache_path(tk, month_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bars, default=str))
        counts[tk] = len(bars)

    return counts


def download_all_stocks(
    from_date: date,
    to_date: date,
    tickers: list[str],
) -> dict[str, int]:
    """Download stock minute bars for the full range, per-month."""
    s3 = _get_s3_client()
    totals: dict[str, int] = {t: 0 for t in tickers}
    months_done = 0

    # Iterate by month
    current = date(from_date.year, from_date.month, 1)
    end_month = date(to_date.year, to_date.month, 1)

    while current <= end_month:
        month_str = f"{current.year}-{current.month:02d}"
        t0 = time.time()
        counts = download_stocks_month(s3, current.year, current.month, tickers)
        elapsed = time.time() - t0

        if counts:
            for tk, cnt in counts.items():
                totals[tk] = totals.get(tk, 0) + cnt
            total_bars = sum(counts.values())
            print(
                f"  Stock {month_str}: {total_bars:,} bars "
                f"({len(counts)} tickers) in {elapsed:.1f}s"
            )
        else:
            print(f"  Stock {month_str}: cached (skipped)")

        months_done += 1

        # Advance to next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return totals


# ── OPRA Options Download ────────────────────────────────────


def download_all_options(
    from_date: date,
    to_date: date,
    tickers: list[str],
    max_dte: int = 5,
    strike_range_pct: float = 0.05,
) -> dict[str, int]:
    """Download OPRA options using the existing OptionsS3Downloader."""
    # Load underlying prices from the expanded stock cache
    print("  Loading underlying closing prices from stock cache...")
    prices_by_date = get_underlying_closes_from_cache(tickers)
    print(f"  Loaded {len(prices_by_date)} trading days of prices")

    downloader = OptionsS3Downloader()
    totals = downloader.download_options_range(
        from_date=from_date,
        to_date=to_date,
        underlying_tickers=tickers,
        underlying_prices_by_date=prices_by_date,
        max_dte=max_dte,
        strike_range_pct=strike_range_pct,
    )
    return totals


# ── S3 Helpers ────────────────────────────────────────────────


def _s3_get_with_retry(s3, key: str) -> bytes | None:
    """Download an S3 object with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
            return resp["Body"].read()
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return None


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical stock + OPRA data",
    )
    parser.add_argument(
        "--tickers",
        default=",".join(ALL_TICKERS),
        help="Comma-separated tickers (default: all 33)",
    )
    parser.add_argument(
        "--from-date", default=HIST_FROM,
        help="Start date (default: 2022-04-12)",
    )
    parser.add_argument(
        "--to-date", default=HIST_TO,
        help="End date (default: 2024-04-11)",
    )
    parser.add_argument(
        "--stocks-only", action="store_true",
        help="Only download stock minute bars",
    )
    parser.add_argument(
        "--options-only", action="store_true",
        help="Only download OPRA options",
    )
    parser.add_argument(
        "--max-dte", type=int, default=5,
        help="Max DTE for options (default: 5)",
    )
    parser.add_argument(
        "--strike-range", type=float, default=0.05,
        help="Strike range fraction (default: 0.05 = 5%%)",
    )
    args = parser.parse_args()

    _load_env()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    from_date = date.fromisoformat(args.from_date)
    to_date = date.fromisoformat(args.to_date)

    do_stocks = not args.options_only
    do_options = not args.stocks_only

    print("=" * 60)
    print("Historical Data Download")
    print("=" * 60)
    print(f"  Tickers:    {len(tickers)}")
    print(f"  Date range: {from_date} -> {to_date}")
    print(f"  Stocks:     {'YES' if do_stocks else 'SKIP'}")
    print(f"  Options:    {'YES' if do_options else 'SKIP'}")
    print()

    t_start = time.time()

    # Phase 1: Stock bars (needed for signal generation AND underlying prices)
    if do_stocks:
        print("Phase 1: Stock Minute Bars")
        print("-" * 40)
        stock_totals = download_all_stocks(from_date, to_date, tickers)
        total_stock = sum(stock_totals.values())
        print(f"\n  Total stock bars: {total_stock:,}")
        print()

    # Phase 2: OPRA options (needs stock cache for underlying prices)
    if do_options:
        print("Phase 2: OPRA Options")
        print("-" * 40)
        opt_totals = download_all_options(
            from_date, to_date, tickers,
            max_dte=args.max_dte,
            strike_range_pct=args.strike_range,
        )
        total_opts = sum(opt_totals.values())
        print(f"\n  Total options bars: {total_opts:,}")
        print()

    elapsed = time.time() - t_start

    print("=" * 60)
    print("Download Complete")
    print("=" * 60)
    print(f"  Time: {elapsed / 60:.1f} minutes")

    # Disk usage
    total_size = 0
    for tk in tickers:
        for subdir in ["1min", "options_1min"]:
            d = CACHE_DIR / tk / subdir
            if d.exists():
                for f in d.glob("*.json"):
                    total_size += f.stat().st_size
    print(f"  Total disk: {total_size / 1024 / 1024 / 1024:.1f} GB")


if __name__ == "__main__":
    main()
