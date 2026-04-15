"""Download 1-min bars for the full flat_files_s3 universe from Polygon REST API.

Usage:
    python scripts/download_universe_polygon.py --from 2026-04-13 --to 2026-04-14 --delay 0.3

Reads the list of tickers from data/flat_files_s3/ (one directory per ticker).
Appends new bars into the existing monthly JSON files, deduplicating by timestamp.

Output format matches existing flat_files_s3 structure:
    data/flat_files_s3/{TICKER}/1min/{TICKER}_1min_{YYYY-MM}.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests

BASE_URL = "https://api.polygon.io"
FLAT_FILES_DIR = Path("data/flat_files_s3")
LIMIT = 5000


def get_universe() -> list[str]:
    """Return sorted list of tickers from flat_files_s3 directories."""
    tickers = []
    for p in sorted(FLAT_FILES_DIR.iterdir()):
        if p.is_dir() and not p.name.endswith("_backup"):
            tickers.append(p.name)
    return tickers


def fetch_bars(ticker: str, from_date: str, to_date: str, api_key: str) -> list[dict]:
    """Fetch 1-min bars from Polygon with automatic pagination."""
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{from_date}/{to_date}"
    params = {
        "apiKey": api_key,
        "limit": str(LIMIT),
        "sort": "asc",
        "adjusted": "true",
    }
    all_results = []

    while url:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            print(f"  [rate limit] sleeping 60s for {ticker}", flush=True)
            time.sleep(60)
            continue
        if resp.status_code != 200:
            print(f"  [error {resp.status_code}] {ticker}: {resp.text[:200]}", flush=True)
            return []

        data = resp.json()
        results = data.get("results") or []
        all_results.extend(results)

        # Handle pagination
        next_url = data.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": api_key}  # next_url already has other params
        else:
            break

    return all_results


def polygon_to_bar(ticker: str, r: dict) -> dict:
    """Convert Polygon result dict to flat_files_s3 bar format."""
    ts_ms = r.get("t", 0)
    dt = datetime.fromtimestamp(ts_ms / 1000)
    return {
        "ticker": ticker,
        "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "date": dt.strftime("%Y-%m-%d"),
        "open": float(r.get("o", 0)),
        "high": float(r.get("h", 0)),
        "low": float(r.get("l", 0)),
        "close": float(r.get("c", 0)),
        "volume": int(r.get("v", 0)),
        "vwap": float(r.get("vw", 0)),
        "transactions": int(r.get("n", 0)),
    }


def get_monthly_file(ticker: str, year: int, month: int) -> Path:
    return FLAT_FILES_DIR / ticker / "1min" / f"{ticker}_1min_{year:04d}-{month:02d}.json"


def load_existing(path: Path) -> list[dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_bars(path: Path, bars: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(bars, f)


def merge_bars(existing: list[dict], new_bars: list[dict]) -> tuple[list[dict], int]:
    """Merge new bars into existing, deduplicating by timestamp. Returns (merged, added_count)."""
    existing_ts = {b["timestamp"] for b in existing}
    to_add = [b for b in new_bars if b["timestamp"] not in existing_ts]
    merged = sorted(existing + to_add, key=lambda b: b["timestamp"])
    return merged, len(to_add)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download 1-min bars from Polygon")
    parser.add_argument("--from", dest="from_date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between requests")
    parser.add_argument("--ticker", help="Single ticker (for testing)")
    args = parser.parse_args()

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    tickers = [args.ticker] if args.ticker else get_universe()
    print(f"Universe: {len(tickers)} tickers | {args.from_date} → {args.to_date}", flush=True)

    # Group the date range by month so we know which monthly files to touch
    from_dt = date.fromisoformat(args.from_date)
    to_dt = date.fromisoformat(args.to_date)

    total_tickers_updated = 0
    total_bars_added = 0
    skipped = 0

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:4d}/{len(tickers)}] {ticker}", end=" ", flush=True)

        raw = fetch_bars(ticker, args.from_date, args.to_date, api_key)
        if not raw:
            print("no data", flush=True)
            skipped += 1
            time.sleep(args.delay)
            continue

        # Convert all bars
        new_bars = [polygon_to_bar(ticker, r) for r in raw]

        # Group new bars by (year, month)
        from collections import defaultdict
        by_month: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for bar in new_bars:
            d = date.fromisoformat(bar["date"])
            by_month[(d.year, d.month)].append(bar)

        ticker_added = 0
        for (year, month), month_bars in by_month.items():
            path = get_monthly_file(ticker, year, month)
            existing = load_existing(path)
            merged, added = merge_bars(existing, month_bars)
            if added > 0:
                save_bars(path, merged)
            ticker_added += added

        print(f"{len(new_bars)} fetched, {ticker_added} new bars added", flush=True)
        total_bars_added += ticker_added
        if ticker_added > 0:
            total_tickers_updated += 1

        time.sleep(args.delay)

    print()
    print("=" * 60)
    print(f"Done.")
    print(f"  Tickers processed : {len(tickers) - skipped}/{len(tickers)}")
    print(f"  Tickers updated   : {total_tickers_updated}")
    print(f"  New bars added    : {total_bars_added:,}")
    print(f"  Skipped (no data) : {skipped}")


if __name__ == "__main__":
    main()
