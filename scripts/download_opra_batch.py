#!/usr/bin/env python3
"""Download OPRA options data for the next N tickers that don't have it yet.

Used by the nightly cron job to incrementally expand the universe.
Resume-safe: skips tickers that already have OPRA data.

Usage:
    python scripts/download_opra_batch.py --count 100
    python scripts/download_opra_batch.py --count 50 --from-date 2022-04-12 --to-date 2026-04-14
"""
import argparse, csv, gzip, io, json, os, re, sys, time
from datetime import date, datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Load .env so MASSIVE_* vars are available to os.getenv() below
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

import boto3
from botocore.config import Config

CACHE = Path("data/flat_files_s3")
OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def get_next_batch(count):
    has_opra = set()
    has_stock = set()
    for d in CACHE.iterdir():
        if d.name == "flat_files_s3_backup": continue
        if (d / "options_1min").exists() and any((d / "options_1min").glob("*.json")):
            has_opra.add(d.name)
        if (d / "1min").exists() and any((d / "1min").glob("*.json")):
            has_stock.add(d.name)
    need = sorted(has_stock - has_opra)
    return need[:count]


def download_batch(tickers, from_date, to_date):
    client = boto3.client("s3", endpoint_url="https://files.massive.com",
        aws_access_key_id=os.getenv("MASSIVE_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MASSIVE_SECRET_KEY"),
        config=Config(signature_version="s3v4"))

    ticker_set = set(tickers)
    current = from_date
    total_bars = 0
    total_days = 0

    while current <= to_date:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        key = f"us_options_opra/minute_aggs_v1/{current.year}/{current.month:02d}/{current.isoformat()}.csv.gz"
        try:
            resp = client.get_object(Bucket="flatfiles", Key=key)
            raw = resp["Body"].read()
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                text = gz.read().decode("utf-8")

            by_underlying = {}
            for row in csv.DictReader(io.StringIO(text)):
                sym = row.get("ticker", "").removeprefix("O:")
                m = OCC_RE.match(sym)
                if not m: continue
                underlying = m.group(1)
                if underlying not in ticker_set: continue
                exp = datetime.strptime(m.group(2), "%y%m%d").date()
                dte = (exp - current).days
                if dte < 0 or dte > 5: continue
                bar = {
                    "contract": "O:" + sym, "underlying": underlying,
                    "expiration": exp.isoformat(), "strike": int(m.group(4)) / 1000.0,
                    "option_type": m.group(3), "dte": dte, "ts": row.get("window_start", "0"),
                    "date": current.isoformat(),
                    "o": float(row.get("open", 0)), "h": float(row.get("high", 0)),
                    "l": float(row.get("low", 0)), "c": float(row.get("close", 0)),
                    "v": int(float(row.get("volume", 0))),
                }
                by_underlying.setdefault(underlying, []).append(bar)

            day_bars = 0
            for underlying, bars in by_underlying.items():
                cache_dir = CACHE / underlying / "options_1min"
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / f"{underlying}_options_1min_{current.isoformat()}.json").write_text(
                    json.dumps(bars, default=str))
                day_bars += len(bars)

            total_bars += day_bars
            total_days += 1
            if total_days % 50 == 0:
                print(f"[{total_days}] {current} — {day_bars} bars, {len(by_underlying)} tickers | cumulative: {total_bars:,}")

        except Exception as e:
            if "NoSuchKey" not in str(e) and "404" not in str(e):
                print(f"  [{current}] {type(e).__name__}: {e}")
            # skip missing/holiday/forbidden days silently

        current += timedelta(days=1)

    print(f"DONE: {total_days} days, {total_bars:,} bars for {len(tickers)} tickers")
    return total_bars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--from-date", default="2022-04-12")
    parser.add_argument("--to-date", default="2026-04-14")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Explicit ticker list (skips auto-detection)")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"Explicit tickers: {tickers}")
    else:
        tickers = get_next_batch(args.count)
        print(f"Downloading OPRA for {len(tickers)} tickers: {tickers[:10]}...")

    download_batch(tickers, date.fromisoformat(args.from_date), date.fromisoformat(args.to_date))


if __name__ == "__main__":
    main()
