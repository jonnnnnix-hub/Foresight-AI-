#!/usr/bin/env python3
"""Download OPRA options minute bars for Trident (SPY, QQQ, IWM).

Downloads BOTH calls and puts from Massive S3 (Polygon OPRA flat files).
Caches to data/flat_files_s3/{TICKER}/options_1min/.
Resume-safe: skips dates already cached.

Usage:
    .venv/bin/python scripts/download_trident_opra.py
    .venv/bin/python scripts/download_trident_opra.py --ticker SPY
    .venv/bin/python scripts/download_trident_opra.py --from-date 2024-01-02 --to-date 2025-12-31
    .venv/bin/python scripts/download_trident_opra.py --check-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.data_feeds.options_s3 import (  # noqa: E402
    OptionsS3Downloader,
    get_underlying_closes_from_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trident.download")

TRIDENT_TICKERS = ["SPY", "QQQ", "IWM"]

# ETFs have tighter spreads → narrower strike range needed
# DTE 0-5 captures daily 0DTE + weekly expirations
ETF_STRIKE_RANGE = 0.05   # ±5% of underlying price
ETF_MAX_DTE = 5


def check_data_status(tickers: list[str]) -> None:
    """Report how much OPRA data is already cached per ticker."""
    cache_dir = Path("data/flat_files_s3")
    for ticker in tickers:
        opts_dir = cache_dir / ticker / "options_1min"
        stock_dir = cache_dir / ticker / "1min"

        stock_files = list(stock_dir.glob("*.json")) if stock_dir.exists() else []
        opts_files = list(opts_dir.glob("*.json")) if opts_dir.exists() else []

        # Extract date range from options files
        opt_dates = sorted(
            f.stem.split("_")[-1] for f in opts_files
        )
        first = opt_dates[0] if opt_dates else "none"
        last = opt_dates[-1] if opt_dates else "none"

        logger.info(
            "%s: %d stock bar files, %d OPRA option days cached [%s → %s]",
            ticker, len(stock_files), len(opts_files), first, last,
        )


def download_trident_data(
    tickers: list[str],
    from_date: date,
    to_date: date,
) -> None:
    """Download OPRA data for Trident tickers.

    Downloads BOTH calls AND puts with DTE 0-5 and
    strikes within 5% of underlying price.
    """
    logger.info("=" * 60)
    logger.info("Trident OPRA Download")
    logger.info("Tickers: %s", tickers)
    logger.info("Date range: %s → %s", from_date, to_date)
    logger.info("Strike range: ±%.0f%%", ETF_STRIKE_RANGE * 100)
    logger.info("Max DTE: %d", ETF_MAX_DTE)
    logger.info("Options: calls + puts")
    logger.info("=" * 60)

    # Load underlying close prices from stock bar cache
    logger.info("Loading underlying close prices from cache ...")
    prices_by_date = get_underlying_closes_from_cache(tickers)
    logger.info("Found prices for %d trading days", len(prices_by_date))

    if not prices_by_date:
        logger.error(
            "No stock bar data found. Download stock bars first with "
            "scripts/download_stock_bars.py or similar."
        )
        return

    downloader = OptionsS3Downloader()

    # Count days to process
    total_trading_days = 0
    current = from_date
    while current <= to_date:
        if current.weekday() < 5:
            total_trading_days += 1
        current += timedelta(days=1)
    logger.info("Total trading days in range: %d", total_trading_days)

    # Download!
    totals = downloader.download_options_range(
        from_date=from_date,
        to_date=to_date,
        underlying_tickers=tickers,
        underlying_prices_by_date=prices_by_date,
        max_dte=ETF_MAX_DTE,
        strike_range_pct=ETF_STRIKE_RANGE,
    )

    logger.info("=" * 60)
    logger.info("Download complete!")
    for ticker, count in totals.items():
        logger.info("  %s: %d option bars cached", ticker, count)
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OPRA options data for Trident backtesting",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Download for a single ticker (default: SPY, QQQ, IWM)",
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default="2022-04-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default="2026-04-10",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check current data status, don't download",
    )
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else TRIDENT_TICKERS

    if args.check_only:
        check_data_status(tickers)
        return

    from_date = date.fromisoformat(args.from_date)
    to_date = date.fromisoformat(args.to_date)

    download_trident_data(tickers, from_date, to_date)


if __name__ == "__main__":
    main()
