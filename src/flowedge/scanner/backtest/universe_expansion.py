"""Universe expansion — download and screen 200+ tickers.

Downloads minute bars from Massive S3 for a large universe of liquid US equities,
runs quick screening (fast grid search), keeps viable tickers, and creates
specialist bots for winners.

Pipeline:
1. Download 1min bars for each ticker batch from S3 (no rate limits)
2. Run quick single-pass backtest with default params to filter viability
3. Run full grid search on viable tickers (WR > 40% on initial screen)
4. Create specialist configs for winners (WR > 55% after optimization)
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import structlog

from flowedge.scanner.backtest.shares_engine import _load_daily, run_shares_backtest
from flowedge.scanner.backtest.specialist import SpecialistConfig, save_specialists

logger = structlog.get_logger()

CACHE_DIR = Path("data/flat_files_s3")
OUTPUT_DIR = Path("data/optimizer")

# ── Universe Definition ──────────────────────────────────────────────────────

# 200+ high-liquidity US equities across sectors
EXPANDED_UNIVERSE: dict[str, list[str]] = {
    # Already cached (33 tickers)
    "existing": [
        "AAPL", "AMD", "AMZN", "ARM", "AVGO", "BAC", "COIN", "COST", "CRM",
        "DIA", "GOOGL", "HOOD", "INTC", "IWM", "JPM", "META", "MSFT", "MSTR",
        "NFLX", "NVDA", "PLTR", "QQQ", "RDDT", "SMCI", "SOFI", "SPY", "TSLA",
        "V", "WMT", "XLE", "XLF", "XLK", "XLV",
    ],
    # Semiconductors
    "semiconductors": [
        "MRVL", "MU", "QCOM", "TXN", "LRCX", "KLAC", "ASML", "AMAT", "ON",
        "MCHP", "ADI", "SWKS", "MPWR",
    ],
    # Big tech / software
    "tech_software": [
        "ORCL", "ADBE", "NOW", "SNOW", "PANW", "CRWD", "ZS", "DDOG",
        "TEAM", "WDAY", "SPLK", "FTNT", "NET", "MDB", "SHOP", "SQ", "UBER",
        "DASH", "ABNB", "LYFT", "PINS", "SNAP", "ROKU", "TTD", "U", "RBLX",
        "PYPL",
    ],
    # Financials
    "financials": [
        "GS", "MS", "C", "WFC", "BLK", "SCHW", "AXP", "USB", "PNC",
        "TFC", "COF", "ALLY", "FIS", "FISV", "MA", "ICE", "CME", "NDAQ",
    ],
    # Healthcare
    "healthcare": [
        "UNH", "JNJ", "PFE", "MRNA", "ABBV", "LLY", "BMY", "MRK", "AMGN",
        "GILD", "BIIB", "REGN", "VRTX", "ISRG", "TMO", "DHR", "ABT",
        "MDT", "SYK", "BSX", "ZTS", "DXCM", "ILMN",
    ],
    # Energy
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "OXY", "DVN", "MPC",
        "VLO", "PSX", "HAL",
    ],
    # Consumer
    "consumer": [
        "SBUX", "MCD", "NKE", "TGT", "HD", "LOW", "TJX", "ROST", "DG",
        "DLTR", "KO", "PEP", "PG", "CL", "KMB", "EL", "LULU", "DECK",
        "BURL", "CMG", "DPZ", "YUM", "DARDEN",
    ],
    # Industrials
    "industrials": [
        "CAT", "DE", "GE", "HON", "LMT", "RTX", "NOC", "BA", "UPS", "FDX",
        "WM", "RSG", "ETN", "EMR", "IR",
    ],
    # REITs / Real Estate
    "reits": [
        "AMT", "PLD", "CCI", "EQIX", "DLR", "O", "SPG", "PSA",
    ],
    # Materials
    "materials": [
        "LIN", "APD", "ECL", "SHW", "NEM", "FCX", "CTVA",
    ],
    # Telecom / Media
    "telecom_media": [
        "T", "VZ", "TMUS", "DIS", "CMCSA", "CHTR", "WBD",
    ],
    # ETFs (additional)
    "etfs": [
        "XLI", "XLY", "XLC", "XLP", "XLU", "XLB", "XLRE",
        "GLD", "SLV", "TLT", "HYG", "EEM", "VXX",
        "ARKK", "ARKF", "ARKG",
        "SMH", "SOXX", "IBB", "KRE", "XBI",
    ],
}


def get_all_new_tickers() -> list[str]:
    """Get all tickers that aren't already cached."""
    existing = set(EXPANDED_UNIVERSE["existing"])
    new_tickers: list[str] = []
    for sector, tickers in EXPANDED_UNIVERSE.items():
        if sector == "existing":
            continue
        for t in tickers:
            if t not in existing:
                new_tickers.append(t)
    return sorted(set(new_tickers))


def get_all_tickers() -> list[str]:
    """Get the full universe (existing + new)."""
    all_tickers: set[str] = set()
    for tickers in EXPANDED_UNIVERSE.values():
        all_tickers.update(tickers)
    return sorted(all_tickers)


# ── Download Pipeline ────────────────────────────────────────────────────────


def download_new_tickers(
    from_date: date | None = None,
    to_date: date | None = None,
    batch_size: int = 20,
) -> dict[str, int]:
    """Download 1min bars for all new tickers from Massive S3.

    Downloads in batches of 20 tickers per S3 day-file scan.
    Returns dict of ticker → bar count downloaded.
    """
    from flowedge.scanner.data_feeds.massive_s3 import MassiveS3Downloader

    new_tickers = get_all_new_tickers()
    logger.info("download_start", new_tickers=len(new_tickers))

    if not from_date:
        from_date = date(2022, 4, 1)
    if not to_date:
        to_date = date(2026, 4, 10)

    downloader = MassiveS3Downloader()
    downloaded: dict[str, int] = {}

    # Download in batches
    for i in range(0, len(new_tickers), batch_size):
        batch = new_tickers[i:i + batch_size]
        logger.info(
            "download_batch",
            batch=i // batch_size + 1,
            tickers=batch,
        )

        try:
            bars = downloader.download_and_cache(
                from_date, to_date, batch,
            )
            for ticker, ticker_bars in bars.items():
                downloaded[ticker] = len(ticker_bars)
        except Exception as e:
            logger.error("download_batch_failed", error=str(e)[:200])
            continue

    logger.info(
        "download_complete",
        tickers_downloaded=len(downloaded),
        total_bars=sum(downloaded.values()),
    )

    return downloaded


# ── Quick Screen ─────────────────────────────────────────────────────────────


def quick_screen_ticker(ticker: str) -> dict[str, Any] | None:
    """Quick screen: run default params on a single ticker.

    Returns basic metrics or None if insufficient data.
    """
    daily = _load_daily(ticker)
    if len(daily) < 60:
        return None

    result = run_shares_backtest(
        mode="precision_shares",
        starting_capital=10_000.0,
        tickers=[ticker],
    )

    if result.total_trades < 5:
        return None

    return {
        "ticker": ticker,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "portfolio_return_pct": result.portfolio_return_pct,
        "profit_factor": result.profit_factor,
        "max_drawdown_pct": result.max_drawdown_pct,
    }


def screen_universe() -> list[dict[str, Any]]:
    """Quick screen all available tickers."""
    available = sorted(
        d.name for d in CACHE_DIR.iterdir()
        if d.is_dir() and (d / "1min").exists()
    )

    logger.info("screening_start", tickers=len(available))
    results: list[dict[str, Any]] = []

    for ticker in available:
        screen = quick_screen_ticker(ticker)
        if screen:
            results.append(screen)
            logger.info(
                "screened",
                ticker=ticker,
                trades=screen["total_trades"],
                wr=screen["win_rate"],
            )

    results.sort(key=lambda r: r["win_rate"], reverse=True)
    return results


# ── Full Pipeline ────────────────────────────────────────────────────────────


def run_full_expansion_pipeline(
    skip_download: bool = False,
    min_screen_wr: float = 0.40,
    min_opt_wr: float = 0.55,
) -> dict[str, Any]:
    """Run the full universe expansion pipeline.

    1. Download new ticker data (unless skip_download=True)
    2. Quick screen all available tickers
    3. Full grid search on viable tickers
    4. Create specialist configs for winners
    5. Save all results
    """
    t0 = time.time()
    pipeline_results: dict[str, Any] = {"steps": []}

    # Step 1: Download
    if not skip_download:
        logger.info("pipeline_step1", step="download")
        downloaded = download_new_tickers()
        pipeline_results["steps"].append({
            "step": "download",
            "tickers_downloaded": len(downloaded),
            "total_bars": sum(downloaded.values()),
        })
    else:
        logger.info("pipeline_step1", step="download_skipped")

    # Step 2: Quick screen
    logger.info("pipeline_step2", step="quick_screen")
    screen_results = screen_universe()
    viable = [r for r in screen_results if r["win_rate"] >= min_screen_wr]

    pipeline_results["steps"].append({
        "step": "quick_screen",
        "total_screened": len(screen_results),
        "viable": len(viable),
    })

    # Print screen results
    print("\n" + "=" * 80)
    print(
        f"QUICK SCREEN — {len(screen_results)} tickers, "
        f"{len(viable)} viable (WR >= {min_screen_wr:.0%})"
    )
    print("=" * 80)
    for r in screen_results[:30]:
        flag = " ***" if r["win_rate"] >= min_screen_wr else ""
        print(
            f"  {r['ticker']:<8} WR={r['win_rate']:.1%}  "
            f"Ret={r['portfolio_return_pct']:+.1f}%  "
            f"PF={r['profit_factor']:.2f}  "
            f"Trades={r['total_trades']}{flag}"
        )
    print("=" * 80 + "\n")

    # Step 3: Full grid search on viable tickers
    logger.info("pipeline_step3", step="grid_search", viable=len(viable))
    from flowedge.scanner.backtest.optimizer import (
        TickerOptResult,
        optimize_shares_ticker,
    )

    opt_results: list[TickerOptResult] = []
    for r in viable:
        ticker = r["ticker"]
        opt = optimize_shares_ticker(ticker)
        if opt:
            opt_results.append(opt)
            logger.info(
                "optimized",
                ticker=ticker,
                baseline_wr=opt.baseline_win_rate,
                optimized_wr=opt.optimized_win_rate,
            )

    opt_results.sort(key=lambda r: r.best_score, reverse=True)

    pipeline_results["steps"].append({
        "step": "grid_search",
        "tickers_optimized": len(opt_results),
    })

    # Step 4: Create specialists for winners
    winners = [o for o in opt_results if o.optimized_win_rate >= min_opt_wr]
    specialists: list[SpecialistConfig] = []
    for opt in winners:
        spec = SpecialistConfig(
            name=f"{opt.ticker}_specialist",
            tickers=[opt.ticker],
            instrument="shares",
            shares_params=opt.best_params,
            baseline_win_rate=opt.baseline_win_rate,
            optimized_win_rate=opt.optimized_win_rate,
            baseline_return_pct=opt.baseline_return_pct,
            optimized_return_pct=opt.optimized_return_pct,
        )
        specialists.append(spec)

    save_specialists(specialists)

    pipeline_results["steps"].append({
        "step": "specialist_creation",
        "specialists_created": len(specialists),
        "tickers": [s.tickers[0] for s in specialists],
    })

    elapsed = time.time() - t0
    pipeline_results["elapsed_seconds"] = round(elapsed, 1)
    pipeline_results["total_specialists"] = len(specialists)

    # Save pipeline results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "expansion_pipeline_results.json").write_text(
        json.dumps(pipeline_results, indent=2),
    )

    # Print summary
    print("\n" + "=" * 80)
    print(f"UNIVERSE EXPANSION COMPLETE — {elapsed:.0f}s")
    print("=" * 80)
    print(f"  Tickers screened:    {len(screen_results)}")
    print(f"  Viable (WR>{min_screen_wr:.0%}):   {len(viable)}")
    print(f"  Optimized:           {len(opt_results)}")
    print(f"  Winners (WR>{min_opt_wr:.0%}):  {len(winners)}")
    print(f"  Specialists created: {len(specialists)}")
    print()
    print("  TOP SPECIALISTS:")
    for s in specialists[:15]:
        print(
            f"    {s.tickers[0]:<8} WR={s.optimized_win_rate:.1%}  "
            f"Ret={s.optimized_return_pct:+.1f}%"
        )
    print("=" * 80 + "\n")

    return pipeline_results


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    from flowedge.config.logging import setup_logging

    setup_logging("INFO")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"

    if cmd == "download":
        download_new_tickers()
    elif cmd == "screen":
        screen_universe()
    elif cmd == "full":
        run_full_expansion_pipeline()
    elif cmd == "local":
        # Skip download, just screen + optimize what's already cached
        run_full_expansion_pipeline(skip_download=True)
    else:
        print("Usage: python -m flowedge.scanner.backtest.universe_expansion "
              "[download|screen|full|local]")
        sys.exit(1)
