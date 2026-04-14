"""Historical tick data downloader + FLUX backtest harness.

Downloads trade ticks and NBBO quotes from Polygon's /v3/trades and
/v3/quotes endpoints for specific date ranges, caches locally as
JSON, and replays through the FLUX engine to compute what FLUX
signals would have been at each historical trade entry.

Usage:
    python -m flowedge.scanner.flux.historical
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv

from flowedge.scanner.flux.consumer import PolygonTradeConsumer
from flowedge.scanner.flux.engine import (
    _classify_block_bias,
    _classify_trades_lee_ready,
    _compute_cumulative_delta,
    _compute_quote_imbalance,
    _detect_block_prints,
    _detect_divergence,
    _determine_bias,
    _score_flux,
)
from flowedge.scanner.flux.schemas import (
    FLUXSignal,
    NBBOQuote,
    TradeTick,
)

logger = structlog.get_logger()

CACHE_DIR = Path("data/flux_ticks")

# Polygon /v3/trades max results per request
_PAGE_LIMIT = 50000

# Rate limit courtesy — 200ms between requests
_RATE_LIMIT_SLEEP = 0.2


# ── Cache Layer ─────────────────────────────────────────────────


def _cache_path(ticker: str, trade_date: str, data_type: str) -> Path:
    """Build cache file path: data/flux_ticks/{ticker}/{date}_{type}.json"""
    return CACHE_DIR / ticker / f"{trade_date}_{data_type}.json"


def _is_cached(ticker: str, trade_date: str, data_type: str) -> bool:
    path = _cache_path(ticker, trade_date, data_type)
    return path.exists() and path.stat().st_size > 100


def _load_cached_trades(ticker: str, trade_date: str) -> list[TradeTick]:
    path = _cache_path(ticker, trade_date, "trades")
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [TradeTick(**t) for t in data]


def _load_cached_quotes(ticker: str, trade_date: str) -> list[NBBOQuote]:
    path = _cache_path(ticker, trade_date, "quotes")
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [NBBOQuote(**q) for q in data]


def _save_cached(
    ticker: str, trade_date: str, data_type: str, items: list[Any],
) -> None:
    path = _cache_path(ticker, trade_date, data_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        [item.model_dump() for item in items],
        default=str,
    ))


# ── Historical Downloader ──────────────────────────────────────


async def download_day_trades(
    consumer: PolygonTradeConsumer,
    ticker: str,
    trade_date: str,
) -> list[TradeTick]:
    """Download all trades for a ticker on a given date.

    Paginates through Polygon /v3/trades with 50K per page.
    Caches results locally to avoid re-downloading.

    Args:
        consumer: Polygon consumer instance
        ticker: e.g. "SPY"
        trade_date: YYYY-MM-DD format

    Returns:
        List of TradeTick sorted by timestamp.
    """
    if _is_cached(ticker, trade_date, "trades"):
        trades = _load_cached_trades(ticker, trade_date)
        logger.info(
            "trades_loaded_from_cache",
            ticker=ticker,
            date=trade_date,
            count=len(trades),
        )
        return trades

    # Build timestamp range for full trading day (9:30-16:00 ET)
    dt = date.fromisoformat(trade_date)
    # 9:30 ET = 13:30 UTC (EDT) or 14:30 UTC (EST)
    # Use 13:00 UTC to be safe (catches pre-market)
    start_dt = datetime(dt.year, dt.month, dt.day, 13, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(dt.year, dt.month, dt.day, 21, 0, 0, tzinfo=timezone.utc)

    start_ns = str(int(start_dt.timestamp() * 1_000_000_000))
    end_ns = str(int(end_dt.timestamp() * 1_000_000_000))

    all_trades: list[TradeTick] = []
    cursor_ns = start_ns
    page = 0

    while True:
        page += 1
        client = await consumer._ensure_client()
        params = {
            "apiKey": consumer._api_key,
            "timestamp.gte": cursor_ns,
            "timestamp.lte": end_ns,
            "limit": str(_PAGE_LIMIT),
            "sort": "timestamp",
            "order": "asc",
        }

        resp = await client.get(
            f"{consumer._base_url}/v3/trades/{ticker}",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break

        # Exclude condition codes
        from flowedge.scanner.flux.consumer import _EXCLUDE_CONDITIONS
        for r in results:
            conditions = r.get("conditions", []) or []
            if any(c in _EXCLUDE_CONDITIONS for c in conditions):
                continue
            size = int(r.get("size", 0))
            if size <= 0:
                continue
            all_trades.append(TradeTick(
                price=float(r.get("price", 0)),
                size=size,
                timestamp=int(r.get("sip_timestamp", r.get("participant_timestamp", 0))),
                conditions=conditions,
                exchange=int(r.get("exchange", 0)),
            ))

        logger.debug(
            "trades_page",
            ticker=ticker,
            date=trade_date,
            page=page,
            raw=len(results),
            total=len(all_trades),
        )

        # Paginate: if we got a full page, advance cursor
        if len(results) < _PAGE_LIMIT:
            break

        # Use last trade timestamp as new cursor
        last_ts = results[-1].get("sip_timestamp", results[-1].get("participant_timestamp", 0))
        cursor_ns = str(int(last_ts) + 1)  # +1 ns to avoid duplicates

        await asyncio.sleep(_RATE_LIMIT_SLEEP)

    # Cache
    _save_cached(ticker, trade_date, "trades", all_trades)
    logger.info(
        "trades_downloaded",
        ticker=ticker,
        date=trade_date,
        pages=page,
        total=len(all_trades),
    )
    return all_trades


async def download_day_quotes(
    consumer: PolygonTradeConsumer,
    ticker: str,
    trade_date: str,
) -> list[NBBOQuote]:
    """Download all NBBO quotes for a ticker on a given date.

    Same pagination approach as trades. Quotes are much denser
    than trades, so we sample every 10th quote to keep cache sizes
    manageable while still having sufficient midpoint resolution
    for Lee-Ready classification.
    """
    if _is_cached(ticker, trade_date, "quotes"):
        quotes = _load_cached_quotes(ticker, trade_date)
        logger.info(
            "quotes_loaded_from_cache",
            ticker=ticker,
            date=trade_date,
            count=len(quotes),
        )
        return quotes

    dt = date.fromisoformat(trade_date)
    start_dt = datetime(dt.year, dt.month, dt.day, 13, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(dt.year, dt.month, dt.day, 21, 0, 0, tzinfo=timezone.utc)

    start_ns = str(int(start_dt.timestamp() * 1_000_000_000))
    end_ns = str(int(end_dt.timestamp() * 1_000_000_000))

    all_quotes: list[NBBOQuote] = []
    cursor_ns = start_ns
    page = 0
    sample_counter = 0

    while True:
        page += 1
        client = await consumer._ensure_client()
        params = {
            "apiKey": consumer._api_key,
            "timestamp.gte": cursor_ns,
            "timestamp.lte": end_ns,
            "limit": str(_PAGE_LIMIT),
            "sort": "timestamp",
            "order": "asc",
        }

        resp = await client.get(
            f"{consumer._base_url}/v3/quotes/{ticker}",
            params=params,
        )

        # Historical quotes may not be available on all Polygon plans
        if resp.status_code == 403:
            logger.debug(
                "quotes_not_available",
                ticker=ticker,
                date=trade_date,
                hint="Polygon plan may not include historical quotes",
            )
            _save_cached(ticker, trade_date, "quotes", [])
            return []

        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            sample_counter += 1
            # Sample every 10th quote (quotes are extremely dense)
            if sample_counter % 10 != 0:
                continue

            bid = float(r.get("bid_price", 0))
            ask = float(r.get("ask_price", 0))
            if bid <= 0 or ask <= 0 or ask < bid:
                continue

            all_quotes.append(NBBOQuote(
                bid=bid,
                bid_size=int(r.get("bid_size", 0)),
                ask=ask,
                ask_size=int(r.get("ask_size", 0)),
                timestamp=int(r.get("sip_timestamp", r.get("participant_timestamp", 0))),
            ))

        logger.debug(
            "quotes_page",
            ticker=ticker,
            date=trade_date,
            page=page,
            raw=len(results),
            sampled=len(all_quotes),
        )

        if len(results) < _PAGE_LIMIT:
            break

        last_ts = results[-1].get("sip_timestamp", results[-1].get("participant_timestamp", 0))
        cursor_ns = str(int(last_ts) + 1)

        await asyncio.sleep(_RATE_LIMIT_SLEEP)

    _save_cached(ticker, trade_date, "quotes", all_quotes)
    logger.info(
        "quotes_downloaded",
        ticker=ticker,
        date=trade_date,
        pages=page,
        total=len(all_quotes),
    )
    return all_quotes


# ── FLUX Replay Engine ─────────────────────────────────────────


def replay_flux_at_timestamp(
    trades: list[TradeTick],
    quotes: list[NBBOQuote],
    ticker: str,
    entry_timestamp_ns: int,
    window_minutes: int = 15,
    price_change_pct: float = 0.0,
) -> FLUXSignal:
    """Replay FLUX analysis at a specific historical timestamp.

    Looks back `window_minutes` from the entry timestamp,
    classifies trades with Lee-Ready, and computes what the
    FLUX signal would have been at that moment.

    Args:
        trades: Full day of trade ticks
        quotes: Full day of NBBO quotes
        ticker: Ticker symbol
        entry_timestamp_ns: Nanosecond timestamp of trade entry
        window_minutes: How far back to look (default 15 min)
        price_change_pct: Price change for divergence detection

    Returns:
        FLUXSignal as it would have appeared at entry time.
    """
    from flowedge.config.settings import get_settings
    settings = get_settings()

    window_ns = window_minutes * 60 * 1_000_000_000
    window_5m_ns = 5 * 60 * 1_000_000_000

    start_ns = entry_timestamp_ns - window_ns
    start_5m_ns = entry_timestamp_ns - window_5m_ns

    # Filter trades and quotes to the window
    trades_15m = [t for t in trades if start_ns <= t.timestamp <= entry_timestamp_ns]
    trades_5m = [t for t in trades if start_5m_ns <= t.timestamp <= entry_timestamp_ns]
    quotes_15m = [q for q in quotes if start_ns <= q.timestamp <= entry_timestamp_ns]
    quotes_5m = [q for q in quotes if start_5m_ns <= q.timestamp <= entry_timestamp_ns]

    # Lee-Ready classification
    classified_5m = _classify_trades_lee_ready(trades_5m, quotes_5m)
    classified_15m = _classify_trades_lee_ready(trades_15m, quotes_15m)

    # Cumulative delta
    delta_5m = _compute_cumulative_delta(classified_5m, 5)
    delta_15m = _compute_cumulative_delta(classified_15m, 15)

    # Quote imbalance
    quote_imbalance = _compute_quote_imbalance(quotes_5m, 5)

    # Block prints
    blocks = _detect_block_prints(
        classified_15m, ticker,
        min_multiple=settings.flux_block_min_multiple,
    )
    block_bias = _classify_block_bias(blocks)

    # Divergence
    divergence = _detect_divergence(delta_5m, delta_15m, price_change_pct)

    # Bias
    bias = _determine_bias(delta_5m, quote_imbalance, block_bias)

    # Score
    strength, rationale = _score_flux(
        delta_5m, delta_15m, quote_imbalance,
        blocks, divergence, settings,
    )

    return FLUXSignal(
        ticker=ticker,
        strength=strength,
        bias=bias,
        delta_5m=delta_5m,
        delta_15m=delta_15m,
        quote_imbalance=quote_imbalance,
        block_prints=blocks[:10],
        block_bias=block_bias,
        divergence=divergence,
        rationale=rationale,
        detected_at=datetime.now(),
    )


# ── Backtest Runner ────────────────────────────────────────────


async def run_flux_backtest(
    backtest_file: str = "data/backtest/backtest_scalp_real.json",
    output_file: str = "data/backtest/flux_backtest_results.json",
) -> dict[str, Any]:
    """Run FLUX analysis on all historical trades and measure impact.

    For each trade in the backtest:
    1. Download tick data for the entry date (cached after first run)
    2. Replay FLUX at the entry timestamp
    3. Record the FLUX signal alongside the trade outcome
    4. Compute aggregate stats: WR with/without FLUX confirmation

    Returns:
        Dict with full results and P&L impact analysis.
    """
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        logger.error("POLYGON_API_KEY not set")
        return {"error": "POLYGON_API_KEY not set"}

    # Load backtest trades
    bt_path = Path(backtest_file)
    if not bt_path.exists():
        logger.error("backtest_file_not_found", path=backtest_file)
        return {"error": f"File not found: {backtest_file}"}

    bt_data = json.loads(bt_path.read_text())
    trades = bt_data.get("trades", [])
    logger.info("flux_backtest_starting", total_trades=len(trades))

    consumer = PolygonTradeConsumer(api_key)
    results: list[dict[str, Any]] = []

    # Collect unique (ticker, date) pairs to download
    ticker_dates: set[tuple[str, str]] = set()
    for trade in trades:
        ticker_dates.add((trade["ticker"], trade["entry_date"]))

    logger.info(
        "downloading_tick_data",
        unique_days=len(ticker_dates),
        tickers=sorted({td[0] for td in ticker_dates}),
    )

    # Download all tick data (cached after first run)
    tick_cache: dict[str, tuple[list[TradeTick], list[NBBOQuote]]] = {}
    downloaded = 0
    for ticker, trade_date in sorted(ticker_dates):
        cache_key = f"{ticker}_{trade_date}"
        try:
            day_trades = await download_day_trades(consumer, ticker, trade_date)
            day_quotes = await download_day_quotes(consumer, ticker, trade_date)
            tick_cache[cache_key] = (day_trades, day_quotes)
            downloaded += 1
            if downloaded % 10 == 0:
                logger.info("download_progress", completed=downloaded, total=len(ticker_dates))
        except Exception as e:
            logger.warning(
                "tick_download_failed",
                ticker=ticker,
                date=trade_date,
                error=str(e),
            )
            tick_cache[cache_key] = ([], [])

        await asyncio.sleep(_RATE_LIMIT_SLEEP)

    await consumer.close()

    # Replay FLUX for each trade
    logger.info("replaying_flux", total_trades=len(trades))
    for i, trade in enumerate(trades):
        ticker = trade["ticker"]
        trade_date = trade["entry_date"]
        cache_key = f"{ticker}_{trade_date}"

        day_trades, day_quotes = tick_cache.get(cache_key, ([], []))
        if not day_trades:
            results.append({
                **trade,
                "flux_available": False,
                "flux_strength": 0.0,
                "flux_bias": "unknown",
                "flux_divergence": "none",
                "flux_aggression": 0.5,
                "flux_blocks": 0,
            })
            continue

        # Estimate entry timestamp: use 10:00 ET as default
        # (most scalp entries happen in first 30 min after open)
        dt = date.fromisoformat(trade_date)
        entry_approx = datetime(
            dt.year, dt.month, dt.day, 14, 0, 0,
            tzinfo=timezone.utc,
        )
        entry_ns = int(entry_approx.timestamp() * 1_000_000_000)

        # If we have trades, use the median timestamp as entry estimate
        if day_trades:
            mid_idx = len(day_trades) // 4  # ~25% into the day (morning)
            entry_ns = day_trades[mid_idx].timestamp

        # Compute price change from bar data
        price_change = 0.0
        if day_trades and len(day_trades) > 100:
            early_price = day_trades[0].price
            entry_price = day_trades[min(len(day_trades) // 4, len(day_trades) - 1)].price
            if early_price > 0:
                price_change = (entry_price - early_price) / early_price

        try:
            flux = replay_flux_at_timestamp(
                day_trades, day_quotes, ticker,
                entry_ns, window_minutes=15,
                price_change_pct=price_change,
            )
            results.append({
                **trade,
                "flux_available": True,
                "flux_strength": flux.strength,
                "flux_bias": flux.bias.value,
                "flux_divergence": flux.divergence.value,
                "flux_aggression": (
                    flux.delta_5m.aggression_ratio if flux.delta_5m else 0.5
                ),
                "flux_net_delta": flux.delta_5m.net_delta if flux.delta_5m else 0,
                "flux_blocks": len(flux.block_prints),
                "flux_rationale": flux.rationale,
            })
        except Exception as e:
            logger.warning(
                "flux_replay_failed",
                ticker=ticker,
                date=trade_date,
                error=str(e),
            )
            results.append({
                **trade,
                "flux_available": False,
                "flux_strength": 0.0,
                "flux_bias": "unknown",
                "flux_divergence": "none",
                "flux_aggression": 0.5,
                "flux_blocks": 0,
            })

        if (i + 1) % 20 == 0:
            logger.info("replay_progress", completed=i + 1, total=len(trades))

    # ── Compute Aggregate Stats ─────────────────────────────────

    analysis = _analyze_flux_impact(results, bt_data)

    # Save full results
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "analysis": analysis,
        "trades": results,
        "source": backtest_file,
        "generated_at": datetime.now().isoformat(),
    }, indent=2, default=str))

    logger.info("flux_backtest_complete", output=output_file)
    return analysis


def _analyze_flux_impact(
    results: list[dict[str, Any]],
    bt_data: dict[str, Any],
) -> dict[str, Any]:
    """Compute the P&L impact of FLUX confirmation on historical trades."""
    baseline_wr = bt_data.get("win_rate", 0)
    baseline_trades = bt_data.get("total_trades", 0)

    # Split by FLUX availability
    flux_available = [r for r in results if r.get("flux_available")]
    no_flux = [r for r in results if not r.get("flux_available")]

    # Split by FLUX confirmation level
    confirmed = [r for r in flux_available if r.get("flux_strength", 0) >= 5.0]
    strong = [r for r in flux_available if r.get("flux_strength", 0) >= 7.0]
    weak = [r for r in flux_available if r.get("flux_strength", 0) < 5.0]

    # Split by bias
    buy_bias = [r for r in flux_available if "buy" in str(r.get("flux_bias", ""))]
    sell_bias = [r for r in flux_available if "sell" in str(r.get("flux_bias", ""))]

    def _wr(trades: list[dict[str, Any]]) -> float:
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.get("outcome") == "win")
        return wins / len(trades)

    def _avg_pnl(trades: list[dict[str, Any]]) -> float:
        if not trades:
            return 0.0
        return sum(float(t.get("pnl_pct", 0)) for t in trades) / len(trades)

    def _total_pnl(trades: list[dict[str, Any]]) -> float:
        return sum(float(t.get("pnl_pct", 0)) for t in trades)

    analysis = {
        "baseline": {
            "total_trades": baseline_trades,
            "win_rate": baseline_wr,
            "total_pnl_pct": bt_data.get("total_pnl_pct", 0),
        },
        "flux_coverage": {
            "flux_available": len(flux_available),
            "no_flux": len(no_flux),
            "coverage_pct": len(flux_available) / len(results) if results else 0,
        },
        "confirmed_vs_weak": {
            "confirmed_trades": len(confirmed),
            "confirmed_wr": round(_wr(confirmed), 4),
            "confirmed_avg_pnl": round(_avg_pnl(confirmed), 2),
            "confirmed_total_pnl": round(_total_pnl(confirmed), 2),
            "strong_trades": len(strong),
            "strong_wr": round(_wr(strong), 4),
            "strong_avg_pnl": round(_avg_pnl(strong), 2),
            "weak_trades": len(weak),
            "weak_wr": round(_wr(weak), 4),
            "weak_avg_pnl": round(_avg_pnl(weak), 2),
            "weak_total_pnl": round(_total_pnl(weak), 2),
        },
        "by_bias": {
            "buy_bias_trades": len(buy_bias),
            "buy_bias_wr": round(_wr(buy_bias), 4),
            "buy_bias_avg_pnl": round(_avg_pnl(buy_bias), 2),
            "sell_bias_trades": len(sell_bias),
            "sell_bias_wr": round(_wr(sell_bias), 4),
            "sell_bias_avg_pnl": round(_avg_pnl(sell_bias), 2),
        },
        "flux_edge": {
            "wr_lift_confirmed": round(_wr(confirmed) - baseline_wr, 4),
            "wr_lift_strong": round(_wr(strong) - baseline_wr, 4),
            "pnl_if_only_confirmed": round(_total_pnl(confirmed), 2),
            "pnl_if_only_weak": round(_total_pnl(weak), 2),
            "trades_saved_by_veto": len([
                t for t in sell_bias
                if t.get("outcome") == "loss" and t.get("flux_strength", 0) >= 7.0
            ]),
        },
    }

    # Summary
    confirmed_wr = _wr(confirmed)
    weak_wr = _wr(weak)
    lift = confirmed_wr - baseline_wr

    analysis["summary"] = (
        f"FLUX data available on {len(flux_available)}/{len(results)} trades "
        f"({len(flux_available) / len(results) * 100:.0f}%). "
        f"Confirmed (score>=5) WR: {confirmed_wr:.1%} vs baseline {baseline_wr:.1%} "
        f"(lift: {lift:+.1%}). "
        f"Strong (score>=7) WR: {_wr(strong):.1%} ({len(strong)} trades). "
        f"Weak (score<5) WR: {weak_wr:.1%} ({len(weak)} trades). "
        f"If only trading FLUX-confirmed setups: "
        f"${_total_pnl(confirmed):.0f}% total PnL on {len(confirmed)} trades."
    )

    return analysis


# ── CLI Entry Point ────────────────────────────────────────────


async def main() -> None:
    """Run the FLUX historical backtest."""
    print("=" * 65)
    print("FLUX HISTORICAL BACKTEST")
    print("=" * 65)
    print("Downloads tick data from Polygon, replays FLUX at each entry,")
    print("and measures WR lift from order flow confirmation.")
    print("=" * 65)
    print()

    analysis = await run_flux_backtest()

    if "error" in analysis:
        print(f"\nERROR: {analysis['error']}")
        return

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    baseline = analysis["baseline"]
    coverage = analysis["flux_coverage"]
    cvw = analysis["confirmed_vs_weak"]
    bias = analysis["by_bias"]
    edge = analysis["flux_edge"]

    print(f"\nBaseline:  {baseline['total_trades']} trades, "
          f"WR {baseline['win_rate']:.1%}, "
          f"PnL {baseline['total_pnl_pct']:.1f}%")
    total_cov = coverage['flux_available'] + coverage['no_flux']
    print(f"Coverage:  {coverage['flux_available']}/{total_cov} "
          f"trades have tick data ({coverage['coverage_pct']:.0%})")

    print("\n--- FLUX Confirmed (score >= 5.0) ---")
    print(f"  Trades:    {cvw['confirmed_trades']}")
    print(f"  Win Rate:  {cvw['confirmed_wr']:.1%}  "
          f"(vs {baseline['win_rate']:.1%} baseline = "
          f"{edge['wr_lift_confirmed']:+.1%} lift)")
    print(f"  Avg PnL:   {cvw['confirmed_avg_pnl']:+.1f}%")
    print(f"  Total PnL: {cvw['confirmed_total_pnl']:+.1f}%")

    print("\n--- FLUX Strong (score >= 7.0) ---")
    print(f"  Trades:    {cvw['strong_trades']}")
    print(f"  Win Rate:  {cvw['strong_wr']:.1%}  "
          f"(vs {baseline['win_rate']:.1%} baseline = "
          f"{edge['wr_lift_strong']:+.1%} lift)")
    print(f"  Avg PnL:   {cvw['strong_avg_pnl']:+.1f}%")

    print("\n--- FLUX Weak (score < 5.0) ---")
    print(f"  Trades:    {cvw['weak_trades']}")
    print(f"  Win Rate:  {cvw['weak_wr']:.1%}")
    print(f"  Avg PnL:   {cvw['weak_avg_pnl']:+.1f}%")
    print(f"  Total PnL: {cvw['weak_total_pnl']:+.1f}%")

    print("\n--- By FLUX Bias ---")
    print(f"  Buy bias:  {bias['buy_bias_trades']} trades, "
          f"WR {bias['buy_bias_wr']:.1%}, "
          f"avg PnL {bias['buy_bias_avg_pnl']:+.1f}%")
    print(f"  Sell bias: {bias['sell_bias_trades']} trades, "
          f"WR {bias['sell_bias_wr']:.1%}, "
          f"avg PnL {bias['sell_bias_avg_pnl']:+.1f}%")

    print("\n--- Edge Summary ---")
    print(f"  Trades saved by FLUX veto: {edge['trades_saved_by_veto']}")
    print(f"  PnL if only confirmed:     {edge['pnl_if_only_confirmed']:+.1f}%")
    print(f"  PnL from weak trades:      {edge['pnl_if_only_weak']:+.1f}%")

    print(f"\n{analysis.get('summary', '')}")
    print("\nFull results saved to: data/backtest/flux_backtest_results.json")


if __name__ == "__main__":
    asyncio.run(main())
