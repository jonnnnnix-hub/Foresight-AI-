"""PHANTOM — runs ghost trades through history to validate the edge.

Uses Polygon intraday bars + Orats historical earnings to simulate:
1. Signal fires on day T (IV rank, catalyst proximity)
2. Buy OTM option at day T close
3. Hold through catalyst or max N days
4. Exit at expiration or when target/stop hit
5. Track P&L, win rate, profit factor
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Any

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.backtest.schemas import (
    BacktestResult,
    BacktestTrade,
    TradeOutcome,
)
from flowedge.scanner.providers.polygon import PolygonProvider
from flowedge.scanner.providers.registry import ProviderRegistry

logger = structlog.get_logger()


async def _fetch_price_history(
    polygon: PolygonProvider,
    ticker: str,
    from_date: str,
    to_date: str,
) -> list[dict[str, Any]]:
    """Fetch daily bars for a ticker over a date range."""
    data = await polygon._get(
        f"{polygon._base_url}/v2/aggs/ticker/{ticker}"
        f"/range/1/day/{from_date}/{to_date}",
        params={"apiKey": polygon._api_key, "limit": "5000", "sort": "asc"},
    )
    return [
        {
            "date": date.fromtimestamp(r["t"] / 1000).isoformat()
            if "t" in r
            else "",
            "open": float(r.get("o", 0)),
            "high": float(r.get("h", 0)),
            "low": float(r.get("l", 0)),
            "close": float(r.get("c", 0)),
            "volume": int(r.get("v", 0)),
        }
        for r in data.get("results", [])
    ]


def _simulate_option_trade(
    entry_price: float,
    underlying_entry: float,
    underlying_bars: list[dict[str, Any]],
    strike: float,
    is_call: bool,
    max_hold_days: int = 10,
    take_profit_pct: float = 100.0,
    stop_loss_pct: float = -80.0,
) -> BacktestTrade:
    """Simulate a single option trade using underlying price movement.

    Simple model: estimates option value from intrinsic + time decay.
    Not Black-Scholes — good enough for signal validation.
    """
    if not underlying_bars or entry_price <= 0:
        return BacktestTrade(
            ticker="",
            entry_date=date.today(),
            entry_price=entry_price,
            outcome=TradeOutcome.EXPIRED,
        )

    hold_days = min(max_hold_days, len(underlying_bars))
    best_pnl_pct = -100.0
    exit_price = 0.0
    exit_idx = hold_days - 1
    underlying_exit = underlying_entry

    for i in range(hold_days):
        bar = underlying_bars[i]
        current_underlying = float(bar.get("close", underlying_entry))

        # Simple intrinsic + time value estimate
        if is_call:
            intrinsic = max(0, current_underlying - strike)
        else:
            intrinsic = max(0, strike - current_underlying)

        # Time decay: linear reduction of extrinsic value
        time_remaining_pct = max(0, 1 - (i + 1) / max(max_hold_days * 1.5, 1))
        extrinsic = entry_price * time_remaining_pct * 0.5
        estimated_price = intrinsic + extrinsic

        pnl_pct = ((estimated_price - entry_price) / entry_price) * 100

        if pnl_pct > best_pnl_pct:
            best_pnl_pct = pnl_pct

        # Check take profit
        if pnl_pct >= take_profit_pct:
            exit_price = estimated_price
            exit_idx = i
            underlying_exit = current_underlying
            break

        # Check stop loss
        if pnl_pct <= stop_loss_pct:
            exit_price = estimated_price
            exit_idx = i
            underlying_exit = current_underlying
            break

        exit_price = estimated_price
        underlying_exit = current_underlying

    pnl = exit_price - entry_price
    pnl_pct_final = (pnl / entry_price) * 100 if entry_price > 0 else 0.0
    underlying_move = (
        (underlying_exit - underlying_entry) / underlying_entry * 100
        if underlying_entry > 0
        else 0.0
    )

    if pnl_pct_final >= 10:
        outcome = TradeOutcome.WIN
    elif pnl_pct_final <= -90:
        outcome = TradeOutcome.EXPIRED
    elif pnl_pct_final < -10:
        outcome = TradeOutcome.LOSS
    else:
        outcome = TradeOutcome.BREAKEVEN

    entry_date_str = underlying_bars[0].get("date", "") if underlying_bars else ""
    exit_date_str = (
        underlying_bars[exit_idx].get("date", "") if exit_idx < len(underlying_bars) else ""
    )

    return BacktestTrade(
        ticker="",  # Set by caller
        entry_date=date.fromisoformat(entry_date_str) if entry_date_str else date.today(),
        exit_date=date.fromisoformat(exit_date_str) if exit_date_str else None,
        entry_price=entry_price,
        exit_price=exit_price,
        underlying_entry=underlying_entry,
        underlying_exit=underlying_exit,
        underlying_move_pct=round(underlying_move, 2),
        pnl_per_contract=round(pnl * 100, 2),
        pnl_pct=round(pnl_pct_final, 2),
        outcome=outcome,
        hold_days=exit_idx + 1,
    )


def _compute_score_buckets(
    trades: list[BacktestTrade],
) -> dict[str, dict[str, float]]:
    """Group trade performance by signal score ranges."""
    buckets: dict[str, list[BacktestTrade]] = {
        "0-3": [],
        "3-5": [],
        "5-7": [],
        "7-10": [],
    }
    for t in trades:
        if t.signal_score < 3:
            buckets["0-3"].append(t)
        elif t.signal_score < 5:
            buckets["3-5"].append(t)
        elif t.signal_score < 7:
            buckets["5-7"].append(t)
        else:
            buckets["7-10"].append(t)

    result: dict[str, dict[str, float]] = {}
    for bucket_name, bucket_trades in buckets.items():
        if not bucket_trades:
            continue
        wins = sum(1 for t in bucket_trades if t.outcome == TradeOutcome.WIN)
        avg_pnl = (
            sum(t.pnl_pct for t in bucket_trades) / len(bucket_trades)
        )
        result[bucket_name] = {
            "count": float(len(bucket_trades)),
            "win_rate": round(wins / len(bucket_trades), 3),
            "avg_pnl_pct": round(avg_pnl, 2),
        }
    return result


async def run_backtest(
    tickers: list[str],
    lookback_days: int = 90,
    entry_premium: float = 2.0,
    otm_pct: float = 0.05,
    max_hold_days: int = 10,
    take_profit_pct: float = 100.0,
    stop_loss_pct: float = -80.0,
    settings: Settings | None = None,
) -> BacktestResult:
    """Run a backtest simulating lotto plays over historical data.

    For each ticker:
    1. Fetch daily bars for lookback period
    2. Simulate entry every N days (weekly)
    3. Buy OTM call at otm_pct above close
    4. Track outcome over max_hold_days
    5. Aggregate statistics

    Args:
        tickers: Tickers to backtest.
        lookback_days: How far back to look.
        entry_premium: Assumed entry premium per share.
        otm_pct: How far OTM (e.g., 0.05 = 5%).
        max_hold_days: Max days to hold before exit.
        take_profit_pct: Exit when option gains this %.
        stop_loss_pct: Exit when option loses this %.
    """
    settings = settings or get_settings()
    registry = ProviderRegistry(settings)
    polygon = PolygonProvider(settings)

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    all_trades: list[BacktestTrade] = []

    for ticker in tickers:
        try:
            bars = await _fetch_price_history(
                polygon,
                ticker,
                start_date.isoformat(),
                end_date.isoformat(),
            )

            if len(bars) < max_hold_days + 5:
                logger.warning(
                    "insufficient_history",
                    ticker=ticker,
                    bars=len(bars),
                )
                continue

            # Simulate weekly entries
            entry_interval = 5  # Every 5 trading days
            for i in range(0, len(bars) - max_hold_days, entry_interval):
                entry_bar = bars[i]
                underlying_entry = float(entry_bar.get("close", 0))
                if underlying_entry <= 0:
                    continue

                strike = underlying_entry * (1 + otm_pct)
                future_bars = bars[i + 1 : i + 1 + max_hold_days]

                trade = _simulate_option_trade(
                    entry_price=entry_premium,
                    underlying_entry=underlying_entry,
                    underlying_bars=future_bars,
                    strike=strike,
                    is_call=True,
                    max_hold_days=max_hold_days,
                    take_profit_pct=take_profit_pct,
                    stop_loss_pct=stop_loss_pct,
                )
                trade.ticker = ticker
                trade.strike = round(strike, 2)
                trade.option_type = "call"
                trade.signal_score = 5.0  # Baseline score

                all_trades.append(trade)

        except Exception as e:
            logger.warning("backtest_ticker_failed", ticker=ticker, error=str(e))

    await polygon.close()
    await registry.close_all()

    # Compute aggregate stats
    wins = sum(1 for t in all_trades if t.outcome == TradeOutcome.WIN)
    losses = sum(
        1 for t in all_trades if t.outcome in (TradeOutcome.LOSS, TradeOutcome.EXPIRED)
    )
    expired = sum(1 for t in all_trades if t.outcome == TradeOutcome.EXPIRED)
    total = len(all_trades)

    win_pnls = [t.pnl_pct for t in all_trades if t.outcome == TradeOutcome.WIN]
    loss_pnls = [t.pnl_pct for t in all_trades if t.outcome != TradeOutcome.WIN]

    gross_profit = sum(t.pnl_pct for t in all_trades if t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct for t in all_trades if t.pnl_pct < 0))

    result = BacktestResult(
        run_id=str(uuid.uuid4())[:12],
        tickers=tickers,
        lookback_days=lookback_days,
        total_trades=total,
        wins=wins,
        losses=losses,
        expired_worthless=expired,
        win_rate=round(wins / total, 3) if total > 0 else 0.0,
        avg_win_pct=round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0.0,
        avg_loss_pct=round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0,
        best_trade_pct=round(max((t.pnl_pct for t in all_trades), default=0), 2),
        worst_trade_pct=round(min((t.pnl_pct for t in all_trades), default=0), 2),
        total_pnl_pct=round(sum(t.pnl_pct for t in all_trades), 2),
        profit_factor=(
            round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0
        ),
        avg_hold_days=(
            round(sum(t.hold_days for t in all_trades) / total, 1)
            if total > 0
            else 0.0
        ),
        trades=all_trades,
        by_score_bucket=_compute_score_buckets(all_trades),
    )

    logger.info(
        "backtest_complete",
        tickers=tickers,
        total_trades=total,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
    )
    return result
