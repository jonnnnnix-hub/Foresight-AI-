"""Catalyst scanner — finds upcoming binary events and insider activity."""

from __future__ import annotations

import contextlib
from datetime import date, datetime, timedelta

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.providers.registry import ProviderRegistry
from flowedge.scanner.schemas.catalyst import CatalystSignal, EarningsEvent

logger = structlog.get_logger()


def _compute_insider_sentiment(
    buys: int, sells: int, net_value: float
) -> str:
    """Determine net insider sentiment from trade counts and value."""
    if net_value > 100_000 and buys > sells:
        return "bullish"
    if net_value < -100_000 and sells > buys:
        return "bearish"
    return "neutral"


def _score_catalyst(
    days_to_catalyst: int | None,
    insider_sentiment: str,
    insider_buy_count: int,
    expected_move_pct: float,
) -> float:
    """Score catalyst signal strength (0-10)."""
    score = 0.0

    # Proximity component (0-4 points) — closer = higher score
    if days_to_catalyst is not None:
        if days_to_catalyst <= 3:
            score += 4.0
        elif days_to_catalyst <= 7:
            score += 3.0
        elif days_to_catalyst <= 14:
            score += 2.0
        elif days_to_catalyst <= 30:
            score += 1.0

    # Insider sentiment component (0-3 points)
    if insider_sentiment == "bullish":
        score += 2.0
        if insider_buy_count >= 3:
            score += 1.0  # Cluster buying
    elif insider_sentiment == "bearish":
        score += 0.5  # Still a catalyst, just directional info

    # Expected move component (0-3 points)
    if expected_move_pct >= 10.0:
        score += 3.0
    elif expected_move_pct >= 7.0:
        score += 2.0
    elif expected_move_pct >= 4.0:
        score += 1.0

    return min(score, 10.0)


async def scan_catalysts(
    registry: ProviderRegistry,
    tickers: list[str],
    settings: Settings | None = None,
) -> list[CatalystSignal]:
    """Scan for upcoming catalysts — earnings and insider activity."""
    settings = settings or get_settings()
    earnings_provider = registry.get_earnings_provider()
    insider_provider = registry.get_insider_provider()
    iv_provider = registry.get_iv_provider()

    today = date.today()
    lookahead = today + timedelta(days=settings.catalyst_lookforward_days)

    # Fetch earnings calendar
    try:
        all_earnings = await earnings_provider.get_earnings_calendar(today, lookahead)
    except Exception as e:
        logger.warning("earnings_fetch_failed", error=str(e))
        all_earnings = []

    # Index earnings by ticker
    earnings_by_ticker: dict[str, list[EarningsEvent]] = {}
    for event in all_earnings:
        earnings_by_ticker.setdefault(event.ticker, []).append(event)

    signals: list[CatalystSignal] = []

    for ticker in tickers:
        try:
            # Get earnings for this ticker
            ticker_earnings = earnings_by_ticker.get(ticker.upper(), [])

            # Get insider trades
            try:
                insider_trades = await insider_provider.get_insider_trades(
                    ticker, days_back=settings.catalyst_lookback_insider_days
                )
            except Exception as e:
                logger.debug("insider_fetch_failed", ticker=ticker, error=str(e))
                insider_trades = []

            # Get expected move if earnings upcoming
            expected_move = None
            if ticker_earnings:
                with contextlib.suppress(Exception):
                    expected_move = await iv_provider.get_expected_move(ticker)

            # Compute insider metrics
            buy_count = sum(
                1 for t in insider_trades if t.transaction_type == "P"
            )
            sell_count = sum(
                1 for t in insider_trades if t.transaction_type == "S"
            )
            buy_value = sum(
                t.total_value for t in insider_trades if t.transaction_type == "P"
            )
            sell_value = sum(
                t.total_value for t in insider_trades if t.transaction_type == "S"
            )
            net_value = buy_value - sell_value
            sentiment = _compute_insider_sentiment(buy_count, sell_count, net_value)

            # Days to nearest catalyst
            days_to_nearest: int | None = None
            if ticker_earnings:
                nearest = min(ticker_earnings, key=lambda e: e.report_date)
                days_to_nearest = (nearest.report_date - today).days

            expected_move_pct = (
                expected_move.expected_move_pct if expected_move else 0.0
            )
            strength = _score_catalyst(
                days_to_nearest, sentiment, buy_count, expected_move_pct
            )

            # Skip tickers with no catalysts and no insider activity
            if not ticker_earnings and not insider_trades:
                continue

            rationale_parts: list[str] = []
            if ticker_earnings:
                rationale_parts.append(
                    f"Earnings in {days_to_nearest}d"
                    if days_to_nearest is not None
                    else "Earnings upcoming"
                )
            if expected_move:
                rationale_parts.append(
                    f"Expected move: {expected_move_pct:.1f}%"
                )
            if insider_trades:
                rationale_parts.append(
                    f"Insiders: {buy_count}B/{sell_count}S, "
                    f"net ${net_value:+,.0f}"
                )

            signals.append(
                CatalystSignal(
                    ticker=ticker,
                    earnings=ticker_earnings,
                    insider_trades=insider_trades,
                    expected_move=expected_move,
                    days_to_nearest_catalyst=days_to_nearest,
                    net_insider_sentiment=sentiment,
                    insider_buy_count=buy_count,
                    insider_sell_count=sell_count,
                    insider_net_value=net_value,
                    strength=strength,
                    rationale="; ".join(rationale_parts),
                    detected_at=datetime.now(),
                )
            )

        except Exception as e:
            logger.warning("catalyst_scan_failed", ticker=ticker, error=str(e))

    signals.sort(key=lambda s: s.strength, reverse=True)
    logger.info("catalyst_scan_complete", signal_count=len(signals))
    return signals
