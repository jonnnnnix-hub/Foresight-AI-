"""FLUX — reads the equity tape to measure real buying/selling pressure.

Validates technical signals (IBS, RSI) with direct observation of
who is aggressing on the tape. Uses Lee-Ready trade classification,
cumulative delta, block print detection, and L1 quote imbalance.

Core thesis: IBS < 0.10 with positive cumulative delta = confirmed
accumulation. IBS < 0.10 without it = falling knife.
"""

from __future__ import annotations

import bisect
from datetime import datetime
from statistics import mean
from typing import Any

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.flux.schemas import (
    BlockPrint,
    ClassifiedTrade,
    CumulativeDelta,
    DeltaDivergence,
    FlowBias,
    FLUXSignal,
    NBBOQuote,
    QuoteImbalance,
    TradeDirection,
    TradeTick,
)

logger = structlog.get_logger()


# ── Lee-Ready Trade Classification ─────────────────────────────


def _classify_trades_lee_ready(
    trades: list[TradeTick],
    quotes: list[NBBOQuote],
) -> list[ClassifiedTrade]:
    """Classify each trade as buyer- or seller-initiated using Lee-Ready.

    Algorithm:
    1. Find the prevailing NBBO quote at the time of each trade
    2. If trade price > midpoint → buyer-initiated (hitting the ask)
    3. If trade price < midpoint → seller-initiated (hitting the bid)
    4. If trade price == midpoint → use tick test (compare to prior trade)

    The tick test fallback:
    - If price > prior trade price → uptick → buyer
    - If price < prior trade price → downtick → seller
    - If price == prior trade price → use direction of last known tick
    """
    if not trades or not quotes:
        return []

    # Build sorted timestamp index for quote lookup
    quote_timestamps = [q.timestamp for q in quotes]

    classified: list[ClassifiedTrade] = []
    last_direction = TradeDirection.UNKNOWN
    last_price = 0.0

    for trade in trades:
        # Find prevailing quote: latest quote with timestamp <= trade timestamp
        idx = bisect.bisect_right(quote_timestamps, trade.timestamp) - 1
        direction = TradeDirection.UNKNOWN

        if idx >= 0:
            quote = quotes[idx]
            mid = quote.midpoint

            if mid > 0:
                if trade.price > mid:
                    direction = TradeDirection.BUY
                elif trade.price < mid:
                    direction = TradeDirection.SELL
                else:
                    # Price at midpoint — use tick test
                    if last_price > 0:
                        if trade.price > last_price:
                            direction = TradeDirection.BUY
                        elif trade.price < last_price:
                            direction = TradeDirection.SELL
                        else:
                            direction = last_direction
                    else:
                        direction = TradeDirection.UNKNOWN

        # Signed volume: positive for buys, negative for sells
        signed_vol = trade.size
        if direction == TradeDirection.SELL:
            signed_vol = -trade.size

        classified.append(ClassifiedTrade(
            price=trade.price,
            size=trade.size,
            timestamp=trade.timestamp,
            direction=direction,
            signed_volume=signed_vol if direction != TradeDirection.UNKNOWN else 0,
        ))

        if direction != TradeDirection.UNKNOWN:
            last_direction = direction
        last_price = trade.price

    return classified


# ── Cumulative Delta ────────────────────────────────────────────


def _compute_cumulative_delta(
    classified: list[ClassifiedTrade],
    window_minutes: int,
) -> CumulativeDelta:
    """Compute cumulative delta (net buying volume) from classified trades."""
    buy_vol = 0
    sell_vol = 0
    buy_count = 0
    sell_count = 0

    for trade in classified:
        if trade.direction == TradeDirection.BUY:
            buy_vol += trade.size
            buy_count += 1
        elif trade.direction == TradeDirection.SELL:
            sell_vol += trade.size
            sell_count += 1

    return CumulativeDelta(
        window_minutes=window_minutes,
        buy_volume=buy_vol,
        sell_volume=sell_vol,
        net_delta=buy_vol - sell_vol,
        total_trades=len(classified),
        buy_trades=buy_count,
        sell_trades=sell_count,
    )


# ── Block Print Detection ──────────────────────────────────────


def _detect_block_prints(
    classified: list[ClassifiedTrade],
    ticker: str,
    min_multiple: float = 10.0,
) -> list[BlockPrint]:
    """Detect unusually large trades (block prints) on the tape.

    A block print is a trade whose size exceeds min_multiple times
    the average trade size for the window. These often indicate
    institutional activity.
    """
    if len(classified) < 10:
        return []

    avg_size = mean(t.size for t in classified)
    threshold = avg_size * min_multiple

    blocks: list[BlockPrint] = []
    for trade in classified:
        if trade.size >= threshold:
            blocks.append(BlockPrint(
                ticker=ticker,
                price=trade.price,
                size=trade.size,
                notional=round(trade.price * trade.size, 2),
                direction=trade.direction,
                size_multiple=round(trade.size / avg_size, 1),
                timestamp=datetime.now(),
            ))

    return sorted(blocks, key=lambda b: b.size, reverse=True)


def _classify_block_bias(blocks: list[BlockPrint]) -> TradeDirection:
    """Determine net direction from block prints."""
    if not blocks:
        return TradeDirection.UNKNOWN

    buy_notional = sum(
        b.notional for b in blocks if b.direction == TradeDirection.BUY
    )
    sell_notional = sum(
        b.notional for b in blocks if b.direction == TradeDirection.SELL
    )

    if buy_notional > sell_notional * 1.5:
        return TradeDirection.BUY
    if sell_notional > buy_notional * 1.5:
        return TradeDirection.SELL
    return TradeDirection.UNKNOWN


# ── L1 Quote Imbalance ──────────────────────────────────────────


def _compute_quote_imbalance(
    quotes: list[NBBOQuote],
    window_minutes: int,
) -> QuoteImbalance:
    """Compute average bid/ask size imbalance from NBBO quotes.

    Positive imbalance = more bid size = buying interest.
    Negative imbalance = more ask size = selling interest.
    """
    if not quotes:
        return QuoteImbalance(window_minutes=window_minutes)

    imbalances = [q.imbalance for q in quotes]
    avg = mean(imbalances)
    max_bid = max(imbalances)
    max_ask = min(imbalances)

    return QuoteImbalance(
        window_minutes=window_minutes,
        avg_imbalance=round(avg, 4),
        max_bid_dominance=round(max_bid, 4),
        max_ask_dominance=round(max_ask, 4),
        snapshots=len(quotes),
    )


# ── Delta Divergence Detection ──────────────────────────────────


def _detect_divergence(
    delta_5m: CumulativeDelta,
    delta_15m: CumulativeDelta,
    price_change_pct: float,
) -> DeltaDivergence:
    """Detect divergence between price action and order flow.

    Bullish divergence: price falling but cumulative delta positive
    (hidden accumulation — this is the key scalp confirmation signal).

    Bearish divergence: price rising but cumulative delta negative
    (hidden distribution — smart money selling into strength).
    """
    # Use 5m delta for sensitivity, 15m for confirmation
    delta_positive = delta_5m.net_delta > 0
    delta_negative = delta_5m.net_delta < 0

    # Need meaningful delta, not noise
    total_vol = delta_5m.buy_volume + delta_5m.sell_volume
    if total_vol == 0:
        return DeltaDivergence.NONE

    # Delta as fraction of total volume — must be > 5% to matter
    delta_pct = abs(delta_5m.net_delta) / total_vol
    if delta_pct < 0.05:
        return DeltaDivergence.CONFIRMED

    price_falling = price_change_pct < -0.05  # Down > 5 bps
    price_rising = price_change_pct > 0.05    # Up > 5 bps

    # Confirm with 15m window for stronger signal
    if price_falling and delta_positive and delta_15m.net_delta > 0:
        return DeltaDivergence.BULLISH
    if price_rising and delta_negative and delta_15m.net_delta < 0:
        return DeltaDivergence.BEARISH

    return DeltaDivergence.CONFIRMED


# ── Flow Bias ───────────────────────────────────────────────────


def _determine_bias(
    delta_5m: CumulativeDelta,
    quote_imbalance: QuoteImbalance,
    block_bias: TradeDirection,
) -> FlowBias:
    """Determine overall flow bias from multiple signals."""
    score = 0.0

    # Cumulative delta
    aggression = delta_5m.aggression_ratio
    if aggression > 0.60:
        score += 2.0
    elif aggression > 0.55:
        score += 1.0
    elif aggression < 0.40:
        score -= 2.0
    elif aggression < 0.45:
        score -= 1.0

    # Quote imbalance
    if quote_imbalance.avg_imbalance > 0.15:
        score += 1.0
    elif quote_imbalance.avg_imbalance < -0.15:
        score -= 1.0

    # Block prints
    if block_bias == TradeDirection.BUY:
        score += 1.0
    elif block_bias == TradeDirection.SELL:
        score -= 1.0

    if score >= 3.0:
        return FlowBias.STRONG_BUY
    if score >= 1.0:
        return FlowBias.BUY
    if score <= -3.0:
        return FlowBias.STRONG_SELL
    if score <= -1.0:
        return FlowBias.SELL
    return FlowBias.NEUTRAL


# ── FLUX Score ──────────────────────────────────────────────────


def _score_flux(
    delta_5m: CumulativeDelta,
    delta_15m: CumulativeDelta,
    quote_imbalance: QuoteImbalance,
    blocks: list[BlockPrint],
    divergence: DeltaDivergence,
    settings: Settings,
) -> tuple[float, str]:
    """Score the order flow signal from 0-10.

    Scores based on:
    1. Trade aggression ratio (0-3 pts) — are trades hitting ask or bid?
    2. Quote imbalance (0-2 pts) — is bid depth > ask depth?
    3. Block print bias (0-2 pts) — are large prints buying or selling?
    4. Delta divergence (0-3 pts) — price vs flow divergence
    """
    score = 0.0
    parts: list[str] = []

    # 1. Trade aggression (0-3 pts)
    aggression = delta_5m.aggression_ratio
    if aggression > 0.65:
        score += 3.0
        parts.append(f"Strong buy aggression {aggression:.0%}")
    elif aggression > 0.55:
        score += 2.0
        parts.append(f"Net buy aggression {aggression:.0%}")
    elif aggression < 0.35:
        score += 3.0  # Strong sell aggression is also a strong signal
        parts.append(f"Strong sell aggression {aggression:.0%}")
    elif aggression < 0.45:
        score += 2.0
        parts.append(f"Net sell aggression {aggression:.0%}")
    else:
        score += 0.5
        parts.append(f"Balanced flow {aggression:.0%}")

    # 2. Quote imbalance (0-2 pts)
    imb = abs(quote_imbalance.avg_imbalance)
    if imb > 0.30:
        score += 2.0
        side = "bid" if quote_imbalance.avg_imbalance > 0 else "ask"
        parts.append(f"Strong {side} depth imbalance {imb:.0%}")
    elif imb > 0.15:
        score += 1.0
        side = "bid" if quote_imbalance.avg_imbalance > 0 else "ask"
        parts.append(f"{side.title()} depth lean {imb:.0%}")

    # 3. Block print bias (0-2 pts)
    if blocks:
        buy_blocks = [b for b in blocks if b.direction == TradeDirection.BUY]
        sell_blocks = [b for b in blocks if b.direction == TradeDirection.SELL]
        total_block_notional = sum(b.notional for b in blocks)

        if buy_blocks and not sell_blocks:
            score += 2.0
            parts.append(
                f"{len(buy_blocks)} buy blocks "
                f"(${total_block_notional:,.0f} notional)"
            )
        elif sell_blocks and not buy_blocks:
            score += 2.0
            parts.append(
                f"{len(sell_blocks)} sell blocks "
                f"(${total_block_notional:,.0f} notional)"
            )
        elif buy_blocks or sell_blocks:
            score += 1.0
            parts.append(
                f"Mixed blocks: {len(buy_blocks)} buy / "
                f"{len(sell_blocks)} sell"
            )

    # 4. Delta divergence (0-3 pts)
    if divergence == DeltaDivergence.BULLISH:
        score += 3.0
        parts.append(
            "BULLISH divergence — price falling but tape buying "
            "(hidden accumulation)"
        )
    elif divergence == DeltaDivergence.BEARISH:
        score += 3.0
        parts.append(
            "BEARISH divergence — price rising but tape selling "
            "(hidden distribution)"
        )
    elif divergence == DeltaDivergence.CONFIRMED:
        score += 1.0
        parts.append("Price and flow confirmed (aligned)")

    return min(score, 10.0), "; ".join(parts)


# ── Main Engine ─────────────────────────────────────────────────


async def _fetch_data(consumer: object, method: str, *args: Any) -> Any:
    """Call a consumer method, handling both sync and async versions.

    WebSocket consumer has sync get_trades/get_quotes (buffer reads).
    REST consumer has async versions (HTTP calls).
    """
    fn = getattr(consumer, method)
    result = fn(*args)
    if hasattr(result, "__await__"):
        return await result
    return result


async def scan_flux(
    consumer: object,
    tickers: list[str] | None = None,
    settings: Settings | None = None,
    price_changes: dict[str, float] | None = None,
) -> list[FLUXSignal]:
    """Scan equity tape for order flow signals.

    Accepts either PolygonTradeConsumer (REST) or
    MassiveWebSocketConsumer (WebSocket) — both provide
    get_trades() and get_quotes() with the same interface.

    Args:
        consumer: Trade/quote consumer (REST or WebSocket)
        tickers: Tickers to scan (default: all scanner tickers)
        settings: App settings
        price_changes: Dict of {ticker: pct_change} for divergence
            detection. Typically the 5-min bar return.

    Returns:
        List of FLUXSignal sorted by strength descending.
    """
    settings = settings or get_settings()
    price_changes = price_changes or {}

    signals: list[FLUXSignal] = []

    for ticker in tickers or []:
        try:
            signal = await _scan_ticker(
                consumer, ticker, settings,
                price_change_pct=price_changes.get(ticker, 0.0),
            )
            signals.append(signal)
        except Exception as e:
            logger.warning("flux_scan_failed", ticker=ticker, error=str(e))

    signals.sort(key=lambda s: s.strength, reverse=True)
    logger.info("flux_scan_complete", signal_count=len(signals))
    return signals


async def _scan_ticker(
    consumer: object,
    ticker: str,
    settings: Settings,
    price_change_pct: float = 0.0,
) -> FLUXSignal:
    """Run full FLUX analysis for a single ticker."""
    # Fetch 5-min and 15-min trade windows (works with both consumer types)
    trades_5m = await _fetch_data(consumer, "get_trades", ticker, 5)
    trades_15m = await _fetch_data(consumer, "get_trades", ticker, 15)

    # Fetch 5-min and 15-min quote windows
    quotes_5m = await _fetch_data(consumer, "get_quotes", ticker, 5)
    quotes_15m = await _fetch_data(consumer, "get_quotes", ticker, 15)

    # Lee-Ready classification
    classified_5m = _classify_trades_lee_ready(trades_5m, quotes_5m)
    classified_15m = _classify_trades_lee_ready(trades_15m, quotes_15m)

    # Cumulative delta
    delta_5m = _compute_cumulative_delta(classified_5m, 5)
    delta_15m = _compute_cumulative_delta(classified_15m, 15)

    # Quote imbalance
    quote_imbalance = _compute_quote_imbalance(quotes_5m, 5)

    # Block prints (use 15m window for better detection)
    blocks = _detect_block_prints(
        classified_15m, ticker,
        min_multiple=settings.flux_block_min_multiple,
    )
    block_bias = _classify_block_bias(blocks)

    # Delta divergence
    divergence = _detect_divergence(delta_5m, delta_15m, price_change_pct)

    # Overall bias
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
        delta_session=None,  # Only computed on demand (expensive)
        quote_imbalance=quote_imbalance,
        block_prints=blocks[:10],  # Top 10 largest
        block_bias=block_bias,
        divergence=divergence,
        rationale=rationale,
        detected_at=datetime.now(),
    )


async def scan_flux_for_snapshot(
    consumer: object,
    ticker: str,
    settings: Settings | None = None,
    price_change_pct: float = 0.0,
) -> FLUXSignal:
    """Convenience wrapper for single-ticker FLUX in scanner loop.

    Returns a single FLUXSignal for integration into the
    ProductionScanner scan cycle.
    """
    settings = settings or get_settings()
    return await _scan_ticker(
        consumer, ticker, settings,
        price_change_pct=price_change_pct,
    )
