"""PULSE — reads the heartbeat of price action across timeframes.

Fetches RSI, MACD, SMA from Polygon indicators API and intraday bars,
then scores momentum alignment across timeframes.

Note: Uses Orats for current price (free, no rate limit) and Polygon
for technical indicators only (minimizes Polygon API calls).
"""

from __future__ import annotations

from datetime import datetime

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.momentum.schemas import (
    MomentumBias,
    MomentumSignal,
    TechnicalSnapshot,
)
from flowedge.scanner.providers.polygon import PolygonProvider

logger = structlog.get_logger()


async def _fetch_indicator(
    polygon: PolygonProvider,
    ticker: str,
    indicator: str,
    window: int = 14,
    timespan: str = "day",
) -> float | None:
    """Fetch a single indicator value (most recent)."""
    try:
        data = await polygon.get_technical_indicator(
            ticker, indicator, window=window, timespan=timespan, limit=1
        )
        if data and data[0].get("value") is not None:
            val = data[0]["value"]
            if isinstance(val, dict):
                inner = val.get("value", 0)
                return float(inner) if isinstance(inner, (int, float, str)) else 0.0
            if isinstance(val, (int, float, str)):
                return float(val)
            return 0.0
    except Exception as e:
        logger.debug(
            "indicator_fetch_failed",
            ticker=ticker,
            indicator=indicator,
            error=str(e),
        )
    return None


async def _fetch_macd(
    polygon: PolygonProvider,
    ticker: str,
    timespan: str = "day",
) -> tuple[float | None, float | None, float | None]:
    """Fetch MACD value, signal, and histogram."""
    try:
        data = await polygon.get_technical_indicator(
            ticker, "macd", timespan=timespan, limit=1
        )
        if data and data[0].get("value"):
            val = data[0]["value"]
            if isinstance(val, dict):
                return (
                    float(val.get("value", 0)),
                    float(val.get("signal", 0)),
                    float(val.get("histogram", 0)),
                )
    except Exception as e:
        logger.debug("macd_fetch_failed", ticker=ticker, error=str(e))
    return None, None, None


async def _build_snapshot(
    polygon: PolygonProvider,
    ticker: str,
    timespan: str,
    timeframe_label: str,
) -> TechnicalSnapshot:
    """Build a technical snapshot for one timeframe."""
    rsi = await _fetch_indicator(polygon, ticker, "rsi", 14, timespan)
    sma_20 = await _fetch_indicator(polygon, ticker, "sma", 20, timespan)
    sma_50 = await _fetch_indicator(polygon, ticker, "sma", 50, timespan)
    ema_9 = await _fetch_indicator(polygon, ticker, "ema", 9, timespan)
    ema_21 = await _fetch_indicator(polygon, ticker, "ema", 21, timespan)
    macd_val, macd_sig, macd_hist = await _fetch_macd(polygon, ticker, timespan)

    # Get recent price — use Orats cores if available, else Polygon prev close
    price = 0.0
    try:
        from flowedge.config.settings import get_settings as _gs
        from flowedge.scanner.providers.orats import OratsProvider

        orats = OratsProvider(_gs())
        price = await orats.get_current_price(ticker)
        await orats.close()
    except Exception:
        pass
    if price <= 0:
        prev = await polygon.get_previous_close(ticker)
        close_val = prev.get("close", 0)
        price = float(close_val) if isinstance(close_val, (int, float, str)) else 0.0

    return TechnicalSnapshot(
        timeframe=timeframe_label,
        rsi=rsi,
        macd_value=macd_val,
        macd_signal=macd_sig,
        macd_histogram=macd_hist,
        sma_20=sma_20,
        sma_50=sma_50,
        ema_9=ema_9,
        ema_21=ema_21,
        current_price=price,
    )


def _classify_bias(snapshots: list[TechnicalSnapshot]) -> MomentumBias:
    """Classify overall momentum bias from multiple timeframes."""
    bullish_count = 0
    bearish_count = 0

    for snap in snapshots:
        tf_bullish = 0
        tf_bearish = 0

        # RSI
        if snap.rsi is not None:
            if snap.rsi > 60:
                tf_bullish += 1
            elif snap.rsi < 40:
                tf_bearish += 1

        # MACD histogram
        if snap.macd_histogram is not None:
            if snap.macd_histogram > 0:
                tf_bullish += 1
            else:
                tf_bearish += 1

        # Price vs SMA
        if snap.sma_20 and snap.current_price > 0:
            if snap.current_price > snap.sma_20:
                tf_bullish += 1
            else:
                tf_bearish += 1

        # EMA crossover
        if snap.ema_9 and snap.ema_21:
            if snap.ema_9 > snap.ema_21:
                tf_bullish += 1
            else:
                tf_bearish += 1

        if tf_bullish > tf_bearish:
            bullish_count += 1
        elif tf_bearish > tf_bullish:
            bearish_count += 1

    total = len(snapshots)
    if total == 0:
        return MomentumBias.NEUTRAL

    if bullish_count == total:
        return MomentumBias.STRONG_BULLISH
    if bearish_count == total:
        return MomentumBias.STRONG_BEARISH
    if bullish_count > bearish_count:
        return MomentumBias.BULLISH
    if bearish_count > bullish_count:
        return MomentumBias.BEARISH
    return MomentumBias.NEUTRAL


def _score_momentum(
    bias: MomentumBias,
    snapshots: list[TechnicalSnapshot],
    trend_aligned: bool,
) -> float:
    """Score momentum signal strength (0-10)."""
    score = 0.0

    # Bias strength (0-4)
    if bias == MomentumBias.STRONG_BULLISH or bias == MomentumBias.STRONG_BEARISH:
        score += 4.0
    elif bias in (MomentumBias.BULLISH, MomentumBias.BEARISH):
        score += 2.5
    else:
        score += 1.0

    # Trend alignment bonus (0-3)
    if trend_aligned:
        score += 3.0

    # RSI extremes (0-2)
    daily = next((s for s in snapshots if s.timeframe == "daily"), None)
    if daily and daily.rsi:
        if daily.rsi < 30 or daily.rsi > 70:
            score += 2.0
        elif daily.rsi < 40 or daily.rsi > 60:
            score += 1.0

    # MACD crossover (0-1)
    if (
        daily
        and daily.macd_histogram
        and daily.macd_signal
        and daily.macd_histogram > 0
        and abs(daily.macd_histogram) < abs(daily.macd_signal) * 0.2
    ):
        score += 1.0

    return min(score, 10.0)


async def analyze_momentum(
    ticker: str,
    settings: Settings | None = None,
) -> MomentumSignal:
    """Analyze multi-timeframe momentum for a ticker.

    Fetches technicals from Polygon for daily timeframe
    (Polygon free tier rate limits prevent multi-timeframe).
    """
    settings = settings or get_settings()
    polygon = PolygonProvider(settings)

    try:
        # Fetch daily snapshot (primary timeframe)
        daily = await _build_snapshot(polygon, ticker, "day", "daily")
        snapshots = [daily]

        # Classify
        bias = _classify_bias(snapshots)
        trend_aligned = bias in (
            MomentumBias.STRONG_BULLISH,
            MomentumBias.STRONG_BEARISH,
        )

        rsi_oversold = daily.rsi is not None and daily.rsi < 30
        rsi_overbought = daily.rsi is not None and daily.rsi > 70
        macd_cross = (
            daily.macd_histogram is not None
            and daily.macd_histogram > 0
        )

        strength = _score_momentum(bias, snapshots, trend_aligned)

        rationale_parts: list[str] = [f"Bias: {bias.value}"]
        if daily.rsi:
            rationale_parts.append(f"RSI: {daily.rsi:.1f}")
        if daily.macd_histogram:
            rationale_parts.append(
                f"MACD hist: {daily.macd_histogram:+.2f}"
            )
        if daily.sma_20 and daily.current_price:
            above = daily.current_price > daily.sma_20
            rationale_parts.append(
                f"{'Above' if above else 'Below'} SMA20 (${daily.sma_20:.2f})"
            )

        signal = MomentumSignal(
            ticker=ticker,
            bias=bias,
            strength=strength,
            timeframes=snapshots,
            trend_alignment=trend_aligned,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            macd_crossover=macd_cross,
            rationale="; ".join(rationale_parts),
            computed_at=datetime.now(),
        )

        logger.info(
            "momentum_analyzed",
            ticker=ticker,
            bias=bias.value,
            strength=strength,
        )
        return signal

    finally:
        await polygon.close()


async def analyze_momentum_batch(
    tickers: list[str],
    settings: Settings | None = None,
) -> list[MomentumSignal]:
    """Analyze momentum for multiple tickers."""
    signals: list[MomentumSignal] = []
    for ticker in tickers:
        try:
            sig = await analyze_momentum(ticker, settings)
            signals.append(sig)
        except Exception as e:
            logger.warning("momentum_failed", ticker=ticker, error=str(e))
    return signals
