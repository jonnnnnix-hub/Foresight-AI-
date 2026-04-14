"""Intraday feature engineering — computes signals from multi-source data.

Takes raw bars, options chains, VIX, breadth, and sentiment data
and produces the IntradaySnapshot that models consume.

Key computed features:
- Multi-timeframe IBS (1m, 5m, daily)
- RSI(3) on 5-minute bars
- VWAP deviation
- Gap percentage (open vs prior close)
- Range position (5-day and intraday)
- Volume ratio (current vs average)
- Breadth capitulation flag
- VIX regime + spike detection
- Sentiment composite score
"""

from __future__ import annotations

from typing import Any

from flowedge.scanner.data_feeds.schemas import (
    BarData,
    IntradaySnapshot,
    MarketBreadth,
    SentimentData,
    VIXData,
)


def compute_ibs(bar: BarData) -> float:
    """Internal Bar Strength at any timeframe."""
    rng = bar.high - bar.low
    if rng <= 0:
        return 0.5
    return (bar.close - bar.low) / rng


def compute_rsi3(closes: list[float]) -> float:
    """Ultra-short RSI(3) from a list of close prices."""
    if len(closes) < 4:
        return 50.0
    gains, losses_ = [], []
    for i in range(-3, 0):
        d = closes[i] - closes[i - 1]
        gains.append(max(0, d))
        losses_.append(max(0, -d))
    ag = sum(gains) / 3
    al = sum(losses_) / 3
    if al == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + ag / al))


def compute_vwap_deviation(bars: list[BarData]) -> float:
    """How far current price is from VWAP (as %)."""
    if not bars:
        return 0.0
    current = bars[-1]
    if current.vwap <= 0:
        return 0.0
    return (current.close - current.vwap) / current.vwap * 100


def compute_gap_pct(
    current_open: float, prior_close: float,
) -> float:
    """Gap percentage from prior close to current open."""
    if prior_close <= 0:
        return 0.0
    return (current_open - prior_close) / prior_close * 100


def compute_range_position(
    bars: list[BarData], period: int = 5,
) -> float:
    """Where current price sits in the last N bars' range (0=low, 1=high)."""
    if len(bars) < period:
        return 0.5
    recent = bars[-period:]
    hi = max(b.high for b in recent)
    lo = min(b.low for b in recent)
    rng = hi - lo
    if rng <= 0:
        return 0.5
    return (bars[-1].close - lo) / rng


def compute_volume_ratio(
    bars: list[BarData], period: int = 20,
) -> float:
    """Current bar volume vs average of prior N bars."""
    if len(bars) < period + 1:
        return 1.0
    avg_vol = sum(b.volume for b in bars[-(period + 1):-1]) / period
    return bars[-1].volume / avg_vol if avg_vol > 0 else 1.0


def build_snapshot(
    ticker: str,
    bars_1m: list[BarData] | None = None,
    bars_5m: list[BarData] | None = None,
    bars_daily: list[BarData] | None = None,
    breadth: MarketBreadth | None = None,
    vix: VIXData | None = None,
    sentiment: SentimentData | None = None,
) -> IntradaySnapshot | None:
    """Build a complete IntradaySnapshot from all available data.

    This is the main entry point for the feature engine. Models
    consume these snapshots to generate signals.
    """
    # Need at least some bars
    if not bars_5m and not bars_daily:
        return None

    from datetime import datetime
    now = datetime.now()

    snapshot = IntradaySnapshot(
        timestamp=now,
        ticker=ticker,
        breadth=breadth,
        vix=vix,
        sentiment=sentiment,
    )

    # 1-minute features
    if bars_1m and len(bars_1m) >= 2:
        snapshot.bar_1m = bars_1m[-1]
        snapshot.ibs_1m = compute_ibs(bars_1m[-1])

    # 5-minute features
    if bars_5m and len(bars_5m) >= 4:
        snapshot.bar_5m = bars_5m[-1]
        snapshot.ibs_5m = compute_ibs(bars_5m[-1])
        closes_5m = [b.close for b in bars_5m]
        snapshot.rsi3_5m = compute_rsi3(closes_5m)
        snapshot.vwap_deviation_pct = round(compute_vwap_deviation(bars_5m), 3)
        snapshot.volume_ratio = round(compute_volume_ratio(bars_5m, 20), 2)

    # Daily features
    if bars_daily and len(bars_daily) >= 2:
        snapshot.bar_daily = bars_daily[-1]
        snapshot.ibs_daily = compute_ibs(bars_daily[-1])
        snapshot.range_position_5d = round(
            compute_range_position(bars_daily, 5), 3,
        )
        # Gap from prior close to today's open
        snapshot.gap_pct = round(
            compute_gap_pct(
                bars_daily[-1].open,
                bars_daily[-2].close,
            ), 3,
        )

    return snapshot


def check_rapid_confluence(snapshot: IntradaySnapshot) -> dict[str, Any] | None:
    """Check if the Rapid model's 4-signal confluence fires.

    The magic confluence (77% WR in backtesting):
    - IBS oversold (< 0.35)
    - RSI(3) snapped (< 30)
    - Price at range low (< 0.20 of 5-day range)
    - Volume surge (> 1.3x average)

    Additional boost from:
    - Breadth capitulation (TICK < -1000)
    - VIX spike (confirms fear)

    Returns signal dict or None if confluence doesn't fire.
    """
    if not snapshot.is_tradeable:
        return None

    signals_hit: list[str] = []
    conviction = 6.0

    # Core 4 signals (all required)
    if snapshot.ibs_5m < 0.35:
        signals_hit.append("ibs_low")
        conviction += 0.5 + (0.35 - snapshot.ibs_5m)

    if snapshot.rsi3_5m < 30.0:
        signals_hit.append("rsi3_snap")
        conviction += 0.5

    if snapshot.range_position_5d < 0.20:
        signals_hit.append("range_low")
        conviction += 0.5

    if snapshot.volume_ratio > 1.3:
        signals_hit.append("volume_surge")
        conviction += 0.5

    # All 4 required
    required = {"ibs_low", "rsi3_snap", "range_low", "volume_surge"}
    if not required.issubset(set(signals_hit)):
        return None

    # Bonus signals (not required but boost conviction)
    if snapshot.breadth and snapshot.breadth.is_capitulation:
        signals_hit.append("breadth_capitulation")
        conviction += 1.5  # Strong confirmation

    if snapshot.vix and snapshot.vix.is_spike:
        signals_hit.append("vix_spike")
        conviction += 1.0  # Fear confirmation

    if snapshot.sentiment and snapshot.sentiment.composite_score < -0.5:
        signals_hit.append("sentiment_bearish")
        conviction += 0.5  # Contrarian buy into extreme fear

    return {
        "ticker": snapshot.ticker,
        "direction": "bullish",
        "signal_type": "+".join(signals_hit),
        "conviction": min(10.0, conviction),
        "signal_count": len(signals_hit),
        "ibs_5m": snapshot.ibs_5m,
        "rsi3": snapshot.rsi3_5m,
        "range_pos": snapshot.range_position_5d,
        "vol_ratio": snapshot.volume_ratio,
        "vwap_dev": snapshot.vwap_deviation_pct,
        "has_breadth": snapshot.breadth is not None and snapshot.breadth.is_capitulation,
        "has_vix_spike": snapshot.vix is not None and snapshot.vix.is_spike,
    }


def check_hybrid_ibs(snapshot: IntradaySnapshot) -> dict[str, Any] | None:
    """Check if the Hybrid model's IBS reversion fires.

    Uses daily IBS (proven 62.5% WR) with intraday confirmation.
    """
    if not snapshot.is_tradeable:
        return None

    # Require daily IBS oversold
    if snapshot.ibs_daily >= 0.20:
        return None

    # Require gap down (confirm dip is real)
    if snapshot.gap_pct >= 0:
        return None

    conviction = 7.0 + (0.20 - snapshot.ibs_daily) / 0.20 * 2.0

    # Intraday confirmation: 5m IBS also oversold
    if snapshot.ibs_5m < 0.30:
        conviction += 0.5

    # VIX boost: fear = stronger reversals
    if snapshot.vix and snapshot.vix.regime in ("elevated", "cautious"):
        conviction += 0.5

    # Breadth boost
    if snapshot.breadth and snapshot.breadth.is_capitulation:
        conviction += 1.0

    return {
        "ticker": snapshot.ticker,
        "direction": "bullish",
        "signal_type": "hybrid_ibs",
        "conviction": min(10.0, conviction),
        "ibs_daily": snapshot.ibs_daily,
        "ibs_5m": snapshot.ibs_5m,
        "gap_pct": snapshot.gap_pct,
    }
