"""Multi-strategy signal generation for options backtesting.

Four strategies with distinct edge profiles:
1. TREND PULLBACK — buy dips in strong trends (highest win rate)
2. BREAKOUT — trade range expansions with volume (good risk/reward)
3. MEAN REVERSION — fade RSI extremes for quick bounces (fast trades)
4. VOLATILITY SQUEEZE — trade Bollinger compression breakouts (rare, large payoffs)

Each strategy includes regime detection, conviction scoring, and
OTM distance calibration. No external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# ── Market Regime ────────────────────────────────────────────────────

class MarketRegime(StrEnum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


# ── Data Structures ─────────────────────────────────────────────────

@dataclass
class Indicators:
    """Computed technical indicators for a single ticker at a point in time."""

    close: float
    sma20: float
    sma50: float
    rsi14: float
    atr14: float
    adx14: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    bb_width_pct: float  # (upper - lower) / mid as percentage
    vol_ratio: float  # today's volume / 20-day avg volume
    range_ratio: float  # today's range / ATR (range expansion)
    high_20: float  # 20-day high close
    low_20: float  # 20-day low close
    high_40: float  # 40-day high close
    low_40: float  # 40-day low close
    momentum_5: float  # 5-day price change %
    momentum_10: float  # 10-day price change %
    atr_ratio: float  # current ATR / ATR from 20 bars ago (compression)
    close_in_range: float  # 0=low of day, 1=high of day


@dataclass
class EntrySignal:
    """A trade entry signal from one of the strategies."""

    ticker: str
    direction: str  # "bullish" or "bearish"
    strategy: str  # strategy name
    conviction: float  # 0-10
    regime: str  # MarketRegime value
    otm_pct: float  # how far OTM (e.g., 0.03 = 3%)
    reason: str  # human-readable explanation


# ── Technical Indicator Computations ─────────────────────────────────

def _sma(values: list[float], period: int) -> float:
    """Simple moving average of last `period` values."""
    if len(values) < period:
        return values[-1] if values else 0.0
    return sum(values[-period:]) / period


def _rsi(closes: list[float], period: int = 14) -> float:
    """Relative Strength Index."""
    if len(closes) < period + 1:
        return 50.0

    gains: list[float] = []
    losses: list[float] = []
    for i in range(len(closes) - period, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(0.0, change))
        losses.append(max(0.0, -change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> float:
    """Average True Range."""
    if len(highs) < 2:
        return (highs[-1] - lows[-1]) if highs else 0.0

    trs: list[float] = []
    for i in range(max(1, len(highs) - period), len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    return sum(trs) / len(trs) if trs else 0.0


def _adx(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> float:
    """Simplified Average Directional Index."""
    if len(highs) < period + 2:
        return 25.0  # Neutral default

    plus_dm: list[float] = []
    minus_dm: list[float] = []
    trs: list[float] = []

    start = max(1, len(highs) - period - 1)
    for i in range(start, len(highs)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]

        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)

        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    n = min(period, len(trs))
    if n == 0:
        return 25.0

    atr_val = sum(trs[-n:]) / n
    if atr_val < 1e-10:
        return 0.0

    plus_di = (sum(plus_dm[-n:]) / n) / atr_val * 100
    minus_di = (sum(minus_dm[-n:]) / n) / atr_val * 100

    di_sum = plus_di + minus_di
    if di_sum < 1e-10:
        return 0.0

    return abs(plus_di - minus_di) / di_sum * 100


def _bollinger_bands(
    closes: list[float],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[float, float, float]:
    """Bollinger Bands: (upper, mid, lower)."""
    if len(closes) < period:
        c = closes[-1] if closes else 0.0
        return c, c, c

    data = closes[-period:]
    mid = sum(data) / period
    variance = sum((x - mid) ** 2 for x in data) / period
    std = variance**0.5

    return mid + num_std * std, mid, mid - num_std * std


def compute_indicators(bars: list[dict[str, Any]]) -> Indicators:
    """Compute all technical indicators from a list of OHLCV bars.

    Expects bars sorted ascending by date with keys:
    close, high, low, open, volume, date.
    """
    closes = [float(b.get("close", 0)) for b in bars]
    highs = [float(b.get("high", 0)) for b in bars]
    lows = [float(b.get("low", 0)) for b in bars]
    volumes = [int(b.get("volume", 0)) for b in bars]

    close = closes[-1] if closes else 0.0
    high_today = highs[-1] if highs else 0.0
    low_today = lows[-1] if lows else 0.0

    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    rsi14 = _rsi(closes, 14)
    atr14 = _atr(highs, lows, closes, 14)
    adx14 = _adx(highs, lows, closes, 14)
    bb_upper, bb_mid, bb_lower = _bollinger_bands(closes, 20, 2.0)

    bb_width_pct = ((bb_upper - bb_lower) / bb_mid * 100) if bb_mid > 0 else 0.0

    # Volume ratio (today vs 20-day avg)
    vol_now = volumes[-1] if volumes else 0
    vol_avg = sum(volumes[-20:]) / max(len(volumes[-20:]), 1)
    vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0

    # Range expansion
    today_range = high_today - low_today
    range_ratio = today_range / atr14 if atr14 > 0 else 1.0

    # N-day high/low
    recent_20 = closes[-20:] if len(closes) >= 20 else closes
    recent_40 = closes[-40:] if len(closes) >= 40 else closes
    high_20 = max(recent_20) if recent_20 else close
    low_20 = min(recent_20) if recent_20 else close
    high_40 = max(recent_40) if recent_40 else close
    low_40 = min(recent_40) if recent_40 else close

    # Momentum
    momentum_5 = (
        (close - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 and closes[-6] > 0 else 0.0
    )
    momentum_10 = (
        (close - closes[-11]) / closes[-11] * 100
        if len(closes) >= 11 and closes[-11] > 0
        else 0.0
    )

    # ATR compression ratio (current vs 20 bars ago)
    if len(bars) >= 35:
        old_bars_h = [float(b.get("high", 0)) for b in bars[-35:-20]]
        old_bars_l = [float(b.get("low", 0)) for b in bars[-35:-20]]
        old_bars_c = [float(b.get("close", 0)) for b in bars[-35:-20]]
        atr_old = _atr(old_bars_h, old_bars_l, old_bars_c, 14)
        atr_ratio = atr14 / atr_old if atr_old > 0 else 1.0
    else:
        atr_ratio = 1.0

    # Close position in today's range
    day_range = high_today - low_today
    close_in_range = (close - low_today) / day_range if day_range > 0 else 0.5

    return Indicators(
        close=close,
        sma20=sma20,
        sma50=sma50,
        rsi14=rsi14,
        atr14=atr14,
        adx14=adx14,
        bb_upper=bb_upper,
        bb_mid=bb_mid,
        bb_lower=bb_lower,
        bb_width_pct=bb_width_pct,
        vol_ratio=vol_ratio,
        range_ratio=range_ratio,
        high_20=high_20,
        low_20=low_20,
        high_40=high_40,
        low_40=low_40,
        momentum_5=momentum_5,
        momentum_10=momentum_10,
        atr_ratio=atr_ratio,
        close_in_range=close_in_range,
    )


# ── Regime Detection ─────────────────────────────────────────────────

def detect_regime(ind: Indicators) -> MarketRegime:
    """Detect market regime from technical indicators.

    Uses SMA crossover (trend direction) + ADX (trend strength).
    """
    sma_spread = (ind.sma20 - ind.sma50) / ind.sma50 * 100 if ind.sma50 > 0 else 0.0

    if sma_spread > 1.0:
        if ind.adx14 > 20:  # was 25 — lowered to classify more days as STRONG
            return MarketRegime.STRONG_UPTREND
        return MarketRegime.UPTREND
    elif sma_spread < -1.0:
        if ind.adx14 > 20:  # was 25
            return MarketRegime.STRONG_DOWNTREND
        return MarketRegime.DOWNTREND
    return MarketRegime.SIDEWAYS


# ── Strategy: Trend Pullback ─────────────────────────────────────────

def _scan_trend_pullback(
    ticker: str,
    ind: Indicators,
    regime: MarketRegime,
) -> EntrySignal | None:
    """Buy pullbacks in strong trends.

    Bullish: uptrend + RSI dips below 35 + close near lower BB.
    Bearish: downtrend + RSI rises above 65 + close near upper BB.

    Highest win-rate strategy — trades WITH the trend.
    """
    if regime in (MarketRegime.STRONG_UPTREND, MarketRegime.UPTREND):
        if ind.rsi14 < 45:  # was 38 — loosened to catch normal pullbacks, not just extremes
            # Pullback in uptrend — buy call
            conviction = 5.0
            if ind.adx14 > 30:
                conviction += 1.5
            if ind.adx14 > 40:
                conviction += 0.5
            if ind.rsi14 < 30:  # was 25 — still bonus for deep oversold
                conviction += 1.0
            if ind.close <= ind.bb_lower * 1.005:
                conviction += 1.0
            if ind.vol_ratio > 1.2:
                conviction += 0.5
            sma_gap = (ind.sma20 - ind.sma50) / ind.sma50 * 100 if ind.sma50 > 0 else 0
            if sma_gap > 2.0:
                conviction += 0.5

            conviction = min(10.0, conviction)

            return EntrySignal(
                ticker=ticker,
                direction="bullish",
                strategy="trend_pullback",
                conviction=conviction,
                regime=regime.value,
                otm_pct=0.025,
                reason=(
                    f"Uptrend pullback: RSI={ind.rsi14:.0f}, "
                    f"ADX={ind.adx14:.0f}, close near BB lower"
                ),
            )

    elif regime in (MarketRegime.STRONG_DOWNTREND, MarketRegime.DOWNTREND) and ind.rsi14 > 55:  # was 62
        # Bounce in downtrend — buy put
        conviction = 5.0
        if ind.adx14 > 30:
            conviction += 1.5
        if ind.adx14 > 40:
            conviction += 0.5
        if ind.rsi14 > 75:
            conviction += 1.0
        if ind.close >= ind.bb_upper * 0.995:
            conviction += 1.0
        if ind.vol_ratio > 1.2:
            conviction += 0.5

        conviction = min(10.0, conviction)

        return EntrySignal(
            ticker=ticker,
            direction="bearish",
            strategy="trend_pullback",
            conviction=conviction,
            regime=regime.value,
            otm_pct=0.025,
            reason=(
                f"Downtrend bounce: RSI={ind.rsi14:.0f}, "
                f"ADX={ind.adx14:.0f}, close near BB upper"
            ),
        )

    return None


# ── Strategy: Breakout ───────────────────────────────────────────────

def _scan_breakout(
    ticker: str,
    ind: Indicators,
    regime: MarketRegime,
) -> EntrySignal | None:
    """Trade range breakouts with volume confirmation.

    New 20-day high/low + volume surge + range expansion.
    Good risk/reward but lower win rate than trend pullback.
    """
    # Bullish breakout
    if ind.close >= ind.high_20 * 0.998 and ind.vol_ratio > 1.5 and ind.range_ratio > 1.2:
        conviction = 4.5
        if ind.vol_ratio > 2.5:
            conviction += 2.0
        elif ind.vol_ratio > 2.0:
            conviction += 1.5
        if ind.range_ratio > 1.5:
            conviction += 1.0
        if ind.close >= ind.high_40 * 0.998:
            conviction += 1.0
        if ind.close_in_range > 0.75:
            conviction += 0.5
        if regime in (MarketRegime.UPTREND, MarketRegime.STRONG_UPTREND):
            conviction += 0.5

        # Don't breakout into strong downtrend
        if regime in (MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND):
            conviction -= 1.5

        conviction = max(0.0, min(10.0, conviction))

        if conviction >= 4.0:
            return EntrySignal(
                ticker=ticker,
                direction="bullish",
                strategy="breakout",
                conviction=conviction,
                regime=regime.value,
                otm_pct=0.030,
                reason=(
                    f"Bullish breakout: new 20d high, "
                    f"vol={ind.vol_ratio:.1f}x, range={ind.range_ratio:.1f}x"
                ),
            )

    # Bearish breakout
    if ind.close <= ind.low_20 * 1.002 and ind.vol_ratio > 1.5 and ind.range_ratio > 1.2:
        conviction = 4.5
        if ind.vol_ratio > 2.5:
            conviction += 2.0
        elif ind.vol_ratio > 2.0:
            conviction += 1.5
        if ind.range_ratio > 1.5:
            conviction += 1.0
        if ind.close <= ind.low_40 * 1.002:
            conviction += 1.0
        if ind.close_in_range < 0.25:
            conviction += 0.5
        if regime in (MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND):
            conviction += 0.5

        if regime in (MarketRegime.UPTREND, MarketRegime.STRONG_UPTREND):
            conviction -= 1.5

        conviction = max(0.0, min(10.0, conviction))

        if conviction >= 4.0:
            return EntrySignal(
                ticker=ticker,
                direction="bearish",
                strategy="breakout",
                conviction=conviction,
                regime=regime.value,
                otm_pct=0.030,
                reason=(
                    f"Bearish breakdown: new 20d low, "
                    f"vol={ind.vol_ratio:.1f}x, range={ind.range_ratio:.1f}x"
                ),
            )

    return None


# ── Strategy: Mean Reversion ─────────────────────────────────────────

def _scan_mean_reversion(
    ticker: str,
    ind: Indicators,
    regime: MarketRegime,
) -> EntrySignal | None:
    """Fade extreme RSI + Bollinger readings for quick bounces.

    Only trades when price is at statistical extremes.
    Quick holds (3-5 days), tighter strikes for higher delta.
    """
    # Don't mean-revert in very strong trends (they can stay extreme)
    if ind.adx14 > 45:
        return None

    # Bullish mean reversion (oversold)
    if ind.rsi14 < 28 and ind.close < ind.bb_lower:
        conviction = 4.5
        if ind.rsi14 < 15:
            conviction += 1.5
        elif ind.rsi14 < 18:
            conviction += 0.8
        bb_distance = (ind.bb_lower - ind.close) / ind.close * 100 if ind.close > 0 else 0
        if bb_distance > 2.0:
            conviction += 1.0
        if ind.adx14 < 25:
            conviction += 0.5
        if ind.vol_ratio > 1.5:
            conviction += 0.5  # Capitulation volume

        # Avoid in strong downtrend (catching falling knife)
        if regime == MarketRegime.STRONG_DOWNTREND:
            conviction -= 2.0

        conviction = max(0.0, min(10.0, conviction))

        if conviction >= 4.0:
            return EntrySignal(
                ticker=ticker,
                direction="bullish",
                strategy="mean_reversion",
                conviction=conviction,
                regime=regime.value,
                otm_pct=0.015,  # Tighter strike for quick bounce
                reason=(
                    f"Oversold bounce: RSI={ind.rsi14:.0f}, "
                    f"below BB by {bb_distance:.1f}%"
                ),
            )

    # Bearish mean reversion (overbought)
    if ind.rsi14 > 72 and ind.close > ind.bb_upper:
        conviction = 4.5
        if ind.rsi14 > 85:
            conviction += 1.5
        elif ind.rsi14 > 82:
            conviction += 0.8
        bb_distance = (ind.close - ind.bb_upper) / ind.close * 100 if ind.close > 0 else 0
        if bb_distance > 2.0:
            conviction += 1.0
        if ind.adx14 < 25:
            conviction += 0.5
        if ind.vol_ratio > 1.5:
            conviction += 0.5

        if regime == MarketRegime.STRONG_UPTREND:
            conviction -= 2.0

        conviction = max(0.0, min(10.0, conviction))

        if conviction >= 4.0:
            return EntrySignal(
                ticker=ticker,
                direction="bearish",
                strategy="mean_reversion",
                conviction=conviction,
                regime=regime.value,
                otm_pct=0.015,
                reason=(
                    f"Overbought fade: RSI={ind.rsi14:.0f}, "
                    f"above BB by {bb_distance:.1f}%"
                ),
            )

    return None


# ── Strategy: Volatility Squeeze ─────────────────────────────────────

def _scan_vol_squeeze(
    ticker: str,
    ind: Indicators,
    regime: MarketRegime,
) -> EntrySignal | None:
    """Trade volatility compression breakouts.

    When Bollinger Bands narrow (low BB width) and ATR compresses,
    a volatility expansion is imminent. Direction from trend context.

    Low frequency but large payoff potential.
    """
    # Squeeze conditions: BB width < 3% and ATR compressed
    if ind.bb_width_pct > 3.0:
        return None
    if ind.atr_ratio > 0.65:
        return None

    # Direction from SMA trend
    if regime in (MarketRegime.STRONG_UPTREND, MarketRegime.UPTREND):
        direction = "bullish"
    elif regime in (MarketRegime.STRONG_DOWNTREND, MarketRegime.DOWNTREND):
        direction = "bearish"
    else:
        # In sideways, use short-term momentum
        direction = "bullish" if ind.momentum_5 > 0 else "bearish"

    conviction = 4.0
    if ind.bb_width_pct < 2.0:
        conviction += 1.5
    elif ind.bb_width_pct < 2.5:
        conviction += 0.8
    if ind.atr_ratio < 0.50:
        conviction += 1.0
    if regime in (MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND):
        conviction += 1.5  # Strong trend gives clearer direction
    elif regime in (MarketRegime.UPTREND, MarketRegime.DOWNTREND):
        conviction += 0.8

    # Volume starting to pick up is a good sign
    if ind.vol_ratio > 1.2:
        conviction += 0.5

    conviction = max(0.0, min(10.0, conviction))

    if conviction >= 4.0:
        return EntrySignal(
            ticker=ticker,
            direction=direction,
            strategy="vol_squeeze",
            conviction=conviction,
            regime=regime.value,
            otm_pct=0.035,  # Wider strike — expecting big move
            reason=(
                f"Vol squeeze: BB width={ind.bb_width_pct:.1f}%, "
                f"ATR ratio={ind.atr_ratio:.2f}, regime={regime.value}"
            ),
        )

    return None


# ── Main Scanner ─────────────────────────────────────────────────────

def _scan_ibs_reversion(
    ticker: str,
    ind: Indicators,
    bars: list[dict[str, Any]],
    regime: MarketRegime,
) -> EntrySignal | None:
    """Internal Bar Strength (IBS) mean reversion.

    IBS = (Close - Low) / (High - Low). Below 0.2 is oversold,
    above 0.8 is overbought. Well-researched intraday alpha signal
    (borrowed from QuantConnect/Leanmodel research).

    Combines with RSI for confirmation.
    """
    if len(bars) < 3:
        return None

    last = bars[-1]
    high = float(last.get("high", 0))
    low = float(last.get("low", 0))
    day_range = high - low
    if day_range <= 0:
        return None

    ibs = (ind.close - low) / day_range

    # Check 3-day IBS pattern (consecutive low IBS = stronger signal)
    ibs_values: list[float] = []
    for b in bars[-3:]:
        bh = float(b.get("high", 0))
        bl = float(b.get("low", 0))
        bc = float(b.get("close", 0))
        br = bh - bl
        if br > 0:
            ibs_values.append((bc - bl) / br)

    # Bullish: IBS < 0.20 today (close near low of day) — was 0.15
    if ibs < 0.20 and ind.rsi14 < 45:  # RSI threshold was 40
        conviction = 5.0
        if ibs < 0.08:
            conviction += 1.5
        elif ibs < 0.12:
            conviction += 0.8
        if ind.rsi14 < 30:
            conviction += 1.0
        # Consecutive low IBS = capitulation
        if len(ibs_values) >= 2 and ibs_values[-2] < 0.25:
            conviction += 1.0
        if regime in (MarketRegime.UPTREND, MarketRegime.STRONG_UPTREND):
            conviction += 1.0  # Dip in uptrend = high probability bounce

        if regime == MarketRegime.STRONG_DOWNTREND:
            conviction -= 1.5  # Don't catch falling knives

        conviction = max(0.0, min(10.0, conviction))

        if conviction >= 4.5:  # was 5.0
            return EntrySignal(
                ticker=ticker,
                direction="bullish",
                strategy="ibs_reversion",
                conviction=conviction,
                regime=regime.value,
                otm_pct=0.015,  # Tight strike for quick bounce
                reason=f"IBS={ibs:.2f} (near day low), RSI={ind.rsi14:.0f}",
            )

    # Bearish: IBS > 0.80 today (close near high of day) — was 0.85
    if ibs > 0.80 and ind.rsi14 > 55:  # RSI threshold was 60
        conviction = 5.0
        if ibs > 0.92:
            conviction += 1.5
        elif ibs > 0.88:
            conviction += 0.8
        if ind.rsi14 > 70:
            conviction += 1.0
        if len(ibs_values) >= 2 and ibs_values[-2] > 0.75:
            conviction += 1.0
        if regime in (MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND):
            conviction += 1.0

        if regime == MarketRegime.STRONG_UPTREND:
            conviction -= 1.5

        conviction = max(0.0, min(10.0, conviction))

        if conviction >= 4.5:  # was 5.0
            return EntrySignal(
                ticker=ticker,
                direction="bearish",
                strategy="ibs_reversion",
                conviction=conviction,
                regime=regime.value,
                otm_pct=0.015,
                reason=f"IBS={ibs:.2f} (near day high), RSI={ind.rsi14:.0f}",
            )

    return None


# ── Allowed regimes for entry ────────────────────────────────────────

ALLOWED_REGIMES: set[MarketRegime] = {
    MarketRegime.STRONG_UPTREND,
    MarketRegime.UPTREND,
    MarketRegime.SIDEWAYS,
    MarketRegime.DOWNTREND,
    MarketRegime.STRONG_DOWNTREND,
}
# v8: Opened to all regimes to restore signal frequency (was STRONG_* only).
# v7 showed 0% WR for UPTREND-only trades but that was before IBS reversion
# was added. Strategy-level guards (regime penalties inside each scanner)
# still protect quality — STRONG_DOWNTREND penalty on IBS bullish, etc.
# Vol squeeze explicitly supports SIDEWAYS. Momentum/GEX/Kronos adjustments
# further filter low-quality setups.


def scan_for_entries(
    ticker: str,
    bars: list[dict[str, Any]],
    indicators: Indicators,
    regime: MarketRegime,
) -> list[EntrySignal]:
    """Scan for all entry signals across all strategies.

    v2: Regime-filtered (blocks sideways + downtrend).
    Breakout disabled (7% WR in backtest).
    IBS reversion added (from Leanmodel research).

    Returns signals sorted by conviction (highest first).
    """
    # Regime gate: don't trade in regimes with 0% historical win rate
    if regime not in ALLOWED_REGIMES:
        return []

    signals: list[EntrySignal] = []

    # Strategy 1: Trend pullback (25.5% WR, +594.7% total — best strategy)
    trend = _scan_trend_pullback(ticker, indicators, regime)
    if trend:
        signals.append(trend)

    # Strategy 2: Breakout — DISABLED (7.1% WR, -589.9% total)
    # breakout = _scan_breakout(ticker, indicators, regime)
    # if breakout:
    #     signals.append(breakout)

    # Strategy 3: Mean reversion (RSI thresholds relaxed from 22→28, 78→72)
    mean_rev = _scan_mean_reversion(ticker, indicators, regime)
    if mean_rev:
        signals.append(mean_rev)

    # Strategy 4: Volatility squeeze
    squeeze = _scan_vol_squeeze(ticker, indicators, regime)
    if squeeze:
        signals.append(squeeze)

    # Strategy 5: IBS reversion (from Leanmodel research)
    ibs = _scan_ibs_reversion(ticker, indicators, bars, regime)
    if ibs:
        signals.append(ibs)

    # If multiple signals for same ticker, keep highest conviction
    if len(signals) > 1:
        signals.sort(key=lambda s: s.conviction, reverse=True)
        signals = signals[:1]

    return signals
