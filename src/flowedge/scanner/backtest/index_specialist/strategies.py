"""Index-specific trading strategies.

Strategies optimized for SPY/QQQ/IWM characteristics:
- High liquidity → tight spreads, fast fills
- Mean-reverting at short horizon (1-3 day scalps)
- Trend-following at medium horizon (10-20 day swings)
- VIX-sensitive regime shifts

Strategies:
1. IBS mean-reversion scalp — buy dips on oversold IBS, hold 1-3 days
2. Momentum continuation swing — buy breakouts with volume confirmation
3. VIX spike reversal — buy after VIX spikes (panic selling overdone)
4. Multi-MA alignment trend — ride sustained trend with tight trail
5. Range breakout — buy/sell breakout from consolidation range
"""

from __future__ import annotations

from typing import Any

from flowedge.scanner.backtest.index_specialist.schemas import (
    IndexRegime,
    IndexSignal,
    TradeHorizon,
)
from flowedge.scanner.backtest.strategies import Indicators


def classify_index_regime(ind: Indicators) -> IndexRegime:
    """Multi-factor index regime classification.

    Uses SMA alignment, ADX strength, RSI zones, and momentum
    to produce a 7-level regime classification optimized for indices.
    """
    price = ind.close
    sma20 = ind.sma20
    sma50 = ind.sma50
    adx = ind.adx14
    rsi = ind.rsi14

    above_20 = price > sma20
    above_50 = price > sma50
    sma20_above_50 = sma20 > sma50

    if above_20 and above_50 and sma20_above_50 and adx > 25:
        if rsi > 55:
            return IndexRegime.STRONG_BULL
        return IndexRegime.BULL

    if above_20 and above_50 and sma20_above_50:
        return IndexRegime.BULL

    if above_20 and not sma20_above_50:
        return IndexRegime.NEUTRAL_BULL

    if not above_20 and sma20_above_50:
        return IndexRegime.NEUTRAL_BEAR

    if not above_20 and not above_50 and not sma20_above_50 and adx > 25:
        if rsi < 45:
            return IndexRegime.STRONG_BEAR
        return IndexRegime.BEAR

    if not above_20 and not above_50 and not sma20_above_50:
        return IndexRegime.BEAR

    return IndexRegime.NEUTRAL


def _compute_volume_surge(bars: list[dict[str, Any]], period: int = 20) -> float:
    """Ratio of current volume to average volume."""
    if len(bars) < period + 1:
        return 1.0
    recent_vols = [float(b.get("volume", 0)) for b in bars[-(period + 1):-1]]
    avg_vol = sum(recent_vols) / len(recent_vols) if recent_vols else 1.0
    current_vol = float(bars[-1].get("volume", 0))
    return current_vol / avg_vol if avg_vol > 0 else 1.0


def _compute_ibs(bar: dict[str, Any]) -> float:
    """Internal Bar Strength = (Close - Low) / (High - Low)."""
    high = float(bar.get("high", 0))
    low = float(bar.get("low", 0))
    close = float(bar.get("close", 0))
    rng = high - low
    if rng <= 0:
        return 0.5
    return (close - low) / rng


def _compute_3day_rsi(closes: list[float]) -> float:
    """Ultra-short RSI(3) for scalp timing."""
    if len(closes) < 4:
        return 50.0
    gains: list[float] = []
    losses_: list[float] = []
    for i in range(-3, 0):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses_.append(0.0)
        else:
            gains.append(0.0)
            losses_.append(abs(diff))
    avg_gain = sum(gains) / 3
    avg_loss = sum(losses_) / 3
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _bb_position(ind: Indicators) -> float:
    """Position within Bollinger Bands (0=lower, 1=upper)."""
    bb_range = ind.bb_upper - ind.bb_lower
    if bb_range <= 0:
        return 0.5
    return (ind.close - ind.bb_lower) / bb_range


def _check_range_breakout(
    bars: list[dict[str, Any]], period: int = 10,
) -> tuple[str, float]:
    """Detect price breakout from recent consolidation range.

    Returns (direction, conviction_modifier).
    """
    if len(bars) < period + 1:
        return "", 0.0

    recent = bars[-(period + 1):-1]
    highs = [float(b.get("high", 0)) for b in recent]
    lows = [float(b.get("low", 0)) for b in recent]
    range_high = max(highs)
    range_low = min(lows)
    rng = range_high - range_low
    avg_price = (range_high + range_low) / 2

    if avg_price <= 0 or rng / avg_price > 0.04:
        return "", 0.0  # Range too wide — not consolidation

    current = float(bars[-1].get("close", 0))
    if current > range_high:
        breakout_pct = (current - range_high) / range_high * 100
        return "bullish", min(2.0, breakout_pct * 2)
    if current < range_low:
        breakout_pct = (range_low - current) / range_low * 100
        return "bearish", min(2.0, breakout_pct * 2)
    return "", 0.0


def scan_index_entries(
    ticker: str,
    bars: list[dict[str, Any]],
    indicators: Indicators,
    regime: IndexRegime,
) -> list[IndexSignal]:
    """Scan for index-specific entry signals across all horizons."""
    signals: list[IndexSignal] = []
    closes = [float(b.get("close", 0)) for b in bars if float(b.get("close", 0)) > 0]

    if len(closes) < 55 or len(bars) < 55:
        return signals

    current_bar = bars[-1]
    ibs = _compute_ibs(current_bar)
    vol_surge = _compute_volume_surge(bars)
    rsi3 = _compute_3day_rsi(closes)
    bb_pos = _bb_position(indicators)

    # ── Strategy 1: IBS Mean-Reversion Scalp ──
    # Buy when IBS < 0.15 (extreme selling) in non-bear regime
    if ibs < 0.15 and regime not in (IndexRegime.STRONG_BEAR, IndexRegime.BEAR):
        conviction = 7.0
        # Boost for extreme oversold
        if ibs < 0.08:
            conviction += 1.0
        if rsi3 < 15:
            conviction += 0.5
        if bb_pos < 0.1:
            conviction += 0.5
        if vol_surge > 1.5:
            conviction += 0.5
        # Regime boost
        if regime in (IndexRegime.STRONG_BULL, IndexRegime.BULL):
            conviction += 0.5

        signals.append(IndexSignal(
            ticker=ticker,
            direction="bullish",
            horizon=TradeHorizon.SCALP,
            conviction=min(10.0, conviction),
            regime=regime,
            strategy="ibs_reversal_scalp",
            otm_pct=0.010,  # Tight OTM for quick moves
            entry_reason=f"IBS={ibs:.2f} RSI3={rsi3:.0f} BB={bb_pos:.2f}",
            momentum_score=min(10.0, (50 - rsi3) / 5),
            volume_score=min(10.0, vol_surge * 2),
        ))

    # IBS > 0.85 → bearish scalp in non-bull regime
    if ibs > 0.85 and regime not in (IndexRegime.STRONG_BULL, IndexRegime.BULL):
        conviction = 7.0
        if ibs > 0.92:
            conviction += 1.0
        if rsi3 > 85:
            conviction += 0.5
        if bb_pos > 0.9:
            conviction += 0.5
        if vol_surge > 1.5:
            conviction += 0.5
        if regime in (IndexRegime.STRONG_BEAR, IndexRegime.BEAR):
            conviction += 0.5

        signals.append(IndexSignal(
            ticker=ticker,
            direction="bearish",
            horizon=TradeHorizon.SCALP,
            conviction=min(10.0, conviction),
            regime=regime,
            strategy="ibs_reversal_scalp",
            otm_pct=0.010,
            entry_reason=f"IBS={ibs:.2f} RSI3={rsi3:.0f} BB={bb_pos:.2f}",
            momentum_score=min(10.0, (rsi3 - 50) / 5),
            volume_score=min(10.0, vol_surge * 2),
        ))

    # ── Strategy 2: Momentum Continuation Swing ──
    # Strong trend with pullback + volume confirmation
    if regime in (IndexRegime.STRONG_BULL, IndexRegime.BULL):
        # Pullback to SMA20 in uptrend
        pct_from_sma20 = (indicators.close - indicators.sma20) / indicators.sma20 * 100
        if -1.5 < pct_from_sma20 < 0.5 and indicators.adx14 > 20:
            conviction = 7.5
            if indicators.adx14 > 30:
                conviction += 0.5
            if vol_surge > 1.2:
                conviction += 0.5
            if regime == IndexRegime.STRONG_BULL:
                conviction += 0.5
            # MACD confirmation
            if len(closes) > 26:
                ema12 = closes[-1]
                ema26 = closes[-1]
                for c in closes[-26:]:
                    ema12 = ema12 * (1 - 2 / 13) + c * (2 / 13)
                    ema26 = ema26 * (1 - 2 / 27) + c * (2 / 27)
                if ema12 > ema26:
                    conviction += 0.5

            signals.append(IndexSignal(
                ticker=ticker,
                direction="bullish",
                horizon=TradeHorizon.SWING,
                conviction=min(10.0, conviction),
                regime=regime,
                strategy="momentum_pullback_swing",
                otm_pct=0.015,
                entry_reason=f"Pullback {pct_from_sma20:.1f}% to SMA20 ADX={indicators.adx14:.0f}",
                momentum_score=min(10.0, indicators.adx14 / 5),
                volume_score=min(10.0, vol_surge * 2),
            ))

    # Bearish continuation in downtrend
    if regime in (IndexRegime.STRONG_BEAR, IndexRegime.BEAR):
        pct_from_sma20 = (indicators.close - indicators.sma20) / indicators.sma20 * 100
        if -0.5 < pct_from_sma20 < 1.5 and indicators.adx14 > 20:
            conviction = 7.5
            if indicators.adx14 > 30:
                conviction += 0.5
            if vol_surge > 1.2:
                conviction += 0.5
            if regime == IndexRegime.STRONG_BEAR:
                conviction += 0.5

            signals.append(IndexSignal(
                ticker=ticker,
                direction="bearish",
                horizon=TradeHorizon.SWING,
                conviction=min(10.0, conviction),
                regime=regime,
                strategy="momentum_pullback_swing",
                otm_pct=0.015,
                entry_reason=f"Bear rally {pct_from_sma20:.1f}% SMA20 ADX={indicators.adx14:.0f}",
                momentum_score=min(10.0, indicators.adx14 / 5),
                volume_score=min(10.0, vol_surge * 2),
            ))

    # ── Strategy 3: Multi-MA Alignment Trend (Medium Term) ──
    # All MAs perfectly aligned → ride the trend
    if (
        regime == IndexRegime.STRONG_BULL
        and indicators.rsi14 > 50
        and indicators.adx14 > 25
        and indicators.rsi14 < 75
    ):
        conviction = 7.5
        if indicators.adx14 > 35:
            conviction += 0.5
        if 55 < indicators.rsi14 < 70:
            conviction += 0.5
        if vol_surge > 1.0:
            conviction += 0.5

        rsi_s = f"{indicators.rsi14:.0f}"
        adx_s = f"{indicators.adx14:.0f}"
        signals.append(IndexSignal(
            ticker=ticker,
            direction="bullish",
            horizon=TradeHorizon.MEDIUM,
            conviction=min(10.0, conviction),
            regime=regime,
            strategy="ma_alignment_trend",
            otm_pct=0.020,
            entry_reason=f"Full MA alignment RSI={rsi_s} ADX={adx_s}",
            momentum_score=min(10.0, indicators.adx14 / 4),
            volume_score=min(10.0, vol_surge * 2),
        ))

    if (
        regime == IndexRegime.STRONG_BEAR
        and indicators.rsi14 < 50
        and indicators.adx14 > 25
        and indicators.rsi14 > 25
    ):
        conviction = 7.5
        if indicators.adx14 > 35:
            conviction += 0.5
        if 30 < indicators.rsi14 < 45:
            conviction += 0.5

        rsi_s = f"{indicators.rsi14:.0f}"
        adx_s = f"{indicators.adx14:.0f}"
        signals.append(IndexSignal(
            ticker=ticker,
            direction="bearish",
            horizon=TradeHorizon.MEDIUM,
            conviction=min(10.0, conviction),
            regime=regime,
            strategy="ma_alignment_trend",
            otm_pct=0.020,
            entry_reason=f"Full bear MA alignment RSI={rsi_s} ADX={adx_s}",
            momentum_score=min(10.0, indicators.adx14 / 4),
            volume_score=min(10.0, vol_surge * 2),
        ))

    # ── Strategy 4: Range Breakout ──
    breakout_dir, breakout_mod = _check_range_breakout(bars, period=10)
    if breakout_dir and breakout_mod > 0.5 and vol_surge > 1.3:
        conviction = 7.0 + breakout_mod
        if vol_surge > 2.0:
            conviction += 0.5

        signals.append(IndexSignal(
            ticker=ticker,
            direction=breakout_dir,
            horizon=TradeHorizon.SWING,
            conviction=min(10.0, conviction),
            regime=regime,
            strategy="range_breakout",
            otm_pct=0.015,
            entry_reason=f"Range breakout {breakout_dir} vol_surge={vol_surge:.1f}x",
            momentum_score=min(10.0, breakout_mod * 3),
            volume_score=min(10.0, vol_surge * 2),
        ))

    return signals
