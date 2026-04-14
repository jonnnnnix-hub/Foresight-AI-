"""Trident signal library — pure functions that score every 5-min bar.

Each signal returns a float in [-1, +1]:
  +1 = strong call signal (bullish)
  -1 = strong put signal (bearish)
   0 = no signal

The backtester combines signals via confluence count and direction gating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ── Bar helpers ───────────────────────────────────────────────────

@dataclass
class Bar:
    """Normalised OHLCV bar."""

    ts: int              # nanosecond epoch
    o: float
    h: float
    lo: float            # renamed from 'l' to avoid E741
    c: float
    v: int
    vwap: float = 0.0    # cumulative VWAP (set externally)
    date: str = ""

    @property
    def ibs(self) -> float:
        """Internal Bar Strength: (close - low) / (high - low)."""
        span = self.h - self.lo
        if span <= 0:
            return 0.5
        return (self.c - self.lo) / span

    @property
    def is_red(self) -> bool:
        return self.c < self.o

    @property
    def is_green(self) -> bool:
        return self.c >= self.o

    @property
    def body_pct(self) -> float:
        if self.o <= 0:
            return 0.0
        return (self.c - self.o) / self.o


@dataclass
class SignalSnapshot:
    """All computed indicator values for a single bar.

    Populated by compute_signals() and consumed by the backtester
    to decide entries.
    """

    bar_idx: int
    bar: Bar

    # Raw indicator values
    rsi3: float = 50.0
    rsi14: float = 50.0
    ibs: float = 0.5
    vwap_distance_pct: float = 0.0    # (price - vwap) / vwap * 100
    volume_ratio: float = 1.0          # current vol / 10-bar avg
    intraday_pct: float = 0.0          # (close - day_open) / day_open * 100
    prior_bar_red: bool = False
    prior_bar_green: bool = False
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    ema_bullish_cross: bool = False    # fast crossed above slow
    ema_bearish_cross: bool = False    # fast crossed below slow
    macd_hist: float = 0.0
    macd_hist_prev: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_mid: float = 0.0
    opening_range_high: float = 0.0
    opening_range_low: float = 0.0
    daily_trend_up: bool = True        # SMA(10) > SMA(20) on daily
    sma5: float = 0.0
    sma10: float = 0.0

    # Directional signal scores [-1, +1]
    call_signals_fired: int = 0
    put_signals_fired: int = 0
    conviction: float = 0.0


# ── Indicator computation ─────────────────────────────────────────

def _rsi(closes: list[float], period: int) -> float:
    """Wilder RSI on a list of close prices."""
    if len(closes) < period + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = closes[-period - 1 + i] - closes[-period - 1 + i - 1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _ema(values: list[float], period: int) -> float:
    """Exponential moving average of last N values."""
    if not values:
        return 0.0
    if len(values) < period:
        return sum(values) / len(values)
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def _sma(values: list[float], period: int) -> float:
    """Simple moving average of last N values."""
    if len(values) < period:
        return sum(values) / len(values) if values else 0.0
    return sum(values[-period:]) / period


def _macd(closes: list[float]) -> tuple[float, float, float]:
    """MACD (12, 26, 9). Returns (macd_line, signal_line, histogram)."""
    if len(closes) < 26:
        return 0.0, 0.0, 0.0
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    # Approximate signal line from recent MACD values
    # (simplified — compute MACD for last 9 bars)
    macd_history = []
    for end in range(max(26, len(closes) - 9), len(closes) + 1):
        chunk = closes[:end]
        if len(chunk) >= 26:
            e12 = _ema(chunk, 12)
            e26 = _ema(chunk, 26)
            macd_history.append(e12 - e26)
    signal = _ema(macd_history, 9) if len(macd_history) >= 9 else macd_line
    return macd_line, signal, macd_line - signal


def _bollinger(
    closes: list[float], period: int = 20, num_std: float = 2.0,
) -> tuple[float, float, float]:
    """Bollinger Bands. Returns (upper, mid, lower)."""
    if len(closes) < period:
        mid = sum(closes) / len(closes) if closes else 0.0
        return mid, mid, mid
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)
    return mid + num_std * std, mid, mid - num_std * std


def _cumulative_vwap(bars: list[Bar]) -> list[float]:
    """Compute cumulative VWAP for a day's bars.

    Returns list of VWAP values aligned with bars.
    Uses typical price * volume accumulation.
    """
    vwaps: list[float] = []
    cum_pv = 0.0
    cum_v = 0
    for b in bars:
        typical = (b.h + b.lo + b.c) / 3.0
        cum_pv += typical * b.v
        cum_v += b.v
        vwaps.append(cum_pv / cum_v if cum_v > 0 else b.c)
    return vwaps


# ── Main signal computation ──────────────────────────────────────

def compute_all_signals(
    bars_5m: list[Bar],
    daily_closes: list[float] | None = None,
    *,
    ema_fast_period: int = 8,
    ema_slow_period: int = 21,
    bollinger_period: int = 20,
    bollinger_std: float = 2.0,
    opening_range_bars: int = 3,  # first 15 min = 3 × 5-min bars
) -> list[SignalSnapshot]:
    """Compute all indicator values for every bar in a day.

    Args:
        bars_5m: Day's 5-minute bars (sorted by timestamp).
        daily_closes: Recent daily close prices (for trend detection).
        ema_fast_period: Fast EMA period.
        ema_slow_period: Slow EMA period.
        bollinger_period: BB lookback.
        bollinger_std: BB standard deviations.
        opening_range_bars: Number of bars for opening range.

    Returns:
        List of SignalSnapshot, one per bar.
    """
    if not bars_5m:
        return []

    # Pre-compute cumulative VWAP
    vwap_values = _cumulative_vwap(bars_5m)
    for i, b in enumerate(bars_5m):
        b.vwap = vwap_values[i]

    day_open = bars_5m[0].o

    # Opening range
    or_bars = bars_5m[:opening_range_bars]
    or_high = max(b.h for b in or_bars) if or_bars else 0.0
    or_low = min(b.lo for b in or_bars) if or_bars else 0.0

    # Daily trend
    daily_trend_up = True
    if daily_closes and len(daily_closes) >= 20:
        sma10d = _sma(daily_closes, 10)
        sma20d = _sma(daily_closes, 20)
        daily_trend_up = sma10d > sma20d

    snapshots: list[SignalSnapshot] = []

    for i, bar in enumerate(bars_5m):
        closes = [b.c for b in bars_5m[: i + 1]]

        # RSI
        rsi3 = _rsi(closes, 3)
        rsi14 = _rsi(closes, 14)

        # VWAP distance
        vwap_dist = ((bar.c - bar.vwap) / bar.vwap * 100.0) if bar.vwap > 0 else 0.0

        # Volume ratio (current vs 10-bar average)
        vol_window = [b.v for b in bars_5m[max(0, i - 10) : i]]
        avg_vol = (sum(vol_window) / len(vol_window)) if vol_window else 1
        vol_ratio = bar.v / avg_vol if avg_vol > 0 else 1.0

        # Intraday % from open
        intraday_pct = ((bar.c - day_open) / day_open * 100.0) if day_open > 0 else 0.0

        # Prior bar colour
        prior_red = bars_5m[i - 1].is_red if i > 0 else False
        prior_green = bars_5m[i - 1].is_green if i > 0 else False

        # EMAs
        ema_f = _ema(closes, ema_fast_period)
        ema_s = _ema(closes, ema_slow_period)
        ema_bull = False
        ema_bear = False
        if i > 0:
            prev_closes = closes[:-1]
            prev_ema_f = _ema(prev_closes, ema_fast_period)
            prev_ema_s = _ema(prev_closes, ema_slow_period)
            if prev_ema_f <= prev_ema_s and ema_f > ema_s:
                ema_bull = True
            if prev_ema_f >= prev_ema_s and ema_f < ema_s:
                ema_bear = True

        # MACD
        _, _, hist = _macd(closes)
        prev_hist = 0.0
        if i > 0:
            _, _, prev_hist = _macd(closes[:-1])

        # Bollinger
        bb_upper, bb_mid, bb_lower = _bollinger(
            closes, bollinger_period, bollinger_std,
        )

        # SMAs for micro-trend
        sma5 = _sma(closes, 5)
        sma10 = _sma(closes, 10)

        snap = SignalSnapshot(
            bar_idx=i,
            bar=bar,
            rsi3=rsi3,
            rsi14=rsi14,
            ibs=bar.ibs,
            vwap_distance_pct=vwap_dist,
            volume_ratio=vol_ratio,
            intraday_pct=intraday_pct,
            prior_bar_red=prior_red,
            prior_bar_green=prior_green,
            ema_fast=ema_f,
            ema_slow=ema_s,
            ema_bullish_cross=ema_bull,
            ema_bearish_cross=ema_bear,
            macd_hist=hist,
            macd_hist_prev=prev_hist,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            bb_mid=bb_mid,
            opening_range_high=or_high,
            opening_range_low=or_low,
            daily_trend_up=daily_trend_up,
            sma5=sma5,
            sma10=sma10,
        )

        snapshots.append(snap)

    return snapshots


# ── Signal evaluation (config-driven) ─────────────────────────────

def evaluate_signals(
    snap: SignalSnapshot,
    cfg: Any,  # EntrySignals
) -> tuple[int, int, float]:
    """Count how many call/put signals fire for this snapshot.

    Returns:
        (call_signals_fired, put_signals_fired, conviction_score)
    """
    call_count = 0
    put_count = 0

    # RSI(3)
    if cfg.use_rsi3:
        if snap.rsi3 < cfg.rsi3_call_threshold:
            call_count += 1
        if snap.rsi3 > cfg.rsi3_put_threshold:
            put_count += 1

    # RSI(14)
    if cfg.use_rsi14:
        if snap.rsi14 < cfg.rsi14_call_threshold:
            call_count += 1
        if snap.rsi14 > cfg.rsi14_put_threshold:
            put_count += 1

    # VWAP position (hard gate — price below for calls, above for puts)
    if cfg.use_vwap_position:
        if snap.vwap_distance_pct < 0:
            call_count += 1
        if snap.vwap_distance_pct > 0:
            put_count += 1

    # VWAP distance
    if cfg.use_vwap_distance:
        if snap.vwap_distance_pct < -cfg.vwap_distance_pct:
            call_count += 1
        if snap.vwap_distance_pct > cfg.vwap_distance_pct:
            put_count += 1

    # IBS
    if cfg.use_ibs:
        if snap.ibs < cfg.ibs_call_threshold:
            call_count += 1
        if snap.ibs > cfg.ibs_put_threshold:
            put_count += 1

    # Volume spike (direction-neutral — confirms both)
    if cfg.use_volume_spike and snap.volume_ratio >= cfg.volume_spike_ratio:
        call_count += 1
        put_count += 1

    # Intraday move
    if cfg.use_intraday_move:
        if snap.intraday_pct <= cfg.intraday_drop_pct:
            call_count += 1
        if snap.intraday_pct >= cfg.intraday_rally_pct:
            put_count += 1

    # Prior bar colour
    if cfg.use_prior_bar_color:
        if snap.prior_bar_red:
            call_count += 1
        if snap.prior_bar_green:
            put_count += 1

    # EMA cross
    if cfg.use_ema_cross:
        if snap.ema_bullish_cross:
            call_count += 1
        if snap.ema_bearish_cross:
            put_count += 1

    # MACD histogram
    if cfg.use_macd:
        if snap.macd_hist > 0 and snap.macd_hist_prev <= 0:
            call_count += 1
        if snap.macd_hist < 0 and snap.macd_hist_prev >= 0:
            put_count += 1

    # Bollinger
    if cfg.use_bollinger:
        if snap.bar.c <= snap.bb_lower:
            call_count += 1
        if snap.bar.c >= snap.bb_upper:
            put_count += 1

    # Opening range breakout/breakdown
    if cfg.use_opening_range:
        if snap.bar.c > snap.opening_range_high:
            call_count += 1
        if snap.bar.c < snap.opening_range_low:
            put_count += 1

    # Daily trend (gate — only boosts the matching direction)
    if cfg.use_daily_trend:
        if snap.daily_trend_up:
            call_count += 1
        else:
            put_count += 1

    # SMA micro-trend
    if cfg.use_sma_micro:
        if snap.sma5 > snap.sma10:
            call_count += 1
        if snap.sma5 < snap.sma10:
            put_count += 1

    # Conviction = max count + extras for extreme readings
    base = max(call_count, put_count)
    bonus = 0.0
    if snap.rsi3 < 15:
        bonus += 0.5
    if snap.rsi3 > 85:
        bonus += 0.5
    if snap.volume_ratio > 3.0:
        bonus += 0.5
    if abs(snap.vwap_distance_pct) > 0.3:
        bonus += 0.25
    conviction = base + bonus

    return call_count, put_count, conviction
