"""Trident backtest configuration — every tunable parameter in one place."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ── Constants ─────────────────────────────────────────────────────
TRIDENT_TICKERS = ["SPY", "QQQ", "IWM"]

CACHE_DIR = Path("data/flat_files_s3")

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05


class Direction(StrEnum):
    LONG = "long"    # buy calls
    SHORT = "short"  # buy puts
    BOTH = "both"


class ExitStyle(StrEnum):
    FIXED = "fixed"        # fixed TP/SL percentages
    TIERED = "tiered"      # tiered targets based on conviction
    TRAILING = "trailing"  # trailing stop only


# ── Signal thresholds ─────────────────────────────────────────────

class EntrySignals(BaseModel, frozen=True):
    """Tunable thresholds for entry signals.

    Each signal can be enabled/disabled independently.
    Call signals fire on oversold/bullish conditions.
    Put signals fire on overbought/bearish conditions (auto-inverted).
    """

    # RSI signals
    use_rsi3: bool = True
    rsi3_call_threshold: float = 25.0     # RSI(3) < X → call entry
    rsi3_put_threshold: float = 75.0      # RSI(3) > X → put entry

    use_rsi14: bool = False
    rsi14_call_threshold: float = 35.0
    rsi14_put_threshold: float = 65.0

    # VWAP signals
    use_vwap_position: bool = True        # price vs VWAP as hard gate
    use_vwap_distance: bool = False
    vwap_distance_pct: float = 0.15       # % away from VWAP to trigger

    # Internal Bar Strength
    use_ibs: bool = True
    ibs_call_threshold: float = 0.20      # IBS < X → call (oversold bar)
    ibs_put_threshold: float = 0.80       # IBS > X → put (overbought bar)

    # Volume
    use_volume_spike: bool = True
    volume_spike_ratio: float = 2.0       # volume > Nx 10-bar avg

    # Momentum
    use_intraday_move: bool = True
    intraday_drop_pct: float = -0.15      # calls: drop from open > X%
    intraday_rally_pct: float = 0.15      # puts: rally from open > X%

    use_prior_bar_color: bool = True      # prior bar red → calls, green → puts

    use_ema_cross: bool = False
    ema_fast: int = 8
    ema_slow: int = 21

    use_macd: bool = False                # MACD histogram sign flip

    use_bollinger: bool = False
    bollinger_period: int = 20
    bollinger_std: float = 2.0

    use_opening_range: bool = False       # first N-min high/low breakout
    opening_range_minutes: int = 15

    use_daily_trend: bool = True          # SMA(10) vs SMA(20) on daily
    use_sma_micro: bool = False           # SMA(5) vs SMA(10) on 5-min

    # Confluence requirement
    min_signals_call: int = 3             # N-of-M for call entry
    min_signals_put: int = 3              # N-of-M for put entry


class ExitParams(BaseModel, frozen=True):
    """Exit rule parameters."""

    # Take profit on option premium
    tp_pct: float = 0.25                  # +25% on option
    tp_tier2_pct: float = 0.50            # +50% (tiered, higher conviction)
    tp_tier3_pct: float = 1.00            # +100% (lottery tier)

    # Stop loss on option premium
    sl_pct: float = -0.35                 # -35% on option

    # Trailing stop (% retracement from peak premium)
    use_trailing: bool = True
    trail_pct: float = 0.30               # 30% retracement from peak

    # Max hold time
    max_hold_bars: int = 18               # 18 × 5-min = 90 min
    max_hold_bars_tier2: int = 12         # 60 min for medium conviction
    max_hold_bars_tier3: int = 6          # 30 min for high conviction (quick TP)

    # End-of-day forced close
    eod_close_minutes_before: int = 10    # close all by 3:50 PM ET

    # Exit on signal reversal (VWAP cross against position)
    use_signal_reversal_exit: bool = False


class OptionsParams(BaseModel, frozen=True):
    """Options selection parameters."""

    min_dte: int = 0
    max_dte: int = 2
    min_delta: float = 0.25
    max_delta: float = 0.50
    min_premium: float = 0.30             # $0.30 min to avoid penny options
    max_spread_pct: float = 0.10          # 10% max bid-ask spread


class PositionParams(BaseModel, frozen=True):
    """Position sizing and risk."""

    risk_per_trade: float = 0.03          # 3% of equity per trade
    max_positions: int = 2
    min_bars_between_trades: int = 3      # 15 min cooldown
    commission_per_contract: float = 0.50
    spread_cents: float = 3.0             # estimated half-spread


class TimeFilter(BaseModel, frozen=True):
    """Time-of-day trading windows."""

    skip_first_n_minutes: int = 5         # avoid opening chaos
    skip_last_n_minutes: int = 15         # no new entries after 3:45 PM
    # Specific windows (all enabled = trade all day)
    use_morning: bool = True              # 9:35 - 11:30
    use_midday: bool = True               # 11:30 - 14:00
    use_afternoon: bool = True            # 14:00 - 15:45


# ── Master config ─────────────────────────────────────────────────

class TridentConfig(BaseModel, frozen=True):
    """Complete Trident backtest configuration."""

    name: str = "trident_default"
    direction: Direction = Direction.BOTH
    tickers: list[str] = Field(default_factory=lambda: list(TRIDENT_TICKERS))

    entry: EntrySignals = Field(default_factory=EntrySignals)
    exit: ExitParams = Field(default_factory=ExitParams)
    options: OptionsParams = Field(default_factory=OptionsParams)
    position: PositionParams = Field(default_factory=PositionParams)
    time_filter: TimeFilter = Field(default_factory=TimeFilter)

    # Aggregation
    bar_size_minutes: int = 5             # aggregate 1-min to N-min

    # Capital
    starting_capital: float = 25_000.0

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TridentConfig:
        return cls(**d)
