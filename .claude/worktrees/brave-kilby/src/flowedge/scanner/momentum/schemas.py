"""Momentum signal schemas."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MomentumBias(StrEnum):
    """Directional momentum bias."""

    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class TechnicalSnapshot(BaseModel):
    """Technical indicator snapshot for one timeframe."""

    timeframe: str = Field(description="1min | 5min | 15min | 1h | 1d")
    rsi: float | None = Field(default=None, ge=0.0, le=100.0)
    macd_value: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    ema_9: float | None = None
    ema_21: float | None = None
    current_price: float = Field(ge=0.0, default=0.0)
    volume_vs_avg: float | None = Field(
        default=None, description="Current volume / 20d avg volume"
    )


class MomentumSignal(BaseModel):
    """Aggregated momentum signal across timeframes."""

    ticker: str
    bias: MomentumBias = MomentumBias.NEUTRAL
    strength: float = Field(ge=0.0, le=10.0, default=0.0)
    timeframes: list[TechnicalSnapshot] = Field(default_factory=list)
    trend_alignment: bool = Field(
        default=False,
        description="All timeframes agree on direction",
    )
    rsi_oversold: bool = False
    rsi_overbought: bool = False
    macd_crossover: bool = Field(
        default=False, description="MACD crossed above signal"
    )
    above_vwap: bool | None = None
    volume_surge: bool = Field(
        default=False, description="Volume > 2x average"
    )
    rationale: str = ""
    computed_at: datetime = Field(default_factory=lambda: datetime.now())
