"""GEX and market structure schemas."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class GEXRegime(StrEnum):
    """Market maker gamma positioning regime."""

    POSITIVE = "positive"  # Dealers long gamma — dampen moves, pin risk
    NEGATIVE = "negative"  # Dealers short gamma — amplify moves, volatile
    NEUTRAL = "neutral"


class StrikeLevel(BaseModel):
    """A significant strike price level with gamma context."""

    strike: float = Field(ge=0.0)
    call_gamma: float = Field(default=0.0)
    put_gamma: float = Field(default=0.0)
    net_gamma: float = Field(default=0.0)
    call_oi: int = Field(ge=0, default=0)
    put_oi: int = Field(ge=0, default=0)
    is_max_pain: bool = False
    is_gex_flip: bool = False


class GEXProfile(BaseModel):
    """Complete gamma exposure profile for a ticker."""

    ticker: str
    underlying_price: float = Field(ge=0.0)
    regime: GEXRegime = GEXRegime.NEUTRAL
    total_gex: float = Field(default=0.0, description="Net gamma exposure")
    gex_flip_price: float | None = Field(
        default=None, description="Price where GEX flips positive/negative"
    )
    max_pain: float | None = Field(
        default=None, description="Max pain strike for nearest expiry"
    )
    key_levels: list[StrikeLevel] = Field(default_factory=list)
    support_levels: list[float] = Field(
        default_factory=list, description="High put gamma = support"
    )
    resistance_levels: list[float] = Field(
        default_factory=list, description="High call gamma = resistance"
    )
    pin_risk_zone: tuple[float, float] | None = Field(
        default=None, description="Price range likely to pin"
    )
    breakout_above: float | None = Field(
        default=None, description="Price above which breakout accelerates"
    )
    breakout_below: float | None = Field(
        default=None, description="Price below which breakdown accelerates"
    )
    lotto_favorable: bool = Field(
        default=False,
        description="True when negative GEX = big moves likely",
    )
    strength: float = Field(ge=0.0, le=10.0, default=0.0)
    rationale: str = ""
    computed_at: datetime = Field(default_factory=lambda: datetime.now())


class NetPremiumTick(BaseModel):
    """A single intraday net premium data point."""

    timestamp: str = ""
    call_volume: int = Field(ge=0, default=0)
    put_volume: int = Field(ge=0, default=0)
    net_call_premium: float = Field(default=0.0)
    net_put_premium: float = Field(default=0.0)
    net_call_volume: int = Field(default=0)
