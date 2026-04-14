"""Options flow and unusual activity models."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from flowedge.scanner.schemas.options import OptionType


class FlowSentiment(StrEnum):
    """Directional bias of the flow."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class FlowType(StrEnum):
    """How the order was executed."""

    SWEEP = "sweep"
    BLOCK = "block"
    SPLIT = "split"
    REGULAR = "regular"


class FlowAlert(BaseModel):
    """A single unusual options flow alert."""

    ticker: str
    option_symbol: str = ""
    option_type: OptionType
    strike: float = Field(ge=0.0)
    expiration: date
    flow_type: FlowType = FlowType.REGULAR
    sentiment: FlowSentiment = FlowSentiment.NEUTRAL
    premium: float = Field(ge=0.0, default=0.0)
    volume: int = Field(ge=0, default=0)
    open_interest: int = Field(ge=0, default=0)
    volume_oi_ratio: float = Field(ge=0.0, default=0.0)
    underlying_price: float = Field(ge=0.0, default=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    source: str = "unusual_whales"


class DarkPoolTrade(BaseModel):
    """A single dark pool print."""

    ticker: str
    price: float = Field(ge=0.0)
    size: int = Field(ge=0)
    notional: float = Field(ge=0.0, default=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    exchange: str = ""


class UOASignal(BaseModel):
    """Aggregated unusual options activity signal for a ticker."""

    ticker: str
    signal_type: str = Field(
        default="",
        description="volume_spike | sweep_cluster | block_trade | skew_shift",
    )
    direction: FlowSentiment = FlowSentiment.NEUTRAL
    strength: float = Field(ge=0.0, le=10.0, default=0.0)
    alerts: list[FlowAlert] = Field(default_factory=list)
    dark_pool_trades: list[DarkPoolTrade] = Field(default_factory=list)
    call_volume: int = Field(ge=0, default=0)
    put_volume: int = Field(ge=0, default=0)
    call_put_ratio: float = Field(ge=0.0, default=0.0)
    total_premium: float = Field(ge=0.0, default=0.0)
    rationale: str = ""
    detected_at: datetime = Field(default_factory=lambda: datetime.now())
