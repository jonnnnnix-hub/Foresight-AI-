"""Implied volatility rank and regime models."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class IVRegime(StrEnum):
    """IV environment classification."""

    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"


class IVRankData(BaseModel):
    """IV rank and percentile data for a ticker."""

    ticker: str
    iv_rank: float = Field(ge=0.0, le=100.0, description="IV rank 0-100")
    iv_percentile: float = Field(
        ge=0.0, le=100.0, default=0.0, description="IV percentile 0-100"
    )
    current_iv: float = Field(ge=0.0, default=0.0)
    hv_20: float | None = Field(default=None, ge=0.0)
    hv_60: float | None = Field(default=None, ge=0.0)
    iv_hv_spread: float | None = Field(
        default=None, description="current_iv - hv_20"
    )
    iv_52w_high: float = Field(ge=0.0, default=0.0)
    iv_52w_low: float = Field(ge=0.0, default=0.0)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now())
    source: str = "orats"


class TermStructurePoint(BaseModel):
    """One point on the IV term structure curve."""

    expiration: date
    iv: float = Field(ge=0.0)
    days_to_expiration: int = Field(ge=0)


class IVSignal(BaseModel):
    """IV-based signal for a ticker."""

    ticker: str
    iv_rank: IVRankData
    term_structure: list[TermStructurePoint] = Field(default_factory=list)
    regime: IVRegime = IVRegime.NORMAL
    is_cheap_premium: bool = False
    is_contango: bool = True
    strength: float = Field(ge=0.0, le=10.0, default=0.0)
    rationale: str = ""
    detected_at: datetime = Field(default_factory=lambda: datetime.now())
