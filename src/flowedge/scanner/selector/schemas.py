"""ARCHITECT v2 — multi-leg strategy schemas."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from flowedge.scanner.schemas.options import OptionContract


class StrategyType(StrEnum):
    """Supported options strategy structures."""

    SINGLE_CALL = "single_call"
    SINGLE_PUT = "single_put"
    CALL_DEBIT_SPREAD = "call_debit_spread"
    PUT_DEBIT_SPREAD = "put_debit_spread"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    PUT_CREDIT_SPREAD = "put_credit_spread"
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"
    IRON_CONDOR = "iron_condor"
    RISK_REVERSAL = "risk_reversal"


class StrategyLeg(BaseModel):
    """A single leg of a multi-leg strategy."""

    contract: OptionContract
    action: str = Field(description="buy | sell")
    qty: int = Field(ge=1, default=1)
    premium: float = Field(ge=0.0, default=0.0, description="Per-share premium")


class StrategyBlueprint(BaseModel):
    """A complete options strategy recommendation from ARCHITECT."""

    ticker: str
    strategy_type: StrategyType
    legs: list[StrategyLeg] = Field(default_factory=list)
    expiration: date = Field(default_factory=date.today)
    net_debit: float = Field(
        default=0.0, description="Net premium paid (positive = debit)"
    )
    max_profit: float = Field(default=0.0, description="Max profit per spread")
    max_loss: float = Field(default=0.0, description="Max loss per spread")
    breakeven_prices: list[float] = Field(
        default_factory=list, description="Breakeven price(s)"
    )
    risk_reward_ratio: float = Field(default=0.0)
    probability_of_profit: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    rationale: str = ""
    tags: list[str] = Field(
        default_factory=list,
        description="lotto | defined_risk | income | volatility",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now())

    @property
    def total_legs(self) -> int:
        return len(self.legs)

    @property
    def is_defined_risk(self) -> bool:
        return self.max_loss > 0 and self.max_loss < float("inf")
