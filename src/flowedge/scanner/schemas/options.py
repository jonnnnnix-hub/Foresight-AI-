"""Unified options chain models normalized across providers."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class OptionType(StrEnum):
    """Call or put."""

    CALL = "call"
    PUT = "put"


class OptionContract(BaseModel):
    """A single options contract with greeks."""

    symbol: str = Field(description="OCC-style option symbol")
    underlying: str
    option_type: OptionType
    strike: float = Field(ge=0.0)
    expiration: date
    bid: float = Field(ge=0.0, default=0.0)
    ask: float = Field(ge=0.0, default=0.0)
    mid: float = Field(ge=0.0, default=0.0)
    last: float | None = None
    volume: int = Field(ge=0, default=0)
    open_interest: int = Field(ge=0, default=0)
    implied_volatility: float | None = Field(default=None, ge=0.0)
    delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    gamma: float | None = Field(default=None, ge=0.0)
    theta: float | None = None
    vega: float | None = Field(default=None, ge=0.0)
    rho: float | None = None
    days_to_expiration: int = Field(ge=0, default=0)
    in_the_money: bool = False
    source: str = Field(default="", description="polygon | tradier")
    fetched_at: datetime = Field(default_factory=lambda: datetime.now())


class OptionsChain(BaseModel):
    """Complete options chain for an underlying."""

    underlying: str
    underlying_price: float = Field(ge=0.0)
    contracts: list[OptionContract] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now())
    source: str = ""

    @property
    def calls(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.option_type == OptionType.CALL]

    @property
    def puts(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.option_type == OptionType.PUT]

    @property
    def total_call_volume(self) -> int:
        return sum(c.volume for c in self.calls)

    @property
    def total_put_volume(self) -> int:
        return sum(c.volume for c in self.puts)
