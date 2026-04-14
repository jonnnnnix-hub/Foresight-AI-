"""Composite scanner output — lotto opportunities and scan results."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from flowedge.scanner.schemas.catalyst import CatalystSignal
from flowedge.scanner.schemas.flow import FlowSentiment, UOASignal
from flowedge.scanner.schemas.iv import IVSignal
from flowedge.scanner.schemas.options import OptionContract

# Avoid circular import — FLUX signal is optional
try:
    from flowedge.scanner.flux.schemas import FLUXSignal as _FLUXSignal
except ImportError:
    _FLUXSignal = None  # type: ignore[assignment, misc]


class ContractPick(BaseModel):
    """A specific contract the AI recommends."""

    symbol: str = ""
    option_type: str = Field(default="call", description="call | put")
    strike: float = Field(ge=0.0, default=0.0)
    expiration: str = ""
    bid: float = Field(ge=0.0, default=0.0)
    ask: float = Field(ge=0.0, default=0.0)
    delta: float | None = None
    volume: int = Field(ge=0, default=0)
    open_interest: int = Field(ge=0, default=0)
    max_loss_per_contract: float = Field(ge=0.0, default=0.0)
    risk_reward: float = Field(ge=0.0, default=0.0)
    reason: str = ""


class LottoOpportunity(BaseModel):
    """A ranked lotto play opportunity."""

    ticker: str
    composite_score: float = Field(ge=0.0, le=10.0, default=0.0)
    score_100: int = Field(ge=0, le=100, default=0, description="Composite score 0-100")
    uoa_score: float = Field(ge=0.0, le=10.0, default=0.0)
    iv_score: float = Field(ge=0.0, le=10.0, default=0.0)
    catalyst_score: float = Field(ge=0.0, le=10.0, default=0.0)
    flux_score: float = Field(ge=0.0, le=10.0, default=0.0)
    uoa_signal: UOASignal | None = None
    iv_signal: IVSignal | None = None
    catalyst_signal: CatalystSignal | None = None
    flux_signal: object | None = Field(
        default=None, description="FLUXSignal — order flow imbalance data",
    )
    suggested_direction: FlowSentiment = FlowSentiment.NEUTRAL
    suggested_contracts: list[OptionContract] = Field(default_factory=list)
    contract_picks: list[ContractPick] = Field(
        default_factory=list,
        description="Specific contracts AI recommends for this setup",
    )
    entry_criteria: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    rationale: str = ""
    scanned_at: datetime = Field(default_factory=lambda: datetime.now())


class ScannerResult(BaseModel):
    """Complete output from a scanner run."""

    scan_id: str
    scanned_at: datetime = Field(default_factory=lambda: datetime.now())
    tickers_scanned: int = 0
    opportunities: list[LottoOpportunity] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    status: str = "complete"

    @property
    def top_opportunities(self) -> list[LottoOpportunity]:
        """Return opportunities sorted by composite score descending."""
        return sorted(
            self.opportunities,
            key=lambda o: o.composite_score,
            reverse=True,
        )
