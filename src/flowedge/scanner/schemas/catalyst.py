"""Catalyst event models — earnings, insider trades, expected moves."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class CatalystType(StrEnum):
    """Type of upcoming catalyst."""

    EARNINGS = "earnings"
    INSIDER_BUY = "insider_buy"
    INSIDER_SELL = "insider_sell"
    FDA_DATE = "fda_date"
    EX_DIVIDEND = "ex_dividend"


class EarningsEvent(BaseModel):
    """An upcoming or recent earnings event."""

    ticker: str
    report_date: date
    fiscal_quarter: str = ""
    eps_estimate: float | None = None
    revenue_estimate: float | None = None
    time_of_day: str = Field(
        default="", description="bmo | amc | during"
    )
    source: str = "fmp"


class InsiderTrade(BaseModel):
    """An insider buy or sell filing (Form 4)."""

    ticker: str
    insider_name: str
    title: str = ""
    transaction_type: str = Field(
        default="", description="P-Purchase | S-Sale | A-Award"
    )
    shares: int = 0
    price_per_share: float = Field(ge=0.0, default=0.0)
    total_value: float = Field(ge=0.0, default=0.0)
    filing_date: date = Field(default_factory=date.today)
    transaction_date: date = Field(default_factory=date.today)
    ownership_type: str = Field(
        default="D", description="D-Direct | I-Indirect"
    )
    source: str = "sec_edgar"


class ExpectedMove(BaseModel):
    """Options-implied expected move around a catalyst."""

    ticker: str
    event_date: date
    expected_move_pct: float = Field(ge=0.0, default=0.0)
    expected_move_dollars: float = Field(ge=0.0, default=0.0)
    straddle_price: float = Field(ge=0.0, default=0.0)
    source: str = "orats"


class CatalystSignal(BaseModel):
    """Aggregated catalyst signal for a ticker."""

    ticker: str
    earnings: list[EarningsEvent] = Field(default_factory=list)
    insider_trades: list[InsiderTrade] = Field(default_factory=list)
    expected_move: ExpectedMove | None = None
    days_to_nearest_catalyst: int | None = None
    net_insider_sentiment: str = Field(
        default="neutral", description="bullish | bearish | neutral"
    )
    insider_buy_count: int = Field(ge=0, default=0)
    insider_sell_count: int = Field(ge=0, default=0)
    insider_net_value: float = Field(
        default=0.0, description="Net $ value (positive = net buying)"
    )
    strength: float = Field(ge=0.0, le=10.0, default=0.0)
    rationale: str = ""
    detected_at: datetime = Field(default_factory=lambda: datetime.now())
