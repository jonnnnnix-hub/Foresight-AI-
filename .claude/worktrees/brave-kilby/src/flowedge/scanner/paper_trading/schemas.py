"""Paper trading schemas."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PaperOrder(BaseModel):
    """A paper trading order."""

    order_id: str = ""
    ticker: str
    side: OrderSide = OrderSide.BUY
    qty: int = Field(ge=1, default=1)
    symbol: str = Field(default="", description="Option symbol or equity")
    order_type: str = Field(default="market")
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float | None = None
    filled_at: datetime | None = None
    submitted_at: datetime = Field(default_factory=lambda: datetime.now())
    signal_score: float = Field(ge=0.0, le=10.0, default=0.0)
    notes: str = ""


class PaperPosition(BaseModel):
    """A current paper position."""

    ticker: str
    symbol: str
    qty: int = Field(ge=0, default=0)
    avg_entry: float = Field(ge=0.0, default=0.0)
    current_price: float = Field(ge=0.0, default=0.0)
    market_value: float = Field(default=0.0)
    unrealized_pnl: float = Field(default=0.0)
    unrealized_pnl_pct: float = Field(default=0.0)


class PaperPortfolio(BaseModel):
    """Paper trading portfolio state."""

    account_id: str = ""
    cash: float = Field(ge=0.0, default=100_000.0)
    portfolio_value: float = Field(ge=0.0, default=100_000.0)
    positions: list[PaperPosition] = Field(default_factory=list)
    orders: list[PaperOrder] = Field(default_factory=list)
    total_pnl: float = Field(default=0.0)
    total_pnl_pct: float = Field(default=0.0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now())
