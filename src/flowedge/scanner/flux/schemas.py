"""Order flow and tape reading schemas."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TradeDirection(StrEnum):
    """Lee-Ready classified trade direction."""

    BUY = "buy"      # Trade at or above midpoint — buyer-initiated
    SELL = "sell"     # Trade below midpoint — seller-initiated
    UNKNOWN = "unknown"


class FlowBias(StrEnum):
    """Net order flow directional bias."""

    STRONG_BUY = "strong_buy"    # Aggressive buying dominance
    BUY = "buy"                  # Net buying pressure
    NEUTRAL = "neutral"
    SELL = "sell"                 # Net selling pressure
    STRONG_SELL = "strong_sell"  # Aggressive selling dominance


class DeltaDivergence(StrEnum):
    """Price vs cumulative delta divergence state."""

    BULLISH = "bullish"    # Price falling but delta rising (hidden accumulation)
    BEARISH = "bearish"    # Price rising but delta falling (hidden distribution)
    CONFIRMED = "confirmed"  # Price and delta moving together
    NONE = "none"


# ── Raw Data Models ─────────────────────────────────────────────


class TradeTick(BaseModel):
    """A single equity trade from the tape."""

    price: float
    size: int
    timestamp: int = Field(description="Unix nanoseconds")
    conditions: list[int] = Field(default_factory=list)
    exchange: int = 0


class NBBOQuote(BaseModel):
    """A single NBBO quote snapshot."""

    bid: float
    bid_size: int
    ask: float
    ask_size: int
    timestamp: int = Field(description="Unix nanoseconds")

    @property
    def midpoint(self) -> float:
        """Midpoint of bid-ask spread."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return 0.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid if self.ask > self.bid else 0.0

    @property
    def imbalance(self) -> float:
        """Bid/ask size imbalance: positive = more bids (buy pressure)."""
        total = self.bid_size + self.ask_size
        if total == 0:
            return 0.0
        return (self.bid_size - self.ask_size) / total


# ── Computed Models ──────────────────────────────────────────────


class ClassifiedTrade(BaseModel):
    """A trade with Lee-Ready direction classification."""

    price: float
    size: int
    timestamp: int
    direction: TradeDirection = TradeDirection.UNKNOWN
    signed_volume: int = 0  # Positive = buy, negative = sell


class BlockPrint(BaseModel):
    """A detected large block trade on the tape."""

    ticker: str
    price: float
    size: int
    notional: float = 0.0
    direction: TradeDirection = TradeDirection.UNKNOWN
    size_multiple: float = Field(
        default=0.0,
        description="How many times larger than avg trade size",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class CumulativeDelta(BaseModel):
    """Cumulative delta (net buying volume) over a time window."""

    window_minutes: int
    buy_volume: int = 0
    sell_volume: int = 0
    net_delta: int = 0  # buy_volume - sell_volume
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0

    @property
    def aggression_ratio(self) -> float:
        """Fraction of volume that is buyer-initiated.

        > 0.55 = net buying aggression
        < 0.45 = net selling aggression
        """
        total = self.buy_volume + self.sell_volume
        if total == 0:
            return 0.5
        return self.buy_volume / total

    @property
    def trade_aggression_ratio(self) -> float:
        """Fraction of trades that are buyer-initiated."""
        total = self.buy_trades + self.sell_trades
        if total == 0:
            return 0.5
        return self.buy_trades / total


class QuoteImbalance(BaseModel):
    """Aggregated L1 bid/ask size imbalance over a window."""

    window_minutes: int
    avg_imbalance: float = 0.0  # -1 to +1
    max_bid_dominance: float = 0.0
    max_ask_dominance: float = 0.0
    snapshots: int = 0


# ── Signal Output ────────────────────────────────────────────────


class FLUXSignal(BaseModel):
    """Order flow signal from the FLUX engine.

    Scores real-time equity tape data to determine if institutional
    buying or selling pressure is present, and whether it confirms
    or diverges from price action.
    """

    ticker: str

    # Scoring
    strength: float = Field(ge=0.0, le=10.0, default=0.0)

    # Direction
    bias: FlowBias = FlowBias.NEUTRAL

    # Cumulative delta
    delta_5m: CumulativeDelta | None = None
    delta_15m: CumulativeDelta | None = None
    delta_session: CumulativeDelta | None = None

    # L1 quote imbalance
    quote_imbalance: QuoteImbalance | None = None

    # Block prints
    block_prints: list[BlockPrint] = Field(default_factory=list)
    block_bias: TradeDirection = TradeDirection.UNKNOWN

    # Divergence
    divergence: DeltaDivergence = DeltaDivergence.NONE

    # Metadata
    rationale: str = ""
    detected_at: datetime = Field(default_factory=lambda: datetime.now())
