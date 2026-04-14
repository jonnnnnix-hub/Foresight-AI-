"""Data feed schemas — typed contracts for all market data sources."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Timeframe(StrEnum):
    """Supported bar timeframes."""

    TICK = "tick"
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    HOUR_1 = "1hour"
    DAILY = "daily"


class BarData(BaseModel):
    """OHLCV bar at any timeframe."""

    ticker: str
    timestamp: datetime
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0  # Volume-weighted average price
    trade_count: int = 0  # Number of trades in bar


class OptionQuote(BaseModel):
    """Real-time option chain quote with Greeks."""

    underlying: str
    contract_symbol: str
    expiration: str
    strike: float
    option_type: str  # "call" or "put"

    # Prices
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    last: float = 0.0

    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0

    # Volume/OI
    volume: int = 0
    open_interest: int = 0

    # Computed
    bid_ask_spread: float = 0.0
    spread_pct: float = 0.0  # Spread as % of mid


class MarketBreadth(BaseModel):
    """NYSE/NASDAQ market breadth internals."""

    timestamp: datetime
    tick: int = 0  # NYSE TICK (< -1000 = capitulation)
    tick_avg: float = 0.0
    add: int = 0  # Advance/Decline
    vold: float = 0.0  # Up volume / Down volume ratio

    @property
    def is_capitulation(self) -> bool:
        """TICK < -1000 indicates institutional panic selling."""
        return self.tick < -1000

    @property
    def is_strong_breadth(self) -> bool:
        """Broad market strength."""
        return self.tick > 500 and self.add > 0


class VIXData(BaseModel):
    """VIX and VVIX real-time data."""

    timestamp: datetime
    vix: float = 0.0
    vvix: float = 0.0
    vix_sma10: float = 0.0
    vix_sma20: float = 0.0

    @property
    def regime(self) -> str:
        """Classify VIX regime."""
        if self.vix > 30:
            return "panic"
        if self.vix > 25:
            return "elevated"
        if self.vix > 20:
            return "cautious"
        if self.vix > 15:
            return "normal"
        return "calm"

    @property
    def is_spike(self) -> bool:
        """VIX spike = current > 1.3x 10-day SMA."""
        if self.vix_sma10 <= 0:
            return False
        return self.vix > self.vix_sma10 * 1.3


class SentimentData(BaseModel):
    """Aggregated sentiment from news + social sources."""

    ticker: str
    timestamp: datetime
    news_score: float = 0.0  # -1.0 (bearish) to +1.0 (bullish)
    social_score: float = 0.0
    composite_score: float = 0.0
    news_volume: int = 0  # Number of articles
    social_volume: int = 0  # Number of posts
    has_earnings_today: bool = False
    has_fed_event: bool = False

    @property
    def is_event_day(self) -> bool:
        """Block trades on event days — technicals don't work."""
        return self.has_earnings_today or self.has_fed_event


class IntradaySnapshot(BaseModel):
    """Complete market snapshot at a point in time.

    Combines all data sources into a single typed contract
    that models can consume.
    """

    timestamp: datetime
    ticker: str

    # Price bars at multiple timeframes
    bar_1m: BarData | None = None
    bar_5m: BarData | None = None
    bar_15m: BarData | None = None
    bar_daily: BarData | None = None

    # Options chain (nearest ATM)
    option_call: OptionQuote | None = None
    option_put: OptionQuote | None = None

    # Market-wide
    breadth: MarketBreadth | None = None
    vix: VIXData | None = None

    # Sentiment
    sentiment: SentimentData | None = None

    # Computed features (filled by feature layer)
    ibs_1m: float = 0.5
    ibs_5m: float = 0.5
    ibs_daily: float = 0.5
    rsi3_5m: float = 50.0
    vwap_deviation_pct: float = 0.0  # Price vs VWAP
    gap_pct: float = 0.0  # Open vs prior close
    range_position_5d: float = 0.5  # 0=low, 1=high
    volume_ratio: float = 1.0

    # FLUX order flow features (filled by FLUX engine)
    flux_aggression_ratio: float = 0.5  # 0=all sells, 1=all buys
    flux_net_delta: int = 0  # Cumulative delta (buy_vol - sell_vol)
    flux_quote_imbalance: float = 0.0  # L1 bid/ask size imbalance
    flux_block_count: int = 0  # Number of block prints detected
    flux_strength: float = 0.0  # FLUX score 0-10

    @property
    def is_tradeable(self) -> bool:
        """Check if conditions allow trading."""
        if self.sentiment and self.sentiment.is_event_day:
            return False
        return not (self.vix and self.vix.regime == "panic")


class DataFeedConfig(BaseModel):
    """Configuration for data feed connections."""

    # Polygon
    polygon_api_key: str = ""
    polygon_tier: str = "paid"  # "free" or "paid"

    # Alpaca
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True  # Paper trading mode

    # Feature settings
    intraday_timeframes: list[Timeframe] = Field(
        default_factory=lambda: [Timeframe.MIN_1, Timeframe.MIN_5, Timeframe.DAILY],
    )
    options_chain_strikes: int = 5  # Number of strikes above/below ATM
    vix_enabled: bool = True
    breadth_enabled: bool = True
    sentiment_enabled: bool = True

    # Tickers to monitor
    tickers: list[str] = Field(
        default_factory=lambda: [
            "SPY", "QQQ", "IWM", "DIA",
            "XLF", "XLK", "XLV",
            "AAPL", "META", "NVDA", "MSFT",
            "JPM", "V", "COST", "NFLX", "AVGO",
        ],
    )
