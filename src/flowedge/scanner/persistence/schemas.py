"""Persistence schemas for watchlists, scan history, and trade tracking."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TradeStatus(StrEnum):
    """Lifecycle of a tracked trade."""

    ALERT = "alert"        # Signal fired, not yet entered
    ENTERED = "entered"    # Position opened
    PARTIAL = "partial"    # Partial exit
    EXITED = "exited"      # Fully closed
    EXPIRED = "expired"    # Option expired


class WatchlistItem(BaseModel):
    """A ticker on the user's watchlist."""

    ticker: str
    added_at: datetime = Field(default_factory=lambda: datetime.now())
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    auto_scan: bool = Field(
        default=True, description="Include in scheduled scans"
    )


class Watchlist(BaseModel):
    """User's watchlist of tickers to monitor."""

    name: str = "default"
    items: list[WatchlistItem] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now())

    @property
    def tickers(self) -> list[str]:
        return [item.ticker for item in self.items]

    @property
    def auto_scan_tickers(self) -> list[str]:
        return [item.ticker for item in self.items if item.auto_scan]


class TrackedAlert(BaseModel):
    """A scanner alert that is being tracked through its lifecycle."""

    alert_id: str
    ticker: str
    scan_id: str = ""
    composite_score: float = Field(ge=0.0, le=10.0, default=0.0)
    uoa_score: float = Field(ge=0.0, le=10.0, default=0.0)
    iv_score: float = Field(ge=0.0, le=10.0, default=0.0)
    catalyst_score: float = Field(ge=0.0, le=10.0, default=0.0)
    direction: str = "neutral"
    strategy_type: str = ""
    entry_price: float | None = None
    exit_price: float | None = None
    underlying_at_alert: float = Field(ge=0.0, default=0.0)
    underlying_at_exit: float | None = None
    status: TradeStatus = TradeStatus.ALERT
    pnl_pct: float | None = None
    was_winner: bool | None = None
    alerted_at: datetime = Field(default_factory=lambda: datetime.now())
    entered_at: datetime | None = None
    exited_at: datetime | None = None
    notes: str = ""


class ScanHistoryEntry(BaseModel):
    """One row in the scan history log."""

    scan_id: str
    scanned_at: datetime = Field(default_factory=lambda: datetime.now())
    tickers_scanned: int = 0
    opportunities_found: int = 0
    top_ticker: str = ""
    top_score: float = Field(ge=0.0, le=10.0, default=0.0)
    engine_versions: dict[str, str] = Field(default_factory=dict)


class LearningStats(BaseModel):
    """Aggregate performance stats for the learning feedback loop."""

    total_alerts: int = 0
    total_entered: int = 0
    total_resolved: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    avg_winner_pnl_pct: float = 0.0
    avg_loser_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    best_engine: str = ""
    best_score_bucket: str = ""
    by_score_bucket: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_engine: dict[str, dict[str, float]] = Field(default_factory=dict)
    computed_at: datetime = Field(default_factory=lambda: datetime.now())
