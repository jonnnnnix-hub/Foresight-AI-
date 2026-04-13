"""ScalpConfig — frozen Pydantic model for scalp backtest parameters.

Externalizes all hardcoded SCALP_* constants into a single, validated,
serializable configuration object.  Defaults match the current v2 model.

Usage:
    cfg = ScalpConfig()                     # all defaults
    cfg = ScalpConfig(dte=2, trail_pct=0.03)  # override a few
    cfg = ScalpConfig.from_json_file("configs/aggressive.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ScalpConfig(BaseModel, frozen=True):
    """Immutable configuration for the scalp v2 backtest engine."""

    # ── Ticker universe ────────────────────────────────────────
    tickers: list[str] = Field(
        default=[
            # Index ETFs (daily 0DTE options, tightest spreads)
            "SPY", "QQQ", "IWM", "DIA",
            # Sector ETFs
            "XLF", "XLK", "XLV", "XLE",
            # Mega-cap tech (highest options volume)
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
            # High-options-volume single stocks
            "AMD", "AVGO", "ARM", "NFLX", "CRM", "COST",
            # High-beta / momentum names
            "PLTR", "SOFI", "COIN", "HOOD", "MSTR", "RDDT", "SMCI",
            # Financials / value
            "BAC", "JPM", "V", "WMT", "INTC",
        ],
        description="Underlying tickers to scan",
    )

    # ── Option selection ───────────────────────────────────────
    dte: int = Field(
        default=5,
        ge=0,
        le=30,
        description="Maximum days to expiration (0-5 captures weekly Fri + daily 0DTE)",
    )
    min_premium: float = Field(
        default=0.30,
        gt=0,
        description="Minimum real option premium ($) to enter a trade",
    )

    # ── Exit parameters ────────────────────────────────────────
    tp_underlying: float = Field(
        default=0.0015,
        gt=0,
        description="Take-profit: underlying move fraction (0.0015 = 0.15%)",
    )
    max_hold_bars: int = Field(
        default=12,
        ge=1,
        le=100,
        description="Maximum hold period in 5-min bars (12 × 5 = 60 min)",
    )
    trail_pct: float = Field(
        default=0.05,
        gt=0,
        le=1.0,
        description="Trailing stop: percentage from peak premium (0.05 = 5%)",
    )

    # ── Entry filters ──────────────────────────────────────────
    ibs_threshold: float = Field(
        default=0.10,
        ge=0,
        le=1.0,
        description="Internal Bar Strength upper limit",
    )
    rsi3_threshold: float = Field(
        default=20.0,
        ge=0,
        le=100,
        description="RSI(3) upper limit",
    )
    vol_spike: float = Field(
        default=2.0,
        gt=0,
        description="Volume spike multiplier (bar vol / avg vol)",
    )
    intraday_drop: float = Field(
        default=-0.003,
        le=0,
        description="Max intraday drop from open (-0.003 = -0.3%)",
    )

    # ── Risk ───────────────────────────────────────────────────
    max_positions: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Maximum concurrent intraday positions",
    )
    risk_per_trade: float = Field(
        default=0.05,
        gt=0,
        le=1.0,
        description="Capital fraction per trade (0.05 = 5%)",
    )

    # ── Costs ──────────────────────────────────────────────────
    spread_cents: float = Field(
        default=0.0,
        ge=0,
        description="Bid-ask spread penalty in cents (applied to both entry and exit)",
    )
    commission_per_contract: float = Field(
        default=0.50,
        ge=0,
        description="Per-contract commission in dollars (entry + exit)",
    )

    # ── Capital ────────────────────────────────────────────────
    starting_capital: float = Field(
        default=25_000.0,
        gt=0,
        description="Portfolio starting value in dollars",
    )

    # ── Convenience ────────────────────────────────────────────

    @classmethod
    def from_json_file(cls, path: str | Path) -> ScalpConfig:
        """Load config from a JSON file."""
        data: dict[str, Any] = json.loads(Path(path).read_text())
        return cls(**data)

    def to_json_file(self, path: str | Path) -> None:
        """Save config to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))
