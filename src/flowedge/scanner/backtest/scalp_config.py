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


# Full 33-ticker universe (use for broad scanning / new ticker discovery)
ALL_33_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLF", "XLK", "XLV", "XLE",
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "AMD", "AVGO", "ARM", "NFLX", "CRM", "COST",
    "PLTR", "SOFI", "COIN", "HOOD", "MSTR", "RDDT", "SMCI",
    "BAC", "JPM", "V", "WMT", "INTC",
]

# Sweep-validated 8-ticker high-WR universe (default for production)
HIGH_WR_TICKERS = ["IWM", "COST", "V", "INTC", "PLTR", "CRM", "WMT", "NVDA"]


class ScalpConfig(BaseModel, frozen=True):
    """Immutable configuration for the scalp v2 backtest engine.

    Defaults are sweep-validated: 90% WR on 4-year OPRA data (2022-2026)
    across 25,600 parameter combinations with walk-forward confirmation
    (train 83% → validate 93%).
    """

    # ── Ticker universe ────────────────────────────────────────
    tickers: list[str] = Field(
        default=[
            # Sweep-validated high-WR tickers (90%+ WR over 4yr backtest)
            "IWM", "COST", "V", "INTC", "PLTR", "CRM", "WMT", "NVDA",
        ],
        description="Underlying tickers to scan (default: sweep-validated 8-ticker universe)",
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
        default=0.002,
        gt=0,
        description="Take-profit: underlying move fraction (0.002 = 0.20%)",
    )
    max_hold_bars: int = Field(
        default=12,
        ge=1,
        le=100,
        description="Maximum hold period in 5-min bars (12 × 5 = 60 min)",
    )
    trail_pct: float = Field(
        default=0.03,
        gt=0,
        le=1.0,
        description="Trailing stop: percentage from peak premium (0.03 = 3%)",
    )

    # ── Entry filters (sweep-validated: 90% WR on 4yr OPRA data) ─
    ibs_threshold: float = Field(
        default=0.12,
        ge=0,
        le=1.0,
        description="Internal Bar Strength upper limit",
    )
    rsi3_threshold: float = Field(
        default=15.0,
        ge=0,
        le=100,
        description="RSI(3) upper limit",
    )
    vol_spike: float = Field(
        default=2.5,
        gt=0,
        description="Volume spike multiplier (bar vol / avg vol)",
    )
    intraday_drop: float = Field(
        default=-0.002,
        le=0,
        description="Max intraday drop from open (-0.002 = -0.2%)",
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
