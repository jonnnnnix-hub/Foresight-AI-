"""Trade thesis schemas from AI interpretation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ConvictionLevel(StrEnum):
    """How confident the AI is in the setup."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    AVOID = "avoid"


class TradeThesis(BaseModel):
    """AI-generated analysis of a lotto opportunity."""

    ticker: str
    conviction: ConvictionLevel = ConvictionLevel.LOW
    thesis_summary: str = Field(
        default="", description="1-2 sentence thesis"
    )
    smart_money_read: str = Field(
        default="", description="What institutional flow suggests"
    )
    catalyst_narrative: str = Field(
        default="", description="Why a big move could happen"
    )
    iv_context: str = Field(
        default="", description="What IV regime means for this trade"
    )
    gex_context: str = Field(
        default="", description="How dealer positioning affects the setup"
    )
    ideal_entry: str = Field(
        default="", description="When and how to enter"
    )
    target_exit: str = Field(
        default="", description="When to take profit"
    )
    stop_logic: str = Field(
        default="", description="When to cut the loss"
    )
    key_risks: list[str] = Field(default_factory=list)
    why_this_could_work: str = ""
    why_this_could_fail: str = ""
    position_sizing_note: str = Field(
        default="", description="How much capital to risk"
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now())
