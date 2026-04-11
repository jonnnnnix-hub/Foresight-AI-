"""Tests for AI signal interpreter."""

from flowedge.scanner.interpreter.engine import _fallback_thesis
from flowedge.scanner.interpreter.schemas import ConvictionLevel, TradeThesis
from flowedge.scanner.schemas.catalyst import CatalystSignal
from flowedge.scanner.schemas.flow import FlowSentiment, UOASignal
from flowedge.scanner.schemas.iv import IVRankData, IVSignal
from flowedge.scanner.schemas.signals import LottoOpportunity


def test_fallback_thesis_high_conviction() -> None:
    opp = LottoOpportunity(
        ticker="TSLA",
        composite_score=8.0,
        uoa_score=9.0,
        iv_score=7.0,
        catalyst_score=6.0,
        suggested_direction=FlowSentiment.BULLISH,
        uoa_signal=UOASignal(
            ticker="TSLA",
            signal_type="sweep_cluster",
            strength=9.0,
            call_put_ratio=3.0,
            total_premium=500_000.0,
            direction=FlowSentiment.BULLISH,
        ),
        iv_signal=IVSignal(
            ticker="TSLA",
            iv_rank=IVRankData(ticker="TSLA", iv_rank=15.0),
            is_cheap_premium=True,
            strength=7.0,
        ),
        catalyst_signal=CatalystSignal(
            ticker="TSLA",
            days_to_nearest_catalyst=5,
            strength=6.0,
        ),
    )
    thesis = _fallback_thesis(opp)
    assert thesis.conviction == ConvictionLevel.HIGH
    assert thesis.ticker == "TSLA"
    assert "cheap premium" in thesis.thesis_summary.lower()


def test_fallback_thesis_low_conviction() -> None:
    opp = LottoOpportunity(ticker="WEAK", composite_score=3.5)
    thesis = _fallback_thesis(opp)
    assert thesis.conviction == ConvictionLevel.LOW


def test_fallback_thesis_avoid() -> None:
    opp = LottoOpportunity(ticker="BAD", composite_score=1.0)
    thesis = _fallback_thesis(opp)
    assert thesis.conviction == ConvictionLevel.AVOID


def test_thesis_schema_roundtrip() -> None:
    thesis = TradeThesis(
        ticker="TEST",
        conviction=ConvictionLevel.MEDIUM,
        thesis_summary="Test thesis",
        key_risks=["Risk 1", "Risk 2"],
    )
    json_str = thesis.model_dump_json()
    rebuilt = TradeThesis.model_validate_json(json_str)
    assert rebuilt.conviction == ConvictionLevel.MEDIUM
    assert len(rebuilt.key_risks) == 2
