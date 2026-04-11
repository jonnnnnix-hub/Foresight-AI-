"""Tests for the composite lotto scorer."""

from flowedge.scanner.schemas.catalyst import CatalystSignal
from flowedge.scanner.schemas.flow import FlowSentiment, UOASignal
from flowedge.scanner.schemas.iv import IVRankData, IVRegime, IVSignal
from flowedge.scanner.scorer.engine import (
    _determine_direction,
    _generate_entry_criteria,
    _generate_risk_flags,
    score_lottos,
)


def _make_uoa(ticker: str, strength: float, direction: FlowSentiment) -> UOASignal:
    return UOASignal(
        ticker=ticker,
        signal_type="sweep_cluster",
        direction=direction,
        strength=strength,
        total_premium=100_000.0,
    )


def _make_iv(ticker: str, strength: float, iv_rank: float) -> IVSignal:
    return IVSignal(
        ticker=ticker,
        iv_rank=IVRankData(ticker=ticker, iv_rank=iv_rank),
        regime=IVRegime.LOW if iv_rank < 30 else IVRegime.NORMAL,
        is_cheap_premium=iv_rank < 25,
        strength=strength,
    )


def _make_catalyst(ticker: str, strength: float) -> CatalystSignal:
    return CatalystSignal(
        ticker=ticker,
        days_to_nearest_catalyst=5,
        net_insider_sentiment="bullish",
        insider_buy_count=3,
        strength=strength,
    )


def test_score_lottos_basic() -> None:
    result = score_lottos(
        uoa_signals=[_make_uoa("AAPL", 8.0, FlowSentiment.BULLISH)],
        iv_signals=[_make_iv("AAPL", 7.0, 20.0)],
        catalyst_signals=[_make_catalyst("AAPL", 6.0)],
    )
    assert len(result.opportunities) == 1
    opp = result.opportunities[0]
    assert opp.ticker == "AAPL"
    assert opp.composite_score > 0


def test_score_lottos_ranking() -> None:
    result = score_lottos(
        uoa_signals=[
            _make_uoa("HIGH", 9.0, FlowSentiment.BULLISH),
            _make_uoa("LOW", 2.0, FlowSentiment.NEUTRAL),
        ],
        iv_signals=[
            _make_iv("HIGH", 8.0, 15.0),
            _make_iv("LOW", 1.0, 80.0),
        ],
        catalyst_signals=[],
    )
    assert result.opportunities[0].ticker == "HIGH"
    assert result.opportunities[-1].ticker == "LOW"


def test_score_lottos_partial_signals() -> None:
    """Tickers with only some signal types should still score."""
    result = score_lottos(
        uoa_signals=[_make_uoa("UOA_ONLY", 7.0, FlowSentiment.BULLISH)],
        iv_signals=[],
        catalyst_signals=[_make_catalyst("CAT_ONLY", 6.0)],
    )
    assert len(result.opportunities) == 2
    assert any(o.ticker == "UOA_ONLY" for o in result.opportunities)
    assert any(o.ticker == "CAT_ONLY" for o in result.opportunities)


def test_determine_direction_from_uoa() -> None:
    uoa = _make_uoa("X", 5.0, FlowSentiment.BEARISH)
    assert _determine_direction(uoa, None) == FlowSentiment.BEARISH


def test_determine_direction_from_catalyst() -> None:
    catalyst = _make_catalyst("X", 5.0)
    assert _determine_direction(None, catalyst) == FlowSentiment.BULLISH


def test_entry_criteria_generation() -> None:
    uoa = _make_uoa("X", 7.0, FlowSentiment.BULLISH)
    iv = _make_iv("X", 6.0, 20.0)
    catalyst = _make_catalyst("X", 5.0)
    criteria = _generate_entry_criteria(uoa, iv, catalyst)
    assert len(criteria) >= 1


def test_risk_flags_high_iv() -> None:
    iv = _make_iv("X", 3.0, 85.0)
    flags = _generate_risk_flags(None, iv, None)
    assert any("expensive" in f.lower() or "iv" in f.lower() for f in flags)


def test_scanner_result_has_scan_id() -> None:
    result = score_lottos([], [], [])
    assert result.scan_id
    assert result.status == "complete"
