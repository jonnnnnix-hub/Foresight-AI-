"""Tests for UOA scanner engine logic."""

from datetime import date

from flowedge.config.settings import Settings
from flowedge.scanner.schemas.flow import FlowAlert, FlowSentiment, FlowType
from flowedge.scanner.schemas.options import OptionType
from flowedge.scanner.uoa.engine import (
    _classify_signal_type,
    _compute_direction,
    _score_uoa,
)


def _settings() -> Settings:
    return Settings(uoa_volume_oi_threshold=3.0, uoa_min_premium=25_000.0)


def _make_alert(
    flow_type: FlowType = FlowType.REGULAR,
    sentiment: FlowSentiment = FlowSentiment.BULLISH,
    premium: float = 10_000.0,
    volume_oi_ratio: float = 1.0,
) -> FlowAlert:
    return FlowAlert(
        ticker="TEST",
        option_type=OptionType.CALL,
        strike=100.0,
        expiration=date(2024, 3, 15),
        flow_type=flow_type,
        sentiment=sentiment,
        premium=premium,
        volume=1000,
        open_interest=500,
        volume_oi_ratio=volume_oi_ratio,
    )


def test_classify_sweep_cluster() -> None:
    alerts = [_make_alert(FlowType.SWEEP) for _ in range(4)]
    assert _classify_signal_type(alerts, _settings()) == "sweep_cluster"


def test_classify_block_trade() -> None:
    alerts = [_make_alert(premium=50_000.0)]
    assert _classify_signal_type(alerts, _settings()) == "block_trade"


def test_classify_volume_spike() -> None:
    alerts = [_make_alert(volume_oi_ratio=5.0)]
    assert _classify_signal_type(alerts, _settings()) == "volume_spike"


def test_compute_direction_bullish() -> None:
    alerts = [
        _make_alert(sentiment=FlowSentiment.BULLISH),
        _make_alert(sentiment=FlowSentiment.BULLISH),
        _make_alert(sentiment=FlowSentiment.BEARISH),
    ]
    assert _compute_direction(alerts) == FlowSentiment.BULLISH


def test_compute_direction_neutral() -> None:
    alerts = [
        _make_alert(sentiment=FlowSentiment.BULLISH),
        _make_alert(sentiment=FlowSentiment.BEARISH),
    ]
    assert _compute_direction(alerts) == FlowSentiment.NEUTRAL


def test_score_high_activity() -> None:
    alerts = [
        _make_alert(
            FlowType.SWEEP,
            premium=100_000.0,
            volume_oi_ratio=10.0,
        )
        for _ in range(12)
    ]
    score = _score_uoa(alerts, "sweep_cluster", _settings())
    assert score >= 8.0


def test_score_low_activity() -> None:
    alerts = [_make_alert(premium=1_000.0, volume_oi_ratio=0.5)]
    score = _score_uoa(alerts, "skew_shift", _settings())
    assert score <= 3.0
