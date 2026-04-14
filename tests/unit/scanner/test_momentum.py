"""Tests for momentum engine logic."""

from flowedge.scanner.momentum.engine import _classify_bias, _score_momentum
from flowedge.scanner.momentum.schemas import (
    MomentumBias,
    MomentumSignal,
    TechnicalSnapshot,
)


def test_classify_all_bullish() -> None:
    snapshots = [
        TechnicalSnapshot(
            timeframe="daily",
            rsi=65.0,
            macd_histogram=0.5,
            sma_20=195.0,
            ema_9=198.0,
            ema_21=196.0,
            current_price=200.0,
        ),
    ]
    assert _classify_bias(snapshots) == MomentumBias.STRONG_BULLISH


def test_classify_all_bearish() -> None:
    snapshots = [
        TechnicalSnapshot(
            timeframe="daily",
            rsi=25.0,
            macd_histogram=-0.5,
            sma_20=210.0,
            ema_9=195.0,
            ema_21=200.0,
            current_price=190.0,
        ),
    ]
    assert _classify_bias(snapshots) == MomentumBias.STRONG_BEARISH


def test_classify_empty() -> None:
    assert _classify_bias([]) == MomentumBias.NEUTRAL


def test_score_strong_trend() -> None:
    snapshots = [
        TechnicalSnapshot(timeframe="daily", rsi=75.0),
    ]
    score = _score_momentum(MomentumBias.STRONG_BULLISH, snapshots, True)
    assert score >= 7.0


def test_score_neutral() -> None:
    snapshots = [
        TechnicalSnapshot(timeframe="daily", rsi=50.0),
    ]
    score = _score_momentum(MomentumBias.NEUTRAL, snapshots, False)
    assert score <= 3.0


def test_momentum_signal_schema() -> None:
    sig = MomentumSignal(
        ticker="TEST",
        bias=MomentumBias.BULLISH,
        strength=6.0,
    )
    json_str = sig.model_dump_json()
    rebuilt = MomentumSignal.model_validate_json(json_str)
    assert rebuilt.bias == MomentumBias.BULLISH
