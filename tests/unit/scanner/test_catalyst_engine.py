"""Tests for catalyst scanner engine logic."""

from flowedge.scanner.catalyst.engine import (
    _compute_insider_sentiment,
    _score_catalyst,
)


def test_insider_sentiment_bullish() -> None:
    assert _compute_insider_sentiment(5, 1, 500_000.0) == "bullish"


def test_insider_sentiment_bearish() -> None:
    assert _compute_insider_sentiment(1, 5, -500_000.0) == "bearish"


def test_insider_sentiment_neutral() -> None:
    assert _compute_insider_sentiment(2, 2, 0.0) == "neutral"


def test_score_catalyst_near_earnings_with_move() -> None:
    score = _score_catalyst(
        days_to_catalyst=3,
        insider_sentiment="bullish",
        insider_buy_count=4,
        expected_move_pct=12.0,
    )
    assert score >= 8.0


def test_score_catalyst_no_events() -> None:
    score = _score_catalyst(
        days_to_catalyst=None,
        insider_sentiment="neutral",
        insider_buy_count=0,
        expected_move_pct=0.0,
    )
    assert score == 0.0


def test_score_catalyst_distant_earnings() -> None:
    score = _score_catalyst(
        days_to_catalyst=25,
        insider_sentiment="neutral",
        insider_buy_count=0,
        expected_move_pct=5.0,
    )
    assert score <= 3.0


def test_score_catalyst_insider_cluster() -> None:
    score = _score_catalyst(
        days_to_catalyst=None,
        insider_sentiment="bullish",
        insider_buy_count=5,
        expected_move_pct=0.0,
    )
    assert score >= 2.0
