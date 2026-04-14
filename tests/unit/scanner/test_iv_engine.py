"""Tests for IV rank engine logic."""

from flowedge.config.settings import Settings
from flowedge.scanner.iv_rank.engine import _is_cheap_premium, _score_iv, classify_regime
from flowedge.scanner.schemas.iv import IVRankData, IVRegime


def _settings() -> Settings:
    return Settings(
        iv_rank_low_threshold=30.0,
        iv_rank_high_threshold=70.0,
    )


def test_classify_regime_low() -> None:
    assert classify_regime(15.0, _settings()) == IVRegime.LOW


def test_classify_regime_normal() -> None:
    assert classify_regime(50.0, _settings()) == IVRegime.NORMAL


def test_classify_regime_elevated() -> None:
    assert classify_regime(80.0, _settings()) == IVRegime.ELEVATED


def test_classify_regime_extreme() -> None:
    assert classify_regime(95.0, _settings()) == IVRegime.EXTREME


def test_is_cheap_premium_low_rank_negative_spread() -> None:
    data = IVRankData(ticker="X", iv_rank=15.0, iv_hv_spread=-0.05)
    assert _is_cheap_premium(data, _settings())


def test_is_cheap_premium_high_rank() -> None:
    data = IVRankData(ticker="X", iv_rank=60.0, iv_hv_spread=-0.10)
    assert not _is_cheap_premium(data, _settings())


def test_score_iv_cheap_premium_scores_high() -> None:
    data = IVRankData(ticker="X", iv_rank=10.0, iv_hv_spread=-0.12)
    score = _score_iv(data, IVRegime.LOW, True, True, _settings())
    assert score >= 7.0


def test_score_iv_expensive_premium_scores_low() -> None:
    data = IVRankData(ticker="X", iv_rank=85.0, iv_hv_spread=0.05)
    score = _score_iv(data, IVRegime.ELEVATED, False, True, _settings())
    assert score <= 2.0
