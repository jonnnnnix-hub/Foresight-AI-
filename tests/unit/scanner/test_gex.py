"""Tests for GEX engine logic."""

from flowedge.scanner.gex.engine import (
    _classify_regime,
    _compute_net_gamma,
    _find_gex_flip,
    _score_gex_for_lottos,
)
from flowedge.scanner.gex.schemas import GEXProfile, GEXRegime, StrikeLevel


def test_compute_net_gamma() -> None:
    strikes = [
        {"strike": 200.0, "callGamma": 0.05, "putGamma": 0.03,
         "callOpenInt": 1000, "putOpenInt": 500},
        {"strike": 210.0, "callGamma": 0.04, "putGamma": 0.06,
         "callOpenInt": 800, "putOpenInt": 1200},
    ]
    levels = _compute_net_gamma(strikes, 205.0)
    assert len(levels) == 2
    assert all(isinstance(lv, StrikeLevel) for lv in levels)


def test_find_gex_flip() -> None:
    levels = [
        StrikeLevel(strike=190.0, net_gamma=-1000),
        StrikeLevel(strike=200.0, net_gamma=500),
        StrikeLevel(strike=210.0, net_gamma=-800),
    ]
    flip = _find_gex_flip(levels, 200.0)
    assert flip is not None


def test_classify_regime_positive() -> None:
    regime = _classify_regime(500_000_000, 200.0)
    assert regime == GEXRegime.POSITIVE


def test_classify_regime_negative() -> None:
    regime = _classify_regime(-500_000_000, 200.0)
    assert regime == GEXRegime.NEGATIVE


def test_score_negative_gex_favorable() -> None:
    score, favorable, rationale = _score_gex_for_lottos(
        GEXRegime.NEGATIVE, 200.0, 198.0, 195.0
    )
    assert favorable is True
    assert score >= 4.0
    assert "amplify" in rationale.lower()


def test_score_positive_gex_unfavorable() -> None:
    score, favorable, rationale = _score_gex_for_lottos(
        GEXRegime.POSITIVE, 200.0, 250.0, 200.0
    )
    assert favorable is False
    assert score < 4.0


def test_gex_profile_schema() -> None:
    profile = GEXProfile(
        ticker="TEST",
        underlying_price=200.0,
        regime=GEXRegime.NEGATIVE,
        total_gex=-1000000.0,
        lotto_favorable=True,
        strength=7.0,
    )
    json_str = profile.model_dump_json()
    rebuilt = GEXProfile.model_validate_json(json_str)
    assert rebuilt.lotto_favorable is True
