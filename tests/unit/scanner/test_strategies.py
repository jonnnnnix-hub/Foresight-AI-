"""Tests for ARCHITECT v2 multi-leg strategy builder."""

from datetime import date, timedelta

from flowedge.scanner.schemas.catalyst import CatalystSignal
from flowedge.scanner.schemas.flow import FlowSentiment
from flowedge.scanner.schemas.options import OptionContract, OptionsChain, OptionType
from flowedge.scanner.schemas.signals import LottoOpportunity
from flowedge.scanner.selector.schemas import StrategyBlueprint, StrategyType
from flowedge.scanner.selector.strategies import (
    build_debit_spread,
    build_single_leg,
    build_straddle,
    build_strangle,
    recommend_strategies,
)


def _make_chain(price: float = 200.0) -> OptionsChain:
    today = date.today()
    exp = today + timedelta(days=21)
    contracts: list[OptionContract] = []
    for strike in range(180, 225, 5):
        for opt_type in [OptionType.CALL, OptionType.PUT]:
            dist = abs(strike - price)
            prem = max(0.5, 10 - dist * 0.3)
            contracts.append(
                OptionContract(
                    symbol=f"T{strike}{opt_type.value[0]}",
                    underlying="TEST",
                    option_type=opt_type,
                    strike=float(strike),
                    expiration=exp,
                    bid=round(prem - 0.1, 2),
                    ask=round(prem + 0.1, 2),
                    mid=round(prem, 2),
                    volume=1000,
                    open_interest=5000,
                    delta=0.4 if opt_type == OptionType.CALL else -0.4,
                    source="test",
                )
            )
    return OptionsChain(
        underlying="TEST",
        underlying_price=price,
        contracts=contracts,
        source="test",
    )


def test_build_single_call() -> None:
    chain = _make_chain(200.0)
    exp = date.today() + timedelta(days=21)
    result = build_single_leg(chain, FlowSentiment.BULLISH, exp)
    assert result is not None
    assert result.strategy_type == StrategyType.SINGLE_CALL
    assert len(result.legs) == 1
    assert result.legs[0].action == "buy"


def test_build_single_put() -> None:
    chain = _make_chain(200.0)
    exp = date.today() + timedelta(days=21)
    result = build_single_leg(chain, FlowSentiment.BEARISH, exp)
    assert result is not None
    assert result.strategy_type == StrategyType.SINGLE_PUT


def test_build_debit_spread() -> None:
    chain = _make_chain(200.0)
    exp = date.today() + timedelta(days=21)
    result = build_debit_spread(chain, FlowSentiment.BULLISH, exp)
    assert result is not None
    assert result.strategy_type == StrategyType.CALL_DEBIT_SPREAD
    assert len(result.legs) == 2
    assert result.legs[0].action == "buy"
    assert result.legs[1].action == "sell"
    assert result.max_loss > 0
    assert "defined_risk" in result.tags


def test_build_straddle() -> None:
    chain = _make_chain(200.0)
    exp = date.today() + timedelta(days=21)
    result = build_straddle(chain, exp)
    assert result is not None
    assert result.strategy_type == StrategyType.LONG_STRADDLE
    assert len(result.legs) == 2
    assert len(result.breakeven_prices) == 2


def test_build_strangle() -> None:
    chain = _make_chain(200.0)
    exp = date.today() + timedelta(days=21)
    result = build_strangle(chain, exp)
    assert result is not None
    assert result.strategy_type == StrategyType.LONG_STRANGLE
    assert len(result.legs) == 2


def test_recommend_strategies_with_catalyst() -> None:
    chain = _make_chain(200.0)
    opp = LottoOpportunity(
        ticker="TEST",
        composite_score=7.0,
        suggested_direction=FlowSentiment.BULLISH,
        catalyst_signal=CatalystSignal(
            ticker="TEST", days_to_nearest_catalyst=5
        ),
    )
    strategies = recommend_strategies(opp, chain)
    # Should have single leg + spread + straddle + strangle (catalyst near)
    assert len(strategies) >= 3
    types = [s.strategy_type for s in strategies]
    assert StrategyType.SINGLE_CALL in types
    assert StrategyType.CALL_DEBIT_SPREAD in types


def test_recommend_strategies_no_catalyst() -> None:
    chain = _make_chain(200.0)
    opp = LottoOpportunity(
        ticker="TEST",
        composite_score=5.0,
        suggested_direction=FlowSentiment.BULLISH,
    )
    strategies = recommend_strategies(opp, chain)
    # Should have single leg + spread only (no catalyst → no straddle)
    assert len(strategies) >= 1


def test_strategy_blueprint_properties() -> None:
    bp = StrategyBlueprint(
        ticker="TEST",
        strategy_type=StrategyType.CALL_DEBIT_SPREAD,
        legs=[],
        max_profit=500.0,
        max_loss=200.0,
    )
    assert bp.is_defined_risk
    assert bp.total_legs == 0
    json_str = bp.model_dump_json()
    rebuilt = StrategyBlueprint.model_validate_json(json_str)
    assert rebuilt.strategy_type == StrategyType.CALL_DEBIT_SPREAD
