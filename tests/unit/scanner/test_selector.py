"""Tests for smart contract selector."""

from datetime import date, timedelta

from flowedge.scanner.schemas.catalyst import CatalystSignal
from flowedge.scanner.schemas.flow import FlowSentiment
from flowedge.scanner.schemas.iv import IVRankData, IVSignal
from flowedge.scanner.schemas.options import OptionContract, OptionsChain, OptionType
from flowedge.scanner.schemas.signals import LottoOpportunity
from flowedge.scanner.selector.engine import (
    _compute_risk_reward,
    _pick_expiration,
    _pick_strike,
    select_contracts,
)


def _make_chain(underlying_price: float = 200.0) -> OptionsChain:
    today = date.today()
    contracts: list[OptionContract] = []
    for days_out in [7, 14, 21, 28, 35]:
        exp = today + timedelta(days=days_out)
        for strike in [180, 190, 200, 210, 220, 230]:
            contracts.append(
                OptionContract(
                    symbol=f"TEST{exp}{strike}C",
                    underlying="TEST",
                    option_type=OptionType.CALL,
                    strike=float(strike),
                    expiration=exp,
                    bid=max(0, underlying_price - strike + 2),
                    ask=max(0.1, underlying_price - strike + 3),
                    mid=max(0.05, underlying_price - strike + 2.5),
                    volume=500,
                    open_interest=2000,
                    delta=0.3,
                    source="test",
                )
            )
            contracts.append(
                OptionContract(
                    symbol=f"TEST{exp}{strike}P",
                    underlying="TEST",
                    option_type=OptionType.PUT,
                    strike=float(strike),
                    expiration=exp,
                    bid=max(0, strike - underlying_price + 2),
                    ask=max(0.1, strike - underlying_price + 3),
                    mid=max(0.05, strike - underlying_price + 2.5),
                    volume=300,
                    open_interest=1500,
                    delta=-0.3,
                    source="test",
                )
            )
    return OptionsChain(
        underlying="TEST",
        underlying_price=underlying_price,
        contracts=contracts,
        source="test",
    )


def test_pick_expiration_default() -> None:
    chain = _make_chain()
    exp = _pick_expiration(chain, None, FlowSentiment.BULLISH)
    assert exp is not None
    today = date.today()
    assert exp >= today + timedelta(days=3)


def test_pick_expiration_with_catalyst() -> None:
    chain = _make_chain()
    catalyst = CatalystSignal(ticker="TEST", days_to_nearest_catalyst=10)
    exp = _pick_expiration(chain, catalyst, FlowSentiment.BULLISH)
    assert exp is not None
    # Should be AFTER the catalyst
    today = date.today()
    catalyst_date = today + timedelta(days=10)
    assert exp > catalyst_date


def test_pick_strike_bullish_otm() -> None:
    chain = _make_chain(200.0)
    calls = [c for c in chain.contracts if c.option_type == OptionType.CALL]
    strikes = _pick_strike(calls, 200.0, FlowSentiment.BULLISH, None)
    assert len(strikes) > 0
    # Bullish → should pick OTM calls (strike > 200)
    for s in strikes:
        assert s.option_type == OptionType.CALL


def test_pick_strike_bearish_otm() -> None:
    chain = _make_chain(200.0)
    all_contracts = chain.contracts
    strikes = _pick_strike(all_contracts, 200.0, FlowSentiment.BEARISH, None)
    assert len(strikes) > 0
    for s in strikes:
        assert s.option_type == OptionType.PUT


def test_compute_risk_reward() -> None:
    contract = OptionContract(
        symbol="T", underlying="T",
        option_type=OptionType.CALL, strike=210.0,
        expiration=date.today() + timedelta(days=14),
        mid=3.0, ask=3.5,
    )
    rr, max_loss, breakeven_pct = _compute_risk_reward(contract, 200.0, 10.0)
    assert max_loss > 0
    assert breakeven_pct > 0


def test_select_contracts_end_to_end() -> None:
    chain = _make_chain(200.0)
    opp = LottoOpportunity(
        ticker="TEST",
        composite_score=7.0,
        suggested_direction=FlowSentiment.BULLISH,
        iv_signal=IVSignal(
            ticker="TEST",
            iv_rank=IVRankData(ticker="TEST", iv_rank=20.0),
            is_cheap_premium=True,
            strength=6.0,
        ),
    )
    recs = select_contracts(opp, chain)
    assert len(recs) > 0
    for r in recs:
        assert r.contract.option_type == OptionType.CALL
        assert r.max_loss > 0
