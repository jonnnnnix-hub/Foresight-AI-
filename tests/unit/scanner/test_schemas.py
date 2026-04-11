"""Tests for scanner schema validation and roundtrips."""

from datetime import date

from flowedge.scanner.schemas.catalyst import (
    CatalystSignal,
    EarningsEvent,
    InsiderTrade,
)
from flowedge.scanner.schemas.flow import (
    FlowAlert,
    FlowSentiment,
    FlowType,
    UOASignal,
)
from flowedge.scanner.schemas.iv import IVRankData, IVRegime, IVSignal
from flowedge.scanner.schemas.options import OptionContract, OptionsChain, OptionType
from flowedge.scanner.schemas.signals import LottoOpportunity, ScannerResult


def test_option_contract_creation() -> None:
    c = OptionContract(
        symbol="AAPL240119C00200000",
        underlying="AAPL",
        option_type=OptionType.CALL,
        strike=200.0,
        expiration=date(2024, 1, 19),
        bid=5.0,
        ask=5.50,
        mid=5.25,
        volume=1000,
        open_interest=5000,
    )
    assert c.strike == 200.0
    assert c.option_type == OptionType.CALL


def test_options_chain_properties() -> None:
    chain = OptionsChain(
        underlying="AAPL",
        underlying_price=195.0,
        contracts=[
            OptionContract(
                symbol="C1", underlying="AAPL",
                option_type=OptionType.CALL, strike=200.0,
                expiration=date(2024, 1, 19), volume=500,
            ),
            OptionContract(
                symbol="P1", underlying="AAPL",
                option_type=OptionType.PUT, strike=190.0,
                expiration=date(2024, 1, 19), volume=300,
            ),
        ],
    )
    assert len(chain.calls) == 1
    assert len(chain.puts) == 1
    assert chain.total_call_volume == 500
    assert chain.total_put_volume == 300


def test_flow_alert_defaults() -> None:
    alert = FlowAlert(
        ticker="TSLA",
        option_type=OptionType.CALL,
        strike=250.0,
        expiration=date(2024, 2, 16),
    )
    assert alert.flow_type == FlowType.REGULAR
    assert alert.sentiment == FlowSentiment.NEUTRAL
    assert alert.source == "unusual_whales"


def test_uoa_signal_scoring() -> None:
    sig = UOASignal(
        ticker="NVDA",
        signal_type="sweep_cluster",
        direction=FlowSentiment.BULLISH,
        strength=8.5,
        call_volume=10000,
        put_volume=2000,
        call_put_ratio=5.0,
        total_premium=500000.0,
    )
    assert 0 <= sig.strength <= 10
    assert sig.call_put_ratio == 5.0


def test_iv_rank_data_bounds() -> None:
    data = IVRankData(ticker="SPY", iv_rank=25.0, iv_percentile=30.0, current_iv=0.18)
    assert 0 <= data.iv_rank <= 100


def test_iv_signal_regime() -> None:
    sig = IVSignal(
        ticker="SPY",
        iv_rank=IVRankData(ticker="SPY", iv_rank=15.0),
        regime=IVRegime.LOW,
        is_cheap_premium=True,
        strength=7.0,
    )
    assert sig.regime == IVRegime.LOW
    assert sig.is_cheap_premium


def test_earnings_event() -> None:
    e = EarningsEvent(
        ticker="AAPL",
        report_date=date(2024, 1, 25),
        eps_estimate=2.10,
        time_of_day="amc",
    )
    assert e.source == "fmp"


def test_insider_trade() -> None:
    t = InsiderTrade(
        ticker="MSFT",
        insider_name="John Doe",
        title="CEO",
        transaction_type="P",
        shares=10000,
        price_per_share=400.0,
        total_value=4_000_000.0,
    )
    assert t.total_value == 4_000_000.0


def test_catalyst_signal() -> None:
    sig = CatalystSignal(
        ticker="AAPL",
        days_to_nearest_catalyst=5,
        net_insider_sentiment="bullish",
        insider_buy_count=3,
        strength=7.0,
    )
    assert sig.days_to_nearest_catalyst == 5


def test_lotto_opportunity() -> None:
    opp = LottoOpportunity(
        ticker="NVDA",
        composite_score=8.5,
        uoa_score=9.0,
        iv_score=7.0,
        catalyst_score=8.0,
        suggested_direction=FlowSentiment.BULLISH,
    )
    assert 0 <= opp.composite_score <= 10


def test_scanner_result_top_opportunities() -> None:
    result = ScannerResult(
        scan_id="test-001",
        opportunities=[
            LottoOpportunity(ticker="A", composite_score=3.0),
            LottoOpportunity(ticker="B", composite_score=8.0),
            LottoOpportunity(ticker="C", composite_score=5.0),
        ],
    )
    top = result.top_opportunities
    assert top[0].ticker == "B"
    assert top[-1].ticker == "A"


def test_all_scanner_schemas_roundtrip() -> None:
    """All scanner schemas must survive JSON roundtrip."""
    schemas = [
        OptionContract(
            symbol="X", underlying="X",
            option_type=OptionType.CALL, strike=100.0,
            expiration=date(2024, 1, 19),
        ),
        FlowAlert(
            ticker="X", option_type=OptionType.PUT,
            strike=90.0, expiration=date(2024, 1, 19),
        ),
        UOASignal(ticker="X"),
        IVRankData(ticker="X", iv_rank=50.0),
        IVSignal(ticker="X", iv_rank=IVRankData(ticker="X", iv_rank=50.0)),
        EarningsEvent(ticker="X", report_date=date(2024, 1, 25)),
        InsiderTrade(ticker="X", insider_name="Test"),
        CatalystSignal(ticker="X"),
        LottoOpportunity(ticker="X"),
        ScannerResult(scan_id="test"),
    ]
    for schema in schemas:
        json_str = schema.model_dump_json()
        rebuilt = type(schema).model_validate_json(json_str)
        assert rebuilt.model_dump() == schema.model_dump(), (
            f"Roundtrip failed for {type(schema).__name__}"
        )
