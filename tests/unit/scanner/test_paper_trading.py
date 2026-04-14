"""Tests for paper trading schemas."""

from flowedge.scanner.paper_trading.schemas import (
    OrderSide,
    OrderStatus,
    PaperOrder,
    PaperPortfolio,
    PaperPosition,
)


def test_paper_order_defaults() -> None:
    order = PaperOrder(ticker="TSLA")
    assert order.side == OrderSide.BUY
    assert order.status == OrderStatus.PENDING
    assert order.qty == 1


def test_paper_position() -> None:
    pos = PaperPosition(
        ticker="AAPL",
        symbol="AAPL",
        qty=10,
        avg_entry=195.0,
        current_price=200.0,
        unrealized_pnl=50.0,
        unrealized_pnl_pct=2.56,
    )
    assert pos.unrealized_pnl > 0


def test_paper_portfolio_defaults() -> None:
    portfolio = PaperPortfolio()
    assert portfolio.cash == 100_000.0
    assert portfolio.portfolio_value == 100_000.0
    assert portfolio.positions == []


def test_portfolio_json_roundtrip() -> None:
    portfolio = PaperPortfolio(
        cash=50_000.0,
        positions=[
            PaperPosition(ticker="TSLA", symbol="TSLA", qty=5, avg_entry=250.0),
        ],
    )
    json_str = portfolio.model_dump_json()
    rebuilt = PaperPortfolio.model_validate_json(json_str)
    assert len(rebuilt.positions) == 1
    assert rebuilt.cash == 50_000.0
