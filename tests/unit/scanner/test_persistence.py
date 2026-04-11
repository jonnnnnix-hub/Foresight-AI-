"""Tests for watchlist and scan history persistence."""

from pathlib import Path

from flowedge.scanner.persistence.schemas import (
    TradeStatus,
    Watchlist,
    WatchlistItem,
)
from flowedge.scanner.persistence.store import ScannerStore
from flowedge.scanner.schemas.flow import FlowSentiment
from flowedge.scanner.schemas.signals import LottoOpportunity, ScannerResult


def test_watchlist_crud(tmp_path: Path) -> None:
    store = ScannerStore(tmp_path)

    # Add tickers
    wl = store.add_to_watchlist("TSLA")
    assert len(wl.items) == 1
    wl = store.add_to_watchlist("NVDA")
    assert len(wl.items) == 2

    # Deduplicate
    wl = store.add_to_watchlist("TSLA")
    assert len(wl.items) == 2

    # Remove
    wl = store.remove_from_watchlist("TSLA")
    assert len(wl.items) == 1
    assert wl.items[0].ticker == "NVDA"


def test_watchlist_persistence(tmp_path: Path) -> None:
    store = ScannerStore(tmp_path)
    store.add_to_watchlist("AAPL", notes="earnings play")

    # Reload from disk
    store2 = ScannerStore(tmp_path)
    wl = store2.get_watchlist()
    assert len(wl.items) == 1
    assert wl.items[0].ticker == "AAPL"
    assert wl.items[0].notes == "earnings play"


def test_scan_history(tmp_path: Path) -> None:
    store = ScannerStore(tmp_path)
    result = ScannerResult(
        scan_id="test-001",
        tickers_scanned=5,
        opportunities=[
            LottoOpportunity(ticker="TSLA", composite_score=7.0),
            LottoOpportunity(ticker="NVDA", composite_score=5.0),
        ],
    )
    entry = store.log_scan(result)
    assert entry.top_ticker == "TSLA"
    assert entry.top_score == 7.0

    history = store.get_scan_history()
    assert len(history) == 1


def test_alert_tracking(tmp_path: Path) -> None:
    store = ScannerStore(tmp_path)
    opp = LottoOpportunity(
        ticker="TSLA",
        composite_score=6.5,
        suggested_direction=FlowSentiment.BULLISH,
    )
    alert = store.track_alert(opp, scan_id="s-001")
    assert alert.status == TradeStatus.ALERT

    # Update to entered
    updated = store.update_alert(
        alert.alert_id,
        status=TradeStatus.ENTERED,
        entry_price=3.50,
    )
    assert updated is not None
    assert updated.status == TradeStatus.ENTERED

    # Update to exited with P&L
    updated = store.update_alert(
        alert.alert_id,
        status=TradeStatus.EXITED,
        exit_price=7.00,
        pnl_pct=100.0,
    )
    assert updated is not None
    assert updated.was_winner is True


def test_learning_stats_empty(tmp_path: Path) -> None:
    store = ScannerStore(tmp_path)
    stats = store.compute_learning_stats()
    assert stats.total_alerts == 0
    assert stats.win_rate == 0.0


def test_watchlist_tickers_property() -> None:
    wl = Watchlist(
        items=[
            WatchlistItem(ticker="AAPL"),
            WatchlistItem(ticker="TSLA", auto_scan=False),
        ]
    )
    assert wl.tickers == ["AAPL", "TSLA"]
    assert wl.auto_scan_tickers == ["AAPL"]
