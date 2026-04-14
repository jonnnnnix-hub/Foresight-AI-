"""Tests for WebSocket streaming connection manager."""


from flowedge.scanner.streaming.engine import ConnectionManager


def test_connection_manager_init() -> None:
    mgr = ConnectionManager()
    assert mgr.client_count == 0
