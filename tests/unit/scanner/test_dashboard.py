"""Tests for dashboard UI endpoint."""

from fastapi.testclient import TestClient

from flowedge.api.app import create_app


def test_dashboard_serves_html() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/dashboard/")
    assert resp.status_code == 200
    assert "FlowEdge Scanner" in resp.text
    assert "Scan Now" in resp.text


def test_dashboard_has_controls() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/dashboard/")
    assert "tickers" in resp.text
    assert "minScore" in resp.text
    assert "Auto-refresh" in resp.text
