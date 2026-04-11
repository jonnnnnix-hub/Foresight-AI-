"""Tests for dashboard UI endpoint."""

from fastapi.testclient import TestClient

from flowedge.api.app import create_app


def test_dashboard_serves_html() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/dashboard/")
    assert resp.status_code == 200
    assert "NEXUS" in resp.text
    assert "SCAN" in resp.text


def test_dashboard_has_controls() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/dashboard/")
    assert "tickers" in resp.text
    assert "minScore" in resp.text
    assert "SPECTER" in resp.text
    assert "ORACLE" in resp.text
    assert "SENTINEL" in resp.text


def test_dashboard_has_scan_overlay() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/dashboard/")
    assert "scan-overlay" in resp.text
    assert "hex-spinner" in resp.text
