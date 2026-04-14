"""Tests for charts dashboard."""

from fastapi.testclient import TestClient

from flowedge.api.app import create_app


def test_charts_dashboard_serves_html() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/charts/")
    assert resp.status_code == 200
    assert "NEXUS Analytics" in resp.text
    assert "ORACLE" in resp.text
    assert "VORTEX" in resp.text
    assert "chart.js" in resp.text.lower() or "Chart" in resp.text
