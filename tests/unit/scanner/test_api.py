"""Tests for scanner API routes."""

from fastapi.testclient import TestClient

from flowedge.api.app import create_app


def test_scanner_health() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/v1/scanner/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
