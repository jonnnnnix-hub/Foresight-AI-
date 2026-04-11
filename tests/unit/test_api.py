"""Tests for API routes."""

from fastapi.testclient import TestClient

from flowedge.api.app import create_app


def test_health_endpoint() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_analyze_endpoint() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/api/v1/analyze",
        json={"repo_urls": ["https://github.com/test/repo"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["repo_count"] == 1
    assert "run_id" in data
