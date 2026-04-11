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


def test_analyze_endpoint_accepts_request() -> None:
    """Verify the analyze endpoint accepts a valid request.

    Uses raise_server_exceptions=False since the background task
    will fail without an API key — we only test the HTTP acceptance.
    """
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/v1/analyze",
        json={"repo_urls": ["https://github.com/test/repo"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["repo_count"] == 1
    assert "run_id" in data


def test_analyze_endpoint_rejects_empty() -> None:
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/v1/analyze", json={"repo_urls": []})
    assert resp.status_code == 422  # Validation error


def test_get_run_not_found() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/v1/runs/nonexistent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "not_found"
