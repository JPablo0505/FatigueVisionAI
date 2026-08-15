"""Tests for the public backend endpoints."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_api_endpoint() -> None:
    """GET /api returns the service identity."""
    response = client.get("/api")
    assert response.status_code == 200
    assert response.json()["service"] == "fatiguevision-backend"
    assert response.json()["version"] == "0.1.0"


def test_healthz_endpoint() -> None:
    """GET /api/healthz reports a healthy service."""
    response = client.get("/api/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"