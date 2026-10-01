"""Simulated microservice: measured metrics, tracing, request-id propagation, auth on failure injection."""

import random

import httpx
import pytest
from fastapi.testclient import TestClient

import services.app as svc


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(svc, "API_KEY", "k")
    monkeypatch.setattr(svc, "DOWNSTREAM_URL", "")
    svc.state.__init__()
    with TestClient(svc.app) as c:
        yield c


def test_metrics_and_health_are_not_logged_as_traffic(client):
    for _ in range(3):
        client.get("/metrics")
        client.get("/health")
        client.get("/traces")
    assert client.get("/traces").json()["recent_requests"] == []
    assert client.get("/metrics").json()["latency"] == 0.0


def test_process_traffic_drives_measured_metrics(client):
    for _ in range(4):
        assert client.post("/process").status_code == 200
    m = client.get("/metrics").json()
    assert m["latency"] >= 10  # /process sleeps 10-50 ms
    assert m["error_rate"] == 0.0
    assert m["requests"] == 4
    assert [t["path"] for t in client.get("/traces").json()["recent_requests"]] == ["/process"] * 4


def test_failure_injection_requires_key_and_produces_errors(client, monkeypatch):
    assert client.post("/simulate_failure").status_code == 401
    assert client.post("/simulate_failure", headers={"X-API-Key": "k"}).status_code == 200
    monkeypatch.setattr(svc, "FAILURE_LATENCY_S", (0.0, 0.0))
    monkeypatch.setattr(svc, "FAILURE_ERROR_PROB", 1.0)
    assert client.post("/process").status_code == 503
    assert client.get("/metrics").json()["error_rate"] == 100.0
    assert client.get("/health").json()["status"] == "degraded"
    assert client.post("/reset", headers={"X-API-Key": "k"}).json()["failure_mode"] is False


def test_request_id_forwarded_downstream(client, monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = request.headers.get("X-Request-ID")
        return httpx.Response(200, json={"service": "order", "status": "processed"})

    monkeypatch.setattr(svc, "DOWNSTREAM_URL", "http://order-service:8000")
    svc.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r = client.post("/process", headers={"X-Request-ID": "trace-42"})
    assert r.status_code == 200
    assert r.headers["X-Request-ID"] == "trace-42"
    assert seen["/process"] == "trace-42"
    assert r.json()["downstream"]["body"]["service"] == "order"

    deps = client.get("/health").json()["dependencies"]
    assert deps == {"order-service": "ok"}


def test_downstream_unreachable_reported(client, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(svc, "DOWNSTREAM_URL", "http://order-service:8000")
    svc.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert client.post("/process").status_code == 502
    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["dependencies"]["order-service"].startswith("unreachable")


def test_compute_metrics_failure_overlay_bounded():
    m = svc.compute_metrics([], now=0.0, cpu=80.0, memory=90.0, failure_mode=True, rng=random.Random(0))
    assert m["cpu"] <= 100 and m["memory"] <= 100
    assert m["latency"] == 0.0 and m["error_rate"] == 0.0
