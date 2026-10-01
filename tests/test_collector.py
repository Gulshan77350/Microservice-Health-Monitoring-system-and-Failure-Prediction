"""Collector: SQLite functions on a temp DB, cleanup cut-offs, anomaly scoring, one full cycle with fakes."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from collector import db
from collector.collector import AnomalyDetector, Collector, Config, create_app

ROW = {"cpu": 40.0, "memory": 55.0, "latency": 120.0, "requests": 30, "error_rate": 0.0}


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "metrics.db"
    assert db.init_db(path) == "wal"
    return path


def _pred(p=0.2, risk="MEDIUM", alert=True):
    return {"failure_probability": p, "risk": risk, "alert": alert, "root_cause": "x", "model_version": "v3"}


def test_timestamps_are_fixed_width_utc():
    ts = db.to_ts(datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
    assert ts == "2026-01-02T03:04:05Z"
    ist = datetime(2026, 1, 2, 14, 4, 5, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert db.to_ts(ist) == "2026-01-02T08:34:05Z"


def test_cleanup_respects_retention_boundary(db_path):
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    old = db.to_ts(now - timedelta(days=7, seconds=1))
    edge = db.to_ts(now - timedelta(days=7))
    fresh = db.to_ts(now - timedelta(hours=1))
    db.insert_metrics(db_path, [{"timestamp": t, "service": "payment", **ROW} for t in (old, edge, fresh)])
    db.insert_predictions(db_path, old, {"payment": _pred()})
    deleted = db.cleanup(db_path, retention_days=7, now=now)
    assert deleted["metrics"] == 1 and deleted["predictions"] == 1
    assert [r["timestamp"] for r in db.history(db_path, "payment", 10)] == [edge, fresh]


def test_history_joins_predictions_and_latest(db_path):
    t1, t2 = "2026-10-01T10:00:00Z", "2026-10-01T10:00:05Z"
    db.insert_metrics(db_path, [{"timestamp": t, "service": s, **ROW} for t in (t1, t2) for s in ("payment", "order")])
    db.insert_predictions(db_path, t1, {"payment": _pred(0.1, "LOW", False), "order": _pred()})
    db.insert_predictions(db_path, t2, {"payment": _pred(0.7, "HIGH", True)})

    hist = db.history(db_path, "payment", 10)
    assert [h["failure_probability"] for h in hist] == [0.1, 0.7]
    assert hist[1]["alert"] is True and hist[1]["model_version"] == "v3"
    assert db.history(db_path, "order", 10)[1]["failure_probability"] is None  # no prediction stored for t2

    latest = db.latest_predictions(db_path)
    assert latest["payment"]["timestamp"] == t2 and latest["payment"]["risk"] == "HIGH"
    assert latest["order"]["timestamp"] == t1
    assert latest["payment"]["cpu"] == ROW["cpu"]


def test_duplicate_metric_rows_ignored(db_path):
    row = {"timestamp": "2026-10-01T10:00:00Z", "service": "payment", **ROW}
    db.insert_metrics(db_path, [row, row])
    assert db.stats(db_path)["row_counts"]["metrics"] == 1


def test_anomaly_point_not_part_of_own_baseline():
    det = AnomalyDetector(window=20, z_threshold=2.0)
    for v in [10, 11, 9, 10, 11, 9, 10]:
        assert det.check("payment", {"cpu": v}, "t") == []
    flagged = det.check("payment", {"cpu": 30}, "t")
    assert len(flagged) == 1 and flagged[0]["metric"] == "cpu"
    assert flagged[0]["mean"] == pytest.approx(10.0)


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, psi=0.05):
        self.calls, self.psi = [], psi

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        if url.endswith("/metrics"):
            return _Resp(ROW)
        if url.endswith("/drift"):
            return _Resp(
                {
                    "status": "ok",
                    "overall_status": "significant_shift" if self.psi > 0.25 else "stable",
                    "max_psi": self.psi,
                    "retrain_recommended": self.psi > 0.25,
                    "model_version": "v3",
                    "per_feature": {},
                }
            )
        raise AssertionError(url)

    def post(self, url, json=None, **kw):
        self.calls.append(("POST", url, kw))
        if url.endswith("/predict/batch"):
            return _Resp({"results": {item["service"]: _pred() for item in json}})
        return _Resp({"status": "processed"})


def test_run_cycle_is_only_batch_caller_and_stores_predictions(db_path):
    cfg = Config(db_path=str(db_path), api_key="k", probes_per_cycle=2, drift_interval=0)
    c = Collector(cfg, session=FakeSession())
    out = c.run_cycle()
    assert out["rows"] == 3 and set(out["predictions"]) == {"payment", "order", "notification"}

    posts = [call for call in c.http.calls if call[0] == "POST"]
    batch = [p for p in posts if p[1].endswith("/predict/batch")]
    assert len(batch) == 1 and batch[0][2]["headers"]["X-API-Key"] == "k"
    probes = [p for p in posts if p[1].endswith("/process")]
    assert len(probes) == 2 and all(p[2]["headers"]["X-Request-ID"].startswith("probe-") for p in probes)

    assert set(db.latest_predictions(db_path)) == {"payment", "order", "notification"}
    assert len(db.drift_snapshots(db_path, 10)) == 1


def test_retrain_event_logged_once_per_episode(db_path):
    cfg = Config(db_path=str(db_path), api_key="k")
    c = Collector(cfg, session=FakeSession(psi=0.4))
    c.snapshot_drift("2026-10-01T10:00:00Z")
    c.snapshot_drift("2026-10-01T10:05:00Z")
    events = db.events(db_path, 10)
    assert len(events) == 1 and events[0]["kind"] == "retrain_recommended"
    c.http.psi = 0.05
    c.snapshot_drift("2026-10-01T10:10:00Z")
    c.http.psi = 0.5
    c.snapshot_drift("2026-10-01T10:15:00Z")
    assert len(db.events(db_path, 10)) == 2


def test_collector_api(db_path):
    cfg = Config(db_path=str(db_path), api_key="k")
    with TestClient(create_app(cfg, start_background=False)) as client:
        assert client.get("/health").json()["status"] == "healthy"
        assert client.get("/history", params={"service": "billing"}).status_code == 422
        assert client.post("/cleanup").status_code == 401
        assert client.post("/cleanup", headers={"X-API-Key": "k"}).status_code == 200
        assert client.get("/db-stats").json()["journal_mode"] == "wal"
        assert client.get("/predictions/latest").json()["predictions"] == {}
