"""Prediction API: model loading, known-input predictions, validation, auth, idempotency, recipients."""

import numpy as np
from fastapi.testclient import TestClient

from api.app import create_app
from api.model_loader import load_bundle
from common.features import FeatureBuffer
from common.risk import classify_risk
from tests.conftest import HEALTHY, ROOT, STRESSED


def test_health_reports_loaded_model_and_disabled_alerts(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_version"] == "v3"
    assert body["alerts_enabled"] is False
    assert body["alerts_disabled_reason"]


def test_health_is_503_when_artifacts_missing(settings, tmp_path):
    from dataclasses import replace

    with TestClient(create_app(replace(settings, artifact_root=tmp_path / "nope"), seed_on_startup=False)) as c:
        r = c.get("/health")
    assert r.status_code == 503
    assert r.json()["model_loaded"] is False


def test_model_info_reports_loaded_version_and_thresholds(client):
    info = client.get("/model-info").json()
    assert info["model_version"] == "v3"
    t = info["thresholds"]
    assert 0 < t["cost_optimal"] < 1 and 0 < t["max_f1"] < 1
    assert info["risk_bands"]["HIGH"][0] == max(t.values())


def test_predict_known_input_matches_direct_model_call(client, auth):
    bundle = load_bundle(ROOT / "artifacts")
    expected = float(bundle.predict_proba(FeatureBuffer().push("payment", STRESSED))[0])
    r = client.post("/predict", json={"service": "payment", "metrics": STRESSED}, headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["failure_probability"] == round(expected, 4)
    assert body["risk"] == classify_risk(expected, bundle.risk_bands)
    assert body["alert"] == (expected >= bundle.thresholds["cost_optimal"])
    assert body["model_version"] == "v3"


def test_stressed_scores_higher_than_healthy(client, auth):
    lo = client.post("/predict", json={"service": "order", "metrics": HEALTHY}, headers=auth).json()
    hi = client.post("/predict", json={"service": "notification", "metrics": STRESSED}, headers=auth).json()
    assert hi["failure_probability"] > lo["failure_probability"]
    assert lo["risk"] == "LOW" and hi["risk"] == "HIGH"


def test_mutating_endpoints_require_api_key(client):
    payload = [{"service": "payment", "metrics": HEALTHY}]
    assert client.post("/predict/batch", json=payload).status_code == 401
    assert client.post("/predict/batch", json=payload, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/alert-config/recipients", json={"email": "ops@example.com"}).status_code == 401
    assert client.delete("/alert-config/recipients/1").status_code == 401
    assert client.post("/alert/test").status_code == 401


def test_auth_fails_closed_when_server_key_unset(settings):
    from dataclasses import replace

    with TestClient(create_app(replace(settings, api_key=""), seed_on_startup=False)) as c:
        r = c.post("/predict", json={"service": "payment", "metrics": HEALTHY}, headers={"X-API-Key": ""})
    assert r.status_code == 503


def test_batch_validation(client, auth):
    item = {"service": "payment", "metrics": HEALTHY}
    assert client.post("/predict/batch", json=[], headers=auth).status_code == 422
    assert client.post("/predict/batch", json=[item] * 11, headers=auth).status_code == 413
    assert client.post("/predict/batch", json=[item, item], headers=auth).status_code == 422
    unknown = {"service": "billing", "metrics": HEALTHY}
    assert client.post("/predict/batch", json=[unknown], headers=auth).status_code == 422
    bad = {"service": "payment", "metrics": {**HEALTHY, "cpu": 150}}
    assert client.post("/predict/batch", json=[bad], headers=auth).status_code == 422


def test_batch_is_idempotent_per_timestamp(client, auth):
    item = {"service": "payment", "metrics": HEALTHY, "timestamp": "2026-10-01T10:00:00Z"}
    first = client.post("/predict/batch", json=[item], headers=auth).json()["results"]["payment"]
    again = client.post("/predict/batch", json=[item], headers=auth).json()["results"]["payment"]
    assert first["buffer_size"] == again["buffer_size"] == 1
    assert again["duplicate"] is True
    newer = {**item, "timestamp": "2026-10-01T10:00:05Z"}
    assert client.post("/predict/batch", json=[newer], headers=auth).json()["results"]["payment"]["buffer_size"] == 2


def test_alert_email_deduplicated_per_episode(client, auth):
    state = client.app.state.serving
    from api.app import ServiceMetrics, ServingState

    assert isinstance(state, ServingState)
    item = ServiceMetrics(service="order", metrics=STRESSED)
    _, send1 = state.predict(item)
    _, send2 = state.predict(item)
    assert send1 is True and send2 is False


def test_recipients_validated_masked_and_persisted(client, auth, settings):
    assert client.post("/alert-config/recipients", json={"email": "not-an-email"}, headers=auth).status_code == 422
    r = client.post("/alert-config/recipients", json={"email": "Ops.Team@example.com"}, headers=auth)
    assert r.status_code == 201
    rid = r.json()["id"]
    listed = client.get("/alert-config").json()["recipients"]
    assert listed == [{"id": rid, "email": "op***@example.com"}]

    # A fresh app instance on the same DB file sees the recipient (survives restarts).
    with TestClient(create_app(settings, seed_on_startup=False)) as c2:
        assert len(c2.get("/alert-config").json()["recipients"]) == 1
        assert c2.delete(f"/alert-config/recipients/{rid}", headers=auth).status_code == 200
        assert c2.delete(f"/alert-config/recipients/{rid}", headers=auth).status_code == 404


def test_test_alert_refused_when_smtp_disabled(client, auth):
    assert client.post("/alert/test", headers=auth).status_code == 409


def test_cors_allows_configured_origin_without_credentials(client):
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-credentials" not in r.headers
    r = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_request_id_propagated(client):
    r = client.get("/", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_feature_importance_and_drift(client, auth):
    fi = client.get("/feature-importance").json()
    assert fi["method"] == "mean_abs_shap"
    assert abs(sum(r["importance_pct"] for r in fi["feature_importance"]) - 100) < 0.5
    d = client.get("/drift").json()
    assert d["status"] == "warming_up" and d["retrain_recommended"] is False


def test_self_check_rejects_tampered_artifact(tmp_path):
    import json
    import shutil

    import pytest

    dst = tmp_path / "v9"
    shutil.copytree(ROOT / "artifacts" / "v3", dst)
    meta = json.loads((dst / "metadata.json").read_text())
    meta["calibration"]["b"] += 0.5  # simulate train/serve skew
    (dst / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="self-check"):
        load_bundle(tmp_path)
    assert np.isfinite(load_bundle(ROOT / "artifacts").self_check())
