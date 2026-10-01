"""
Prediction API — serves the calibrated XGBoost failure model from artifacts/v{N}/.

Callers: the collector is the ONLY caller of /predict/batch (every poll cycle).
The dashboard reads predictions from the collector, and only uses this API for
model info, feature importance, drift and alert-recipient management.

State (per process — run a single worker; see README "Limitations"):
  * per-service ring buffer of recent raw metrics for rolling features,
    seeded from collector history on startup so a restart does not reset
    the rolling window to a single point;
  * last timestamp per service, which makes /predict/batch idempotent;
  * rolling window of live inputs for PSI drift;
  * per-service alert state for de-duplicating emails.
"""

from __future__ import annotations

import hmac
import json
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field

from api.alerts import Mailer, RecipientStore, alert_html, mask_email
from api.config import Settings
from api.model_loader import ModelBundle, load_bundle
from common import drift
from common.features import BUFFER_SIZE, RAW_FEATURES, WINDOW, FeatureBuffer
from common.logging_utils import install_request_logging, setup_logging
from common.risk import classify_risk, root_cause, should_alert

API_VERSION = "3.0.0"
LIVE_WINDOW = 500

log = setup_logging("prediction-api")


# ── Schemas ─────────────────────────────────────────────────────────────────
class Metrics(BaseModel):
    cpu: float = Field(..., ge=0, le=100, description="CPU usage percent")
    memory: float = Field(..., ge=0, le=100, description="Memory usage percent")
    latency: float = Field(..., ge=0, le=600_000, description="Response latency in ms")
    requests: int = Field(..., ge=0, description="Request volume proxy")
    error_rate: float = Field(..., ge=0, le=100, description="Error rate percent")


class ServiceMetrics(BaseModel):
    service: str = Field(..., min_length=1, max_length=64)
    metrics: Metrics
    timestamp: datetime | None = Field(
        default=None, description="Observation time (UTC). Repeated/older timestamps are not re-buffered."
    )


class EmailIn(BaseModel):
    email: EmailStr


class EmailListIn(BaseModel):
    emails: list[EmailStr] = Field(..., max_length=50)


def _utc(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


# ── Serving state ───────────────────────────────────────────────────────────
class ServingState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bundle: ModelBundle | None = None
        self.load_error: str | None = None
        self.buffers = FeatureBuffer(BUFFER_SIZE)
        self.last_ts: dict[str, datetime] = {}
        self.live: dict[str, deque] = {f: deque(maxlen=LIVE_WINDOW) for f in RAW_FEATURES}
        self.alert_state: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.mailer = Mailer(settings)
        self.store: RecipientStore | None = None
        self.seed_status: dict = {"status": "not_started"}
        self.started_at = datetime.now(UTC).isoformat(timespec="seconds")

    def load_model(self) -> None:
        try:
            self.bundle = load_bundle(self.settings.artifact_root, self.settings.model_version)
            self.load_error = None
            log.info("model loaded", extra={"model_version": self.bundle.version, "path": str(self.bundle.path)})
        except Exception as exc:
            self.bundle, self.load_error = None, str(exc)
            log.exception("model failed to load")

    def open_store(self) -> None:
        self.store = RecipientStore(self.settings.alert_db_path)
        if self.settings.default_recipient and not self.store.list():
            self.store.add(self.settings.default_recipient)

    # Buffer seeding — runs in a background thread so startup never blocks on the collector.
    def seed_from_collector(self, attempts: int = 12, delay_s: float = 5.0) -> None:
        url = self.settings.collector_url.rstrip("/")
        if not url:
            self.seed_status = {"status": "disabled", "reason": "COLLECTOR_URL not set"}
            return
        for attempt in range(1, attempts + 1):
            try:
                seeded = {}
                for svc in self.settings.allowed_services:
                    query = urllib.parse.urlencode({"service": svc, "limit": BUFFER_SIZE})
                    with urllib.request.urlopen(f"{url}/history?{query}", timeout=3) as resp:
                        rows = json.load(resp)["data"]
                    with self.lock:
                        if self.buffers.size(svc) == 0 and rows:
                            seeded[svc] = self.buffers.seed(svc, rows)
                            self.last_ts[svc] = _utc(datetime.fromisoformat(rows[-1]["timestamp"]))
                self.seed_status = {"status": "ok", "seeded_rows": seeded, "attempts": attempt}
                log.info("feature buffers seeded from collector", extra=self.seed_status)
                return
            except Exception as exc:
                self.seed_status = {"status": "retrying", "attempts": attempt, "error": str(exc)}
                time.sleep(delay_s)
        self.seed_status["status"] = "failed"
        log.warning("could not seed feature buffers from collector", extra=self.seed_status)

    def predict(self, item: ServiceMetrics) -> tuple[dict, bool]:
        bundle = self.bundle
        svc, raw, ts = item.service, item.metrics.model_dump(), _utc(item.timestamp)
        with self.lock:
            last = self.last_ts.get(svc)
            duplicate = ts is not None and last is not None and ts <= last
            if duplicate:
                feats = self.buffers.current(svc)
            else:
                feats = self.buffers.push(svc, raw)
                if ts is not None:
                    self.last_ts[svc] = ts
                for f in RAW_FEATURES:
                    self.live[f].append(raw[f])
            probability = float(bundle.predict_proba(feats)[0])
            alert = should_alert(probability, bundle.thresholds)
            send_email = False
            if not duplicate:
                st = self.alert_state.setdefault(svc, {"alerting": False, "last_sent": 0.0})
                now = time.monotonic()
                if alert and not st["alerting"] and now - st["last_sent"] >= self.settings.alert_cooldown_s:
                    send_email, st["last_sent"] = True, now
                st["alerting"] = alert
            size = self.buffers.size(svc)
        result = {
            "service": svc,
            "failure_probability": round(probability, 4),
            "alert": alert,
            "risk": classify_risk(probability, bundle.risk_bands),
            "root_cause": root_cause(raw),
            "model_version": bundle.version,
            "horizon_steps": bundle.metadata["target"]["horizon_steps"],
            "thresholds": bundle.thresholds,
            "buffer_size": size,
            "buffer_warm": size >= WINDOW,
            "duplicate": duplicate,
        }
        return result, send_email


# ── App factory ─────────────────────────────────────────────────────────────
def create_app(settings: Settings | None = None, seed_on_startup: bool = True) -> FastAPI:
    settings = settings or Settings()
    state = ServingState(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.load_model()
        state.open_store()
        if seed_on_startup:
            threading.Thread(target=state.seed_from_collector, daemon=True, name="seed-buffers").start()
        log.info(
            "startup complete",
            extra={"alerts_enabled": state.mailer.enabled, "auth_configured": bool(settings.api_key)},
        )
        yield

    app = FastAPI(
        title="Microservice Failure Prediction API",
        description="Calibrated XGBoost failure-risk model with PSI drift monitoring and email alerts.",
        version=API_VERSION,
        lifespan=lifespan,
    )
    app.state.serving = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )
    install_request_logging(app, log, quiet_paths=frozenset({"/health"}))

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if not settings.api_key:
            raise HTTPException(503, "API_KEY is not configured on the server; mutating endpoints are disabled")
        if not x_api_key or not hmac.compare_digest(x_api_key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "Missing or invalid X-API-Key")

    def require_model() -> ModelBundle:
        if state.bundle is None:
            raise HTTPException(503, f"Model not loaded: {state.load_error}")
        return state.bundle

    def require_store() -> RecipientStore:
        if state.store is None:
            raise HTTPException(503, "Recipient store not initialised")
        return state.store

    def check_service(name: str) -> None:
        if name not in settings.allowed_services:
            raise HTTPException(422, f"Unknown service '{name}'. Allowed: {settings.allowed_services}")

    def queue_alert_email(bg: BackgroundTasks, item: ServiceMetrics, result: dict) -> None:
        recipients = state.store.emails() if state.store else []
        if not (state.mailer.enabled and recipients):
            return
        subject = f"[{result['risk']}] Failure risk for '{item.service}'"
        bg.add_task(state.mailer.send, recipients, subject, alert_html(item.service, result, item.metrics.model_dump()))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error", extra={"path": request.url.path})
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # ── Read-only endpoints ──
    @app.get("/")
    def home():
        return {"service": "prediction-api", "version": API_VERSION, "docs": "/docs"}

    @app.get("/health")
    def health():
        body = {
            "status": "healthy" if state.bundle else "unhealthy",
            "api_version": API_VERSION,
            "model_loaded": state.bundle is not None,
            "model_version": state.bundle.version if state.bundle else None,
            "model_error": state.load_error,
            "started_at": state.started_at,
            "alerts_enabled": state.mailer.enabled,
            "alerts_disabled_reason": None
            if state.mailer.enabled
            else "SMTP_SERVER/SMTP_USER/SMTP_PASSWORD/SENDER_EMAIL not all set",
            "alert_recipients": len(state.store.list()) if state.store else 0,
            "auth_configured": bool(settings.api_key),
            "feature_buffers": state.buffers.sizes(),
            "buffer_seed": state.seed_status,
        }
        return JSONResponse(status_code=200 if state.bundle else 503, content=body)

    @app.get("/model-info")
    def model_info(bundle: ModelBundle = Depends(require_model)):
        meta = bundle.metadata
        xgb = meta["metrics"]["xgboost"]
        return {
            "model_version": bundle.version,
            "algorithm": meta["algorithm"],
            "created_at_utc": meta["created_at_utc"],
            "git_sha": meta["git"]["sha"],
            "dataset_md5": meta["dataset"]["md5"],
            "target": meta["target"],
            "thresholds": meta["thresholds"],
            "risk_bands": meta["risk_bands"],
            "calibration": meta["calibration"]["method"],
            "test_metrics": {
                **xgb["test"],
                "at_cost_optimal": xgb["test_at_cost_optimal"],
                "at_max_f1": xgb["test_at_max_f1"],
                "early_warning": xgb["early_warning"],
            },
            "baselines": {
                name: {
                    **meta["metrics"][name]["test"],
                    "at_cost_optimal": meta["metrics"][name]["test_at_cost_optimal"],
                }
                for name in ("rule", "logistic_regression", "random_forest")
            },
            "trivial": meta["metrics"]["trivial"],
            "split": meta["split"],
            "per_service_test": meta["per_service_test"],
            "libraries": meta["libraries"],
        }

    @app.get("/feature-importance")
    def feature_importance(bundle: ModelBundle = Depends(require_model)):
        imp = bundle.metadata["feature_importance"]
        method = "mean_abs_shap" if imp.get("mean_abs_shap") else "gain"
        values = imp[method]
        total = sum(values.values()) or 1.0
        ranked = sorted(
            (
                {"feature": k, "importance": round(v, 6), "importance_pct": round(100 * v / total, 2)}
                for k, v in values.items()
            ),
            key=lambda r: -r["importance"],
        )
        return {"method": method, "model_version": bundle.version, "feature_importance": ranked}

    @app.get("/drift")
    def drift_report(bundle: ModelBundle = Depends(require_model)):
        with state.lock:
            live = {f: list(v) for f, v in state.live.items()}
        return {
            **drift.drift_report(bundle.drift_reference, live),
            "window_size": LIVE_WINDOW,
            "model_version": bundle.version,
        }

    @app.get("/alert-config")
    def get_alert_config(store: RecipientStore = Depends(require_store)):
        return {
            "alerts_enabled": state.mailer.enabled,
            "smtp_server": settings.smtp_server or None,
            "cooldown_seconds": settings.alert_cooldown_s,
            "recipients": [{"id": r["id"], "email": mask_email(r["email"])} for r in store.list()],
        }

    # ── Mutating endpoints (X-API-Key required) ──
    @app.post("/predict", dependencies=[Depends(require_api_key)])
    def predict(item: ServiceMetrics, bg: BackgroundTasks, bundle: ModelBundle = Depends(require_model)):
        check_service(item.service)
        result, send = state.predict(item)
        if send:
            queue_alert_email(bg, item, result)
        return result

    @app.post("/predict/batch", dependencies=[Depends(require_api_key)])
    def predict_batch(items: list[ServiceMetrics], bg: BackgroundTasks, bundle: ModelBundle = Depends(require_model)):
        if not items:
            raise HTTPException(422, "Batch is empty")
        if len(items) > settings.max_batch:
            raise HTTPException(413, f"Batch too large: {len(items)} > {settings.max_batch}")
        names = [i.service for i in items]
        if len(set(names)) != len(names):
            raise HTTPException(422, "Each service may appear at most once per batch")
        for name in names:
            check_service(name)
        results = {}
        for item in items:
            result, send = state.predict(item)
            if send:
                queue_alert_email(bg, item, result)
            results[item.service] = result
        return {"results": results, "count": len(results), "model_version": bundle.version}

    @app.post("/alert-config/recipients", status_code=201, dependencies=[Depends(require_api_key)])
    def add_recipient(body: EmailIn, store: RecipientStore = Depends(require_store)):
        row = store.add(body.email)
        return {"id": row["id"], "email": mask_email(row["email"])}

    @app.delete("/alert-config/recipients/{recipient_id}", dependencies=[Depends(require_api_key)])
    def remove_recipient(recipient_id: int, store: RecipientStore = Depends(require_store)):
        if not store.remove(recipient_id):
            raise HTTPException(404, "Recipient not found")
        return {"removed": recipient_id}

    @app.put("/alert-config/recipients", dependencies=[Depends(require_api_key)])
    def replace_recipients(body: EmailListIn, store: RecipientStore = Depends(require_store)):
        store.replace([str(e) for e in body.emails])
        return {"recipients": [{"id": r["id"], "email": mask_email(r["email"])} for r in store.list()]}

    @app.post("/alert/test", dependencies=[Depends(require_api_key)])
    def send_test_alert(bg: BackgroundTasks, store: RecipientStore = Depends(require_store)):
        if not state.mailer.enabled:
            raise HTTPException(409, "Alerts are disabled: SMTP is not configured")
        recipients = store.emails()
        if not recipients:
            raise HTTPException(409, "No alert recipients configured")
        body = "<p>Test notification from the Microservice Failure Prediction API.</p>"
        bg.add_task(state.mailer.send, recipients, "[TEST] Failure prediction alert channel", body)
        return {"queued": True, "recipient_count": len(recipients)}

    return app


app = create_app()
