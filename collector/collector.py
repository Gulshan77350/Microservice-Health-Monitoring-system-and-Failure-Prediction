"""
Telemetry collector — the only component that writes to SQLite and the ONLY
caller of the prediction API's /predict/batch.

Every POLL_INTERVAL seconds (one writer thread):
  1. send PROBES_PER_CYCLE end-to-end probes to payment /process
     (payment -> order -> notification, one X-Request-ID per probe)
  2. scrape /metrics from each service and store them (UTC timestamps)
  3. flag z-score anomalies (|z| > Z_THRESHOLD, default 3) against each
     metric's previous ROLLING_WIN values
  4. call /predict/batch once for all services and store the predictions
  5. every DRIFT_INTERVAL: store a PSI snapshot from the API's /drift and log
     a `retrain_recommended` event when max PSI crosses 0.25
  6. maintenance: retention cleanup hourly; VACUUM daily (it rewrites the
     whole file and blocks the writer for its duration, so it is not run
     after every cleanup — deleted pages are reused by SQLite anyway)

The read API serves history (metrics joined with predictions), latest
predictions, anomalies, drift snapshots and events to the dashboard.
"""

from __future__ import annotations

import hmac
import os
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query

from collector import db
from common.logging_utils import install_request_logging, request_id_var, setup_logging

ROOT = Path(__file__).resolve().parents[1]
log = setup_logging("collector")


def _services() -> tuple[tuple[str, str], ...]:
    return (
        ("payment", os.getenv("PAYMENT_SERVICE_URL", "http://localhost:8001")),
        ("order", os.getenv("ORDER_SERVICE_URL", "http://localhost:8002")),
        ("notification", os.getenv("NOTIFICATION_SERVICE_URL", "http://localhost:8003")),
    )


@dataclass(frozen=True)
class Config:
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", str(ROOT / "collector" / "metrics.db")))
    services: tuple[tuple[str, str], ...] = field(default_factory=_services)
    prediction_api_url: str = field(default_factory=lambda: os.getenv("PREDICTION_API_URL", "http://localhost:8004"))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    probe_url: str = field(default_factory=lambda: os.getenv("PROBE_URL", "http://localhost:8001/process"))
    probes_per_cycle: int = field(default_factory=lambda: int(os.getenv("PROBES_PER_CYCLE", "3")))
    poll_interval: float = field(default_factory=lambda: float(os.getenv("POLL_INTERVAL", "5")))
    retention_days: int = field(default_factory=lambda: int(os.getenv("RETENTION_DAYS", "7")))
    cleanup_interval: float = field(default_factory=lambda: float(os.getenv("CLEANUP_INTERVAL", "3600")))
    vacuum_interval: float = field(default_factory=lambda: float(os.getenv("VACUUM_INTERVAL", "86400")))
    drift_interval: float = field(default_factory=lambda: float(os.getenv("DRIFT_INTERVAL", "300")))
    rolling_win: int = 20
    z_threshold: float = field(default_factory=lambda: float(os.getenv("Z_THRESHOLD", "3.0")))

    @property
    def service_names(self) -> list[str]:
        return [s for s, _ in self.services]


class AnomalyDetector:
    """Flags a value whose z-score against the PREVIOUS `window` values exceeds `z_threshold`."""

    def __init__(self, window: int, z_threshold: float, min_points: int = 5):
        self.windows: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=window))
        self.z_threshold = z_threshold
        self.min_points = min_points

    def check(self, service: str, data: dict, ts: str) -> list[dict]:
        flagged = []
        for metric in db.METRIC_COLS:
            val = data.get(metric)
            if val is None:
                continue
            win = self.windows[(service, metric)]
            if len(win) >= self.min_points:
                mean, std = statistics.mean(win), statistics.stdev(win)
                if std > 0 and abs(val - mean) / std > self.z_threshold:
                    flagged.append(
                        {
                            "timestamp": ts,
                            "service": service,
                            "metric": metric,
                            "value": round(val, 3),
                            "mean": round(mean, 3),
                            "std": round(std, 3),
                            "z_score": round(abs(val - mean) / std, 3),
                        }
                    )
            win.append(val)  # append AFTER scoring so a point is not part of its own baseline
        return flagged


class Collector:
    def __init__(self, cfg: Config, session: requests.Session | None = None):
        self.cfg = cfg
        self.http = session or requests.Session()
        self.detector = AnomalyDetector(cfg.rolling_win, cfg.z_threshold)
        self.stop = threading.Event()
        self.status: dict = {"cycles": 0, "last_cycle_at": None, "last_prediction_at": None, "last_error": None}
        now = time.monotonic()
        self._next_cleanup = now
        self._next_vacuum = now + cfg.vacuum_interval
        self._next_drift = now + cfg.drift_interval

    def _api_headers(self) -> dict:
        return {"X-API-Key": self.cfg.api_key, "X-Request-ID": request_id_var.get()}

    def probe(self) -> None:
        if not self.cfg.probe_url:
            return
        for _ in range(self.cfg.probes_per_cycle):
            rid = f"probe-{uuid.uuid4().hex[:12]}"
            try:
                r = self.http.post(self.cfg.probe_url, headers={"X-Request-ID": rid}, timeout=5)
                log.debug("probe", extra={"probe_id": rid, "status": r.status_code})
            except requests.RequestException as exc:
                log.warning("probe failed", extra={"probe_id": rid, "error": type(exc).__name__})

    def scrape(self, ts: str) -> list[dict]:
        rows = []
        for name, base in self.cfg.services:
            try:
                r = self.http.get(f"{base.rstrip('/')}/metrics", timeout=3)
                r.raise_for_status()
                data = r.json()
                rows.append({"timestamp": ts, "service": name, **{k: data[k] for k in db.METRIC_COLS}})
            except (requests.RequestException, KeyError, ValueError) as exc:
                log.warning("scrape failed", extra={"target": name, "error": str(exc)})
        return rows

    def predict(self, ts: str, rows: list[dict]) -> dict | None:
        payload = [
            {"service": r["service"], "timestamp": ts, "metrics": {k: r[k] for k in db.METRIC_COLS}} for r in rows
        ]
        try:
            r = self.http.post(
                f"{self.cfg.prediction_api_url}/predict/batch", json=payload, headers=self._api_headers(), timeout=5
            )
            r.raise_for_status()
        except requests.RequestException as exc:
            log.warning("prediction request failed", extra={"error": str(exc)})
            return None
        results = r.json()["results"]
        db.insert_predictions(self.cfg.db_path, ts, results)
        self.status["last_prediction_at"] = ts
        return results

    def snapshot_drift(self, ts: str) -> dict | None:
        try:
            r = self.http.get(f"{self.cfg.prediction_api_url}/drift", headers=self._api_headers(), timeout=5)
            r.raise_for_status()
            report = r.json()
        except requests.RequestException as exc:
            log.warning("drift snapshot failed", extra={"error": str(exc)})
            return None
        previous = db.last_drift_snapshot(self.cfg.db_path)
        db.insert_drift_snapshot(self.cfg.db_path, ts, report)
        if report.get("retrain_recommended") and not (previous and previous["retrain_recommended"]):
            msg = f"Retrain recommended: max PSI {report.get('max_psi')} > 0.25"
            db.insert_event(self.cfg.db_path, ts, "warning", "retrain_recommended", msg, report.get("per_feature"))
            log.warning(msg, extra={"model_version": report.get("model_version")})
        return report

    def maintenance(self) -> None:
        now = time.monotonic()
        if now >= self._next_cleanup:
            deleted = db.cleanup(self.cfg.db_path, self.cfg.retention_days)
            log.info("retention cleanup", extra={"deleted": deleted, "retention_days": self.cfg.retention_days})
            self._next_cleanup = now + self.cfg.cleanup_interval
        if now >= self._next_vacuum:
            started = time.perf_counter()
            db.vacuum(self.cfg.db_path)
            log.info("vacuum", extra={"duration_ms": round((time.perf_counter() - started) * 1000, 1)})
            self._next_vacuum = now + self.cfg.vacuum_interval

    def run_cycle(self) -> dict:
        ts = db.to_ts(db.utc_now())
        token = request_id_var.set(f"cycle-{uuid.uuid4().hex[:12]}")
        try:
            self.probe()
            rows = self.scrape(ts)
            results = None
            if rows:
                db.insert_metrics(self.cfg.db_path, rows)
                for row in rows:
                    for a in self.detector.check(row["service"], row, ts):
                        db.insert_anomaly(self.cfg.db_path, a)
                        log.info("anomaly", extra=a)
                results = self.predict(ts, rows)
            if time.monotonic() >= self._next_drift:
                self.snapshot_drift(ts)
                self._next_drift = time.monotonic() + self.cfg.drift_interval
            self.status.update(cycles=self.status["cycles"] + 1, last_cycle_at=ts, last_error=None)
            return {"timestamp": ts, "rows": len(rows), "predictions": results}
        finally:
            request_id_var.reset(token)

    def run_forever(self) -> None:
        log.info("collector loop started", extra={"poll_interval": self.cfg.poll_interval})
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                self.run_cycle()
                self.maintenance()
            except Exception as exc:
                self.status["last_error"] = str(exc)
                log.exception("collector cycle failed")
            self.stop.wait(max(0.0, self.cfg.poll_interval - (time.monotonic() - started)))


def create_app(cfg: Config | None = None, start_background: bool = True) -> FastAPI:
    cfg = cfg or Config()
    collector = Collector(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mode = db.init_db(cfg.db_path)
        log.info("database ready", extra={"db_path": cfg.db_path, "journal_mode": mode})
        thread = None
        if start_background:
            thread = threading.Thread(target=collector.run_forever, daemon=True, name="collector-loop")
            thread.start()
        yield
        collector.stop.set()
        if thread:
            thread.join(timeout=10)

    app = FastAPI(title="Metrics Collector API", version="3.0.0", lifespan=lifespan)
    app.state.collector = collector
    install_request_logging(app, log, quiet_paths=frozenset({"/health"}))

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if not cfg.api_key:
            raise HTTPException(503, "API_KEY not configured; mutating endpoints disabled")
        if not x_api_key or not hmac.compare_digest(x_api_key.encode(), cfg.api_key.encode()):
            raise HTTPException(401, "Missing or invalid X-API-Key")

    def check_service(service: str | None) -> None:
        if service is not None and service not in cfg.service_names:
            raise HTTPException(422, f"Unknown service '{service}'. Allowed: {cfg.service_names}")

    @app.get("/")
    def home():
        return {
            "service": "collector",
            "endpoints": [
                "/health",
                "/history",
                "/predictions/latest",
                "/anomalies",
                "/drift/snapshots",
                "/events",
                "/summary",
                "/db-stats",
            ],
        }

    @app.get("/health")
    def health():
        try:
            rows = db.stats(cfg.db_path)["row_counts"]["metrics"]
        except Exception as exc:
            raise HTTPException(503, f"database unavailable: {exc}") from exc
        return {
            "status": "healthy",
            "total_metric_rows": rows,
            "poll_interval": cfg.poll_interval,
            "z_threshold": cfg.z_threshold,
            **collector.status,
        }

    @app.get("/history")
    def history(service: str | None = Query(default=None), limit: int = Query(default=100, ge=1, le=2000)):
        """Metrics joined with the prediction stored for the same (service, timestamp). Oldest first."""
        check_service(service)
        data = db.history(cfg.db_path, service, limit)
        return {"service": service, "count": len(data), "data": data}

    @app.get("/predictions/latest")
    def predictions_latest():
        return {
            "predictions": db.latest_predictions(cfg.db_path, cfg.service_names),
            "last_prediction_at": collector.status["last_prediction_at"],
        }

    @app.get("/anomalies")
    def anomalies(service: str | None = Query(default=None), limit: int = Query(default=50, ge=1, le=500)):
        check_service(service)
        data = db.anomalies(cfg.db_path, service, limit)
        return {"count": len(data), "anomalies": data}

    @app.get("/drift/snapshots")
    def drift_snapshots(limit: int = Query(default=50, ge=1, le=1000)):
        return {"snapshots": db.drift_snapshots(cfg.db_path, limit)}

    @app.get("/events")
    def events(limit: int = Query(default=50, ge=1, le=500)):
        return {"events": db.events(cfg.db_path, limit)}

    @app.get("/summary")
    def summary():
        return db.summary(cfg.db_path, cfg.service_names)

    @app.get("/db-stats")
    def db_stats():
        return {
            **db.stats(cfg.db_path),
            "retention_days": cfg.retention_days,
            "estimated_metric_rows_per_day": len(cfg.services) * int(86400 / cfg.poll_interval),
        }

    @app.post("/cleanup", dependencies=[Depends(require_api_key)])
    def manual_cleanup():
        return {"deleted": db.cleanup(cfg.db_path, cfg.retention_days), "retention_days": cfg.retention_days}

    return app


app = create_app()
