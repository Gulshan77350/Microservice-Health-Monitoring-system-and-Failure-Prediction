"""
Micro-benchmarks quoted in the README. Run from the repo root:

    python -m scripts.benchmark            # writes docs/benchmark.json

Measures, on the local machine:
  1. model inference: shared feature engineering + calibrated XGBoost Booster, one row
  2. /predict/batch for 3 services through FastAPI's TestClient (in-process HTTP stack, no network)
  3. collector SQLite queries on a DB sized for the full retention window
     (7 days x 3 services x one row per 5 s = 362,880 metric rows + as many predictions)
"""

from __future__ import annotations

import json
import os
import platform
import random
import statistics
import tempfile
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOG_LEVEL", "WARNING")


def pct(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]  # noqa: E731
    return {
        "n": len(s),
        "p50_ms": round(q(0.50), 3),
        "p95_ms": round(q(0.95), 3),
        "p99_ms": round(q(0.99), 3),
        "mean_ms": round(statistics.mean(s), 3),
    }


def timed(fn, n: int, warmup: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return pct(out)


def bench_inference() -> dict:
    from api.model_loader import load_bundle
    from common.features import FeatureBuffer

    bundle = load_bundle(ROOT / "artifacts")
    buf = FeatureBuffer()
    rng = random.Random(0)

    def one():
        raw = {
            "cpu": rng.uniform(5, 95),
            "memory": rng.uniform(20, 95),
            "latency": rng.uniform(50, 2000),
            "requests": rng.randint(100, 9000),
            "error_rate": rng.uniform(0, 30),
        }
        bundle.predict_proba(buf.push("payment", raw))

    return timed(one, 2000)


def bench_batch_endpoint(tmp: Path) -> dict:
    from fastapi.testclient import TestClient

    from api.app import create_app
    from api.config import Settings

    settings = Settings(api_key="bench", alert_db_path=tmp / "alerts.db", collector_url="")
    rng = random.Random(1)
    with TestClient(create_app(settings, seed_on_startup=False)) as client:

        def call():
            payload = [
                {
                    "service": s,
                    "metrics": {
                        "cpu": rng.uniform(5, 95),
                        "memory": rng.uniform(20, 95),
                        "latency": rng.uniform(50, 2000),
                        "requests": rng.randint(100, 9000),
                        "error_rate": rng.uniform(0, 30),
                    },
                }
                for s in ("payment", "order", "notification")
            ]
            r = client.post("/predict/batch", json=payload, headers={"X-API-Key": "bench"})
            assert r.status_code == 200

        return timed(call, 500)


def bench_collector_queries(tmp: Path) -> dict:
    from collector import db

    path = tmp / "bench.db"
    db.init_db(path)
    services = ["payment", "order", "notification"]
    start = datetime.now(UTC) - timedelta(days=7)
    n_steps = 7 * 86400 // 5
    rng = random.Random(2)

    t0 = time.perf_counter()
    batch_m, batch_p = [], []
    with closing(db.connect(path)) as con, con:
        for i in range(n_steps):
            ts = db.to_ts(start + timedelta(seconds=5 * i))
            for s in services:
                batch_m.append(
                    (ts, s, rng.uniform(5, 95), rng.uniform(20, 95), rng.uniform(20, 900), rng.randint(0, 60), 0.0)
                )
                p = rng.random()
                batch_p.append((ts, s, p, "HIGH" if p > 0.4 else "LOW", int(p > 0.11), "Normal Operation", "v3", 3))
        con.executemany(
            "INSERT INTO metrics (timestamp, service, cpu, memory, latency, requests, error_rate) "
            "VALUES (?,?,?,?,?,?,?)",
            batch_m,
        )
        con.executemany(
            "INSERT INTO predictions (timestamp, service, failure_probability, risk, alert, root_cause, "
            "model_version, horizon_steps) VALUES (?,?,?,?,?,?,?,?)",
            batch_p,
        )
    load_s = time.perf_counter() - t0
    rows = len(batch_m)

    return {
        "metric_rows": rows,
        "prediction_rows": len(batch_p),
        "bulk_load_seconds": round(load_s, 2),
        "db_size_mb": round(db.stats(path)["size_kb"] / 1024, 1),
        "history_service_limit_100": timed(lambda: db.history(path, "payment", 100), 300),
        "history_all_limit_2000": timed(lambda: db.history(path, None, 2000), 100),
        "latest_predictions": timed(lambda: db.latest_predictions(path, ["payment", "order", "notification"]), 300),
    }


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        results = {
            "machine": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "python": platform.python_version(),
            },
            "run_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "inference_single_row": bench_inference(),
            "predict_batch_endpoint_3_services": bench_batch_endpoint(tmp),
            "collector_sqlite": bench_collector_queries(tmp),
        }
    out = ROOT / "docs" / "benchmark.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nwritten to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
