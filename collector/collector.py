"""
collector.py  —  Tier 2 upgrade
- Writes metrics to SQLite instead of CSV
- Exposes a FastAPI server on port 8005 with:
    GET /history?service=payment&limit=100   → historical data
    GET /anomalies                           → recent anomaly flags
    GET /health                              → collector health
- Anomaly detection: flags any metric > 2 std deviations from
  its rolling 20-point mean
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import requests
import sqlite3
import time
import statistics
from datetime import datetime
from collections import deque
import os

# ── Config ─────────────────────────────────────────────────────────────────
DB_FILE       = "metrics.db"
POLL_INTERVAL    = 5      # seconds between scrapes
ROLLING_WIN      = 20     # window size for anomaly z-score
RETENTION_DAYS   = 7      # delete rows older than this many days
CLEANUP_INTERVAL = 3600   # run cleanup every 1 hour (in seconds)
Z_THRESHOLD   = 2.0        # standard deviations to flag as anomaly

SERVICES = [
    ("payment",      os.getenv("PAYMENT_SERVICE_URL", "http://localhost:8001/metrics")),
    ("order",        os.getenv("ORDER_SERVICE_URL", "http://localhost:8002/metrics")),
    ("notification", os.getenv("NOTIFICATION_SERVICE_URL", "http://localhost:8003/metrics")),
]

METRIC_COLS = ["cpu", "memory", "latency", "requests", "error_rate"]

# ── Rolling windows for anomaly detection (per service per metric) ──────────
# Structure: windows[service][metric] = deque of last ROLLING_WIN values
windows: dict[str, dict[str, deque]] = {
    svc: {col: deque(maxlen=ROLLING_WIN) for col in METRIC_COLS}
    for svc, _ in SERVICES
}

# ── SQLite setup ─────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    NOT NULL,
            service    TEXT    NOT NULL,
            cpu        REAL,
            memory     REAL,
            latency    REAL,
            requests   INTEGER,
            error_rate REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT NOT NULL,
            service    TEXT NOT NULL,
            metric     TEXT NOT NULL,
            value      REAL,
            mean       REAL,
            std        REAL,
            z_score    REAL
        )
    """)

    # Index for fast time-range queries
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_metrics_service_time
        ON metrics (service, timestamp)
    """)

    con.commit()
    con.close()
    print(f"[DB] Initialised SQLite → {DB_FILE}")
    cleanup_old_data()  # clean stale data on every startup


def insert_metrics(rows: list[dict]):
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.executemany("""
        INSERT INTO metrics (timestamp, service, cpu, memory, latency, requests, error_rate)
        VALUES (:timestamp, :service, :cpu, :memory, :latency, :requests, :error_rate)
    """, rows)
    con.commit()
    con.close()


def insert_anomaly(a: dict):
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO anomalies (timestamp, service, metric, value, mean, std, z_score)
        VALUES (:timestamp, :service, :metric, :value, :mean, :std, :z_score)
    """, a)
    con.commit()
    con.close()


# ── Cleanup old data ─────────────────────────────────────────────────────────
def cleanup_old_data():
    """Delete rows older than RETENTION_DAYS. Runs every hour."""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("""
        DELETE FROM metrics
        WHERE timestamp < datetime('now', ? || ' days')
    """, (f"-{RETENTION_DAYS}",))
    metrics_deleted = cur.rowcount

    cur.execute("""
        DELETE FROM anomalies
        WHERE timestamp < datetime('now', ? || ' days')
    """, (f"-{RETENTION_DAYS}",))
    anomalies_deleted = cur.rowcount

    # Reclaim disk space after deletes
    # cur.execute("VACUUM")
    con.commit()

    # Get current DB file size in KB
    db_size_kb = round(os.path.getsize(DB_FILE) / 1024, 1)

    con.close()
    print(f"[CLEANUP] Deleted {metrics_deleted} metric rows + "
          f"{anomalies_deleted} anomaly rows older than {RETENTION_DAYS} days")
    print(f"[CLEANUP] DB size after vacuum: {db_size_kb} KB")
    return metrics_deleted, anomalies_deleted


# ── Cleanup loop (runs in its own thread) ─────────────────────────────────────
def cleanup_loop():
    """Runs cleanup_old_data() once per hour."""
    while True:
        time.sleep(CLEANUP_INTERVAL)
        try:
            cleanup_old_data()
        except Exception as e:
            print(f"[CLEANUP ERROR] {e}")


# ── Anomaly detection ─────────────────────────────────────────────────────────
def check_anomalies(service: str, data: dict, ts: str) -> list[dict]:
    flagged = []
    for metric in METRIC_COLS:
        val = data.get(metric)
        if val is None:
            continue

        win = windows[service][metric]
        win.append(val)

        if len(win) < 5:          # need at least 5 points
            continue

        mean = statistics.mean(win)
        std  = statistics.stdev(win)

        if std == 0:
            continue

        z = abs(val - mean) / std

        if z > Z_THRESHOLD:
            anomaly = {
                "timestamp": ts,
                "service":   service,
                "metric":    metric,
                "value":     round(val, 3),
                "mean":      round(mean, 3),
                "std":       round(std, 3),
                "z_score":   round(z, 3),
            }
            flagged.append(anomaly)
            insert_anomaly(anomaly)
            print(f"[ANOMALY] {service}.{metric} = {val:.2f}  "
                  f"(mean={mean:.2f}, z={z:.2f})")
    return flagged


# ── Collector loop (runs in background thread) ───────────────────────────────
def collector_loop():
    init_db()
    print("[Collector] Started — polling every 5s\n")

    while True:
        rows = []
        ts   = datetime.now().isoformat(timespec="seconds")

        for service_name, url in SERVICES:
            try:
                resp = requests.get(url, timeout=5)
                data = resp.json()

                row = {
                    "timestamp":  ts,
                    "service":    service_name,
                    "cpu":        data["cpu"],
                    "memory":     data["memory"],
                    "latency":    data["latency"],
                    "requests":   data["requests"],
                    "error_rate": data["error_rate"],
                }
                rows.append(row)

                check_anomalies(service_name, data, ts)

                print(f"[{ts}] {service_name:<15} "
                      f"cpu={data['cpu']:.1f}%  "
                      f"mem={data['memory']:.1f}%  "
                      f"lat={data['latency']:.0f}ms  "
                      f"err={data['error_rate']:.1f}%")

            except Exception as e:
                print(f"[ERROR] {service_name}: {e}")

        if rows:
            insert_metrics(rows)

            # Autonomous ML prediction & Email Alert trigger via Prediction API
            pred_url = os.getenv("PREDICTION_API_URL", "http://prediction-api:8004/predict/batch")
            try:
                payload = [
                    {
                        "service": r["service"],
                        "metrics": {
                            "cpu": r["cpu"],
                            "memory": r["memory"],
                            "latency": r["latency"],
                            "requests": r["requests"],
                            "error_rate": r["error_rate"],
                        }
                    }
                    for r in rows
                ]
                requests.post(pred_url, json=payload, timeout=3)
            except Exception as e:
                print(f"[COLLECTOR WARNING] Failed to post metrics to Prediction API ({pred_url}): {e}")

        print("-" * 60)
        time.sleep(POLL_INTERVAL)


# ── FastAPI server (history + anomaly endpoints) ──────────────────────────────
app = FastAPI(
    title="Metrics Collector API",
    description="Historical metrics and anomaly data from SQLite",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "service": "metrics-collector",
        "message": "Metrics Collector API is running",
        "endpoints": {
            "health": "/health",
            "history": "/history?service=payment&limit=30",
            "anomalies": "/anomalies?limit=20",
            "summary": "/summary",
            "db_stats": "/db-stats",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM metrics")
    total = cur.fetchone()[0]
    con.close()
    return {
        "status":       "healthy",
        "db":           DB_FILE,
        "total_rows":   total,
        "poll_interval": POLL_INTERVAL,
    }


@app.get("/history")
def history(
    service: str = Query(default=None, description="Filter by service name"),
    limit:   int = Query(default=100,  ge=1, le=2000, description="Max rows to return"),
):
    """
    Returns historical metrics from SQLite.
    Used by the dashboard to pre-populate charts on page load.

    Examples:
        /history                        → last 100 rows all services
        /history?service=payment        → last 100 rows for payment
        /history?service=order&limit=50 → last 50 rows for order
    """
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    if service:
        cur.execute("""
            SELECT timestamp, service, cpu, memory, latency, requests, error_rate
            FROM metrics
            WHERE service = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (service, limit))
    else:
        cur.execute("""
            SELECT timestamp, service, cpu, memory, latency, requests, error_rate
            FROM metrics
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

    rows = cur.fetchall()
    con.close()

    cols = ["timestamp", "service", "cpu", "memory", "latency", "requests", "error_rate"]
    data = [dict(zip(cols, row)) for row in reversed(rows)]  # oldest first

    return {"service": service, "count": len(data), "data": data}


@app.get("/anomalies")
def anomalies(
    service: str = Query(default=None),
    limit:   int = Query(default=50, ge=1, le=500),
):
    """Returns recent anomaly detections."""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    if service:
        cur.execute("""
            SELECT timestamp, service, metric, value, mean, std, z_score
            FROM anomalies WHERE service = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (service, limit))
    else:
        cur.execute("""
            SELECT timestamp, service, metric, value, mean, std, z_score
            FROM anomalies ORDER BY timestamp DESC LIMIT ?
        """, (limit,))

    rows = cur.fetchall()
    con.close()

    cols  = ["timestamp", "service", "metric", "value", "mean", "std", "z_score"]
    data  = [dict(zip(cols, row)) for row in rows]
    return {"count": len(data), "anomalies": data}


@app.get("/summary")
def summary():
    """Per-service averages over the last 100 data points each."""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    result = {}

    for svc, _ in SERVICES:
        cur.execute("""
            SELECT
                ROUND(AVG(cpu),2)        AS avg_cpu,
                ROUND(AVG(memory),2)     AS avg_memory,
                ROUND(AVG(latency),2)    AS avg_latency,
                ROUND(AVG(error_rate),2) AS avg_error_rate,
                COUNT(*)                 AS sample_count
            FROM (
                SELECT cpu, memory, latency, error_rate
                FROM metrics WHERE service = ?
                ORDER BY timestamp DESC LIMIT 100
            )
        """, (svc,))
        row = cur.fetchone()
        result[svc] = {
            "avg_cpu":        row[0],
            "avg_memory":     row[1],
            "avg_latency":    row[2],
            "avg_error_rate": row[3],
            "sample_count":   row[4],
        }

    con.close()
    return result


@app.get("/db-stats")
def db_stats():
    """DB size, row counts, oldest/newest entry, retention info."""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM metrics")
    m = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM anomalies")
    a = cur.fetchone()
    cur.execute("SELECT service, COUNT(*) FROM metrics GROUP BY service")
    per_service = dict(cur.fetchall())
    con.close()

    db_size_kb = round(os.path.getsize(DB_FILE) / 1024, 1) if os.path.exists(DB_FILE) else 0
    daily_rows = len(SERVICES) * int(86400 / POLL_INTERVAL)

    return {
        "db_file":             DB_FILE,
        "db_size_kb":          db_size_kb,
        "db_size_mb":          round(db_size_kb / 1024, 3),
        "total_metric_rows":   m[0],
        "total_anomaly_rows":  a[0],
        "oldest_entry":        m[1],
        "newest_entry":        m[2],
        "rows_per_service":    per_service,
        "retention_days":      RETENTION_DAYS,
        "estimated_daily_rows": daily_rows,
        "estimated_rows_at_7days": daily_rows * RETENTION_DAYS,
    }


@app.post("/cleanup")
def manual_cleanup():
    """Manually trigger data cleanup right now."""
    deleted_m, deleted_a = cleanup_old_data()
    db_size_kb = round(os.path.getsize(DB_FILE) / 1024, 1)
    return {
        "message":           "Cleanup complete",
        "deleted_metrics":   deleted_m,
        "deleted_anomalies": deleted_a,
        "db_size_kb":        db_size_kb,
        "retention_days":    RETENTION_DAYS,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start collector loop in background thread
    t = threading.Thread(target=collector_loop, daemon=True)
    t.start()

    # Start hourly cleanup thread
    c = threading.Thread(target=cleanup_loop, daemon=True)
    c.start()

    # Start FastAPI on port 8005
    print("[API] Starting collector API on http://localhost:8005")
    uvicorn.run(app, host="0.0.0.0", port=8005)


# ── DB Stats ──────────────────────────────────────────────────────────────────
# (inserted after summary endpoint)
