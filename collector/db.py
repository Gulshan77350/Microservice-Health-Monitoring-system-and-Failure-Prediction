"""
SQLite storage for the collector.

Conventions:
  * WAL journal mode (readers never block the single writer thread).
  * Every timestamp is UTC, stored as ISO-8601 'YYYY-MM-DDTHH:MM:SSZ'. All
    comparisons (retention cleanup, ordering) are plain string comparisons
    on that one fixed-width format, computed in Python. The v2 collector
    stored local time with a 'T' separator but compared against SQLite's
    datetime('now'), which is UTC with a space separator — so cleanup cut-offs
    were wrong.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
METRIC_COLS = ["cpu", "memory", "latency", "requests", "error_rate"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL, service TEXT NOT NULL,
    cpu REAL, memory REAL, latency REAL, requests INTEGER, error_rate REAL,
    UNIQUE (service, timestamp)
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL, service TEXT NOT NULL,
    failure_probability REAL NOT NULL, risk TEXT NOT NULL, alert INTEGER NOT NULL,
    root_cause TEXT, model_version TEXT NOT NULL, horizon_steps INTEGER,
    UNIQUE (service, timestamp)
);
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL, service TEXT NOT NULL, metric TEXT NOT NULL,
    value REAL, mean REAL, std REAL, z_score REAL
);
CREATE TABLE IF NOT EXISTS drift_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL, model_version TEXT, status TEXT, overall_status TEXT,
    max_psi REAL, retrain_recommended INTEGER NOT NULL DEFAULT 0, payload TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL, level TEXT NOT NULL, kind TEXT NOT NULL, message TEXT, payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_service_time ON metrics (service, timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_time ON metrics (timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_service_time ON predictions (service, timestamp);
CREATE INDEX IF NOT EXISTS idx_anomalies_time ON anomalies (timestamp);
CREATE INDEX IF NOT EXISTS idx_drift_time ON drift_snapshots (timestamp);
CREATE INDEX IF NOT EXISTS idx_events_time ON events (timestamp);
"""

PRUNABLE_TABLES = ("metrics", "predictions", "anomalies", "drift_snapshots", "events")


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_ts(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime(TS_FORMAT)


def connect(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout = 5000")
    con.execute("PRAGMA synchronous = NORMAL")
    return con


def init_db(path: str | Path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(path)) as con:
        mode = con.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        con.executescript(SCHEMA)
        con.commit()
    return mode


def insert_metrics(path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    with closing(connect(path)) as con, con:
        con.executemany(
            "INSERT OR IGNORE INTO metrics (timestamp, service, cpu, memory, latency, requests, error_rate) "
            "VALUES (:timestamp, :service, :cpu, :memory, :latency, :requests, :error_rate)",
            rows,
        )
    return len(rows)


def insert_predictions(path, timestamp: str, results: dict[str, dict]) -> int:
    rows = [
        {
            "timestamp": timestamp,
            "service": svc,
            "failure_probability": r["failure_probability"],
            "risk": r["risk"],
            "alert": int(bool(r["alert"])),
            "root_cause": r.get("root_cause"),
            "model_version": r["model_version"],
            "horizon_steps": r.get("horizon_steps"),
        }
        for svc, r in results.items()
    ]
    with closing(connect(path)) as con, con:
        con.executemany(
            "INSERT OR REPLACE INTO predictions (timestamp, service, failure_probability, risk, alert, root_cause, "
            "model_version, horizon_steps) VALUES (:timestamp, :service, :failure_probability, :risk, :alert, "
            ":root_cause, :model_version, :horizon_steps)",
            rows,
        )
    return len(rows)


def insert_anomaly(path, a: dict) -> None:
    with closing(connect(path)) as con, con:
        con.execute(
            "INSERT INTO anomalies (timestamp, service, metric, value, mean, std, z_score) "
            "VALUES (:timestamp, :service, :metric, :value, :mean, :std, :z_score)",
            a,
        )


def insert_drift_snapshot(path, timestamp: str, report: dict) -> None:
    with closing(connect(path)) as con, con:
        con.execute(
            "INSERT INTO drift_snapshots (timestamp, model_version, status, overall_status, max_psi, "
            "retrain_recommended, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp,
                report.get("model_version"),
                report.get("status"),
                report.get("overall_status"),
                report.get("max_psi"),
                int(bool(report.get("retrain_recommended"))),
                json.dumps(report.get("per_feature", {})),
            ),
        )


def last_drift_snapshot(path) -> dict | None:
    with closing(connect(path)) as con:
        row = con.execute("SELECT * FROM drift_snapshots ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def insert_event(path, timestamp: str, level: str, kind: str, message: str, payload: dict | None = None) -> None:
    with closing(connect(path)) as con, con:
        con.execute(
            "INSERT INTO events (timestamp, level, kind, message, payload) VALUES (?, ?, ?, ?, ?)",
            (timestamp, level, kind, message, json.dumps(payload or {})),
        )


# ── Queries ─────────────────────────────────────────────────────────────────
_HISTORY_SQL = """
SELECT m.timestamp, m.service, m.cpu, m.memory, m.latency, m.requests, m.error_rate,
       p.failure_probability, p.risk, p.alert, p.root_cause, p.model_version
FROM metrics m
LEFT JOIN predictions p ON p.service = m.service AND p.timestamp = m.timestamp
{where}
ORDER BY m.timestamp DESC, m.service
LIMIT ?
"""


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    if d.get("alert") is not None:
        d["alert"] = bool(d["alert"])
    return d


def history(path, service: str | None, limit: int) -> list[dict]:
    where, params = ("WHERE m.service = ?", [service]) if service else ("", [])
    with closing(connect(path)) as con:
        rows = con.execute(_HISTORY_SQL.format(where=where), [*params, limit]).fetchall()
    return [_row(r) for r in reversed(rows)]  # oldest first


def latest_predictions(path, services: list[str] | None = None) -> dict[str, dict]:
    """Newest prediction per service: one indexed (service, timestamp DESC) lookup each.

    The v3 first draft used GROUP BY MAX(timestamp), a full scan (~53 ms at 7-day retention).
    """
    sql = """
    SELECT p.timestamp, p.service, p.failure_probability, p.risk, p.alert, p.root_cause, p.model_version,
           p.horizon_steps, m.cpu, m.memory, m.latency, m.requests, m.error_rate
    FROM predictions p
    LEFT JOIN metrics m ON m.service = p.service AND m.timestamp = p.timestamp
    WHERE p.service = ?
    ORDER BY p.timestamp DESC
    LIMIT 1
    """
    with closing(connect(path)) as con:
        if services is None:
            services = [r[0] for r in con.execute("SELECT DISTINCT service FROM metrics").fetchall()]
        out = {}
        for svc in services:
            row = con.execute(sql, (svc,)).fetchone()
            if row:
                out[svc] = _row(row)
        return out


def anomalies(path, service: str | None, limit: int) -> list[dict]:
    where, params = ("WHERE service = ?", [service]) if service else ("", [])
    sql = (
        f"SELECT timestamp, service, metric, value, mean, std, z_score FROM anomalies {where} ORDER BY id DESC LIMIT ?"
    )
    with closing(connect(path)) as con:
        return [dict(r) for r in con.execute(sql, [*params, limit]).fetchall()]


def drift_snapshots(path, limit: int) -> list[dict]:
    with closing(connect(path)) as con:
        rows = con.execute(
            "SELECT timestamp, model_version, status, overall_status, max_psi, retrain_recommended "
            "FROM drift_snapshots ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{**dict(r), "retrain_recommended": bool(r["retrain_recommended"])} for r in rows]


def events(path, limit: int) -> list[dict]:
    with closing(connect(path)) as con:
        rows = con.execute(
            "SELECT timestamp, level, kind, message, payload FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{**dict(r), "payload": json.loads(r["payload"] or "{}")} for r in rows]


def summary(path, services: list[str], window: int = 100) -> dict:
    out = {}
    with closing(connect(path)) as con:
        for svc in services:
            r = con.execute(
                "SELECT ROUND(AVG(cpu),2), ROUND(AVG(memory),2), ROUND(AVG(latency),2), ROUND(AVG(error_rate),2), "
                "COUNT(*) FROM (SELECT * FROM metrics WHERE service = ? ORDER BY timestamp DESC LIMIT ?)",
                (svc, window),
            ).fetchone()
            out[svc] = dict(
                zip(["avg_cpu", "avg_memory", "avg_latency", "avg_error_rate", "sample_count"], r, strict=True)
            )
    return out


def stats(path) -> dict:
    with closing(connect(path)) as con:
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in PRUNABLE_TABLES}
        oldest, newest = con.execute("SELECT MIN(timestamp), MAX(timestamp) FROM metrics").fetchone()
        per_service = dict(con.execute("SELECT service, COUNT(*) FROM metrics GROUP BY service").fetchall())
        journal = con.execute("PRAGMA journal_mode").fetchone()[0]
    size = sum(os.path.getsize(p) for p in (str(path), f"{path}-wal") if os.path.exists(p))
    return {
        "row_counts": counts,
        "oldest_metric": oldest,
        "newest_metric": newest,
        "metrics_per_service": per_service,
        "journal_mode": journal,
        "size_kb": round(size / 1024, 1),
    }


# ── Maintenance ─────────────────────────────────────────────────────────────
def cleanup(path, retention_days: int, now: datetime | None = None) -> dict[str, int]:
    cutoff = to_ts((now or utc_now()) - timedelta(days=retention_days))
    deleted = {}
    with closing(connect(path)) as con, con:
        for table in PRUNABLE_TABLES:
            deleted[table] = con.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,)).rowcount
    with closing(connect(path)) as con:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return deleted


def vacuum(path) -> None:
    with closing(connect(path)) as con:
        con.execute("VACUUM")
