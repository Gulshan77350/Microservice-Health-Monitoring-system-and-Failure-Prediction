"""Alert recipients (persisted in SQLite) and SMTP delivery."""

from __future__ import annotations

import html
import logging
import smtplib
import sqlite3
import threading
from contextlib import closing
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from api.config import Settings

log = logging.getLogger("prediction-api.alerts")


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:2]}***@{domain}"


class RecipientStore:
    """Tiny SQLite-backed set of email addresses. Survives restarts via a Docker volume."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as con, con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS recipients (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def list(self) -> list[dict]:
        with closing(self._conn()) as con:
            rows = con.execute("SELECT id, email FROM recipients ORDER BY id").fetchall()
        return [{"id": r[0], "email": r[1]} for r in rows]

    def emails(self) -> list[str]:
        return [r["email"] for r in self.list()]

    def add(self, email: str) -> dict:
        email = email.strip().lower()
        with self._lock, closing(self._conn()) as con, con:
            con.execute("INSERT OR IGNORE INTO recipients (email) VALUES (?)", (email,))
            row = con.execute("SELECT id, email FROM recipients WHERE email = ?", (email,)).fetchone()
        return {"id": row[0], "email": row[1]}

    def remove(self, recipient_id: int) -> bool:
        with self._lock, closing(self._conn()) as con, con:
            return con.execute("DELETE FROM recipients WHERE id = ?", (recipient_id,)).rowcount > 0

    def replace(self, emails: list[str]) -> None:
        with self._lock, closing(self._conn()) as con, con:
            con.execute("DELETE FROM recipients")
            con.executemany(
                "INSERT OR IGNORE INTO recipients (email) VALUES (?)", [(e.strip().lower(),) for e in emails]
            )


class Mailer:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.smtp_configured

    def send(self, recipients: list[str], subject: str, html_body: str) -> bool:
        if not self.enabled:
            log.warning("email skipped: SMTP not configured", extra={"subject": subject})
            return False
        if not recipients:
            log.warning("email skipped: no recipients", extra={"subject": subject})
            return False
        s = self.settings
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = subject, s.sender_email, ", ".join(recipients)
        msg.attach(MIMEText(html_body, "html"))
        try:
            with smtplib.SMTP(s.smtp_server, s.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(s.smtp_user, s.smtp_password)
                server.sendmail(s.sender_email, recipients, msg.as_string())
        except Exception:
            log.exception("email send failed", extra={"subject": subject})
            return False
        log.info("email sent", extra={"subject": subject, "recipient_count": len(recipients)})
        return True


def alert_html(service: str, result: dict, metrics: dict) -> str:
    e = html.escape
    rows = "".join(
        f"<tr><td>{e(k)}</td><td>{e(str(v))}</td></tr>"
        for k, v in [
            ("Failure probability (calibrated)", f"{result['failure_probability'] * 100:.1f}%"),
            ("Risk band", result["risk"]),
            ("Heuristic root cause", result["root_cause"]),
            ("Model version", result["model_version"]),
            *[(k, v) for k, v in metrics.items()],
        ]
    )
    return (
        f"<html><body style='font-family:Arial,sans-serif'>"
        f"<h2>Failure risk alert: {e(service)}</h2>"
        f"<p>The model predicts elevated risk of failure within the next "
        f"{result['horizon_steps']} collection intervals.</p>"
        f"<table border='1' cellpadding='6' style='border-collapse:collapse'>{rows}</table>"
        f"</body></html>"
    )
