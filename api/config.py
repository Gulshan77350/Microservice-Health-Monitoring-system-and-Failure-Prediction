"""Environment-driven settings for the prediction API. No secrets or real addresses as defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _csv(name: str, default: str) -> list[str]:
    return [v.strip() for v in os.getenv(name, default).split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    artifact_root: Path = field(default_factory=lambda: Path(os.getenv("ARTIFACT_ROOT", str(ROOT / "artifacts"))))
    model_version: str | None = field(default_factory=lambda: os.getenv("MODEL_VERSION") or None)
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    cors_origins: list[str] = field(default_factory=lambda: _csv("CORS_ORIGINS", "http://localhost:5173"))
    allowed_services: list[str] = field(default_factory=lambda: _csv("ALLOWED_SERVICES", "payment,order,notification"))
    max_batch: int = field(default_factory=lambda: int(os.getenv("MAX_BATCH", "10")))
    collector_url: str = field(default_factory=lambda: os.getenv("COLLECTOR_URL", ""))
    alert_db_path: Path = field(
        default_factory=lambda: Path(os.getenv("ALERT_DB_PATH", str(ROOT / "api" / "alerts.db")))
    )
    alert_cooldown_s: int = field(default_factory=lambda: int(os.getenv("ALERT_COOLDOWN_SECONDS", "600")))
    smtp_server: str = field(default_factory=lambda: os.getenv("SMTP_SERVER", ""))
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_user: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    sender_email: str = field(default_factory=lambda: os.getenv("SENDER_EMAIL", ""))
    default_recipient: str = field(default_factory=lambda: os.getenv("DEFAULT_RECIPIENT", ""))

    @property
    def smtp_configured(self) -> bool:
        return all([self.smtp_server, self.smtp_user, self.smtp_password, self.sender_email])
