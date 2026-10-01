from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import Settings

ROOT = Path(__file__).resolve().parents[1]
API_KEY = "test-key"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        artifact_root=ROOT / "artifacts",
        api_key=API_KEY,
        cors_origins=["http://localhost:5173"],
        alert_db_path=tmp_path / "alerts.db",
        collector_url="",
        smtp_server="",
        smtp_user="",
        smtp_password="",
        sender_email="",
        default_recipient="",
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, seed_on_startup=False)) as c:
        yield c


@pytest.fixture
def auth() -> dict:
    return {"X-API-Key": API_KEY}


HEALTHY = {"cpu": 35.0, "memory": 50.0, "latency": 150.0, "requests": 2500, "error_rate": 0.8}
STRESSED = {"cpu": 97.0, "memory": 95.0, "latency": 2400.0, "requests": 9500, "error_rate": 28.0}
