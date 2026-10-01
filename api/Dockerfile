# Build context: repository root (needs common/ and artifacts/).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /data && chown appuser /data

COPY common/ common/
COPY api/ api/
COPY artifacts/ artifacts/

ENV ARTIFACT_ROOT=/app/artifacts ALERT_DB_PATH=/data/alerts.db
USER appuser
EXPOSE 8004

# Single worker on purpose: feature buffers, drift window and alert state are in-process.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]
