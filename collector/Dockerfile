# Build context: repository root.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY collector/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /data && chown appuser /data

COPY common/__init__.py common/logging_utils.py common/
COPY collector/ collector/

ENV DB_PATH=/data/metrics.db
USER appuser
EXPOSE 8005
# Single process: the collector loop is the only SQLite writer.
CMD ["uvicorn", "collector.collector:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "1"]
