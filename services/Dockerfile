# Build context: repository root. One image, run as payment/order/notification via SERVICE_NAME.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY services/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY common/__init__.py common/logging_utils.py common/
COPY services/ services/

USER appuser
EXPOSE 8000
CMD ["uvicorn", "services.app:app", "--host", "0.0.0.0", "--port", "8000"]
