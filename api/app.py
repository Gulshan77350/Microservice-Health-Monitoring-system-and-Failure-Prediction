from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, EmailStr
import pandas as pd
import joblib
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

import drift

# ── SMTP / SendGrid Configuration ───────────────────────────────────────────
SMTP_SERVER   = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SENDER_EMAIL  = os.getenv("SENDER_EMAIL", "noreply@example.com")
DEFAULT_RECIPIENT = os.getenv("DEFAULT_RECIPIENT", "alerts@example.com")

ALERT_RECIPIENTS: list[str] = [DEFAULT_RECIPIENT]
PREVIOUS_RISK_STATES: dict[str, str] = {}  # Deduplication state per service

# ── Helper: Send Email via SMTP ──────────────────────────────────────────────
def send_email_notification(recipients: list[str], subject: str, html_body: str):
    if not SMTP_USER or not SMTP_PASSWORD:
        print(f"[SMTP WARNING] Email notification skipped — SMTP_USER or SMTP_PASSWORD not set. Subject: '{subject}'")
        return False

    if not recipients:
        print("[SMTP WARNING] No recipients specified for email notification.")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = ", ".join(recipients)
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipients, msg.as_string())

        print(f"[SMTP SUCCESS] Alert email sent to {recipients} | Subject: '{subject}'")
        return True
    except Exception as e:
        print(f"[SMTP ERROR] Failed to send email: {e}")
        return False


def build_alert_html(service: str, risk: str, root_cause: str, probability: float, m: dict) -> str:
    risk_color = "#ef4444" if risk == "HIGH" else "#f59e0b"
    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #0f172a; color: #e2e8f0; padding: 20px;">
      <div style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 8px; padding: 24px; border: 1px solid #334155;">
        <h2 style="color: {risk_color}; margin-top: 0;">⚠️ Microservice Failure Alert — {service.upper()}</h2>
        <p>Service <strong>{service}</strong> has reached <span style="color: {risk_color}; font-weight: bold;">{risk} RISK</span>.</p>
        <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
          <tr style="background-color: #334155;"><th style="padding: 8px; text-align: left;">Metric</th><th style="padding: 8px; text-align: left;">Value</th></tr>
          <tr><td style="padding: 8px; border-bottom: 1px solid #334155;">Failure Probability</td><td style="padding: 8px; border-bottom: 1px solid #334155;"><strong>{(probability*100):.1f}%</strong></td></tr>
          <tr><td style="padding: 8px; border-bottom: 1px solid #334155;">Root Cause</td><td style="padding: 8px; border-bottom: 1px solid #334155;">{root_cause}</td></tr>
          <tr><td style="padding: 8px; border-bottom: 1px solid #334155;">CPU Usage</td><td style="padding: 8px; border-bottom: 1px solid #334155;">{m.get('cpu')}%</td></tr>
          <tr><td style="padding: 8px; border-bottom: 1px solid #334155;">Memory Usage</td><td style="padding: 8px; border-bottom: 1px solid #334155;">{m.get('memory')}%</td></tr>
          <tr><td style="padding: 8px; border-bottom: 1px solid #334155;">Latency</td><td style="padding: 8px; border-bottom: 1px solid #334155;">{m.get('latency')} ms</td></tr>
          <tr><td style="padding: 8px; border-bottom: 1px solid #334155;">Error Rate</td><td style="padding: 8px; border-bottom: 1px solid #334155;">{m.get('error_rate')}%</td></tr>
        </table>
        <p style="font-size: 12px; color: #94a3b8;">Automated alert from Microservice Failure Prediction System v3.0</p>
      </div>
    </body>
    </html>
    """

# ── Load model ───────────────────────────────────────────────────────────────
model = joblib.load("failure_predictor.pkl")  # swapped to v2 model — see README

FEATURE_NAMES = ["cpu", "memory", "latency", "requests", "error_rate"]
MODEL_LOADED_AT = datetime.now().isoformat()

# Cost-optimal decision threshold (replaces hardcoded 0.5/0.8 from v1).
# Selected via train_v2.py to minimize expected cost assuming a missed
# failure (false negative) costs 10x an unnecessary alert (false positive).
# Falls back to this literal if model_meta_v2.json isn't present.
DEFAULT_THRESHOLD = 0.1736
HIGH_RISK_MULTIPLIER = 2.0  # HIGH risk band = 2x the base decision threshold


def _load_threshold() -> float:
    meta_path = "model_meta_v2.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return float(meta.get("chosen_threshold", DEFAULT_THRESHOLD))
    return DEFAULT_THRESHOLD


DECISION_THRESHOLD = _load_threshold()

app = FastAPI(
    title="Microservice Failure Prediction API",
    description="Predicts microservice failures using XGBoost. Includes batch prediction, feature importance, model metadata, and live drift monitoring.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """
    Load the training data distribution as the drift reference baseline.
    If the v2 dataset isn't present, drift endpoints report not_initialized
    instead of crashing the API.
    """
    try:
        drift.load_reference("final_dataset_v2.csv")
        print("[startup] Drift reference distribution loaded from final_dataset_v2.csv")
    except FileNotFoundError:
        print("[startup] WARNING: final_dataset_v2.csv not found — /drift will report not_initialized")


# ── Schemas ──────────────────────────────────────────────────────────────────
class Metrics(BaseModel):
    cpu:        float = Field(..., ge=0, le=100,   description="CPU usage percent")
    memory:     float = Field(..., ge=0, le=100,   description="Memory usage percent")
    latency:    float = Field(..., ge=0,            description="Response latency in ms")
    requests:   int   = Field(..., ge=0,            description="Number of active requests")
    error_rate: float = Field(..., ge=0, le=100,   description="Error rate percent")

    @field_validator("cpu", "memory", "error_rate")
    @classmethod
    def must_be_percentage(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("Must be between 0 and 100")
        return v


class ServiceMetrics(BaseModel):
    service: str
    metrics: Metrics


class EmailRecipientRequest(BaseModel):
    email: str


class RecipientListRequest(BaseModel):
    emails: list[str]


import numpy as np

RECENT_METRICS_BUFFER: dict[str, list[dict]] = {}

def build_feature_dataframe(metrics_dict: dict, service_name: str = "default") -> pd.DataFrame:
    buffer = RECENT_METRICS_BUFFER.setdefault(service_name, [])
    buffer.append(metrics_dict)
    if len(buffer) > 10:
        buffer.pop(0)
    
    buf_df = pd.DataFrame(buffer)
    row = metrics_dict.copy()
    for col in ["cpu", "memory", "latency", "requests", "error_rate"]:
        vals = buf_df[col].tolist()
        row[f"{col}_roll5_mean"] = float(np.mean(vals[-5:]))
        row[f"{col}_roll5_std"] = float(np.std(vals[-5:])) if len(vals) > 1 else 0.0
        row[f"{col}_diff1"] = float(vals[-1] - vals[-2]) if len(vals) > 1 else 0.0

    row["cpu_x_mem"] = float(row["cpu"] * row["memory"])
    row["lat_x_err"] = float(row["latency"] * row["error_rate"])
    row["load_index"] = float((row["requests"] * row["latency"]) / 1000.0)
    
    df = pd.DataFrame([row])
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        df = df.reindex(columns=cols, fill_value=0.0)
    return df


# ── Shared prediction logic ───────────────────────────────────────────────────
def run_prediction(m: Metrics, service_name: str = "unknown", background_tasks: BackgroundTasks = None) -> dict:
    metrics_dict = {
        "cpu":        m.cpu,
        "memory":     m.memory,
        "latency":    m.latency,
        "requests":   m.requests,
        "error_rate": m.error_rate,
    }
    data = build_feature_dataframe(metrics_dict, service_name)

    probability = float(model.predict_proba(data)[0][1])
    prediction = int(probability >= DECISION_THRESHOLD)

    high_cutoff = min(DECISION_THRESHOLD * 1.25, 0.85)
    if probability >= high_cutoff:
        risk = "HIGH"
    elif probability >= DECISION_THRESHOLD:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    root_cause = "Normal Operation"
    if m.memory > 90:
        root_cause = "Memory Saturation"
    elif m.cpu > 90:
        root_cause = "CPU Overload"
    elif m.error_rate > 10:
        root_cause = "Error Spike"
    elif m.latency > 1000:
        root_cause = "Latency Surge"

    # Feed live drift monitor
    drift.record_live_sample({
        "cpu": m.cpu, "memory": m.memory, "latency": m.latency,
        "requests": m.requests, "error_rate": m.error_rate,
    })

    # Trigger background email alert on risk escalation (LOW -> MEDIUM/HIGH or MEDIUM -> HIGH)
    prev_risk = PREVIOUS_RISK_STATES.get(service_name, "LOW")
    if risk in ["MEDIUM", "HIGH"] and prev_risk != risk:
        if background_tasks and ALERT_RECIPIENTS:
            subject = f"[{risk} ALERT] Service '{service_name}' Failure Risk Warning"
            html_body = build_alert_html(service_name, risk, root_cause, probability, m.model_dump())
            background_tasks.add_task(send_email_notification, list(ALERT_RECIPIENTS), subject, html_body)
    PREVIOUS_RISK_STATES[service_name] = risk

    return {
        "prediction":          prediction,
        "failure_probability": round(probability, 4),
        "risk":                risk,
        "root_cause":          root_cause,
        "decision_threshold":  round(DECISION_THRESHOLD, 4),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Microservice Failure Prediction API v3.0 is running"}


@app.get("/health")
def health():
    return {
        "status":            "healthy",
        "model_loaded":      True,
        "version":           "3.0.0",
        "loaded_at":         MODEL_LOADED_AT,
        "decision_threshold": DECISION_THRESHOLD,
        "smtp_configured":   bool(SMTP_USER and SMTP_PASSWORD),
        "alert_recipients":  len(ALERT_RECIPIENTS),
    }


# Single prediction
@app.post("/predict")
def predict(metrics: Metrics, background_tasks: BackgroundTasks):
    return run_prediction(metrics, service_name="unknown", background_tasks=background_tasks)


# Batch prediction — all services in 1 HTTP call
@app.post("/predict/batch")
def predict_batch(services: list[ServiceMetrics], background_tasks: BackgroundTasks):
    """
    Accept metrics for multiple services in one request.
    Reduces 3 round-trips to 1 — used by the dashboard.
    """
    if len(services) > 10:
        raise HTTPException(status_code=400, detail="Max 10 services per batch")

    results = {}
    for item in services:
        results[item.service] = run_prediction(item.metrics, service_name=item.service, background_tasks=background_tasks)

    return {"results": results, "count": len(results)}


# ── Alert Configuration Endpoints ─────────────────────────────────────────────
@app.get("/alert-config")
def get_alert_config():
    return {
        "smtp_server":     SMTP_SERVER,
        "smtp_port":       SMTP_PORT,
        "smtp_user":       SMTP_USER or "Not configured",
        "smtp_configured": bool(SMTP_USER and SMTP_PASSWORD),
        "recipients":      ALERT_RECIPIENTS,
    }


@app.post("/alert-config/add")
def add_recipient(req: EmailRecipientRequest):
    email = req.email.strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email cannot be empty")
    if email not in ALERT_RECIPIENTS:
        ALERT_RECIPIENTS.append(email)
    return {"message": f"Added {email} to alert recipients", "recipients": ALERT_RECIPIENTS}


@app.post("/alert-config/remove")
def remove_recipient(req: EmailRecipientRequest):
    email = req.email.strip()
    if email in ALERT_RECIPIENTS:
        ALERT_RECIPIENTS.remove(email)
    return {"message": f"Removed {email} from alert recipients", "recipients": ALERT_RECIPIENTS}


@app.post("/alert-config/set")
def set_recipients(req: RecipientListRequest):
    global ALERT_RECIPIENTS
    ALERT_RECIPIENTS = [e.strip() for e in req.emails if e.strip()]
    return {"message": "Updated alert recipient list", "recipients": ALERT_RECIPIENTS}


@app.post("/alert/test")
def send_test_alert(background_tasks: BackgroundTasks):
    if not ALERT_RECIPIENTS:
        raise HTTPException(status_code=400, detail="No alert recipients configured")

    subject = "[TEST ALERT] Microservice Observability System Test Email"
    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #0f172a; color: #e2e8f0; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 8px; padding: 24px; border: 1px solid #334155;">
          <h2 style="color: #38bdf8; margin-top: 0;">✅ Test Email Delivery Success</h2>
          <p>This is a test notification from the <strong>Microservice Failure Prediction API</strong>.</p>
          <p>SMTP host <code>{SMTP_SERVER}:{SMTP_PORT}</code> is correctly wired!</p>
          <p style="font-size: 12px; color: #94a3b8;">Sent at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
      </body>
    </html>
    """

    background_tasks.add_task(send_email_notification, list(ALERT_RECIPIENTS), subject, html_body)
    return {
        "message": "Test email triggered in background task",
        "recipients": ALERT_RECIPIENTS,
        "smtp_server": SMTP_SERVER,
        "smtp_configured": bool(SMTP_USER and SMTP_PASSWORD),
    }


# Feature importance — which metric drives predictions most
@app.get("/feature-importance")
def feature_importance():
    """
    Returns XGBoost feature importances.
    Tells you which metric (cpu/memory/latency/etc) matters most for failure prediction.
    """
    feature_list = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else FEATURE_NAMES
    importances = model.feature_importances_
    total_imp = float(np.sum(importances)) or 1.0
    ranked = sorted(
        [
            {
                "feature":    name,
                "importance": round(float(imp), 6),
                "importance_pct": round(float(imp) / total_imp * 100, 2),
            }
            for name, imp in zip(feature_list, importances)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )
    return {
        "feature_importance": ranked,
        "most_important":     ranked[0]["feature"],
        "note": "Higher value = stronger driver of failure prediction.",
    }


# Model metadata — algorithm, training info, evaluation metrics
@app.get("/model-info")
def model_info():
    if os.path.exists("model_meta_v2.json"):
        with open("model_meta_v2.json") as f:
            meta = json.load(f)

        comparison = meta.get("model_comparison", {})
        xgb_tuned = comparison.get("xgboost_tuned", {})
        xgb_max_f1 = comparison.get("xgboost_max_f1_threshold", {})
        f1_metrics = xgb_max_f1.get("metrics", {}) or xgb_tuned.get("metrics", {})

        flattened = {
            "algorithm": "XGBoostClassifier (tuned with feature engineering & optimal threshold)",
            "accuracy": f1_metrics.get("accuracy"),
            "f1_score": f1_metrics.get("f1"),
            "precision": f1_metrics.get("precision"),
            "recall": f1_metrics.get("recall"),
            "cv_f1_mean": xgb_tuned.get("cv_f1"),
            "train_samples": meta.get("train_rows"),
            "test_samples": meta.get("test_rows"),
        }

        return {**flattened, **meta}

    if os.path.exists("model_meta.json"):
        with open("model_meta.json") as f:
            return json.load(f)

    return {
        "algorithm":        "XGBoostClassifier",
        "note":             "model_meta_v2.json not found — run model/train_v2.py to generate it",
        "features":         FEATURE_NAMES,
        "trained_at":       "unknown",
    }


# Drift monitoring — compares live request distributions to training baseline
@app.get("/drift")
def drift_report():
    """
    Population Stability Index (PSI) drift report.

    Compares the rolling window of recent live /predict inputs against the
    training data distribution. PSI < 0.1 = stable, 0.1-0.25 = moderate
    shift worth investigating, > 0.25 = significant shift, model likely
    needs retraining.

    This closes the loop on the project's premise: it's not enough to
    predict failure once at training time, you also need to know when the
    model itself has stopped applying to current conditions.
    """
    return drift.compute_drift_report()
