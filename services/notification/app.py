from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import psutil
import time
import uuid
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
failure_mode = False
start_time   = time.time()
request_log  = []

SERVICE_NAME    = "notification-service"
SERVICE_VERSION = "2.0.0"


# ── Middleware: attach X-Request-ID + measure duration ──────────────────────
@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"]    = request_id
    response.headers["X-Response-Time"] = f"{duration_ms}ms"
    response.headers["X-Service-Name"]  = SERVICE_NAME

    request_log.append({
        "request_id":  request_id,
        "path":        request.url.path,
        "duration_ms": duration_ms,
    })
    if len(request_log) > 50:
        request_log.pop(0)

    return response


@app.get("/")
def home():
    return {
        "service": SERVICE_NAME,
        "message": f"{SERVICE_NAME} is running",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "traces": "/traces",
            "process": "/process",
            "docs": "/docs"
        }
    }


# ── Health (rich) ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    uptime_seconds = round(time.time() - start_time, 1)
    return {
        "status":          "healthy",
        "service":         SERVICE_NAME,
        "version":         SERVICE_VERSION,
        "uptime_seconds":  uptime_seconds,
        "failure_mode":    failure_mode,
        "dependencies":    {
            "email_provider": "ok",
            "queue":          "ok",
        }
    }


# ── Process ──────────────────────────────────────────────────────────────────
@app.get("/process")
def process():
    time.sleep(random.uniform(0.01, 0.05))
    return {"service": SERVICE_NAME, "status": "processed"}


# ── Simulate failure / Reset ─────────────────────────────────────────────────
@app.post("/simulate_failure")
def simulate_failure():
    global failure_mode
    failure_mode = True
    return {"message": "Failure mode enabled"}


@app.post("/reset")
def reset():
    global failure_mode
    failure_mode = False
    return {"message": "Normal mode enabled"}


# ── Metrics (real psutil + failure overlay) ───────────────────────────────────
@app.get("/metrics")
def metrics():
    real_cpu    = psutil.cpu_percent(interval=0.1)
    real_memory = psutil.virtual_memory().percent

    recent = request_log[-10:] if request_log else []
    avg_latency = (
        round(sum(r["duration_ms"] for r in recent) / len(recent), 2)
        if recent else 0.0
    )

    try:
        connections = len(psutil.net_connections())
    except Exception:
        connections = random.randint(50, 200)

    if failure_mode:
        return {
            "cpu":        min(100.0, round(real_cpu + random.uniform(40, 60), 2)),
            "memory":     min(100.0, round(real_memory + random.uniform(20, 30), 2)),
            "latency":    round(avg_latency + random.uniform(900, 2000), 2),
            "requests":   connections * random.randint(20, 50),
            "error_rate": round(random.uniform(10, 20), 2),
        }

    return {
        "cpu":        real_cpu,
        "memory":     real_memory,
        "latency":    max(avg_latency, round(random.uniform(10, 80), 2)),
        "requests":   connections * random.randint(1, 5),
        "error_rate": round(random.uniform(0, 2), 2),
    }


# ── Trace log ────────────────────────────────────────────────────────────────
@app.get("/traces")
def traces():
    return {"service": SERVICE_NAME, "recent_requests": request_log[-20:]}
