"""
Simulated microservice — ONE codebase, run as three instances (payment, order,
notification) configured by environment variables:

  SERVICE_NAME    payment | order | notification
  DOWNSTREAM_URL  base URL of the next service in the chain (empty = leaf)
  API_KEY         required in X-API-Key for /simulate_failure and /reset

Call chain for /process:  payment -> order -> notification
X-Request-ID is generated at the edge (or accepted from the caller) and
forwarded downstream, so one id appears in all three services' logs/traces.

/metrics reports:
  cpu, memory   psutil readings. NOTE: inside a container psutil reports the
                host/VM's CPU and memory, shared by every container, so the
                three services see near-identical values. In failure mode a
                synthetic overlay is added because a container cannot force
                its host CPU/memory up on demand.
  latency       mean duration (ms) of recent /process requests — measured
  error_rate    % of recent /process requests that returned 5xx — measured
  requests      /process requests in the last 60 s, per minute — measured
"""

from __future__ import annotations

import asyncio
import hmac
import os
import random
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

import httpx
import psutil
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from common.logging_utils import request_id_var, setup_logging

SERVICE_NAME = os.getenv("SERVICE_NAME", "payment")
SERVICE_VERSION = "3.0.0"
DOWNSTREAM_URL = os.getenv("DOWNSTREAM_URL", "").rstrip("/")
API_KEY = os.getenv("API_KEY", "")

UNTRACKED_PATHS = frozenset({"/metrics", "/health", "/livez", "/traces", "/docs", "/openapi.json"})
METRIC_WINDOW = 20  # /process requests used for latency and error_rate
FAILURE_LATENCY_S = (0.3, 0.8)
FAILURE_ERROR_PROB = 0.25
DEPENDENCY_CACHE_S = 5.0

log = setup_logging(f"{SERVICE_NAME}-service")


class ServiceState:
    def __init__(self):
        self.failure_mode = False
        self.started = time.time()
        self.process_log: deque[dict] = deque(maxlen=200)  # /process only — feeds /metrics
        self.trace_log: deque[dict] = deque(maxlen=50)  # every tracked request — feeds /traces
        self.dependency_cache: tuple[float, dict] = (0.0, {})
        self.http: httpx.AsyncClient | None = None


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    psutil.cpu_percent(interval=None)  # prime: the first non-blocking call always returns 0.0
    state.http = httpx.AsyncClient(timeout=httpx.Timeout(3.0))
    log.info("started", extra={"downstream": DOWNSTREAM_URL or None})
    yield
    await state.http.aclose()


app = FastAPI(title=f"{SERVICE_NAME}-service", version=SERVICE_VERSION, lifespan=lifespan)


@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time"] = f"{duration_ms}ms"
    response.headers["X-Service-Name"] = SERVICE_NAME
    path = request.url.path
    if path not in UNTRACKED_PATHS:
        entry = {
            "ts": time.time(),
            "request_id": rid,
            "path": path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        }
        state.trace_log.append(entry)
        if path == "/process":
            state.process_log.append(entry)
        log.info("request", extra={k: v for k, v in entry.items() if k != "ts"})
    request_id_var.reset(token)
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_KEY:
        raise HTTPException(503, "API_KEY not configured; failure injection disabled")
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(401, "Missing or invalid X-API-Key")


@app.get("/")
def home():
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "downstream": DOWNSTREAM_URL or None,
        "endpoints": ["/process", "/metrics", "/health", "/traces", "/simulate_failure", "/reset"],
    }


@app.get("/livez")
def livez():
    """Shallow liveness (no dependency checks). Used by upstream services' /health."""
    return {"status": "alive", "service": SERVICE_NAME}


async def _check_downstream() -> dict:
    if not DOWNSTREAM_URL:
        return {}
    cached_at, cached = state.dependency_cache
    if time.monotonic() - cached_at < DEPENDENCY_CACHE_S:
        return cached
    name = DOWNSTREAM_URL.split("//")[-1].split(":")[0]
    try:
        r = await state.http.get(f"{DOWNSTREAM_URL}/livez", timeout=1.0)
        result = {name: "ok" if r.status_code == 200 else f"http_{r.status_code}"}
    except httpx.HTTPError as exc:
        result = {name: f"unreachable ({type(exc).__name__})"}
    state.dependency_cache = (time.monotonic(), result)
    return result


@app.get("/health")
async def health():
    deps = await _check_downstream()
    degraded = state.failure_mode or any(v != "ok" for v in deps.values())
    return {
        "status": "degraded" if degraded else "healthy",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "uptime_seconds": round(time.time() - state.started, 1),
        "failure_mode": state.failure_mode,
        "dependencies": deps,
    }


@app.post("/process")
async def process(request: Request):
    """Do some 'work', then call the next service in the chain with the same X-Request-ID."""
    await asyncio.sleep(random.uniform(0.01, 0.05))
    if state.failure_mode:
        await asyncio.sleep(random.uniform(*FAILURE_LATENCY_S))
        if random.random() < FAILURE_ERROR_PROB:
            return JSONResponse(status_code=503, content={"service": SERVICE_NAME, "error": "injected failure"})

    downstream = None
    if DOWNSTREAM_URL:
        try:
            r = await state.http.post(f"{DOWNSTREAM_URL}/process", headers={"X-Request-ID": request_id_var.get()})
            downstream = {"status_code": r.status_code, "body": r.json()}
        except (httpx.HTTPError, ValueError) as exc:
            downstream = {"status_code": None, "error": type(exc).__name__}
        if downstream["status_code"] != 200:
            log.warning("downstream call failed", extra={"downstream": DOWNSTREAM_URL})
            return JSONResponse(
                status_code=502,
                content={"service": SERVICE_NAME, "error": "downstream failed", "downstream": downstream},
            )
    return {
        "service": SERVICE_NAME,
        "request_id": request_id_var.get(),
        "status": "processed",
        "downstream": downstream,
    }


@app.post("/simulate_failure", dependencies=[Depends(require_api_key)])
def simulate_failure():
    state.failure_mode = True
    log.warning("failure mode enabled")
    return {"service": SERVICE_NAME, "failure_mode": True}


@app.post("/reset", dependencies=[Depends(require_api_key)])
def reset():
    state.failure_mode = False
    log.info("failure mode disabled")
    return {"service": SERVICE_NAME, "failure_mode": False}


def compute_metrics(process_log, now: float, cpu: float, memory: float, failure_mode: bool, rng=random) -> dict:
    recent = list(process_log)[-METRIC_WINDOW:]
    latency = sum(r["duration_ms"] for r in recent) / len(recent) if recent else 0.0
    error_rate = 100.0 * sum(r["status"] >= 500 for r in recent) / len(recent) if recent else 0.0
    per_minute = sum(1 for r in process_log if now - r["ts"] <= 60)
    if failure_mode:
        cpu = min(100.0, cpu + rng.uniform(40, 60))
        memory = min(100.0, memory + rng.uniform(20, 30))
    return {
        "cpu": round(cpu, 2),
        "memory": round(memory, 2),
        "latency": round(latency, 2),
        "requests": per_minute,
        "error_rate": round(error_rate, 2),
    }


@app.get("/metrics")
def metrics():
    return compute_metrics(
        state.process_log,
        time.time(),
        psutil.cpu_percent(interval=None),
        psutil.virtual_memory().percent,
        state.failure_mode,
    )


@app.get("/traces")
def traces():
    return {"service": SERVICE_NAME, "recent_requests": list(state.trace_log)[-20:]}
