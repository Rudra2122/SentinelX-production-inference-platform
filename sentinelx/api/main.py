# sentinelx/api/main.py
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Async Redis (for SSE + Phase-5 async flow)
import redis.asyncio as redis_async

# ✅ OTel tracing + FastAPI instrumentation
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from sentinelx.telemetry import init_tracing

from sentinelx.core.config import (
    REDIS_URL,
    config,
    progress_channel,
    result_key,
    status_key,
)
from sentinelx.core.router import BackpressureError, NoHealthyVersionError, router
from sentinelx.core.scheduler import autoscaler
from sentinelx.observability.logging import logger
from sentinelx.observability.metrics import REQUESTS_TOTAL, REQUEST_LATENCY_SECONDS
from sentinelx.registry.registry import registry


app = FastAPI(
    title="SentinelX — Real-Time AI Inference Platform",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ============================================================
# Models (OLD sync REST)
# ============================================================
class PredictRequest(BaseModel):
    inputs: List[float]
    metadata: Dict[str, Any] | None = None
    # ✅ (optional) allow clients to pass route hint like screenshot expects: req.route
    route: Optional[str] = None


class PredictResponse(BaseModel):
    outputs: List[float]
    logical_model: str
    physical_model: str
    version: str
    latency_ms: float
    job_id: str


class TrafficConfigRequest(BaseModel):
    primary_version: str
    canary_version: str | None = None
    canary_percentage: float = 0.0


# ============================================================
# Models (NEW Phase-5 async + SSE)
# ============================================================
class AsyncPredictOut(BaseModel):
    request_id: str
    status: str  # queued


# ============================================================
# Lifecycle
# ============================================================
@app.on_event("startup")
async def on_startup():
    # ✅ Initialize tracing on startup
    init_tracing("sentinelx-gateway")

    # ✅ Instrument FastAPI automatically (root request span)
    FastAPIInstrumentor.instrument_app(app)

    # ✅ Instrument Redis calls (optional but nice)
    RedisInstrumentor().instrument()

    # ✅ Async Redis client for Phase-5 SSE/pubsub/result polling
    app.state.redis_async = redis_async.from_url(REDIS_URL, decode_responses=False)

    # ✅ Start autoscaler (optional)
    autoscaler.start()

    logger.info("[API] startup complete")


@app.on_event("shutdown")
async def on_shutdown():
    r = getattr(app.state, "redis_async", None)
    if r is not None:
        await r.close()


# ============================================================
# Helpers (Phase-5)
# ============================================================
async def _publish_status(
    r: redis_async.Redis,
    request_id: str,
    status: str,
    detail: str | None = None,
    **extra: Any,
) -> None:
    evt: Dict[str, Any] = {"request_id": request_id, "status": status, "ts": time.time()}
    if detail:
        evt["detail"] = detail
    evt.update(extra)

    payload = json.dumps(evt).encode("utf-8")
    await r.set(status_key(request_id), payload)
    await r.publish(progress_channel(request_id), payload)


def _route_queue_name(physical_model: str, version: str) -> str:
    """
    Your existing system uses:
      sentinelx:requests:<physical_model>:<version>
    via config.request_queue_prefix and route_queue_name() in config.py.

    We keep it consistent with the rest of your codebase.
    """
    return f"{config.request_queue_prefix}{physical_model}:{version}"


async def _enqueue_job(
    r: redis_async.Redis,
    request_id: str,
    logical_model: str,
    physical_model: str,
    version: str,
    ab_hint: Optional[str],
    inputs: List[float],
    metadata: Dict[str, Any],
) -> None:
    qname = _route_queue_name(physical_model, version)
    job = {
        "job_id": request_id,  # keep your existing naming convention
        "request_id": request_id,
        "logical_model": logical_model,
        "physical_model": physical_model,
        "version": version,
        "inputs": inputs,
        "metadata": metadata,
        "ab_hint": ab_hint,
        "enqueued_ts": time.time(),
    }
    await r.rpush(qname, json.dumps(job).encode("utf-8"))


async def _wait_for_result(r: redis_async.Redis, request_id: str, timeout_s: int) -> dict:
    """
    Worker writes final result to: sentinelx:result:{request_id}
    """
    key = result_key(request_id)
    deadline = time.time() + max(1, timeout_s)

    while time.time() < deadline:
        raw = await r.get(key)
        if raw:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            try:
                return json.loads(raw)
            except Exception:
                return {"error": "Malformed result payload"}
        await asyncio.sleep(0.05)

    return {"error": f"Timeout waiting for result (>{timeout_s}s)"}


# ============================================================
# Basic endpoints
# ============================================================
@app.get("/")
def root():
    return {
        "service": "sentinelx",
        "default_model": config.default_model,
        "message": "SentinelX gateway is running",
    }


@app.get("/health")
async def health():
    """
    Health endpoint used by Docker healthcheck.
    Also verifies Redis connectivity if redis_async is initialized.
    """
    r = getattr(app.state, "redis_async", None)
    if r is not None:
        await r.ping()
    return {"ok": True, "ts": time.time()}


@app.get("/api/v1/models")
def list_models():
    models_by_logical = registry.list_models()
    traffic = {
        name: traffic_cfg.dict()
        for name, traffic_cfg in registry._traffic.items()  # type: ignore[attr-defined]
    }
    return {"models": models_by_logical, "traffic": traffic}


@app.post("/api/v1/models/{logical_name}/activate")
def activate_model_version(logical_name: str, version: str):
    try:
        registry.activate_version(logical_name, version)
        return {"status": "ok", "logical_model": logical_name, "version": version}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/models/{logical_name}/traffic")
def configure_canary(logical_name: str, req: TrafficConfigRequest):
    try:
        registry.configure_canary(
            logical_name,
            primary_version=req.primary_version,
            canary_version=req.canary_version or req.primary_version,
            canary_percentage=req.canary_percentage,
        )
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# Inference (OLD: synchronous wait via router) + ✅ gateway metrics
# ============================================================
@app.post("/api/v1/predict/{logical_model}", response_model=PredictResponse)
def predict(
    logical_model: str,
    req: PredictRequest,
    ab: str | None = Query(
        default=None,
        description="A/B override: 'primary' or 'canary'. If unset, use configured split.",
    ),
    x_sentinelx_ab: str | None = Header(default=None, alias="X-SentinelX-Ab"),
    x_request_id: str | None = Header(default=None),
):
    """
    Synchronous inference endpoint (legacy).
    Uses router.enqueue_request + router.wait_for_response.
    """
    start = time.time()
    status = "success"

    # Defaults for metrics (overwritten on successful routing)
    physical_model = "unknown"
    version = "unknown"
    route_label = "http"

    try:
        # Validate logical model + route existence
        traffic = registry.get_traffic(logical_model)
        if traffic is None:
            status = "bad_request"
            raise HTTPException(status_code=404, detail="Unknown model")

        model_cfg = registry.get_physical(logical_model, traffic.primary_version)
        if model_cfg is None:
            status = "error"
            raise HTTPException(status_code=500, detail="Model config missing for primary route")

        expected_dim = model_cfg.input_dim
        if len(req.inputs) != expected_dim:
            status = "bad_request"
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input length: expected {expected_dim}, got {len(req.inputs)}",
            )

        ab_hint = ab or x_sentinelx_ab
        if ab_hint is not None:
            ab_hint = ab_hint.lower()
            if ab_hint not in ("primary", "canary"):
                logger.warning(f"[API] ignoring invalid A/B hint '{ab_hint}'")
                ab_hint = None

        # Enqueue via router
        try:
            enqueue_result = router.enqueue_request(
                logical_model=logical_model,
                inputs=req.inputs,
                metadata=req.metadata or {},
                request_id=x_request_id,
                ab_hint=ab_hint,
            )
        except BackpressureError as e:
            status = "backpressure"
            raise HTTPException(status_code=429, detail=str(e))
        except NoHealthyVersionError as e:
            status = "error"
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            status = "bad_request"
            raise HTTPException(status_code=404, detail=str(e))

        job_id = enqueue_result["job_id"]
        physical_model = enqueue_result["physical_model"]
        version = enqueue_result["version"]
        route = f"{physical_model}:{version}"

        logger.info(
            f"[API] received request_id={job_id} logical={logical_model} route={route} "
            f"input_len={len(req.inputs)} ab_hint={ab_hint}"
        )

        try:
            resp = router.wait_for_response(
                job_id=job_id,
                logical_model=logical_model,
                physical_model=physical_model,
                version=version,
                timeout_seconds=5.0,
            )
        except TimeoutError as e:
            status = "timeout"
            raise HTTPException(status_code=504, detail=str(e))

        latency_ms = (time.time() - start) * 1000.0
        logger.info(
            f"[API] completed request_id={job_id} logical={logical_model} "
            f"route={route} latency_ms={latency_ms:.2f}"
        )

        return PredictResponse(
            outputs=resp["outputs"],
            logical_model=resp["logical_model"],
            physical_model=resp["physical_model"],
            version=resp["version"],
            latency_ms=latency_ms,
            job_id=resp["job_id"],
        )

    finally:
        # ✅ Canonical gateway metrics (always exactly once per request)
        REQUESTS_TOTAL.labels(
            logical_model=logical_model,
            physical_model=physical_model,
            version=version,
            status=status,
        ).inc()

        # ✅ latency with status label (as in screenshot)
        REQUEST_LATENCY_SECONDS.labels(
            logical_model=logical_model,
            physical_model=physical_model,
            version=version,
            route=route_label,
            status=status,  # ✅ REQUIRED now
        ).observe(time.time() - start)


# ============================================================
# Inference (NEW: async enqueue + SSE progress) + ✅ gateway metrics
# ============================================================
@app.post("/api/v1/predict/{logical_model}/async", response_model=AsyncPredictOut)
async def predict_async(
    logical_model: str,
    req: PredictRequest,
    ab: str | None = Query(default=None),
    x_sentinelx_ab: str | None = Header(default=None, alias="X-SentinelX-Ab"),
    x_request_id: str | None = Header(default=None),
):
    """
    Phase-5: enqueue and return request_id immediately.
    Client listens on /api/v1/predict/{request_id}/events for progress updates.
    """
    start = time.time()
    status = "success"
    request_id = x_request_id or str(uuid.uuid4())

    # Defaults for metrics (overwritten if router picks a route)
    physical_model = "unknown"
    version = "unknown"
    route_label = "http_async"

    try:
        # Use router to enforce health + backpressure
        try:
            enqueue_result = router.enqueue_request(
                logical_model=logical_model,
                inputs=req.inputs,
                metadata=req.metadata or {},
                request_id=request_id,
                ab_hint=req.route or ab or x_sentinelx_ab,
            )
        except BackpressureError as e:
            status = "backpressure"
            raise HTTPException(status_code=503, detail=str(e))
        except NoHealthyVersionError as e:
            status = "error"
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            status = "bad_request"
            raise HTTPException(status_code=404, detail=str(e))

        physical_model = enqueue_result["physical_model"]
        version = enqueue_result["version"]
        job_id = enqueue_result["job_id"]

        # Publish queued (SSE consumers can start immediately)
        r: redis_async.Redis = app.state.redis_async
        await _publish_status(
            r,
            job_id,
            "queued",
            logical_model=logical_model,
            physical_model=physical_model,
            version=version,
            ab_hint=(req.route or ab or x_sentinelx_ab),
        )

        return AsyncPredictOut(request_id=job_id, status="queued")

    finally:
        # ✅ Canonical gateway metrics (always exactly once per request)
        REQUESTS_TOTAL.labels(
            logical_model=logical_model,
            physical_model=physical_model,
            version=version,
            status=status,
        ).inc()

        # ✅ latency with status label (as in screenshot)
        REQUEST_LATENCY_SECONDS.labels(
            logical_model=logical_model,
            physical_model=physical_model,
            version=version,
            route=route_label,
            status=status,  # ✅ REQUIRED now
        ).observe(time.time() - start)


@app.get("/api/v1/predict/{request_id}/events")
async def predict_events(request_id: str):
    """
    SSE stream:
      - immediately emits latest status if present
      - then subscribes to pubsub channel sentinelx:progress:{request_id}
      - ends when status is finished/error
    """
    r: redis_async.Redis = app.state.redis_async

    async def gen() -> AsyncGenerator[dict, None]:
        # 1) Immediately send latest status (if any)
        latest = await r.get(status_key(request_id))
        if latest:
            if isinstance(latest, (bytes, bytearray)):
                latest = latest.decode("utf-8")
            yield {"event": "status", "data": latest}

        # 2) Subscribe to pubsub updates
        pubsub = r.pubsub()
        await pubsub.subscribe(progress_channel(request_id))

        try:
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg.get("type") == "message":
                    data = msg["data"]
                    if isinstance(data, (bytes, bytearray)):
                        data = data.decode("utf-8")

                    yield {"event": "status", "data": data}

                    # stop on finished/error
                    try:
                        evt = json.loads(data)
                        if evt.get("status") in ("finished", "error"):
                            break
                    except Exception:
                        pass

                await asyncio.sleep(0.05)
        finally:
            await pubsub.unsubscribe(progress_channel(request_id))
            await pubsub.close()

    async def event_stream():
        async for msg in gen():
            # SSE wire format: "event: x\ndata: y\n\n"
            event = msg.get("event", "message")
            data = msg.get("data", "")
            yield f"event: {event}\n".encode()
            yield f"data: {data}\n\n".encode()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================
# Metrics
# ============================================================
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)
