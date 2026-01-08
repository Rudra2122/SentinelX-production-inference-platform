# sentinelx/inference/worker.py
from __future__ import annotations

import json
import math
import os
import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import redis
import torch
from prometheus_client import start_http_server

from sentinelx.core.config import (
    REDIS_URL,
    ModelConfig,
    WORKER_HEARTBEAT_INTERVAL_S,
    WORKER_HEARTBEAT_TTL_S,
    config,
    progress_channel,
    result_key,
    route_queue_name,
    status_key,
    worker_heartbeat_key,
)
from sentinelx.inference.loader import load_model
from sentinelx.observability.logging import logger
from sentinelx.observability.metrics import (
    BATCH_SIZE_HIST,
    WORKER_HEALTH_GAUGE,  # timestamp gauge
    WORKER_INFER_LATENCY_SECONDS,  # worker-side infer latency histogram
)
from sentinelx.registry.registry import registry

# Tracing
from opentelemetry import trace
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.status import Status, StatusCode
from sentinelx.telemetry import init_tracing

# ✅ GPU metrics poller (Phase 4)
from sentinelx.worker.gpu_metrics import start_gpu_metrics_poller

# ------------------------------------------------------------
# FORCE autoscaling (DEV only): Artificial worker delay
# Set env: SENTINELX_WORKER_SLOW_MS=50  (50ms per request)
# Default: 0 (no slowdown)
# ------------------------------------------------------------
SLOW_MS = int(os.getenv("SENTINELX_WORKER_SLOW_MS", "0"))

# ✅ Phase-5: TTL for final results in Redis
RESULT_TTL_SECONDS = int(os.getenv("SENTINELX_RESULT_TTL_SECONDS", "300"))

# 🔥 REQUIRED by screenshot/spec: heartbeat TTL must be <= 15s
MAX_WORKER_HEARTBEAT_TTL_S = 15


def connect_redis() -> redis.Redis:
    # Use shared REDIS_URL (prefers SENTINELX_REDIS_URL)
    return redis.from_url(REDIS_URL)


# ============================================================
# ✅ Worker heartbeat (NEW: background TTL heartbeat key)
# ============================================================
def start_heartbeat(redis_client: redis.Redis, worker_id: str, meta: dict) -> None:
    """
    Writes a TTL heartbeat key in Redis so autoscaler/router can detect stale workers.
    """
    # ✅ REQUIRED by your screenshot: log that heartbeat started
    logger.info(f"[HEARTBEAT] worker {worker_id} started")

    # ✅ Enforce TTL <= 15s no matter what config says (screenshots require this)
    ttl_s = int(WORKER_HEARTBEAT_TTL_S)
    if ttl_s <= 0:
        ttl_s = 10
        logger.warning("[HEARTBEAT] WORKER_HEARTBEAT_TTL_S was <= 0; defaulting to 10s")
    if ttl_s > MAX_WORKER_HEARTBEAT_TTL_S:
        logger.warning(
            f"[HEARTBEAT] WORKER_HEARTBEAT_TTL_S={ttl_s}s too long; clamping to {MAX_WORKER_HEARTBEAT_TTL_S}s"
        )
        ttl_s = MAX_WORKER_HEARTBEAT_TTL_S

    interval_s = float(WORKER_HEARTBEAT_INTERVAL_S)
    if interval_s <= 0:
        interval_s = 3.0
        logger.warning("[HEARTBEAT] WORKER_HEARTBEAT_INTERVAL_S was <= 0; defaulting to 3s")
    if interval_s >= ttl_s:
        # ensure we refresh before expiry (safe heuristic: ~ TTL/3)
        interval_s = max(1.0, float(math.floor(ttl_s / 3)))
        logger.warning(f"[HEARTBEAT] Interval >= TTL; adjusting interval to {interval_s}s (TTL={ttl_s}s)")

    def _loop() -> None:
        while True:
            # ✅ REQUIRED payload keys (screenshots)
            payload = {
                "worker_id": worker_id,
                "logical_model": meta.get("logical_model", ""),
                "version": meta.get("version", ""),
                "ts": time.time(),
                **meta,  # keep extra metadata too
            }
            try:
                # ✅ MUST be setex(value + TTL)
                redis_client.setex(
                    worker_heartbeat_key(worker_id),
                    ttl_s,
                    json.dumps(payload),
                )

                # ✅ REQUIRED by your screenshot: debug log inside loop
                logger.info(f"[HEARTBEAT] worker_id={worker_id} ts={time.time()}")

            except Exception as e:
                # Best effort (don’t crash worker)
                logger.warning(f"[Worker] heartbeat write failed worker_id={worker_id}: {e}")

            time.sleep(interval_s)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


# ============================================================
# ✅ Phase-5 helpers (publish status + write results)
# ============================================================
def publish_status(
    r: redis.Redis,
    request_id: str,
    status: str,
    detail: str | None = None,
    **extra: Any,
) -> None:
    """
    Writes latest status to status_key(request_id) AND publishes to progress_channel(request_id).
    This supports SSE + gRPC streaming consumers.
    """
    evt: Dict[str, Any] = {"request_id": request_id, "status": status, "ts": time.time()}
    if detail:
        evt["detail"] = detail
    evt.update(extra)

    payload = json.dumps(evt)
    r.set(status_key(request_id), payload)
    r.publish(progress_channel(request_id), payload)


def write_final_result(
    r: redis.Redis,
    request_id: str,
    payload: Dict[str, Any],
) -> None:
    """
    Final result location expected by Phase-5 gateway/grpc:
      sentinelx:result:{request_id}
    """
    r.set(result_key(request_id), json.dumps(payload), ex=RESULT_TTL_SECONDS)


def fetch_batch(
    r: redis.Redis,
    queue: str,
    model_cfg: ModelConfig,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Collect a batch for a specific *physical model family*.
    """
    batch: List[Tuple[str, Dict[str, Any]]] = []

    res = r.blpop(queue, timeout=1)
    if res is None:
        return batch
    _, raw = res
    first = json.loads(raw)

    # Safety: only accept jobs for this physical model family
    if first.get("physical_model") != model_cfg.name:
        r.lpush(queue, json.dumps(first))
        return batch

    batch.append((first["job_id"], first))
    start = time.time()

    while len(batch) < model_cfg.max_batch_size:
        raw2 = r.lpop(queue)
        if raw2 is None:
            elapsed_ms = (time.time() - start) * 1000.0
            if elapsed_ms >= model_cfg.max_batch_wait_ms:
                break
            time.sleep(0.001)
            continue

        item = json.loads(raw2)

        # Safety: keep only matching physical model jobs
        if item.get("physical_model") != model_cfg.name:
            r.rpush(queue, json.dumps(item))
            elapsed_ms = (time.time() - start) * 1000.0
            if elapsed_ms >= model_cfg.max_batch_wait_ms:
                break
            continue

        batch.append((item["job_id"], item))

        elapsed_ms = (time.time() - start) * 1000.0
        if elapsed_ms >= model_cfg.max_batch_wait_ms:
            break

    return batch


def run_inference(model, device, batch_inputs: List[List[float]]) -> List[List[float]]:
    """
    Runs inference on a batch and returns plain Python lists.
    """
    with torch.no_grad():
        x = torch.tensor(batch_inputs, dtype=torch.float32, device=device)
        logits = model(x)
        if logits.ndim == 2:
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().tolist()
        return logits.cpu().tolist()


def heartbeat_metric_and_legacy_key(r: redis.Redis, worker_id: str, model_cfg: ModelConfig) -> None:
    """
    Keeps your existing per-route health key + Prometheus gauge (Phase-4 style).
    (This is separate from the NEW worker_heartbeat_key TTL key.)
    """
    key = f"{config.health_prefix}{model_cfg.name}:{model_cfg.version}:{worker_id}"
    r.set(key, str(time.time()), ex=30)

    WORKER_HEALTH_GAUGE.labels(
        physical_model=model_cfg.name,
        version=model_cfg.version,
        worker_id=worker_id,
    ).set_to_current_time()


def _extract_parent_ctx(job: Dict[str, Any]):
    """
    Continue the SAME trace from the gateway via W3C trace context header `traceparent`.
    """
    carrier: Dict[str, str] = {}
    tp = job.get("traceparent")
    if tp:
        carrier["traceparent"] = str(tp)
    ts = job.get("tracestate")
    if ts:
        carrier["tracestate"] = str(ts)
    return TraceContextTextMapPropagator().extract(carrier)


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _enqueued_time_ns(job: Dict[str, Any]) -> int:
    """
    Prefer enqueued_at_ns (int) from gateway/router.
    Fallback to enqueued_at (seconds float) if needed.
    """
    enq_ns = _safe_int(job.get("enqueued_at_ns"))
    if enq_ns is not None:
        return enq_ns

    enq_s = job.get("enqueued_at")
    if enq_s is not None:
        try:
            return int(float(enq_s) * 1_000_000_000)
        except Exception:
            pass

    return time.time_ns()


def worker_loop(physical_model_name: str, version: str) -> None:
    # ✅ Initialize tracing in worker process
    init_tracing("sentinelx-worker")

    # ✅ Instrument Redis calls
    RedisInstrumentor().instrument()

    tracer = trace.get_tracer(__name__)

    # ✅ Phase 4: start GPU poller ONCE per worker process
    start_gpu_metrics_poller()

    # ✅ MUST create redis client first, then start heartbeat
    r = connect_redis()

    model_cfg = registry.get_physical(physical_model_name, version)
    if model_cfg is None:
        raise RuntimeError(f"Model {physical_model_name} v={version} not in registry")

    model, device = load_model(model_cfg)

    # ✅ Worker ID per screenshot pattern
    worker_id = os.getenv("SENTINELX_WORKER_ID") or f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

    # ✅ Start background TTL heartbeat key (NEW)
    meta = {
        "hostname": socket.gethostname(),
        "service": os.getenv("SENTINELX_SERVICE_NAME", "worker"),
        "logical_model": os.getenv("SENTINELX_LOGICAL_MODEL", ""),
        "version": os.getenv("SENTINELX_MODEL_VERSION", model_cfg.version),
        "route": os.getenv("SENTINELX_ROUTE", f"{model_cfg.name}:{model_cfg.version}"),
        "physical_model": model_cfg.name,
    }
    start_heartbeat(r, worker_id, meta)

    # Start worker /metrics endpoint (Prometheus scrapes this)
    metrics_port = int(os.getenv("SENTINELX_WORKER_METRICS_PORT", "8001"))
    start_http_server(metrics_port)
    logger.info(f"[Worker] metrics server listening on :{metrics_port}")

    logger.info(f"[Worker] started worker_id={worker_id} route={model_cfg.name}:{model_cfg.version}")

    # per-route queue (physical + version)
    queue_name = route_queue_name(model_cfg.name, model_cfg.version)

    while True:
        try:
            # Keep legacy health + Prom gauge updated
            heartbeat_metric_and_legacy_key(r, worker_id, model_cfg)

            batch = fetch_batch(r, queue_name, model_cfg)
            if not batch:
                continue

            dequeue_time_ns = time.time_ns()

            job_ids = [jid for (jid, _) in batch]
            logger.info(
                f"[Worker] Dequeued batch size={len(batch)} "
                f"route={model_cfg.name}:{model_cfg.version} "
                f"job_ids={job_ids}"
            )

            BATCH_SIZE_HIST.labels(
                physical_model=model_cfg.name,
                version=model_cfg.version,
            ).observe(len(batch))

            # --------- ✅ Phase-5: publish "started" for EACH job ---------
            for (job_id, job) in batch:
                req_id = str(job.get("request_id") or job_id)
                publish_status(
                    r,
                    req_id,
                    "started",
                    physical_model=model_cfg.name,
                    version=model_cfg.version,
                    job_id=str(job_id),
                )

            inputs: List[List[float]] = [job[1]["inputs"] for job in batch]

            # Artificial delay
            if SLOW_MS > 0:
                time.sleep((SLOW_MS / 1000.0) * len(inputs))

            # Trace context from first job
            first_job = batch[0][1]
            parent_ctx = _extract_parent_ctx(first_job)

            # queue_wait span
            enq_ns = _enqueued_time_ns(first_job)
            qw_span = tracer.start_span("queue_wait", context=parent_ctx, start_time=enq_ns)
            try:
                qw_span.set_attribute("job.id", str(first_job.get("job_id")))
                qw_span.set_attribute("queue.key", queue_name)
                qw_span.set_attribute("model.physical", str(first_job.get("physical_model")))
                qw_span.set_attribute("model.version", str(first_job.get("version")))
            except Exception as e:
                qw_span.record_exception(e)
                qw_span.set_status(Status(StatusCode.ERROR, str(e)))
            finally:
                try:
                    qw_span.end(end_time=dequeue_time_ns)
                except TypeError:
                    qw_span.end()

            # worker_infer span
            infer_start_wall = time.time()
            with tracer.start_as_current_span("worker_infer", context=parent_ctx) as infer_span:
                try:
                    infer_span.set_attribute("batch.size", len(batch))
                    infer_span.set_attribute("model.physical", model_cfg.name)
                    infer_span.set_attribute("model.version", model_cfg.version)

                    outputs = run_inference(model, device, inputs)

                except Exception as e:
                    infer_span.record_exception(e)
                    infer_span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

            finished_at = time.time()
            infer_latency_s = finished_at - infer_start_wall

            WORKER_INFER_LATENCY_SECONDS.labels(
                physical_model=model_cfg.name,
                version=model_cfg.version,
            ).observe(infer_latency_s)

            # response_write span + write BOTH (old + new)
            for (job_id, job), out in zip(batch, outputs):
                response_key = f"{config.response_prefix}{job_id}"
                req_id = str(job.get("request_id") or job_id)

                old_payload = {
                    "job_id": job_id,
                    "logical_model": job["logical_model"],
                    "physical_model": model_cfg.name,
                    "version": model_cfg.version,
                    "outputs": out,
                    "started_at": job.get("enqueued_at", time.time()),
                    "finished_at": finished_at,
                }

                new_payload = {
                    "output": {
                        "job_id": str(job_id),
                        "logical_model": str(job.get("logical_model", "")),
                        "physical_model": model_cfg.name,
                        "version": model_cfg.version,
                        "outputs": out,
                    }
                }

                parent_ctx_job = _extract_parent_ctx(job)

                with tracer.start_as_current_span("response_write", context=parent_ctx_job) as rw_span:
                    try:
                        rw_span.set_attribute("job.id", str(job_id))
                        rw_span.set_attribute("redis.key", response_key)

                        # ✅ Old location (kept)
                        r.set(response_key, json.dumps(old_payload), ex=30)

                        # ✅ New Phase-5 location (required)
                        write_final_result(r, req_id, new_payload)

                    except Exception as e:
                        rw_span.record_exception(e)
                        rw_span.set_status(Status(StatusCode.ERROR, str(e)))

                        # ✅ publish error + write error result for Phase-5
                        write_final_result(r, req_id, {"error": str(e)})
                        publish_status(r, req_id, "error", detail=str(e))
                        raise

                # ✅ Phase-5 finished event
                publish_status(
                    r,
                    req_id,
                    "finished",
                    physical_model=model_cfg.name,
                    version=model_cfg.version,
                    job_id=str(job_id),
                )

                latency_ms = (finished_at - float(job.get("enqueued_at", finished_at))) * 1000.0
                logger.info(
                    f"[Worker] completed job_id={job_id} "
                    f"route={model_cfg.name}:{model_cfg.version} "
                    f"batch={len(batch)} latency={latency_ms:.1f}ms "
                    f"infer_latency_ms={infer_latency_s * 1000.0:.2f}"
                )

        except Exception as e:
            # If a batch-level crash happens, still try to mark jobs as error (best-effort)
            try:
                if "batch" in locals():
                    for (job_id, job) in batch:
                        req_id = str(job.get("request_id") or job_id)
                        write_final_result(r, req_id, {"error": str(e)})
                        publish_status(r, req_id, "error", detail=str(e))
            except Exception:
                pass

            logger.exception(f"[Worker] error route={physical_model_name}:{version} err={e}")
            time.sleep(1)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--physical-model", type=str, default="demo_classifier")
    parser.add_argument("--version", type=str, default="v1.0.0")
    args = parser.parse_args()

    worker_loop(args.physical_model, args.version)
