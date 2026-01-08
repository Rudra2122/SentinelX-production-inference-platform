# sentinelx/core/router.py
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import redis

from sentinelx.core.config import config, route_queue_name
from sentinelx.observability.logging import logger
from sentinelx.observability.metrics import (
    QUEUE_LENGTH_GAUGE,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
)
from sentinelx.registry.registry import registry

# ============================
# OpenTelemetry (gateway spans + propagation)
# ============================
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


# ============================================================
# Helper: worker liveness gate (TTL-based heartbeat existence)
# ============================================================
def has_healthy_workers(redis_client: redis.Redis, ttl_grace_s: int = 5) -> bool:
    """
    Simple, robust worker-liveness gate.

    We consider the system to have a healthy worker if ANY heartbeat key exists
    and has a TTL > ttl_grace_s.

    Uses Redis TTL only (no JSON parsing).
    """
    try:
        keys = redis_client.keys("sentinelx:worker_heartbeat:*")
        if not keys:
            return False

        for k in keys:
            ttl = redis_client.ttl(k)
            if ttl is None:
                continue
            if ttl > ttl_grace_s:
                return True

        return False
    except Exception as e:
        # Fail-closed: better to refuse than "queue forever"
        logger.warning(f"[Router] has_healthy_workers check failed: {e}")
        return False


class BackpressureError(Exception):
    pass


class NoHealthyVersionError(Exception):
    """
    Raised when there is no healthy route / no healthy workers.
    Your API layer should map this to HTTP 503.
    """


class RequestRouter:
    """
    Handles:
      - Logical → physical routing (primary/canary A/B + deterministic canary pick)
      - Health enforcement (route away from unhealthy versions)
      - Pre-enqueue worker-heartbeat enforcement (refuse if no healthy workers)
      - Backpressure checks
      - Enqueueing + waiting for responses
      - Metrics emission:
          * enqueue path: counts + enqueue latency (route="enqueue")
          * wait path: counts + end-to-end wait latency (route="<physical>:<version>")
    """

    def __init__(self) -> None:
        self.redis = redis.from_url(config.redis_url)
        self._tracer = trace.get_tracer(__name__)
        self._prop = TraceContextTextMapPropagator()

    # ----------------------------
    # Deterministic canary bucketing
    # ----------------------------
    def _stable_canary_pick(self, request_id: str, pct: float) -> bool:
        h = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        return bucket < pct

    # ----------------------------
    # Version health (registry-backed)
    # ----------------------------
    def _is_version_healthy(self, logical_model: str, version: str) -> bool:
        """
        If registry has health methods, use them. Otherwise default to healthy.
        """
        try:
            fn = getattr(registry, "is_healthy", None)
            if callable(fn):
                return bool(fn(logical_model, version))
        except Exception as e:
            logger.warning(f"[Router] registry health check failed for {logical_model}:{version}: {e}")
        return True

    # ----------------------------
    # Routing: primary/canary + registry health enforcement
    # ----------------------------
    def _choose_physical(
        self,
        logical_model: str,
        request_id: str,
        ab_hint: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Returns (physical_model, version) while enforcing registry version health.

        Rules:
          - If canary is unhealthy → route 100% to primary
          - If primary is unhealthy → route 100% to canary (if configured)
          - If both unhealthy → raise NoHealthyVersionError (should become 503)
        """
        traffic = registry.get_traffic(logical_model)
        if not traffic:
            raise ValueError(f"Unknown logical model {logical_model}")

        physical_model = logical_model  # your design: physical == logical

        # normalize hint
        if ab_hint is not None:
            ab_hint = ab_hint.lower()
            if ab_hint not in ("primary", "canary"):
                ab_hint = None

        primary = traffic.primary_version
        canary = traffic.canary_version

        candidates: List[str] = []

        if ab_hint == "primary" or not canary:
            candidates = [primary] + ([canary] if canary else [])
        elif ab_hint == "canary" and canary:
            candidates = [canary, primary]
        else:
            preferred = primary
            other = None
            if canary and traffic.canary_percentage > 0.0:
                if self._stable_canary_pick(request_id, traffic.canary_percentage):
                    preferred, other = canary, primary
                else:
                    preferred, other = primary, canary
            candidates = [preferred] + ([other] if other else [])

        for v in candidates:
            if v is None:
                continue
            if self._is_version_healthy(logical_model, v):
                return physical_model, v
            logger.warning(f"[Router] version unhealthy -> skipping {logical_model}:{v}")

        logger.error(
            f"[Router] no healthy versions for logical_model={logical_model} "
            f"candidates={candidates} request_id={request_id}"
        )
        raise NoHealthyVersionError(f"No healthy versions available for {logical_model}")

    # ----------------------------
    # Enqueue
    # ----------------------------
    def enqueue_request(
        self,
        logical_model: str,
        inputs: Any,
        metadata: Dict[str, Any],
        request_id: str | None = None,
        ab_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enqueue a job onto Redis queue.

        ✅ Adds the "photo style" wrapper:
          - start = time.time()
          - status = ...
          - try/except/finally emits:
              REQUESTS_TOTAL (counts)
              REQUEST_LATENCY_SECONDS (enqueue latency, route="enqueue")
        """
        start = time.time()
        status = "success"

        # Defaults so metrics always have values even if we fail early
        request_id = request_id or str(uuid.uuid4())
        physical_model = "unknown"
        version = "unknown"
        route_label = "enqueue"

        try:
            # Decide final route/version
            physical_model, version = self._choose_physical(
                logical_model=logical_model,
                request_id=request_id,
                ab_hint=ab_hint,
            )

            # Fail fast if no healthy workers
            ttl_grace_s = 5
            if not has_healthy_workers(self.redis, ttl_grace_s=ttl_grace_s):
                status = "no_healthy_workers"
                msg = "No healthy workers available"
                logger.error(
                    f"[Router] {msg} request_id={request_id} logical={logical_model} version={version}"
                )
                raise BackpressureError(msg)

            queue_name = route_queue_name(physical_model, version)

            # queue depth metric
            qlen = int(self.redis.llen(queue_name))
            QUEUE_LENGTH_GAUGE.labels(queue_name=queue_name).set(qlen)

            # backpressure
            if qlen >= config.max_queue_length:
                status = "backpressure"
                logger.warning(
                    f"[Router] backpressure request_id={request_id} "
                    f"route={physical_model}:{version} qlen={qlen} "
                    f"max={config.max_queue_length}"
                )
                raise BackpressureError("Queue overloaded")

            # ---- tracing: inject W3C context into job payload ----
            carrier: Dict[str, str] = {}
            self._prop.inject(carrier)
            traceparent = carrier.get("traceparent")

            span_ctx = trace.get_current_span().get_span_context()
            trace_id_hex = (
                format(span_ctx.trace_id, "032x") if span_ctx and span_ctx.trace_id else None
            )

            job = {
                "job_id": request_id,  # request_id == job_id
                "logical_model": logical_model,
                "physical_model": physical_model,
                "version": version,
                "inputs": inputs,
                "metadata": metadata or {},
                "enqueued_at": time.time(),
                "traceparent": traceparent,
            }

            with self._tracer.start_as_current_span("enqueue") as span:
                span.set_attribute("job.id", request_id)
                span.set_attribute("model.logical", logical_model)
                span.set_attribute("model.physical", physical_model)
                span.set_attribute("model.version", version)
                span.set_attribute("queue.name", queue_name)
                span.set_attribute("queue.qlen_before", qlen)
                if traceparent:
                    span.set_attribute("w3c.traceparent", traceparent)

                self.redis.rpush(queue_name, json.dumps(job))
                QUEUE_LENGTH_GAUGE.labels(queue_name=queue_name).set(qlen + 1)
                span.set_attribute("queue.qlen_after", qlen + 1)

            logger.info(
                f"[Router] enqueued request_id={request_id} logical={logical_model} "
                f"route={physical_model}:{version} queue={queue_name} qlen={qlen + 1} "
                f"ab_hint={ab_hint} trace_id={trace_id_hex}"
            )

            return {
                "job_id": request_id,
                "physical_model": physical_model,
                "version": version,
                "trace_id": trace_id_hex,
                "traceparent": traceparent,
            }

        except ValueError:
            # unknown logical model etc.
            status = "bad_request"
            raise
        except NoHealthyVersionError:
            status = "no_healthy_version"
            raise
        except BackpressureError:
            # status already set to backpressure / no_healthy_workers above
            raise
        except Exception:
            status = "error"
            raise
        finally:
            # ✅ PHOTO STYLE: always emit metrics for the enqueue path
            try:
                REQUESTS_TOTAL.labels(
                    logical_model=logical_model,
                    physical_model=physical_model,
                    version=version,
                    status=status,
                ).inc()

                REQUEST_LATENCY_SECONDS.labels(
                    logical_model=logical_model,
                    physical_model=physical_model,
                    version=version,
                    route=route_label,   # "enqueue"
                    status=status,       # ✅ NEW
                ).observe(time.time() - start)
            except Exception as e:
                logger.warning(f"[Router] metrics emit failed (enqueue): {e}")

    # ----------------------------
    # Wait for response
    # ----------------------------
    def wait_for_response(
        self,
        job_id: str,
        logical_model: str,
        physical_model: str,
        version: str,
        timeout_seconds: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Waits for worker response and emits:
          - REQUEST_LATENCY_SECONDS (end-to-end from start of wait loop)
          - REQUESTS_TOTAL (success/timeout)
        """
        key = f"{config.response_prefix}{job_id}"
        start = time.time()

        with self._tracer.start_as_current_span("gateway_wait_response") as span:
            span.set_attribute("job.id", job_id)
            span.set_attribute("model.logical", logical_model)
            span.set_attribute("model.physical", physical_model)
            span.set_attribute("model.version", version)
            span.set_attribute("redis.response_key", key)
            span.set_attribute("timeout_s", timeout_seconds)

            while True:
                raw = self.redis.get(key)
                if raw is not None:
                    resp = json.loads(raw)

                    resp_physical = resp.get("physical_model", physical_model)
                    resp_version = resp.get("version", version)
                    route = f"{resp_physical}:{resp_version}"

                    if "finished_at" in resp and "started_at" in resp:
                        latency_s = float(resp["finished_at"]) - float(resp["started_at"])
                    else:
                        latency_s = time.time() - start

                    REQUEST_LATENCY_SECONDS.labels(
                        logical_model=logical_model,
                        physical_model=resp_physical,
                        version=resp_version,
                        route=route,
                        status="success",   # ✅ NEW (only success writes latency)
                    ).observe(latency_s)


                    REQUESTS_TOTAL.labels(
                        logical_model=logical_model,
                        physical_model=resp_physical,
                        version=resp_version,
                        status="success",
                    ).inc()

                    span.set_attribute("result.found", True)
                    span.set_attribute("route", route)
                    span.set_attribute("latency_ms", latency_s * 1000.0)

                    logger.info(
                        f"[Router] completed job_id={job_id} logical={logical_model} "
                        f"route={route} latency_ms={latency_s * 1000:.2f}"
                    )
                    return resp

                if time.time() - start > timeout_seconds:
                    REQUESTS_TOTAL.labels(
                        logical_model=logical_model,
                        physical_model=physical_model,
                        version=version,
                        status="timeout",
                    ).inc()

                    span.set_attribute("result.found", False)
                    span.set_attribute("error", True)
                    span.set_attribute("timeout", True)

                    logger.warning(
                        f"[Router] timeout job_id={job_id} logical={logical_model} "
                        f"route={physical_model}:{version} after {timeout_seconds:.2f}s"
                    )
                    raise TimeoutError(f"Timed out waiting for {job_id}")

                time.sleep(0.005)


router = RequestRouter()
