# sentinelx/grpc/server.py
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Optional

import grpc
import redis.asyncio as redis_async

from sentinelx.core.config import (
    REDIS_URL,
    progress_channel,
    result_key,
    status_key,
    route_queue_name,
)
from sentinelx.core.router import router, BackpressureError
from sentinelx.observability.logging import logger
from sentinelx.grpc import inference_pb2, inference_pb2_grpc


# ============================================================
# Phase-5 helpers (Redis result + streaming)
# ============================================================
async def _publish_status(
    r: redis_async.Redis,
    request_id: str,
    status: str,
    detail: str | None = None,
    **extra,
) -> None:
    evt = {"request_id": request_id, "status": status, "ts": time.time()}
    if detail:
        evt["detail"] = detail
    evt.update(extra)
    payload = json.dumps(evt).encode("utf-8")
    await r.set(status_key(request_id), payload)
    await r.publish(progress_channel(request_id), payload)


async def _enqueue_job(
    r: redis_async.Redis,
    request_id: str,
    logical_model: str,
    version: str,
    route: str,
    input_json: dict,
) -> None:
    qname = route_queue_name(logical_model, version, route)
    job = {
        "request_id": request_id,
        "logical_model": logical_model,
        "version": version,
        "route": route,
        "input_json": input_json,
    }
    await r.rpush(qname, json.dumps(job).encode("utf-8"))


async def _wait_for_result(r: redis_async.Redis, request_id: str, timeout_s: int) -> dict:
    key = result_key(request_id)
    deadline = time.time() + max(1, int(timeout_s))

    while time.time() < deadline:
        raw = await r.get(key)
        if raw:
            try:
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                return json.loads(raw)
            except Exception:
                return {"error": "Malformed result payload"}
        await asyncio.sleep(0.05)

    return {"error": f"Timeout waiting for result (>{timeout_s}s)"}


# ============================================================
# NEW (Phase-5) gRPC Gateway Service (JSON payload + streaming)
# ============================================================
class InferenceGatewayServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, r: redis_async.Redis):
        self.r = r

    async def Predict(
        self,
        request: inference_pb2.PredictRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.PredictResponse:
        request_id = request.request_id or str(uuid.uuid4())
        timeout_s = request.timeout_s if request.timeout_s > 0 else 30

        # Parse input JSON
        try:
            input_dict = json.loads(request.input_json or "{}")
        except Exception:
            return inference_pb2.PredictResponse(
                request_id=request_id,
                status="error",
                error="input_json is not valid JSON",
                total_ms=0.0,
            )

        t0 = time.time()

        await _publish_status(
            self.r,
            request_id,
            "queued",
            logical_model=request.logical_model,
            version=request.version,
            route=request.route,
        )
        await _enqueue_job(
            self.r,
            request_id,
            request.logical_model,
            request.version,
            request.route,
            input_dict,
        )

        result = await _wait_for_result(self.r, request_id, int(timeout_s))
        total_ms = (time.time() - t0) * 1000.0

        if result.get("error"):
            return inference_pb2.PredictResponse(
                request_id=request_id,
                status="error",
                error=str(result["error"]),
                total_ms=total_ms,
            )

        return inference_pb2.PredictResponse(
            request_id=request_id,
            status="finished",
            output_json=json.dumps(result.get("output", {})),
            total_ms=total_ms,
        )

    async def PredictStream(
        self,
        request: inference_pb2.PredictRequest,
        context: grpc.aio.ServicerContext,
    ):
        request_id = request.request_id or str(uuid.uuid4())
        timeout_s = request.timeout_s if request.timeout_s > 0 else 30

        try:
            input_dict = json.loads(request.input_json or "{}")
        except Exception:
            yield inference_pb2.PredictUpdate(
                request_id=request_id,
                status="error",
                error="input_json is not valid JSON",
            )
            return

        # publish queued, enqueue
        await _publish_status(
            self.r,
            request_id,
            "queued",
            logical_model=request.logical_model,
            version=request.version,
            route=request.route,
        )
        await _enqueue_job(
            self.r,
            request_id,
            request.logical_model,
            request.version,
            request.route,
            input_dict,
        )

        pubsub = self.r.pubsub()
        await pubsub.subscribe(progress_channel(request_id))

        # Send latest status immediately if exists
        latest = await self.r.get(status_key(request_id))
        if latest and isinstance(latest, (bytes, bytearray)):
            try:
                evt = json.loads(latest.decode("utf-8"))
                yield inference_pb2.PredictUpdate(
                    request_id=request_id,
                    status=str(evt.get("status", "queued")),
                    detail=str(evt.get("detail", "")),
                )
            except Exception:
                pass
        else:
            yield inference_pb2.PredictUpdate(request_id=request_id, status="queued")

        deadline = time.time() + max(1, int(timeout_s))

        try:
            while time.time() < deadline:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg.get("type") == "message":
                    data = msg["data"]
                    if isinstance(data, (bytes, bytearray)):
                        data = data.decode("utf-8")

                    try:
                        evt = json.loads(data)
                    except Exception:
                        continue

                    st = str(evt.get("status", ""))
                    detail = str(evt.get("detail", ""))

                    if st == "finished":
                        # Grab final result and return it in stream
                        result = await _wait_for_result(self.r, request_id, timeout_s=5)
                        yield inference_pb2.PredictUpdate(
                            request_id=request_id,
                            status="finished",
                            output_json=json.dumps(result.get("output", {})),
                        )
                        return

                    if st == "error":
                        yield inference_pb2.PredictUpdate(
                            request_id=request_id,
                            status="error",
                            error=detail or "unknown error",
                        )
                        return

                    # started / queued / other progress
                    yield inference_pb2.PredictUpdate(
                        request_id=request_id,
                        status=st or "unknown",
                        detail=detail,
                    )

                await asyncio.sleep(0.05)

            yield inference_pb2.PredictUpdate(
                request_id=request_id,
                status="error",
                error=f"Timeout waiting for completion (>{timeout_s}s)",
            )

        finally:
            await pubsub.unsubscribe(progress_channel(request_id))
            await pubsub.close()


# ============================================================
# OLD (Legacy) gRPC Service (router.enqueue_request + float inputs)
# ============================================================
class LegacyInferenceServiceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """
    This implements your OLD proto:
      service InferenceService { rpc Predict(PredictRequest) returns (PredictResponse) }

    It uses the existing router pipeline (same as FastAPI path earlier).
    """

    async def Predict(self, request, context):
        logical_model = request.logical_model

        # optional tiny simulated overhead (kept from old)
        await asyncio.sleep(0.05)

        try:
            enqueue_result = router.enqueue_request(
                logical_model=logical_model,
                inputs=list(request.inputs),
                metadata={"client": "grpc-legacy"},
            )
        except BackpressureError as e:
            # IMPORTANT: do NOT count backpressure here.
            # Router.enqueue_request() already increments REQUESTS_TOTAL{status="backpressure"}.
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(e))
        except ValueError as e:
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))

        job_id = enqueue_result["job_id"]
        physical_model = enqueue_result["physical_model"]
        version = enqueue_result["version"]

        try:
            resp = router.wait_for_response(
                job_id=job_id,
                logical_model=logical_model,
                physical_model=physical_model,
                version=version,
                timeout_seconds=5.0,
            )
        except TimeoutError as e:
            # Router.wait_for_response() already increments REQUESTS_TOTAL{status="timeout"}.
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(e))

        latency_ms = (resp["finished_at"] - resp["started_at"]) * 1000.0

        return inference_pb2.PredictResponse(
            outputs=resp["outputs"],
            logical_model=logical_model,
            physical_model=physical_model,
            version=version,
            latency_ms=latency_ms,
            job_id=job_id,
        )


# ============================================================
# One server that hosts BOTH services
# ============================================================
async def serve_grpc(host: str = "0.0.0.0", port: int = 50051) -> None:
    r = redis_async.from_url(REDIS_URL, decode_responses=False)

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ]
    )

    # NEW gateway service (Phase-5)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceGatewayServicer(r), server
    )

    # OLD legacy service
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        LegacyInferenceServiceServicer(), server
    )

    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    logger.info(f"[gRPC] listening on {host}:{port} (InferenceGateway + InferenceService)")

    try:
        await server.wait_for_termination()
    finally:
        await r.close()
