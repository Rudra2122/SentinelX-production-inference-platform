# sentinelx/core/config.py
from __future__ import annotations

import os
from typing import Dict, Literal, Optional

from pydantic import BaseModel

# -------------------------------------------------------------------
# Redis URL (shared)
# Prefer SENTINELX_REDIS_URL, then REDIS_URL, then default.
# -------------------------------------------------------------------
REDIS_URL = os.getenv("SENTINELX_REDIS_URL", os.getenv("REDIS_URL", "redis://redis:6379/0"))

# -------------------------------------------------------------------
# Phase-5: Result + progress plumbing
# -------------------------------------------------------------------
RESULT_KEY_PREFIX = "sentinelx:result:"
STATUS_KEY_PREFIX = "sentinelx:status:"
PROGRESS_CH_PREFIX = "sentinelx:progress:"


def result_key(request_id: str) -> str:
    return f"{RESULT_KEY_PREFIX}{request_id}"


def status_key(request_id: str) -> str:
    return f"{STATUS_KEY_PREFIX}{request_id}"


def progress_channel(request_id: str) -> str:
    return f"{PROGRESS_CH_PREFIX}{request_id}"


# -------------------------------------------------------------------
# App-level health keys (strong): Worker heartbeat keys + thresholds
# (Updated to match screenshot recommendations)
# -------------------------------------------------------------------
WORKER_HEARTBEAT_KEY_PREFIX = "sentinelx:worker_heartbeat:"

# ✅ Screenshot-recommended defaults:
#   WORKER_HEARTBEAT_TTL_S = 10
#   WORKER_HEARTBEAT_INTERVAL_S = 3
WORKER_HEARTBEAT_TTL_S = int(os.getenv("SENTINELX_WORKER_HEARTBEAT_TTL_S", "10"))
WORKER_HEARTBEAT_INTERVAL_S = int(os.getenv("SENTINELX_WORKER_HEARTBEAT_INTERVAL_S", "3"))

# Keep stale-after separate (router can use this as an additional guard),
# but set a safer default that aligns with TTL.
WORKER_STALE_AFTER_S = int(os.getenv("SENTINELX_WORKER_STALE_AFTER_S", "15"))


def worker_heartbeat_key(worker_id: str) -> str:
    return f"{WORKER_HEARTBEAT_KEY_PREFIX}{worker_id}"


# -------------------------------------------------------------------
# Existing config models (Phase-1/2/3/4)
# -------------------------------------------------------------------
class SLOConfig(BaseModel):
    target_p95_ms: float = 200.0
    max_error_rate: float = 0.01  # 1%


class ModelConfig(BaseModel):
    name: str
    version: str
    device: Literal["cpu", "cuda"] = "cpu"
    input_dim: int = 8  # used for gateway validation + model build
    max_batch_size: int = 64
    max_batch_wait_ms: int = 5
    slo: SLOConfig = SLOConfig()


class TrafficSplit(BaseModel):
    primary_version: str
    canary_version: Optional[str] = None
    canary_percentage: float = 0.0  # 0–1 (e.g. 0.2 = 20%)


class SentinelXConfig(BaseModel):
    # Core infra / Redis
    redis_url: str = REDIS_URL

    # Version-aware routing (queue per physical_model:version)
    request_queue_prefix: str = "sentinelx:requests:"
    response_prefix: str = "sentinelx:response:"
    health_prefix: str = "sentinelx:worker_health:"

    default_model: str = "demo_classifier"

    # Global backpressure limit
    max_queue_length: int = 1000

    # Physical model configs (multiple versions can share the same name)
    models: Dict[str, ModelConfig] = {
        "demo_classifier": ModelConfig(
            name="demo_classifier",
            version="v1.0.0",
            device="cpu",
            input_dim=8,
            max_batch_size=32,
            max_batch_wait_ms=10,
            slo=SLOConfig(target_p95_ms=200, max_error_rate=0.01),
        ),
        "demo_classifier_v1_1": ModelConfig(
            name="demo_classifier",  # same physical model, newer version
            version="v1.1.0",
            device="cpu",
            input_dim=8,
            max_batch_size=32,
            max_batch_wait_ms=10,
            slo=SLOConfig(target_p95_ms=180, max_error_rate=0.01),
        ),
    }

    # Logical model → traffic policy (A/B / canary)
    traffic: Dict[str, TrafficSplit] = {
        "demo_classifier": TrafficSplit(
            primary_version="v1.0.0",
            canary_version="v1.1.0",
            canary_percentage=0.20,  # 20% canary traffic
        )
    }


config = SentinelXConfig()


def route_queue_name(physical_model: str, version: str) -> str:
    # sentinelx:requests:<physical_model>:<version>
    return f"{config.request_queue_prefix}{physical_model}:{version}"
