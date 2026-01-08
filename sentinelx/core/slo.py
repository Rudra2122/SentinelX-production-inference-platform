from __future__ import annotations
from sentinelx.core.config import SLOConfig
from sentinelx.observability.metrics import SLO_VIOLATION_COUNTER
from sentinelx.observability.logging import logger


def record_slo_violation(logical_model: str, slo_type: str) -> None:
    logger.warning(f"[SLO] Violation for {logical_model}: {slo_type}")
    SLO_VIOLATION_COUNTER.labels(
        logical_model=logical_model,
        type=slo_type,
    ).inc()


def check_latency_slo(logical_model: str, observed_p95_ms: float, slo: SLOConfig) -> None:
    if observed_p95_ms > slo.target_p95_ms:
        record_slo_violation(logical_model, "latency")
