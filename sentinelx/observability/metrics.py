# sentinelx/observability/metrics.py
from prometheus_client import Counter, Histogram, Gauge, REGISTRY


def _get_or_create(name: str, factory):
    """
    Avoid duplicate metric registration when the module is imported multiple times
    (e.g., uvicorn reload).
    """
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing
    return factory()


# ------------------------------------------------------------
# Canonical request metrics (GATEWAY ONLY)
# ------------------------------------------------------------
REQUESTS_TOTAL = _get_or_create(
    "sentinelx_requests_total",
    lambda: Counter(
        "sentinelx_requests_total",
        "Total inference requests by logical/physical/version and status",
        ["logical_model", "physical_model", "version", "status"],  # success|timeout|error|backpressure|bad_request
    ),
)

REQUEST_LATENCY_SECONDS = _get_or_create(
    "sentinelx_request_latency_seconds",
    lambda: Histogram(
        "sentinelx_request_latency_seconds",
        "End-to-end request latency in seconds (gateway enqueue -> response)",
        # ✅ Add status label so p95/p99 can be computed on success-only (or filtered by status)
        ["logical_model", "physical_model", "version", "route", "status"],
        buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    ),
)



# ------------------------------------------------------------
# Worker metrics (WORKER ONLY)
# ------------------------------------------------------------
WORKER_INFER_LATENCY_SECONDS = _get_or_create(
    "sentinelx_worker_infer_latency_seconds",
    lambda: Histogram(
        "sentinelx_worker_infer_latency_seconds",
        "Model forward-pass inference latency (seconds), worker-side",
        ["physical_model", "version"],
        buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    ),
)

BATCH_SIZE_HIST = _get_or_create(
    "sentinelx_batch_size",
    lambda: Histogram(
        "sentinelx_batch_size",
        "Batch sizes processed by workers",
        ["physical_model", "version"],
        buckets=[1, 2, 4, 8, 16, 32, 64],
    ),
)

QUEUE_LENGTH_GAUGE = _get_or_create(
    "sentinelx_queue_length",
    lambda: Gauge(
        "sentinelx_queue_length",
        "Length of Redis request queue",
        ["queue_name"],
    ),
)

ACTIVE_WORKERS_GAUGE = _get_or_create(
    "sentinelx_active_workers",
    lambda: Gauge(
        "sentinelx_active_workers",
        "Desired worker count (autoscaler)",
    ),
)

# NEW (canonical): unix timestamp of last heartbeat
WORKER_HEALTH_GAUGE = _get_or_create(
    "sentinelx_worker_health",
    lambda: Gauge(
        "sentinelx_worker_health",
        "Worker last heartbeat time (unix seconds)",
        ["physical_model", "version", "worker_id"],
    ),
)

# OLD (backward-compat): heartbeat age in seconds (keep if dashboards already shipped)
# NOTE: you must set this gauge to (now - last_heartbeat_ts) in code that updates it.
WORKER_HEALTH_SECONDS_GAUGE = _get_or_create(
    "sentinelx_worker_health_seconds",
    lambda: Gauge(
        "sentinelx_worker_health_seconds",
        "Worker last heartbeat age (seconds)",
        ["physical_model", "version", "worker_id"],
    ),
)


# ------------------------------------------------------------
# SLO tracking (typically gateway / control-plane)
# ------------------------------------------------------------
SLO_VIOLATION_COUNTER = _get_or_create(
    "sentinelx_slo_violations_total",
    lambda: Counter(
        "sentinelx_slo_violations_total",
        "SLO violations (p95 latency or error rate)",
        ["logical_model", "type"],  # latency|error_rate
    ),
)


# ------------------------------------------------------------
# GPU metrics (NVML)
# ------------------------------------------------------------
GPU_UTILIZATION_PERCENT = _get_or_create(
    "sentinelx_gpu_utilization_percent",
    lambda: Gauge(
        "sentinelx_gpu_utilization_percent",
        "GPU utilization percent reported by NVML",
        ["gpu"],
    ),
)

GPU_MEM_USED_BYTES = _get_or_create(
    "sentinelx_gpu_mem_used_bytes",
    lambda: Gauge(
        "sentinelx_gpu_mem_used_bytes",
        "GPU memory used in bytes reported by NVML",
        ["gpu"],
    ),
)

GPU_TEMP_CELSIUS = _get_or_create(
    "sentinelx_gpu_temp_celsius",
    lambda: Gauge(
        "sentinelx_gpu_temp_celsius",
        "GPU temperature in celsius reported by NVML",
        ["gpu"],
    ),
)

GPU_PRESENT = _get_or_create(
    "sentinelx_gpu_present",
    lambda: Gauge(
        "sentinelx_gpu_present",
        "1 if NVML is available and at least one GPU detected, else 0",
    ),
)

GPU_NVML_OK = _get_or_create(
    "sentinelx_gpu_nvml_ok",
    lambda: Gauge(
        "sentinelx_gpu_nvml_ok",
        "1 if NVML init succeeded, else 0",
    ),
)


# ------------------------------------------------------------
# Backward-compatible aliases (so older imports keep working)
# ------------------------------------------------------------
REQUEST_COUNTER = REQUESTS_TOTAL
REQUEST_LATENCY = REQUEST_LATENCY_SECONDS
REQUESTS_BY_MODEL = REQUESTS_TOTAL
END_TO_END_LATENCY = REQUEST_LATENCY_SECONDS
SLO_VIOLATIONS = SLO_VIOLATION_COUNTER
BATCH_SIZE_HISTOGRAM = BATCH_SIZE_HIST
