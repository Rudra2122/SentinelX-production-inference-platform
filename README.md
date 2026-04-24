# SentinelX: Production ML Inference Backend

## Why Serving ML Models Is Harder Than Training Them

Training a model is a solved workflow. You write a loop, wait, get a checkpoint. Serving that model to real traffic is where the hard problems actually live.

When a worker crashes mid-inference, do pending requests fail immediately or queue forever? When you ship a new model version, do you need to take the system down? When traffic spikes 10x in 30 seconds, does the system scale or fall over? When something breaks at 2am, do you have a path to follow or are you reading raw logs?

These are the questions SentinelX was built to answer. Not a demo wrapper around a PyTorch model. A serving system I stress-tested under real failure conditions and had to make actually work.

---

## Key Results

| Metric | Value | Methodology |
|---|---|---|
| Throughput | 300+ RPS sustained | 200 concurrent requests, 20 worker threads, `scripts/load_test.py` |
| p50 latency | ~40ms | Measured under burst load, Prometheus histogram |
| p99 latency | under 200ms | Held during autoscaling events |
| Scale-up time | under 30 seconds | 2 to 8 workers, triggered by queue depth |
| Cascading failures | 0 | Verified under fault injection across all failure modes |

---

## Architecture

```
Client
  │
  ▼
API Gateway (FastAPI + gRPC)
  │  W3C trace context injected here, propagated to worker
  ▼
Request Router
  ├── Worker liveness gate (TTL heartbeat check, fail-closed)
  ├── Deterministic canary bucketing (SHA256 hash of request_id)
  ├── Health-aware version routing (primary / canary fallback)
  ├── Backpressure enforcement (queue depth threshold)
  │
  ▼
Redis Job Queue (per physical model + version)
  │
  ▼
Inference Workers
  ├── Dynamic batching (max_batch_size + max_batch_wait_ms)
  ├── TTL-based heartbeat (clamped to 15s, refreshed every ~3s)
  ├── W3C trace context extracted from job payload
  ├── Per-job status publishing (Redis pub/sub, SSE + gRPC streaming)
  │
  ▼
Result Store (Redis, TTL-scoped)
  │
  ▼
Streaming Results (SSE)

Observability (always on):
Prometheus metrics, Grafana dashboards, OpenTelemetry distributed tracing
```

---

## Engineering Decisions Worth Explaining

### 1. Fail-closed worker liveness gate

The router checks for healthy workers before enqueuing anything. If no heartbeat key exists with a TTL above the grace threshold, the request gets an immediate HTTP 503. It does not queue.

This was a deliberate choice over the alternative: let requests queue and timeout. Silent queue buildup is worse than a fast failure. The client knows immediately, the queue stays clean, and the on-call runbook is simple.

```python
def has_healthy_workers(redis_client, ttl_grace_s=5):
    keys = redis_client.keys("sentinelx:worker_heartbeat:*")
    for k in keys:
        if redis_client.ttl(k) > ttl_grace_s:
            return True
    return False
```

### 2. Deterministic canary bucketing via SHA256

Canary routing needed to be sticky, not random. If a request goes to the canary, every subsequent request from that session should go to the canary too. Random sampling breaks that guarantee.

The solution: SHA256 hash of the request ID, take the first 8 hex digits, normalize to [0,1], compare to the canary percentage. Same request ID always routes the same way. No session state required.

```python
def _stable_canary_pick(self, request_id, pct):
    h = hashlib.sha256(request_id.encode()).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return bucket < pct
```

### 3. TTL-clamped heartbeats

Worker heartbeat TTL is hard-clamped to 15 seconds regardless of config. The refresh interval is automatically set to TTL/3 if the configured interval would let the key expire before the next refresh.

This prevents a class of subtle bugs where misconfigured TTLs cause false-positive health failures. The system enforces its own invariants rather than trusting operator configuration.

### 4. Metrics emission in finally blocks

Every request path emits Prometheus metrics in a finally block, not inline. This guarantees metrics are emitted even when exceptions happen mid-request. The status label (success, timeout, backpressure, no\_healthy\_workers, error) makes it possible to compute p95 on success-only paths in Grafana without filtering noise from failures.

### 5. Dual result writing for backward compatibility

Workers write results to two Redis locations simultaneously: the old response key (used by the polling gateway) and the new Phase-5 result key (used by SSE and gRPC streaming consumers). This kept the existing gateway working while the streaming layer was being built, without a flag day migration.

### 6. Dynamic batching with time-bounded collection

The batching loop collects requests up to max\_batch\_size but exits early if max\_batch\_wait\_ms elapses. This bounds the worst-case latency for the first request in a batch: it never waits longer than the window regardless of how full the batch is.

---

## What I Built and Why

### Gateway and routing layer

FastAPI gateway with an optional gRPC endpoint. The router handles version selection, health enforcement, backpressure, and job enqueuing. All routing decisions happen before the job hits the queue, not after.

OpenTelemetry spans are started at the gateway boundary and the W3C traceparent header is injected into the job payload so workers can continue the same trace without a shared context object.

### Worker pool

Workers run as separate processes, each owning a specific physical model and version. They pull from a per-route Redis queue, collect batches up to the configured size and wait window, run PyTorch inference, and write results back to Redis.

Each worker emits a TTL heartbeat key every few seconds. If a worker crashes, the key expires and the router stops sending traffic within one heartbeat window. No manual intervention required.

### Autoscaler

A control-plane daemon watches queue depth and worker heartbeat count. It scales workers up when queue depth exceeds a threshold and scales back down after a cooldown window. Scale-up from 2 to 8 workers happens in under 30 seconds under burst load.

Version health is tracked separately in Redis. If a version has zero healthy workers, the router automatically falls back to the primary version. The canary is marked unhealthy without operator intervention.

### Observability stack

Prometheus metrics across both gateway and worker processes. Grafana dashboards showing p50/p95/p99 latency, queue depth, worker health, batch size distribution, and error rates. OpenTelemetry distributed traces with latency breakdown across gateway enqueue, queue wait, worker inference, and response write spans.

Alerts fire on: high p95 latency, error rate above budget, and no healthy workers. Each alert has a corresponding runbook.

---

## Failure Modes Explicitly Tested

| Scenario | Expected behavior | Verified |
|---|---|---|
| Worker crashes mid-inference | Requests on that worker fail, others continue | Yes |
| All workers unavailable | Immediate HTTP 503, no queue buildup | Yes |
| Heartbeat TTL expires | Router stops routing to that worker within one TTL window | Yes |
| Queue overload | Backpressure kicks in, requests rejected early | Yes |
| Canary version unhealthy | 100% traffic routed to primary automatically | Yes |
| SLO violation (p95 exceeded) | Violation counter incremented, alert fires | Yes |

---

## Repository Structure

```
sentinelx/
├── sentinelx/
│   ├── api/
│   │   └── main.py                  # FastAPI gateway, request lifecycle
│   ├── core/
│   │   ├── config.py                # Config, Redis key naming conventions
│   │   ├── health.py                # Worker heartbeat queries
│   │   ├── router.py                # Routing, health enforcement, backpressure
│   │   ├── scaler.py                # Autoscaler, version health, Docker Compose scaling
│   │   ├── scheduler.py             # Job scheduling logic
│   │   └── slo.py                   # SLO violation tracking
│   ├── grpc/
│   │   ├── dual_server.py           # REST + gRPC dual server
│   │   ├── inference.proto          # gRPC service definition
│   │   ├── inference_pb2.py         # Generated protobuf bindings
│   │   ├── inference_pb2_grpc.py    # Generated gRPC bindings
│   │   └── server.py                # gRPC server entrypoint
│   ├── inference/
│   │   ├── loader.py                # Model loading and warmup
│   │   └── worker.py                # Worker loop, batching, tracing, result writing
│   ├── observability/
│   │   ├── logging.py               # Structured logging
│   │   └── metrics.py               # Prometheus metrics, idempotent registration
│   ├── registry/
│   │   └── registry.py              # Model registry, traffic config, version management
│   ├── worker/
│   │   └── gpu_metrics.py           # NVML GPU metrics poller
│   └── telemetry.py                 # OpenTelemetry initialization
├── docker/
│   ├── Dockerfile
│   ├── prometheus.yml
│   └── prometheus/
│       ├── alerts.yml
│       └── prometheus.yml
├── docs/
│   ├── architecture.md
│   ├── decisions.md
│   └── metrics.md                   # Metric definitions and Grafana query reference
├── scripts/
│   ├── load_test.py                 # Load test: 200 requests, 20 concurrent workers
│   ├── run_gateway.sh
│   └── run_worker.sh
├── docker-compose.yml
└── requirements.txt
```

---

## Reproduce

```bash
# Start the full stack: gateway, workers, Redis, Prometheus, Grafana
docker compose up --build

# Run the gateway individually
bash scripts/run_gateway.sh

# Run a worker individually
bash scripts/run_worker.sh

# Run load test: 200 requests, 20 concurrent workers
python scripts/load_test.py

# Access
# Gateway:    http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Gateway | FastAPI, gRPC |
| Inference | PyTorch |
| Queue | Redis |
| Autoscaling | Custom control-plane daemon |
| Infrastructure | Docker, Docker Compose |
| Observability | Prometheus, Grafana, OpenTelemetry |
| GPU Metrics | NVML |

---

## Limitations

- Autoscaling uses Docker Compose service scaling, not Kubernetes. Horizontal scaling is bounded by single-host resources.
- Dynamic batching collects requests on a time window, not a token-budget window. For LLM serving, token-aware batching (as in vLLM) would be more efficient.
- The canary bucketing is request-ID-stable but not user-stable. A user making requests with different IDs may hit different versions.
- GPU scheduling is not aware of KV cache pressure. Under high concurrency, memory contention is not explicitly managed.

---

## What I Would Do Next

- Replace Docker Compose autoscaling with a Kubernetes operator and HPA
- Add token-budget-aware batching for LLM workloads, similar to vLLM's continuous batching
- Add KV cache memory tracking per active session to prevent eviction under high concurrency
- Add DCGM exporter for production-grade GPU observability across multi-GPU nodes

---

## Author

**Rudra Brahmbhatt**
MS Computer Science, Texas State University, May 2026
ML Inference Infrastructure
