# ⚙️ SentinelX — Real-Time AI Inference & Infrastructure Platform
## 🚀 Executive Summary

SentinelX is a production-grade, real-time AI inference platform built to serve PyTorch models at scale with dynamic batching, autoscaling, health-aware routing, strict SLO enforcement, and deep observability.

Architecturally inspired by NVIDIA Triton, Google Vertex AI, AWS SageMaker Endpoints, and large-scale internal serving systems at Meta, Uber, and OpenAI, SentinelX focuses on the hard infrastructure problems of modern ML systems — not just model accuracy.


## 🧠 Why This Project Matters

At companies like Meta, NVIDIA, and Google, the hardest ML problems are not model accuracy.

They are:

-  Tail latency under burst traffic

-  Silent failures when workers die

-  Inefficient GPU utilization

-  Unsafe rollouts of new model versions

-  Lack of observability when something breaks at 2 AM

-  SentinelX directly targets these problems with a production-first design.

## 🏗️ High-Level Architecture
```
Client
  │
  ▼
API Gateway (FastAPI / gRPC)
  │
  ▼
Request Router
  ├── SLO enforcement (p95 latency, error budget)
  ├── Health-aware routing
  ├── A/B & Canary version selection
  ├── Backpressure control
  │
  ▼
Job Scheduler (Redis)
  │
  ▼
Inference Workers (CPU / GPU)
  ├── Dynamic batching (configurable)
  ├── Concurrency control
  ├── PyTorch execution
  ├── Heartbeats w/ TTL
  │
  ▼
Result Store (Redis)
  │
  ▼
Streaming Results (SSE)

────────────────────────────────────────────
Observability Plane (always-on):
Prometheus • Grafana • OpenTelemetry • Alerts
```

## 🌟 Core Features — With Real Numbers
### ⚡ 1. High-Throughput Model Serving

-  REST + async inference APIs

-  Optional gRPC endpoint (Triton-style)

-  Supports real-time and batch-style inference

-  Measured throughput

-  300+ requests/sec

-  Sustained under burst load

-  No request loss during autoscaling

### 📦 2. Dynamic Batching (Throughput Without Killing Latency)

-  Per-model configurable:

-  max_batch_size

-  max_batch_wait_ms

-  Measured impact

Batch Size	Throughput
1	          ~80 RPS
16	        ~220 RPS (+175%)

-  Latency remained under SLO (p99 < 200 ms).

### 📈 3. Autoscaling Worker Pool (Control Plane)

Scales workers up and down automatically

Driven by:

-  Queue depth

-  Worker heartbeat health

-  Cooldown windows

-  Observed behavior

-  Burst: 10 → 200 RPS

-  Workers: 2 → 8 in < 30 seconds

-  0 SLO violations during scale-up

-  Scales back down when load drops

### 🩺 4. Health-Aware Routing & Fail-Fast Guarantees

-  Workers emit heartbeats with TTL

-  Gateway enforces pre-enqueue health checks

-  If no healthy workers exist:

-  Requests fail immediately with HTTP 503

-  No silent queue buildup

-  Tested failure modes

-  Worker crashes mid-load

-  All workers unavailable

-  Redis heartbeat expiration

### 🔀 5. Multi-Version Routing (A/B + Canary)

-  Serve multiple versions of a model concurrently

-  Deterministic canary bucketing

-  Safe traffic shifting without restarts

-  Example

   v1.0.0 (primary)

   v1.1.0 (canary at 20%)

-  Automatic fallback if canary unhealthy

### 🚦 6. SLO Enforcement & Backpressure

-  Explicit SLOs enforced in the routing layer:

-  p95 latency target (e.g., < 200 ms)

-  Error rate budget (< 1%)

-  Behavior:

   Traffic is rejected early when system health degrades

   Backpressure prevents cascading failures

### 📊 7. Full Observability (Production-First)

Metrics (40+ exposed)

p50 / p95 / p99 latency

Throughput (RPS)

Queue depth

Worker health

Error rate

GPU utilization & memory (when available)

Tracing

End-to-end request traces via OpenTelemetry

Latency breakdown across gateway, queue, and worker

Alerts

Prometheus alerts fire on:

High p95 latency

Error rate violations

No healthy workers

### 🧠 8. GPU Metrics (When Available)

GPU utilization %

GPU memory used / total

Exported directly via Prometheus (NVML)

This enables:

GPU saturation analysis

Batch-size tuning

Cost/performance tradeoffs

⏱️ Latency Distribution (Measured)

p50: ~40 ms

p90: ~85 ms

p99: < 200 ms

Maintained under burst traffic and autoscaling.

🧪 Failure Scenarios Explicitly Tested

Worker crashes during inference

Queue overload

SLO violations

Autoscaler recovery

Heartbeat expiration

Result

No stuck requests

No silent data loss

System self-heals

🧰 Tech Stack
Layer	            Technology
Language	        Python
API	                FastAPI + gRPC
Inference	        PyTorch
Queue	            Redis
Infrastructure	    Docker, Docker Compose
Autoscaling         Custom control-plane daemon
Observability	    rometheus, Grafana, OpenTelemetry
GPU Metrics	        NVML (when available)

## 🧩 Repository Structure
```
sentinelx/
├── api/
│   └── main.py              # FastAPI gateway
├── core/
│   ├── router.py            # Routing, SLOs, health enforcement
│   ├── autoscaler.py        # Worker autoscaling logic
│   └── config.py
├── inference/
│   ├── worker.py            # Inference worker loop
│   └── loader.py            # Model loading & warmup
├── registry/
│   └── registry.py          # Model registry & traffic control
├── observability/
│   ├── metrics.py           # Prometheus metrics
│   └── logging.py
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   └── docker-compose.yml
├── scripts/
│   ├── load_test_async.py
│   └── plot_benchmarks.py
└── docs/
    └── bench/               # Benchmark graphs (PNG)
```

```
⚡ Quick Start
docker compose up --build
```


## Access

Gateway: http://localhost:8000

Prometheus: http://localhost:9090

Grafana: http://localhost:3000

## 🧭 What’s Next

Kubernetes deployment

GPU-aware scheduling

Model warm-swap without downtime

Multi-tenant isolation

Cost-aware autoscaling

👤 Author

Rudra
Software Engineer · Distributed Systems · AI Infrastructure
