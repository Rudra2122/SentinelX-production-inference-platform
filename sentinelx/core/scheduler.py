# sentinelx/core/scheduler.py
from __future__ import annotations

import math
import os
import time
import threading
import subprocess

import redis

from sentinelx.core.config import config, route_queue_name
from sentinelx.observability.logging import logger
from sentinelx.observability.metrics import ACTIVE_WORKERS_GAUGE, QUEUE_LENGTH_GAUGE


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


class AutoScaler:
    """
    REAL autoscaling for docker-compose (scales one worker service).

    IMPORTANT FIX (mismatch):
    - The Redis queue key MUST match what the gateway/router enqueues into.
    - Your gateway/router uses route_queue_name(physical_model, version).
      So we use the SAME helper here to guarantee an exact match.

    Scaling action:
      docker compose up -d --scale <service>=<N> --no-recreate

    NOTE:
    - This must run on the HOST, OR inside a container with:
        * docker CLI installed
        * /var/run/docker.sock mounted
    """

    def __init__(
        self,
        service: str = "worker_v1",
        physical_model: str = "demo_classifier",
        version: str = "v1.0.0",
        min_replicas: int = 2,
        max_replicas: int = 6,
        upscale_at: int = 50,
        downscale_at: int = 5,
        poll_sec: float = 1.0,
        cooldown_sec: int = 10,
    ) -> None:
        self.service = service
        self.physical_model = physical_model
        self.version = version

        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.upscale_at = max(1, upscale_at)
        self.downscale_at = max(0, downscale_at)
        self.poll_sec = poll_sec
        self.cooldown_sec = cooldown_sec

        self.redis = redis.from_url(config.redis_url)

        # compose execution settings
        self.compose_file = os.getenv("SENTINELX_COMPOSE_FILE", "docker-compose.yml")
        self.project_dir = os.getenv("SENTINELX_PROJECT_DIR", os.getcwd())

        # track replicas we last set (compose doesn't give us a clean "get replicas" API)
        self.current_replicas = self.min_replicas
        ACTIVE_WORKERS_GAUGE.set(self.current_replicas)

        self._thread: threading.Thread | None = None
        self._stop = False
        self._last_scale_ts = 0.0

    # ✅ EXACT queue key match with gateway/router
    def _queue_key(self) -> str:
        return route_queue_name(self.physical_model, self.version)

    def _queue_len(self) -> int:
        q = self._queue_key()
        qlen = int(self.redis.llen(q))
        QUEUE_LENGTH_GAUGE.labels(queue_name=q).set(qlen)
        return qlen

    def _scale_compose(self, replicas: int) -> None:
        replicas = max(self.min_replicas, min(self.max_replicas, replicas))

        cmd = [
            "docker",
            "compose",
            "-p",
            os.getenv("COMPOSE_PROJECT_NAME", "sentinelx"),
            "-f",
            self.compose_file,  # MUST be like "/app/docker-compose.yml"
            "up",
            "-d",
            "--scale",
            f"{self.service}={replicas}",
            "--no-recreate",
        ]

        logger.info(
            f"[Autoscaler] scale {self.service} -> {replicas} | "
            f"queue={self._queue_key()} | "
            f"compose_file={self.compose_file} project_dir={self.project_dir} | "
            f"cmd={' '.join(cmd)}"
        )

        subprocess.run(cmd, cwd=self.project_dir, check=False)

    def _desired_from_queue(self, qlen: int) -> int:
        """
        Faster scaling: jump to a replica count that can drain bursts.
        Heuristic: treat upscale_at as "queue items per worker we tolerate".
        """
        # If queue is above upscale threshold, scale to ceil(qlen / upscale_at)
        if qlen > self.upscale_at:
            return int(math.ceil(qlen / float(self.upscale_at)))

        # If queue is tiny, allow downscale by 1 (bounded by min_replicas)
        if qlen < self.downscale_at:
            return self.current_replicas - 1

        # Otherwise keep current
        return self.current_replicas

    def _loop(self) -> None:
        logger.info(
            "[Autoscaler] Started (REAL compose scaling) | "
            f"service={self.service} queue={self._queue_key()} | "
            f"up>{self.upscale_at} down<{self.downscale_at} | "
            f"min={self.min_replicas} max={self.max_replicas} | "
            f"poll={self.poll_sec}s cooldown={self.cooldown_sec}s"
        )

        while not self._stop:
            try:
                now = time.time()
                if now - self._last_scale_ts < self.cooldown_sec:
                    time.sleep(self.poll_sec)
                    continue

                qlen = self._queue_len()
                desired = self._desired_from_queue(qlen)

                # Bound
                desired = max(self.min_replicas, min(self.max_replicas, desired))

                # Extra safety: never change by more than needed, but allow jump-up on bursts
                if desired > self.current_replicas:
                    # allow jump-up (burst protection)
                    pass
                elif desired < self.current_replicas:
                    # be conservative on downscale (already -1 only)
                    desired = self.current_replicas - 1
                    desired = max(self.min_replicas, desired)

                if desired != self.current_replicas:
                    self._scale_compose(desired)
                    self.current_replicas = desired
                    self._last_scale_ts = time.time()

                ACTIVE_WORKERS_GAUGE.set(self.current_replicas)
                time.sleep(self.poll_sec)

            except Exception as e:
                logger.exception(f"[Autoscaler] Error: {e}")
                time.sleep(2.0)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True


# -----------------------------
# Env wiring (supports BOTH naming styles)
# -----------------------------

# ✅ SLO-friendly defaults (scale earlier)
UPSCALE_AT = _env_int("SENTINELX_UPSCALE_AT", _env_int("SENTINELX_SCALE_UP_THRESHOLD", 50))
DOWNSCALE_AT = _env_int("SENTINELX_DOWNSCALE_AT", _env_int("SENTINELX_SCALE_DOWN_THRESHOLD", 5))
MIN_REPLICAS = _env_int("SENTINELX_MIN_REPLICAS", _env_int("SENTINELX_MIN_WORKERS", 2))
MAX_REPLICAS = _env_int("SENTINELX_MAX_REPLICAS", _env_int("SENTINELX_MAX_WORKERS", 6))
COOLDOWN_SEC = _env_int("SENTINELX_COOLDOWN_SEC", _env_int("SENTINELX_SCALE_COOLDOWN_S", 10))
POLL_SEC = _env_float("SENTINELX_POLL_SEC", _env_float("SENTINELX_SCALE_POLL_S", 1.0))

autoscaler = AutoScaler(
    service=os.getenv("SENTINELX_WORKER_SERVICE", os.getenv("SENTINELX_SCALE_SERVICE", "worker_v1")),
    physical_model=os.getenv("SENTINELX_SCALE_PHYSICAL_MODEL", "demo_classifier"),
    version=os.getenv("SENTINELX_SCALE_VERSION", "v1.0.0"),
    min_replicas=MIN_REPLICAS,
    max_replicas=MAX_REPLICAS,
    upscale_at=UPSCALE_AT,
    downscale_at=DOWNSCALE_AT,
    poll_sec=POLL_SEC,
    cooldown_sec=COOLDOWN_SEC,
)


if __name__ == "__main__":
    autoscaler.start()
    while True:
        time.sleep(60)
