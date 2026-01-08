# sentinelx/registry/registry.py
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import redis

from sentinelx.core.config import ModelConfig, TrafficSplit, config
from sentinelx.observability.logging import logger

# -------------------------------------------------------------------
# Redis-backed version health (recommended; works across containers)
# Key:
#   sentinelx:version_health:<logical_model>:<version>
# Value (JSON):
#   {"status":"healthy"|"unhealthy","reason":"...","ts": <unix seconds>}
# -------------------------------------------------------------------
VERSION_HEALTH_KEY_PREFIX = "sentinelx:version_health:"


def version_health_key(logical_model: str, version: str) -> str:
    return f"{VERSION_HEALTH_KEY_PREFIX}{logical_model}:{version}"


def _redis_url() -> str:
    # keep consistent with your config
    return os.getenv("SENTINELX_REDIS_URL", os.getenv("REDIS_URL", config.redis_url))


def _connect_redis() -> redis.Redis:
    return redis.from_url(_redis_url(), decode_responses=True)


class ModelRegistry:
    """
    Simple registry with:
      - model configs in-memory
      - traffic policy in-memory
      - version health in Redis (shared across gateway/scaler containers)
    """

    def __init__(self) -> None:
        # physical model instances keyed as (name, version)
        self._models: Dict[tuple[str, str], ModelConfig] = {}
        for m in config.models.values():
            self._models[(m.name, m.version)] = m

        # logical traffic config
        self._traffic: Dict[str, TrafficSplit] = dict(config.traffic)

        # redis client (for health)
        self._r = _connect_redis()

    # ---------- models ----------
    def list_models(self) -> Dict[str, Dict[str, ModelConfig]]:
        by_logical: Dict[str, Dict[str, ModelConfig]] = {}
        for (name, version), cfg in self._models.items():
            by_logical.setdefault(name, {})[version] = cfg
        return by_logical

    def get_physical(self, name: str, version: str) -> Optional[ModelConfig]:
        return self._models.get((name, version))

    def register_model(self, cfg: ModelConfig) -> None:
        logger.info(f"[Registry] Register model {cfg.name} v={cfg.version}")
        self._models[(cfg.name, cfg.version)] = cfg

    # ---------- traffic / canary ----------
    def get_traffic(self, logical_name: str) -> Optional[TrafficSplit]:
        return self._traffic.get(logical_name)

    def set_traffic(self, logical_name: str, traffic: TrafficSplit) -> None:
        logger.info(
            f"[Registry] Update traffic for {logical_name}: "
            f"primary={traffic.primary_version}, "
            f"canary={traffic.canary_version}, "
            f"canary%={traffic.canary_percentage}"
        )
        self._traffic[logical_name] = traffic

    def activate_version(self, logical_name: str, version: str) -> None:
        """
        Hard switch: 100% traffic to this version; canary off.
        """
        traffic = self._traffic.get(logical_name)
        if not traffic:
            raise ValueError(f"Unknown logical model {logical_name}")

        if not self.get_physical(logical_name, version):
            raise ValueError(f"Version {version} not registered for {logical_name}")

        self._traffic[logical_name] = TrafficSplit(
            primary_version=version,
            canary_version=None,
            canary_percentage=0.0,
        )
        logger.info(f"[Registry] Activated {logical_name} -> {version} (100%)")

    def configure_canary(
        self,
        logical_name: str,
        primary_version: str,
        canary_version: str,
        canary_percentage: float,
    ) -> None:
        if not (0.0 <= canary_percentage <= 1.0):
            raise ValueError("canary_percentage must be in [0,1]")
        if not self.get_physical(logical_name, primary_version):
            raise ValueError(f"Primary version {primary_version} not registered")
        if not self.get_physical(logical_name, canary_version):
            raise ValueError(f"Canary version {canary_version} not registered")

        self._traffic[logical_name] = TrafficSplit(
            primary_version=primary_version,
            canary_version=canary_version,
            canary_percentage=canary_percentage,
        )
        logger.info(
            f"[Registry] Canary for {logical_name}: "
            f"{primary_version} (1-{canary_percentage}) / "
            f"{canary_version} ({canary_percentage})"
        )

    # ============================================================
    # ✅ Health methods (Redis-backed)
    # ============================================================
    def mark_unhealthy(self, logical_model: str, version: str, reason: str = "unknown") -> None:
        payload = {"status": "unhealthy", "reason": reason, "ts": time.time()}
        try:
            self._r.set(version_health_key(logical_model, version), json.dumps(payload))
        except Exception as e:
            logger.exception(
                f"[Registry] mark_unhealthy failed logical={logical_model} version={version}: {e}"
            )

    def mark_healthy(self, logical_model: str, version: str) -> None:
        payload = {"status": "healthy", "reason": "ok", "ts": time.time()}
        try:
            self._r.set(version_health_key(logical_model, version), json.dumps(payload))
        except Exception as e:
            logger.exception(
                f"[Registry] mark_healthy failed logical={logical_model} version={version}: {e}"
            )

    def is_healthy(self, logical_model: str, version: str) -> bool:
        """
        Default behavior:
          - If key missing => assume healthy (don't accidentally blackhole traffic)
          - If key present => read and enforce
        """
        try:
            raw = self._r.get(version_health_key(logical_model, version))
            if not raw:
                return True
            data = json.loads(raw)
            return str(data.get("status", "healthy")) == "healthy"
        except Exception:
            return True


registry = ModelRegistry()
