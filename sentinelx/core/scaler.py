# sentinelx/core/scaler.py
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import redis

from sentinelx.observability.logging import logger
from sentinelx.core.config import (
    REDIS_URL,
    WORKER_HEARTBEAT_KEY_PREFIX,
    WORKER_STALE_AFTER_S,
)

# -------------------------------------------------------------------
# Version health stored in Redis (recommended for multi-container setup)
# -------------------------------------------------------------------
VERSION_HEALTH_KEY_PREFIX = "sentinelx:version_health:"


def version_health_key(logical_model: str, version: str) -> str:
    return f"{VERSION_HEALTH_KEY_PREFIX}{logical_model}:{version}"


# -------------------------------------------------------------------
# Docker compose scaling
# -------------------------------------------------------------------
def scale_compose(service: str, replicas: int) -> None:
    """
    Scale a docker-compose service to the desired number of replicas.
    """

    compose_file = os.getenv("SENTINELX_COMPOSE_FILE", "docker-compose.yml")
    project_dir = os.getenv("SENTINELX_PROJECT_DIR", os.getcwd())

    cmd = [
        "docker",
        "compose",
        "-f",
        compose_file,
        "up",
        "-d",
        "--scale",
        f"{service}={replicas}",
        "--no-recreate",
    ]

    logger.info(
        f"[Scaler] docker compose scale -> service={service} replicas={replicas} "
        f"(compose_file={compose_file}, project_dir={project_dir})"
    )

    subprocess.run(cmd, cwd=project_dir, check=False)


# -------------------------------------------------------------------
# Health enforcement (Phase: "enforce health")
# -------------------------------------------------------------------
def _connect_redis() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        out = json.loads(s)
        if isinstance(out, dict):
            return out
        return None
    except Exception:
        return None


def list_worker_heartbeats(redis_client: redis.Redis) -> List[Dict[str, Any]]:
    """
    Reads all heartbeat values stored at keys:
      sentinelx:worker_heartbeat:<worker_id>

    Each value is expected to be JSON like:
      {"worker_id": "...", "ts": 1234567.89, "hostname": "...", "service": "...",
       "logical_model": "...", "version": "...", "route": "..."}
    """
    hbs: List[Dict[str, Any]] = []
    pattern = f"{WORKER_HEARTBEAT_KEY_PREFIX}*"

    try:
        for key in redis_client.scan_iter(match=pattern, count=200):
            raw = redis_client.get(key)
            if not raw:
                continue
            hb = _safe_json_loads(raw)
            if hb:
                hbs.append(hb)
    except Exception as e:
        logger.exception(f"[Scaler][Health] failed to scan heartbeats: {e}")

    return hbs


def is_stale(hb: Dict[str, Any]) -> bool:
    """
    Heartbeat is stale if it's older than WORKER_STALE_AFTER_S.
    """
    try:
        ts = float(hb.get("ts", 0.0))
    except Exception:
        return True
    return (time.time() - ts) > float(WORKER_STALE_AFTER_S)


def _extract_known_versions_from_registry(registry_obj) -> List[Tuple[str, str]]:
    """
    Best-effort extraction of (logical_model, version) pairs from your registry.

    Your codebase uses:
      models_by_logical = registry.list_models()
    We treat that return as:
      { "logical_model": [ {..version..} or "v1.0.0" ... ] }  OR
      { "logical_model": ["v1.0.0", "v1.1.0"] }  OR
      { "logical_model": {...} }

    If we can't parse, we return [] (and we still write health for versions we see in heartbeats).
    """
    out: List[Tuple[str, str]] = []

    try:
        models_by_logical = registry_obj.list_models()
    except Exception:
        return out

    if not isinstance(models_by_logical, dict):
        return out

    for logical_model, versions in models_by_logical.items():
        # versions might be list[str], list[dict], dict, etc.
        if isinstance(versions, list):
            for v in versions:
                if isinstance(v, str):
                    out.append((logical_model, v))
                elif isinstance(v, dict):
                    ver = v.get("version") or v.get("primary_version") or v.get("name")
                    if isinstance(ver, str) and ver:
                        out.append((logical_model, ver))
        elif isinstance(versions, dict):
            # sometimes already keyed by version
            for ver in versions.keys():
                if isinstance(ver, str):
                    out.append((logical_model, ver))

    # de-dupe
    return list({(lm, ver) for (lm, ver) in out})


def enforce_version_health(
    *,
    redis_client: redis.Redis,
    registry_obj,
    route_default: str = "default",
) -> None:
    """
    Goal:
      If a (logical_model, version, route) has 0 healthy workers -> mark it unhealthy.

    We store state in Redis:
      sentinelx:version_health:<logical_model>:<version>
        JSON: {"status":"healthy"|"unhealthy", "reason": "...", "ts": ...}

    NOTE:
      This does NOT automatically change routing unless your gateway/router checks this key.
      (But it matches the screenshot recommendation: store version health in Redis.)
    """
    hbs = list_worker_heartbeats(redis_client)

    healthy_keys: Set[Tuple[str, str, str]] = set()
    stale_count = 0
    healthy_count = 0

    for hb in hbs:
        if is_stale(hb):
            stale_count += 1
            continue

        healthy_count += 1
        lm = hb.get("logical_model") or hb.get("logicalModel") or hb.get("model")
        ver = hb.get("version")
        route = hb.get("route") or route_default

        if isinstance(lm, str) and lm and isinstance(ver, str) and ver:
            healthy_keys.add((lm, ver, str(route)))

    # Determine "known" versions from registry, and also include any versions observed in heartbeats
    known_versions = _extract_known_versions_from_registry(registry_obj)
    observed_versions = {(hb.get("logical_model"), hb.get("version")) for hb in hbs}
    for lm, ver in list(observed_versions):
        if isinstance(lm, str) and lm and isinstance(ver, str) and ver:
            known_versions.append((lm, ver))

    known_versions = list({(lm, ver) for (lm, ver) in known_versions})

    now = time.time()
    logger.info(
        f"[Scaler][Health] heartbeats total={len(hbs)} healthy={healthy_count} stale={stale_count} "
        f"known_versions={len(known_versions)}"
    )

    # For MVP we use route_default for registry-known versions.
    for (logical_model, version) in known_versions:
        key = (logical_model, version, route_default)

        if key in healthy_keys:
            payload = {"status": "healthy", "reason": "has_healthy_workers", "ts": now}
        else:
            payload = {"status": "unhealthy", "reason": "no_healthy_workers", "ts": now}

        try:
            redis_client.set(version_health_key(logical_model, version), json.dumps(payload))
        except Exception as e:
            logger.exception(
                f"[Scaler][Health] failed to write version health "
                f"logical_model={logical_model} version={version}: {e}"
            )


def enforce_health_once() -> None:
    """
    Convenience wrapper if you want to call this periodically from the autoscaler loop.
    """
    try:
        from sentinelx.registry.registry import registry  # your existing singleton
    except Exception as e:
        logger.exception(f"[Scaler][Health] cannot import registry: {e}")
        return

    r = _connect_redis()
    enforce_version_health(redis_client=r, registry_obj=registry)
