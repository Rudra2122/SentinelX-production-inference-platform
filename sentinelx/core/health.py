# sentinelx/core/health.py
from __future__ import annotations

import json
import time
from typing import Dict, Any, List

import redis.asyncio as redis_async

from sentinelx.core.config import WORKER_STALE_AFTER_S, WORKER_HEARTBEAT_KEY_PREFIX


async def list_worker_heartbeats(r: redis_async.Redis) -> List[Dict[str, Any]]:
    keys = await r.keys(f"{WORKER_HEARTBEAT_KEY_PREFIX}*")
    if not keys:
        return []
    vals = await r.mget(keys)
    out = []
    for v in vals:
        if not v:
            continue
        try:
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8")
            out.append(json.loads(v))
        except Exception:
            continue
    return out


def is_stale(hb: Dict[str, Any]) -> bool:
    ts = float(hb.get("ts", 0.0))
    return (time.time() - ts) > WORKER_STALE_AFTER_S
