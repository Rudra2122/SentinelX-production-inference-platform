# sentinelx/grpc/dual_server.py
from __future__ import annotations

import asyncio
import os

import uvicorn

from sentinelx.api.main import app
from sentinelx.grpc.server import serve_grpc


async def _run_uvicorn() -> None:
    host = os.getenv("SENTINELX_HTTP_HOST", "0.0.0.0")
    port = int(os.getenv("SENTINELX_HTTP_PORT", "8000"))

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main() -> None:
    grpc_host = os.getenv("SENTINELX_GRPC_HOST", "0.0.0.0")
    grpc_port = int(os.getenv("SENTINELX_GRPC_PORT", "50051"))

    grpc_task = asyncio.create_task(serve_grpc(host=grpc_host, port=grpc_port), name="grpc")
    http_task = asyncio.create_task(_run_uvicorn(), name="http")

    done, pending = await asyncio.wait(
        {grpc_task, http_task},
        return_when=asyncio.FIRST_EXCEPTION,
    )

    # If one exits (or errors), cancel the other
    for t in pending:
        t.cancel()

    # Re-raise exceptions if any
    for t in done:
        exc = t.exception()
        if exc:
            raise exc


if __name__ == "__main__":
    asyncio.run(main())
