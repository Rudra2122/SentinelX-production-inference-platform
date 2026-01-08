#!/usr/bin/env bash
set -e

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

export PYTHONPATH=.

uvicorn sentinelx.api.main:app --host 0.0.0.0 --port 8000 --reload
