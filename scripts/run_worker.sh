#!/usr/bin/env bash
set -e

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

export PYTHONPATH=.

PHYSICAL_MODEL=${1:-demo_classifier}
VERSION=${2:-v1.0.0}

python -m sentinelx.inference.worker --physical-model "$PHYSICAL_MODEL" --version "$VERSION"
