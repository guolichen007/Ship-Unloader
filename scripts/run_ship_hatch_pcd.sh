#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
echo "[ENV] Checking offline numerical dependencies..."
if ! "$PYTHON_BIN" -s -c 'import numpy, scipy, matplotlib; from jsonschema import Draft202012Validator'; then
    echo "ERROR: Install Python 3, numpy, scipy, matplotlib and jsonschema. See the field guide." >&2
    exit 1
fi
cd -- "$REPO_ROOT"
echo "[ENV] READY transport=PCD"
exec "$PYTHON_BIN" -s -m ship_perception.perception.cli "$@"
