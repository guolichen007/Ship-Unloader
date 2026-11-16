#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-python3.10}"
command -v "$PY" >/dev/null 2>&1 || { echo 'PYTHON_NOT_FOUND: set PYTHON_BIN to a compatible interpreter.' >&2; exit 2; }
"$PY" -c 'import sys; assert (3,10) <= sys.version_info < (3,13), "PYTHON_VERSION_UNSUPPORTED: use Python 3.10 to 3.12."'
"$PY" -m venv "$ROOT/.local/review-venv"
"$ROOT/.local/review-venv/bin/python" -m pip install -r "$ROOT/requirements-ros.txt"
"$ROOT/.local/review-venv/bin/python" -c 'import numpy, scipy, matplotlib, jsonschema, yaml; print("REVIEW_ENVIRONMENT_READY")'
