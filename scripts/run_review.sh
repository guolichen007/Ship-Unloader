#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-$ROOT/.local/review-venv/bin/python}"
command -v "$PY" >/dev/null 2>&1 || { echo 'REVIEW_ENVIRONMENT_MISSING: run scripts/install_review_env.sh or set PYTHON_BIN.' >&2; exit 2; }
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
exec "$PY" -s "$ROOT/tools/r3h_oneclick.py" "$@"
