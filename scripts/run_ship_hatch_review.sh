#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$REPO_ROOT/.local/review-venv/bin/python" ]]; then
        PYTHON_BIN="$REPO_ROOT/.local/review-venv/bin/python"
    elif [[ -x "$REPO_ROOT/.local/ros2-venv/bin/python" ]]; then
        PYTHON_BIN="$REPO_ROOT/.local/ros2-venv/bin/python"
    else
        PYTHON_BIN=python3
    fi
fi
export PYTHON_BIN PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
cd -- "$REPO_ROOT"
exec "$PYTHON_BIN" -s scripts/hatch_review_runner.py "$@"
