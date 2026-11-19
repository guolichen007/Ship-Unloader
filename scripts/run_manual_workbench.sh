#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$root/.local/review-venv/bin/python}"
command -v "$python_bin" >/dev/null 2>&1 || { echo 'PYTHON_NOT_FOUND: run scripts/install_review_env.sh or set PYTHON_BIN.' >&2; exit 2; }
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$python_bin" -s -m ship_perception.manual_review.workbench_server "$@"
