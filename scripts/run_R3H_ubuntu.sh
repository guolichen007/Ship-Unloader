#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
exec "${PYTHON_BIN:-python3}" "$repo_root/tools/r3h_oneclick.py" "$@"
