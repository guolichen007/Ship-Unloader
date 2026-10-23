"""Process-local numerical scheduling for command-line entry points."""

import os


def configure_numerical_environment():
    threads = int(os.environ.get("SHIP_BLAS_THREADS", "1"))
    if threads < 1:
        raise ValueError("SHIP_BLAS_THREADS must be positive")
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = str(threads)
