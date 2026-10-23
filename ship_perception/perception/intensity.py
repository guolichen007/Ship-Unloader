"""Admission statistics before any optional auxiliary review."""

import numpy as np


def admission_stats(intensity):
    values = np.array([]) if intensity is None else np.asarray(intensity)
    values = values[np.isfinite(values)]
    result = dict(status="UNAVAILABLE_NO_FINITE_INTENSITY", used=False,
                  source_calibration="UNKNOWN", finite_count=int(len(values)),
                  unique_count=0, effective_unique_count=0, p5=None, p50=None,
                  p95=None, mad=None, robust_range=None)
    if not len(values):
        return result
    p5, p50, p95 = np.percentile(values, (5, 50, 95))
    tolerance = np.finfo(values.dtype if values.dtype.kind == "f" else float).eps * max(1., abs(p5), abs(p95))
    ordered = np.unique(values)
    result.update(unique_count=len(ordered),
        effective_unique_count=int(1 + np.count_nonzero(np.diff(ordered) > tolerance)),
        p5=float(p5), p50=float(p50), p95=float(p95),
        mad=float(np.median(abs(values-p50))), robust_range=float(p95-p5),
        status=("UNAVAILABLE_CONSTANT_DEVICE_SIGNAL" if p95-p5 <= tolerance else "AVAILABLE"))
    return result
