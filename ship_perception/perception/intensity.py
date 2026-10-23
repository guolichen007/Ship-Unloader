"""Admission statistics before any optional auxiliary review."""

import numpy as np


def admission_stats(intensity, source_dtype=None):
    values = np.array([]) if intensity is None else np.asarray(intensity)
    values = values[np.isfinite(values)]
    result = dict(
        status="UNAVAILABLE_NO_FINITE_INTENSITY",
        used=False,
        source_calibration="UNKNOWN",
        finite_count=int(len(values)),
        unique_count=0,
        effective_unique_count=0,
        p5=None,
        p50=None,
        p95=None,
        mad=None,
        robust_range=None,
    )
    if not len(values):
        return result
    p5, p50, p95 = np.percentile(values, (5, 50, 95))
    precision = np.dtype(source_dtype) if source_dtype else values.dtype
    tolerance = (
        np.finfo(precision).eps * max(1.0, abs(p5), abs(p95))
        if precision.kind == "f"
        else 1.0
    )
    ordered = np.unique(values)
    effective, anchor = 0, None
    for value in ordered:
        if anchor is None or value - anchor > tolerance:
            effective, anchor = effective + 1, value
    result.update(
        unique_count=len(ordered),
        effective_unique_count=effective,
        source_dtype=precision.str,
        constant_signal_tolerance=float(tolerance),
        p5=float(p5),
        p50=float(p50),
        p95=float(p95),
        mad=float(np.median(abs(values - p50))),
        robust_range=float(p95 - p5),
        status=(
            "UNAVAILABLE_CONSTANT_DEVICE_SIGNAL"
            if p95 - p5 <= tolerance
            else "AVAILABLE"
        ),
    )
    return result
