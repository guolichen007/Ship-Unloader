"""Intern normals only within one measurement, with exact immutable Raw contracts."""

import hashlib
import json
import os

import numpy as np


def pool_enabled():
    value = os.environ.get("SHIP_REVIEW_EXACT_NORMAL_POOL", "1")
    if value not in ("0", "1"):
        raise ValueError("SHIP_REVIEW_EXACT_NORMAL_POOL must be 0 or 1")
    return value == "1"


def exact_normal_instance(measurement, points, raw_ids, config, axes, create):
    """Equal coordinates with different Raw IDs, neighbourhoods or frames cannot hit."""
    if axes is None:
        return create()
    points, raw_ids, axes = np.asarray(points), np.asarray(raw_ids), np.asarray(axes)
    if not pool_enabled() or points.flags.writeable or raw_ids.flags.writeable:
        return create()
    if len(points) != len(raw_ids):
        raise ValueError("NORMAL_POOL_RAW_INDEX_LENGTH_MISMATCH")
    config_json = json.dumps(
        config, sort_keys=True, allow_nan=False, separators=(",", ":")
    )
    key = (
        points.dtype.str,
        points.shape,
        hashlib.sha256(points.tobytes()).hexdigest(),
        raw_ids.dtype.str,
        hashlib.sha256(raw_ids.tobytes()).hexdigest(),
        config_json,
        axes.dtype.str,
        axes.shape,
        axes.tobytes(),
    )
    entries = getattr(measurement, "_exact_normal_pool", {})
    previous = entries.get(key)
    if previous is not None:
        old_points, old_ids, normals = previous
        if np.array_equal(old_points, points) and np.array_equal(old_ids, raw_ids):
            measurement.counters["normal_exact_pool_hits"] = (
                measurement.counters.get("normal_exact_pool_hits", 0) + 1
            )
            return normals
    normals = create()
    entries[key] = (points, raw_ids, normals)
    measurement._exact_normal_pool = entries
    measurement.counters["normal_exact_pool_entries"] = len(entries)
    return normals
