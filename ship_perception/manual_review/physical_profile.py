"""R3 experiment B: corroborate quarantined high-interior profiles from Raw structure."""

import numpy as np

from .side_candidates import ExecutionStop, profile

CELL_M = 0.08
MIN_STATION_M = 0.16
MIN_CAP_DEPTH_M = 0.13
WALL_CAP_JOIN_M = 0.12
MAX_SURFACE_SLOPE = 0.22


def balanced(points):
    # Equal weight per occupied physical voxel. Raw points remain available for audit.
    if not len(points):
        return points
    bins = np.floor((points - points.min(axis=0)) / CELL_M).astype("int64")
    _, idx = np.unique(bins, axis=0, return_index=True)
    return points[idx]


def surface(points, axes, dependent, slope_limit, tolerance):
    positions = balanced(points)
    if len(positions) < 5 or min(np.ptp(positions[:, axes], axis=0)) < MIN_CAP_DEPTH_M:
        return None
    center = positions.mean(axis=0)
    shifted = positions - center
    design = np.column_stack((shifted[:, axes], np.ones(len(shifted))))
    q, _, rank, _ = np.linalg.lstsq(design, shifted[:, dependent], rcond=None)
    if rank != 3 or np.linalg.norm(q[:2]) > slope_limit:
        return None

    def residual(values):
        return np.abs(
            values[:, dependent]
            - center[dependent]
            - q[2]
            - (values[:, axes] - center[list(axes)]) @ q[:2]
        )

    if np.quantile(residual(points), 0.85) > tolerance:
        return None
    return float(center[dependent] + q[2] - center[list(axes)] @ q[:2]), q[:2]


def transition(local, ids, rows, observation, cfg):
    face, cap, lower, upper = [
        local[np.searchsorted(ids, observation[name])]
        for name in ("face", "cap", "interior", "competing_interior")
    ]
    if min(len(face), len(cap), len(lower)) < 5:
        return None
    wall = surface(face, (0, 2), 1, MAX_SURFACE_SLOPE, cfg.cap_band_m / 2)
    top = surface(cap, (0, 1), 2, MAX_SURFACE_SLOPE, cfg.cap_band_m / 2)
    if wall is None or top is None:
        return None
    if np.ptp(lower[:, 0]) < MIN_STATION_M or np.ptp(cap[:, 0]) < MIN_STATION_M:
        return None
    s = observation["t_m"]
    d0, (ds, dz) = wall
    z0, (zs, zd) = top
    divisor = 1 - dz * zd
    d = (d0 + ds * s + dz * (z0 + zs * s)) / divisor
    z = z0 + zs * s + zd * d
    if abs(d - observation["d_m"]) > cfg.residual_m:
        return None
    if (
        max(face[:, 2]) < z - WALL_CAP_JOIN_M
        or np.any(cap[:, 1] >= d)
        or min(d - cap[:, 1]) > WALL_CAP_JOIN_M + 0.12
    ):
        return None
    # A continuation towards the interior is an actual veto, even with cargo gaps.
    inner = local[rows]
    expected = z0 + inner[:, 0] * zs + inner[:, 1] * zd
    extension = inner[
        (inner[:, 1] > d + 0.12)
        & (inner[:, 1] < d + 0.62)
        & (abs(inner[:, 2] - expected) <= cfg.cap_band_m)
    ]
    if (
        len(extension) >= 4
        and np.ptp(extension[:, 0]) >= MIN_STATION_M
        and np.ptp(extension[:, 1]) >= MIN_CAP_DEPTH_M
    ):
        return None
    if len(upper) and min(upper[:, 1]) < d + WALL_CAP_JOIN_M:
        return None
    if np.ptp(face[:, 2]) < cfg.relief_m or min(z - face[:, 2]) < -cfg.cap_band_m:
        return None
    return dict(
        observation,
        role="INNER_RIM_SUPPORTED",
        d_m=float(d),
        top_m=float(z),
        physical_source="WALL_CAP_INTERSECTION",
        physical_profile=dict(
            wall_d=[d0, ds, dz],
            cap_z=[z0, *top[1]],
            determinant=float(divisor),
            station_overlap_m=float(
                min(max(cap[:, 0]), max(face[:, 0]))
                - max(min(cap[:, 0]), min(face[:, 0]))
            ),
        ),
    )


def augmented_profile(points, ids, edge, tangent, normal, length, search, cfg, check):
    sections, original = profile(
        points, ids, edge, tangent, normal, length, search, cfg, check
    )
    if not original:
        return sections, original
    order = np.argsort(ids)
    ids = ids[order]
    points = points[order]
    delta = points[:, :2] - edge
    local = np.column_stack((delta @ tangent, delta @ normal, points[:, 2]))
    support = {row["slab"] for row in original if row["role"] == "INNER_RIM_SUPPORTED"}
    slab_rows = {
        part["slab"]: (local[:, 0] >= part["t_m"][0]) & (local[:, 0] < part["t_m"][1])
        for part in sections
        if part["slab"] not in support
    }
    nominations = []
    for observation in original:
        check()
        slab = observation["slab"]
        if slab not in slab_rows or observation["role"] != "CONFLICTING":
            continue
        verified = transition(local, ids, slab_rows[slab], observation, cfg)
        if (
            verified
            and verified["physical_profile"]["station_overlap_m"] >= MIN_STATION_M
        ):
            nominations.append(verified)
            if len(original) + len(nominations) > cfg.max_observations:
                raise ExecutionStop("PHYSICAL_PROFILE_LIMIT")
    return sections, original + nominations
