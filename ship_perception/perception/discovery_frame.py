"""Condition B1's lattice in a current-Raw rigid frame, preserving point IDs.

The scene covariance defines a numerical lattice basis, not a vessel direction
or an opening prior. The existing frozen B1 still discovers every support and
vessel. All measurement consumers continue to use original input XYZ.
"""

import numpy as np

from ship_perception.r1_static.scene_vessel import infer_scene_vessels


def conditioned_discovery(points, config, research):
    raw = np.asarray(points, dtype=np.float64)
    center = raw.mean(axis=0)
    xy = raw[:, :2] - center[:2]
    values, vectors = np.linalg.eigh(xy.T @ xy / len(xy))
    along = vectors[:, -1]
    projected = xy @ along
    third = projected * projected * projected
    sign = float(third.sum())
    numerical = 64 * np.finfo(float).eps * float(abs(third).sum())
    if abs(sign) <= numerical:
        sign = float(np.sum(projected * (raw[:, 2] - center[2]) ** 2))
    if sign < 0:
        along = -along
    axes = np.array([along, [-along[1], along[0]]])
    local = np.column_stack((xy @ axes.T, raw[:, 2] - center[2]))
    offset = local.min(axis=0)
    local = local - offset
    scene, private = infer_scene_vessels(local, config, research)
    bounds = {
        row["support_id"]: row["bbox_xy"] for row in scene["scene_support_candidates"]
    }

    def to_input(p):
        return (np.asarray(p) + offset[:2]) @ axes + center[:2]

    for support in scene["scene_support_candidates"]:
        box = support["bbox_xy"]
        corners = to_input(
            [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
        )
        support["lattice_bounds_xy"] = list(box)
        support["bbox_xy"] = np.r_[corners.min(0), corners.max(0)].tolist()
        support["spatial_centroid"] = to_input(support["spatial_centroid"]).tolist()
        support["candidate_axis_families"] = [
            (np.asarray(a) @ axes).tolist() for a in support["candidate_axis_families"]
        ]
    origin_xy = center[:2] + offset[:2] @ axes
    for vessel in scene["vessel_hypotheses"]:
        vessel["local_axes"] = (np.asarray(vessel["local_axes"]) @ axes).tolist()
        owned = raw[private["vessel_point_indexes"][vessel["vessel_hypothesis_id"]], :2]
        vessel["bbox_xy"] = np.r_[owned.min(0), owned.max(0)].tolist()
        for seam in vessel["split_seams"]:
            seam["lattice_rho"] = seam["rho"]
            seam["rho"] = float(
                seam["rho"] + origin_xy @ np.asarray(vessel["local_axes"])[1]
            )
    scene["discovery_lattice_frame"] = dict(
        role="NUMERICAL_LATTICE_BASIS_NOT_VESSEL_OR_HATCH_PRIOR",
        origin_xyz=[*origin_xy.tolist(), float(center[2] + offset[2])],
        axes_xy=axes.tolist(),
        covariance_eigenvalues=values.tolist(),
        point_order_preserved=True,
        measurement_xyz="ORIGINAL_UNMODIFIED_RAW",
    )
    # Mixed-support attribution must use the very lattice that assigned B1
    # cells, rather than a rotated world AABB. Only its original IDs leave here.
    private.update(membership_xyz=local, support_bounds=bounds)
    return scene, private
