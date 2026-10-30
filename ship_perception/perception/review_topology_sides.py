"""Current Raw3D/BEV side-family constraints for an existing opening identity.

These are topology-supported review lines, not independently measured steel
coordinates. Distinct normal-face measurements remain separate downstream.
"""

import hashlib
import numpy as np
from .sections import physical_sections
from .review_geometry import SIDES


def current_side_constraints(cache, measurement, region, config):
    candidates = cache.side_candidates(region.conservative_bounds)
    output = {name: [] for name, _, _, _ in SIDES}
    for name, axis, pos, sign in SIDES:
        rows = sorted(
            (
                c
                for c in candidates[name]
                if min(c["primary_raw3d"], c["primary_bev"])
                >= config["roi"]["min_enclosure_ratio"]
            ),
            key=lambda c: c["rho"],
        )
        groups = []
        for row in rows:
            if (
                groups
                and row["rho"] - groups[-1][-1]["rho"]
                <= config["boundary"]["corner_join_m"]
            ):
                groups[-1].append(row)
            else:
                groups.append([row])
        if not groups:
            continue
        quality = [
            max(min(r["primary_raw3d"], r["primary_bev"]) for r in group)
            for group in groups
        ]
        winners = [i for i, q in enumerate(quality) if q == max(quality)]
        if len(winners) != 1:
            continue
        group = groups[winners[0]]
        # Normal-coordinate aliases form one current structural support band.
        # Select its opening-facing extent, never the average of competing Raw
        # measurements. The full alias width remains review uncertainty.
        rho = float(
            max(r["rho"] for r in group) if sign < 0 else min(r["rho"] for r in group)
        )
        span = tuple(region.conservative_bounds[i] for i in (1 - axis, 3 - axis))
        band = config["geometry"]["coarse_voxel_m"] / 2
        box = [0.0] * 4
        box[axis] = rho - band
        box[axis + 2] = rho + band
        box[1 - axis], box[3 - axis] = span
        ids = measurement.spatial.query_indexes(box)
        points = measurement.raw_uvz[ids]
        if not len(points):
            continue
        primary = getattr(cache, "primary_cache", cache)
        keep = np.isin(measurement.raw_point_indexes[ids], primary.vessel.point_indexes)
        points = points[keep]
        ids = ids[keep]
        edges = physical_sections(span, config["geometry"]["refine_voxel_m"], config)
        counts = [
            int(
                np.count_nonzero(
                    (points[:, 1 - axis] >= lo) & (points[:, 1 - axis] <= hi)
                )
            )
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        sections = sum(n >= config["boundary"]["line_min_points"] for n in counts)
        if sections < config["boundary"]["profile_min_sections"]:
            continue
        normal = np.eye(2)[axis]
        direction = np.array([-normal[1], normal[0]])
        along = points[:, :2] @ direction
        raw_ids = measurement.raw_point_indexes[ids]
        identity = hashlib.sha256(
            np.asarray(sorted(raw_ids), dtype="<i8").tobytes()
        ).hexdigest()
        providers = sorted({pid for r in group for pid in r.get("provider_ids", [])})
        output[name].append(
            dict(
                mode_id="TOPOLOGY:" + identity,
                source="TOPOLOGY_SUPPORTED",
                state="TOPOLOGY_SUPPORTED",
                rho=rho,
                normal_uv=normal.tolist(),
                support_span=[float(along.min()), float(along.max())],
                uncertainty_m=max(
                    config["geometry"]["plane_inlier_m"],
                    max(r["rho"] for r in group) - min(r["rho"] for r in group),
                ),
                independent_raw_sections=sections,
                physical_role_improved=True,
                physical_role="CURRENT_RAW3D_BEV_ENCLOSURE_SIDE_NOT_STEEL",
                raw_support_ids=["RAW_TOPOLOGY_INPUT:" + identity],
                source_point_indexes=raw_ids.tolist(),
                raw_support_scope="CURRENT_PRIMARY_QUERY_POINTS_NOT_FITTED_STEEL_INLIERS",
                topology_support_ids=providers,
                candidate_band_rhos=[r["rho"] for r in group],
                competing_band_rhos=[
                    [r["rho"] for r in g]
                    for j, g in enumerate(groups)
                    if j != winners[0]
                ],
                source_selection="UNIQUE_HIGHEST_JOINT_RAW3D_BEV_FAMILY_SUPPORT",
            )
        )
    return output
