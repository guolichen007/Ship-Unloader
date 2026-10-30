"""Role-qualified local rim-top elevation, independent of steel certification."""

import hashlib
import math
import numpy as np

from ship_perception.r1_static.perimeter_deck import _gravity_hypotheses

SEMANTIC = "HATCH_INNER_RIM_TOP_ELEVATION"
TIERS = (
    "DIRECT_RIM_TOP",
    "LOCAL_RIM_PLANE",
    "DECK_PLANE",
    "SAME_SIDE_RIM_PLANE",
    "VESSEL_DECK_REFERENCE",
)
FORBIDDEN = frozenset(
    ("CARGO", "QUAY", "OTHER_VESSEL", "CONFIRMED_EXTERIOR", "UNKNOWN")
)


def cross_segment_consensus(sources, corner_uv, config, vessel_id):
    """Low residual alone cannot identify a deck; all semantic gates precede it."""
    good = [
        s
        for s in sources
        if s.get("vessel_id") == vessel_id
        and s.get("plane_role") not in FORBIDDEN
        and s.get("role_confirmed", False)
        and not set(s.get("source_roles", [])).intersection(FORBIDDEN)
    ]
    segments = {s["segment_id"] for s in good}
    if len(segments) < config["boundary"]["profile_min_sections"]:
        return None
    tolerance = config["geometry"]["plane_inlier_m"]
    cosine = math.cos(math.radians(config["geometry"]["normal_angle_deg"]))
    heights = []
    normals = []
    for s in good:
        n = np.asarray(s["normal_uvz"], float)
        n = n / np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if n[2] < cosine:
            return None
        heights.append(float(-(n[:2] @ corner_uv + s["offset_m"]) / n[2]))
        normals.append(n)
    if np.ptp(heights) > tolerance or any(
        float(a @ b) < cosine for a in normals for b in normals
    ):
        return None
    # Retain a measured plane. Do not average distinct height modes.
    selected = min(good, key=lambda s: (s["fit_residual"], s["segment_id"]))
    return dict(
        selected,
        segment_vote_count=len(segments),
        source_segment_ids=sorted(segments),
        source_point_indexes=sorted(
            {i for s in good for i in s["source_point_indexes"]}
        ),
    )


class ReviewCornerHeight:
    def __init__(self, measurement, config):
        self.measurement, self.config = measurement, config

    def _plane(self, points, indexes, segment_id, role):
        if len(points) < self.config["geometry"]["min_plane_points"]:
            return None
        hypotheses = _gravity_hypotheses(points, self.config)
        if not hypotheses:
            return None
        count, plane, mask = hypotheses[0]
        normal, offset = plane
        tolerance = self.config["geometry"]["plane_inlier_m"]
        # Reject multiple supported plane heights rather than choosing the
        # largest cargo patch as a compulsory deck.
        heights = [
            -p[1] / p[0][2]
            for n, p, m in hypotheses
            if n >= count * self.config["roi"]["min_enclosure_ratio"]
        ]
        if heights and np.ptp(heights) > tolerance:
            return None
        residual = abs(points[mask] @ normal + offset)
        return dict(
            segment_id=segment_id,
            vessel_id=self.measurement.local.vessel.vessel_id,
            plane_role=role,
            role_confirmed=True,
            source_roles=[role],
            normal_uvz=normal.tolist(),
            offset_m=float(offset),
            fit_residual=float(np.percentile(residual, 95)),
            fit_extent=[
                points[mask, :2].min(0).tolist(),
                points[mask, :2].max(0).tolist(),
            ],
            source_point_indexes=np.asarray(indexes)[mask].astype(int).tolist(),
        )

    def measure(self, result, plane_sources=()):
        polygon = np.asarray(result["boundary_uv"])
        output = []
        primary = getattr(
            self.measurement.local, "primary_cache", self.measurement.local
        )
        owned = primary.vessel.point_indexes
        for index, corner in enumerate(polygon):
            row = dict(
                z_m=None,
                z=None,
                z_state="UNOBSERVED",
                z_status="UNOBSERVED",
                z_observed=None,
                z_estimate=None,
                z_source=None,
                z_uncertainty_m=None,
                z_definition=SEMANTIC,
                source_point_indexes=[],
                source_segment_ids=[],
                fit_extent=None,
                fit_residual=None,
                plane_role=None,
                segment_vote_count=0,
                extrapolation_distance_m=None,
                fit_uncertainty_m=None,
                extrapolation_uncertainty_m=None,
                role_uncertainty="UNQUANTIFIED",
                final_uncertainty_m=None,
                steel_certified=False,
                used_for_detection=False,
            )
            sources = list(plane_sources)
            adjacent = (("U0", "V0"), ("U1", "V0"), ("U1", "V1"), ("U0", "V1"))[index]
            if all(
                result["sides"][s].get("independent_geometry_support") for s in adjacent
            ):
                radius = self.config["roi"]["support_search_m"]
                ids = self.measurement.spatial.query_indexes(
                    [*(corner - radius), *(corner + radius)]
                )
                ids = ids[np.isin(self.measurement.raw_point_indexes[ids], owned)]
                points = self.measurement.raw_uvz[ids]
                exterior = np.zeros(len(points), bool)
                for s in adjacent:
                    side = result["sides"][s]
                    n = np.asarray(side["normal_uv"])
                    distance = points[:, :2] @ n - side["final_rho"]
                    exterior |= (distance * side["outward_sign"] >= 0) & (
                        abs(distance) <= self.config["roi"]["deck_patch_width_m"]
                    )
                points = points[exterior]
                ids = ids[exterior]
                source = self._plane(
                    points,
                    self.measurement.raw_point_indexes[ids],
                    result["opening_identity"] + ":corner:" + str(index),
                    "RIM_TOP",
                )
                if source:
                    near = (
                        np.linalg.norm(
                            self.measurement.raw_uvz[ids, :2] - corner, axis=1
                        )
                        <= self.config["boundary"]["corner_join_m"]
                    )
                    source.update(
                        tier=(
                            "DIRECT_RIM_TOP"
                            if near.sum() >= self.config["boundary"]["line_min_points"]
                            else "LOCAL_RIM_PLANE"
                        )
                    )
                    sources.append(source)
            chosen = None
            tier_used = None
            for tier in TIERS:
                eligible = [
                    s
                    for s in sources
                    if s.get("tier") == tier
                    and s.get("vessel_id") == self.measurement.local.vessel.vessel_id
                    and s.get("role_confirmed", False)
                    and s.get("plane_role") not in FORBIDDEN
                    and not set(s.get("source_roles", [])).intersection(FORBIDDEN)
                ]
                if tier in ("DECK_PLANE", "VESSEL_DECK_REFERENCE"):
                    source = cross_segment_consensus(
                        eligible,
                        corner,
                        self.config,
                        self.measurement.local.vessel.vessel_id,
                    )
                    eligible = [source] if source else []
                if eligible:
                    heights = [
                        -(np.asarray(s["normal_uvz"])[:2] @ corner + s["offset_m"])
                        / s["normal_uvz"][2]
                        for s in eligible
                    ]
                    if np.ptp(heights) > self.config["geometry"]["plane_inlier_m"]:
                        row["reason"] = "MULTIPLE_CURRENT_SAME_ROLE_HEIGHT_MODES"
                        break
                    chosen = min(
                        eligible, key=lambda s: (s["fit_residual"], s["segment_id"])
                    )
                    tier_used = tier
                    break
            if chosen:
                n = np.asarray(chosen["normal_uvz"])
                z = float(-(n[:2] @ corner + chosen["offset_m"]) / n[2])
                extent = np.asarray(chosen["fit_extent"])
                extra = float(
                    np.linalg.norm(
                        np.maximum(
                            0, np.maximum(extent[0] - corner, corner - extent[1])
                        )
                    )
                )
                fit = max(
                    self.config["geometry"]["plane_inlier_m"],
                    chosen["fit_residual"] / abs(n[2]),
                )
                xy = result["corners"][index]["uncertainty_m"]
                extrap = float((extra + xy) * np.linalg.norm(n[:2]) / abs(n[2]))
                uncertainty = fit + extrap
                measured = tier_used == "DIRECT_RIM_TOP"
                row.update(
                    z_m=z,
                    z=z,
                    z_state="MEASURED" if measured else "ESTIMATED",
                    z_status="Z_MEASURED" if measured else "Z_ESTIMATED",
                    z_observed=z if measured else None,
                    z_estimate=None if measured else z,
                    z_source=tier_used,
                    z_uncertainty_m=uncertainty,
                    source_point_indexes=chosen["source_point_indexes"],
                    source_segment_ids=chosen.get(
                        "source_segment_ids", [chosen["segment_id"]]
                    ),
                    fit_extent=chosen["fit_extent"],
                    fit_residual=chosen["fit_residual"],
                    plane_role=chosen["plane_role"],
                    segment_vote_count=chosen.get("segment_vote_count", 1),
                    extrapolation_distance_m=extra,
                    fit_uncertainty_m=fit,
                    extrapolation_uncertainty_m=extrap,
                    final_uncertainty_m=uncertainty,
                    local_plane_uvz=dict(
                        normal=n.tolist(), offset_m=chosen["offset_m"]
                    ),
                    reason="CURRENT_SAME_VESSEL_ROLE_QUALIFIED_RIM_TOP_ELEVATION",
                )
            else:
                row.setdefault("reason", "NO_RELIABLE_CURRENT_RIM_TOP_SOURCE")
            output.append(row)
        return output
