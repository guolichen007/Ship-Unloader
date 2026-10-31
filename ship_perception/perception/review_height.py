"""Role-qualified rim-top Z, independent of post-rectangle XY certification."""

import math
import numpy as np

from ship_perception.r1_static.perimeter_deck import _gravity_hypotheses
from .rim_top_height import RimTopHeightEstimator

SEMANTIC = "HATCH_INNER_RIM_TOP_ELEVATION"
TIERS = (
    "DIRECT_RIM_TOP",
    "LOCAL_RIM_PLANE",
    "SAME_SIDE_RIM_PLANE",
    "DECK_PLANE",
    "VESSEL_DECK_REFERENCE",
)
FORBIDDEN = frozenset(
    ("CARGO", "QUAY", "OTHER_VESSEL", "CONFIRMED_EXTERIOR", "UNKNOWN")
)
RIM_ROLES = frozenset(("RIM_TOP", "COAMING_TOP"))
DECK_ROLES = frozenset(("TRUE_DECK", "DECK"))


def cross_segment_consensus(sources, corner_uv, config, vessel_id):
    """Require distinct current semantic/physical supports before deck use."""
    good = [
        s
        for s in sources
        if s.get("vessel_id") == vessel_id
        and s.get("plane_role") in RIM_ROLES | DECK_ROLES
        and s.get("role_confirmed", False)
        and not set(s.get("source_roles", [])).intersection(FORBIDDEN)
    ]
    segments = {}
    for source in good:
        evidence = source.get("independent_segment_evidence") or [source]
        for segment in evidence:
            segments.setdefault(segment["segment_id"], set()).update(
                segment.get("source_point_indexes", [])
            )
    unique, seen = {}, set()
    for ident, indexes in sorted(segments.items()):
        key = tuple(sorted(indexes))
        if not key or key in seen:
            continue
        seen.add(key)
        unique[ident] = indexes
    if len(unique) < config["boundary"]["profile_min_sections"]:
        return None
    cosine = math.cos(math.radians(config["geometry"]["normal_angle_deg"]))
    heights, normals = [], []
    for source in good:
        original = np.asarray(source["normal_uvz"], dtype=float)
        scale = np.linalg.norm(original)
        if not scale:
            return None
        normal = original / scale
        offset = source["offset_m"] / scale
        if normal[2] < 0:
            normal, offset = -normal, -offset
        if normal[2] < cosine:
            return None
        heights.append(float(-(normal[:2] @ corner_uv + offset) / normal[2]))
        normals.append(normal)
    if (
        not good
        or np.ptp(heights) > config["geometry"]["plane_inlier_m"]
        or any(float(a @ b) < cosine for a in normals for b in normals)
    ):
        return None
    selected = min(good, key=lambda s: (s["fit_residual"], s["segment_id"]))
    return dict(
        selected,
        segment_vote_count=len(unique),
        source_segment_ids=sorted(unique),
        source_point_indexes=sorted({i for ids in unique.values() for i in ids}),
    )


class ReviewCornerHeight:
    def __init__(self, measurement, config):
        self.measurement, self.config = measurement, config
        self.audit = {}
        self._eligibility_cache = {}

    def _plane_candidates(self, points, indexes, segment_id, role):
        if len(points) < self.config["geometry"]["min_plane_points"]:
            return []
        hypotheses = _gravity_hypotheses(points, self.config)
        if not hypotheses:
            return []
        threshold = self.config["geometry"]["plane_inlier_m"]
        anchor = points[:, :2].mean(0)
        unique, output = [], []
        for count, (normal, offset), mask in hypotheses:
            if count < hypotheses[0][0] * self.config["roi"]["min_enclosure_ratio"]:
                continue
            height = -(normal[:2] @ anchor + offset) / normal[2]
            if any(
                np.count_nonzero(mask & old_mask)
                / max(1, min(count, int(old_mask.sum())))
                >= self.config["frame"]["same_plane_overlap_min"]
                or (
                    abs(height - old_height) <= threshold
                    and float(normal @ old_normal)
                    >= math.cos(math.radians(self.config["frame"]["normal_refine_deg"]))
                )
                for old_height, old_normal, old_mask in unique
            ):
                continue
            selected = points[mask]
            origin = selected[:, :2].mean(0)
            design = np.column_stack((selected[:, :2] - origin, np.ones(len(selected))))
            if np.linalg.matrix_rank(design) < 3:
                continue
            residual = selected @ normal + offset
            z_residual = residual / normal[2]
            variance = float(z_residual @ z_residual / max(1, len(selected) - 3))
            covariance = variance * np.linalg.pinv(design.T @ design)
            output.append(
                dict(
                    segment_id=segment_id,
                    vessel_id=self.measurement.local.vessel.vessel_id,
                    plane_role=role,
                    role_confirmed=role in RIM_ROLES | DECK_ROLES,
                    source_roles=[role],
                    normal_uvz=normal.tolist(),
                    offset_m=float(offset),
                    fit_residual=float(np.percentile(abs(residual), 95)),
                    fit_extent=[
                        selected[:, :2].min(0).tolist(),
                        selected[:, :2].max(0).tolist(),
                    ],
                    source_point_indexes=np.asarray(indexes)[mask].astype(int).tolist(),
                    plane_parameter_origin_uv=origin.tolist(),
                    plane_parameter_covariance=covariance.tolist(),
                    plane_covariance_semantic="CURRENT_LINEAR_FIT_RESIDUAL_COVARIANCE_NOT_SENSOR_CALIBRATED",
                )
            )
            unique.append((height, normal, mask))
            if len(output) >= self.config["geometry"]["max_plane_hypotheses"]:
                break
        return output

    def _plane(self, points, indexes, segment_id, role):
        candidates = self._plane_candidates(points, indexes, segment_id, role)
        return candidates[0] if len(candidates) == 1 else None

    def _eligible(self, source, owned, roles):
        key = (id(source), tuple(sorted(roles)))
        if key in self._eligibility_cache:
            return self._eligibility_cache[key]
        indexes = source.get("source_point_indexes", [])
        valid = bool(
            source.get("vessel_id") == self.measurement.local.vessel.vessel_id
            and source.get("plane_role") in roles
            and source.get("role_confirmed", False)
            and not set(source.get("source_roles", [])).intersection(FORBIDDEN)
            and indexes
            and np.isin(indexes, owned).all()
        )
        if valid:
            points = self._support_points(source)
            normal = np.asarray(source["normal_uvz"], dtype=float)
            scale = np.linalg.norm(normal)
            valid = len(points) == len(indexes) and scale > 0
            if valid:
                residual = abs(points @ normal + source["offset_m"]) / scale
                valid = bool(
                    np.percentile(residual, 95)
                    <= self.config["geometry"]["plane_inlier_m"]
                    + 64 * np.finfo(float).eps * max(1.0, float(np.max(abs(points))))
                )
        self._eligibility_cache[key] = valid
        return valid

    def _support_points(self, source):
        ids = np.asarray(source["source_point_indexes"], dtype=np.int64)
        if not hasattr(self, "_raw_index_order"):
            self._raw_index_order = np.argsort(self.measurement.raw_point_indexes)
            self._sorted_raw_indexes = self.measurement.raw_point_indexes[
                self._raw_index_order
            ]
        positions = np.searchsorted(self._sorted_raw_indexes, ids)
        valid = positions < len(self._sorted_raw_indexes)
        valid[valid] &= self._sorted_raw_indexes[positions[valid]] == ids[valid]
        return self.measurement.raw_uvz[self._raw_index_order[positions[valid]]]

    def _empty(self):
        return dict(
            z_m=None,
            z=None,
            z_state="UNOBSERVED",
            z_status="Z_UNOBSERVED",
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
            z_probe_attempted=True,
            role_qualified_z_candidate_count=0,
            deck_reference_z=None,
        )

    def measure(self, result, plane_sources=()):
        probe = RimTopHeightEstimator(
            self.measurement, self.config, self._plane_candidates
        )
        sources = list(plane_sources) + probe.collect(result)
        self.audit = dict(
            probe.audit,
            z_logic_dead_path_removed=True,
            xy_certification_flag_used_for_z=False,
        )
        self.audit["role_qualified_sources"] = sources
        primary = getattr(
            self.measurement.local, "primary_cache", self.measurement.local
        )
        owned = primary.vessel.point_indexes
        output = []
        for index, corner in enumerate(np.asarray(result["boundary_uv"])):
            row, local = self._empty(), []
            for source in sources:
                if not self._eligible(source, owned, RIM_ROLES):
                    continue
                extent = np.asarray(source["fit_extent"])
                extra = float(
                    np.linalg.norm(
                        np.maximum(
                            0, np.maximum(extent[0] - corner, corner - extent[1])
                        )
                    )
                )
                if extra > self.config["roi"]["support_search_m"]:
                    continue
                points = self._support_points(source)
                near = int(
                    np.count_nonzero(
                        np.linalg.norm(points[:, :2] - corner, axis=1)
                        <= self.config["boundary"]["corner_join_m"]
                    )
                )
                tier = (
                    "DIRECT_RIM_TOP"
                    if near >= self.config["boundary"]["line_min_points"]
                    else source.get("tier", "LOCAL_RIM_PLANE")
                )
                if (
                    tier == "LOCAL_RIM_PLANE"
                    and extra > self.config["boundary"]["corner_join_m"]
                ):
                    continue
                local.append(
                    dict(
                        source,
                        tier=tier,
                        current_corner_extrapolation_m=extra,
                        direct_corner_point_count=near,
                    )
                )
            row["role_qualified_z_candidate_count"] = len(local)
            chosen = None
            for tier in TIERS[:3]:
                eligible = [s for s in local if s["tier"] == tier]
                if not eligible:
                    continue
                heights = [
                    float(
                        -(np.asarray(s["normal_uvz"])[:2] @ corner + s["offset_m"])
                        / s["normal_uvz"][2]
                    )
                    for s in eligible
                ]
                if np.ptp(heights) > self.config["geometry"]["plane_inlier_m"]:
                    row.update(
                        z_state="AMBIGUOUS",
                        z_status="Z_AMBIGUOUS",
                        reason="MULTIPLE_CURRENT_SAME_ROLE_HEIGHT_MODES",
                        z_candidates=[
                            dict(
                                z=z,
                                source_segment_ids=s.get(
                                    "source_segment_ids", [s["segment_id"]]
                                ),
                                fit_residual=s["fit_residual"],
                                plane_role=s["plane_role"],
                            )
                            for z, s in zip(heights, eligible)
                        ],
                    )
                    break
                chosen = min(
                    eligible,
                    key=lambda s: (
                        s["fit_residual"],
                        s["current_corner_extrapolation_m"],
                        s["segment_id"],
                    ),
                )
                break
            if chosen:
                normal = np.asarray(chosen["normal_uvz"])
                z = float(-(normal[:2] @ corner + chosen["offset_m"]) / normal[2])
                covariance = chosen.get("plane_parameter_covariance")
                fit_sigma = extra_sigma = z_sigma = None
                if covariance is not None:
                    cov = np.asarray(covariance)
                    station = np.r_[
                        corner - np.asarray(chosen["plane_parameter_origin_uv"]), 1.0
                    ]
                    prediction_variance = float(max(0, station @ cov @ station))
                    fit_sigma = float(np.sqrt(max(0, cov[2, 2])))
                    extra_sigma = float(
                        np.sqrt(max(0, prediction_variance - fit_sigma**2))
                    )
                    xy_cov = result.get("corners", [{}] * 4)[index].get(
                        "corner_xy_model_covariance"
                    )
                    if xy_cov is not None:
                        gradient = -normal[:2] / normal[2]
                        z_sigma = float(
                            np.sqrt(
                                max(
                                    0,
                                    prediction_variance
                                    + gradient @ np.asarray(xy_cov) @ gradient,
                                )
                            )
                        )
                measured = chosen["tier"] == "DIRECT_RIM_TOP"
                row.update(
                    z_m=z,
                    z=z,
                    z_state="MEASURED" if measured else "ESTIMATED",
                    z_status="Z_MEASURED" if measured else "Z_ESTIMATED",
                    z_observed=z if measured else None,
                    z_estimate=None if measured else z,
                    z_source=chosen["tier"],
                    z_uncertainty_m=z_sigma,
                    source_point_indexes=chosen["source_point_indexes"],
                    source_segment_ids=chosen.get(
                        "source_segment_ids", [chosen["segment_id"]]
                    ),
                    fit_extent=chosen["fit_extent"],
                    fit_residual=chosen["fit_residual"],
                    plane_role=chosen["plane_role"],
                    segment_vote_count=chosen.get("segment_vote_count", 1),
                    extrapolation_distance_m=chosen["current_corner_extrapolation_m"],
                    fit_uncertainty_m=fit_sigma,
                    extrapolation_uncertainty_m=extra_sigma,
                    final_uncertainty_m=z_sigma,
                    uncertainty_status=(
                        "PROPAGATED_MODEL_COVARIANCE"
                        if z_sigma is not None
                        else "XY_MODEL_COVARIANCE_PENDING"
                    ),
                    local_plane_uvz=dict(
                        normal=normal.tolist(), offset_m=chosen["offset_m"]
                    ),
                    reason="CURRENT_OWNED_ROLE_QUALIFIED_RIM_TOP_INDEPENDENT_OF_XY_CERTIFICATION",
                )
            deck = [s for s in sources if self._eligible(s, owned, DECK_ROLES)]
            deck_source = cross_segment_consensus(
                deck, corner, self.config, self.measurement.local.vessel.vessel_id
            )
            if deck_source:
                n = np.asarray(deck_source["normal_uvz"])
                extent = np.asarray(deck_source["fit_extent"])
                extra = float(
                    np.linalg.norm(
                        np.maximum(
                            0, np.maximum(extent[0] - corner, corner - extent[1])
                        )
                    )
                )
                if extra <= self.config["roi"]["support_search_m"]:
                    row.update(
                        deck_reference_z=float(
                            -(n[:2] @ corner + deck_source["offset_m"]) / n[2]
                        ),
                        deck_reference_semantic="DECK_REFERENCE_Z",
                        deck_reference_source=deck_source,
                    )
            if chosen is None:
                row.setdefault(
                    "reason",
                    (
                        "DECK_REFERENCE_ONLY_NOT_RIM_TOP"
                        if row["deck_reference_z"] is not None
                        else "NO_RELIABLE_CURRENT_RIM_TOP_SOURCE"
                    ),
                )
            output.append(row)
        self.audit.update(
            z_measured_count=sum(r["z_state"] == "MEASURED" for r in output),
            z_estimated_count=sum(r["z_state"] == "ESTIMATED" for r in output),
            z_ambiguous_count=sum(r["z_state"] == "AMBIGUOUS" for r in output),
            z_unobserved_count=sum(r["z_state"] == "UNOBSERVED" for r in output),
        )
        return output
