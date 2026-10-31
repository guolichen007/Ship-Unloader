"""Current owned rim-top sources for the existing corner-height consumer.

This probes existing opening sides; it cannot create or move XY geometry.
Interpolated line stations are never reported as original Raw observations.
"""

import hashlib
import numpy as np

from .sections import physical_sections

SIDE_LAYOUT = (("U0", 0, -1), ("U1", 0, 1), ("V0", 1, -1), ("V1", 1, 1))


def owned_measurement_mask(measurement):
    if not hasattr(measurement, "_height_owned_mask"):
        primary = getattr(measurement.local, "primary_cache", measurement.local)
        measurement._height_owned_mask = np.isin(
            measurement.raw_point_indexes,
            primary.vessel.point_indexes,
            assume_unique=True,
        )
    return measurement._height_owned_mask


def absolute_surface_evidence(measurement, axis, rho, span, inward_sign, config):
    """Attach absolute Z/ownership to a selected profile, without changing it."""
    primary = getattr(
        getattr(measurement, "local", None),
        "primary_cache",
        getattr(measurement, "local", None),
    )
    if not all(
        hasattr(measurement, name)
        for name in ("spatial", "raw_uvz", "raw_point_indexes")
    ) or not hasattr(getattr(primary, "vessel", None), "point_indexes"):
        return dict(
            absolute_z_scope="UNAVAILABLE_ORIGINAL_POINT_OWNERSHIP",
            inside_surface_z=None,
            outside_surface_z=None,
            upper_surface_z=None,
            source_point_indexes=[],
            upper_surface_point_indexes=[],
        )
    width = config["boundary"]["side_probe_m"]
    box = [0.0] * 4
    box[axis], box[axis + 2] = rho - width, rho + width
    box[1 - axis], box[3 - axis] = span
    ids = measurement.spatial.query_indexes(box)
    ids = ids[owned_measurement_mask(measurement)[ids]]
    points = measurement.raw_uvz[ids]
    distance = inward_sign * (points[:, axis] - rho)
    inside = distance >= 0
    outside = distance < 0
    minimum = config["boundary"]["profile_min_points"]
    values = [points[m, 2] for m in (inside, outside)]
    heights = [float(np.median(v)) if len(v) >= minimum else None for v in values]
    available = [v for v in heights if v is not None]
    upper = max(available) if available else None
    top = (
        np.zeros(len(ids), dtype=bool)
        if upper is None
        else abs(points[:, 2] - upper) <= config["geometry"]["plane_inlier_m"]
    )
    return dict(
        absolute_z_scope="CURRENT_VESSEL_OWNED_RAW_ONLY",
        inside_surface_z=heights[0],
        outside_surface_z=heights[1],
        upper_surface_z=upper,
        inside_surface_point_count=int(inside.sum()),
        outside_surface_point_count=int(outside.sum()),
        source_point_indexes=measurement.raw_point_indexes[ids].astype(int).tolist(),
        upper_surface_point_indexes=measurement.raw_point_indexes[ids[top]]
        .astype(int)
        .tolist(),
        upper_surface_role="UNCLASSIFIED_UPPER_OF_CURRENT_PROFILE_NOT_RIM_CERTIFICATION",
    )


class RimTopHeightEstimator:
    def __init__(self, measurement, config, plane_fitter):
        self.measurement = measurement
        self.config = config
        self.plane_fitter = plane_fitter
        self.audit = dict(
            probed_side_count=0,
            probed_section_count=0,
            role_qualified_candidate_count=0,
            rim_candidate_count=0,
            deck_candidate_count=0,
            rejected_sections=[],
            section_summaries=[],
            local_rim_candidates_before_persistence=0,
        )

    def _reference(self, result, name):
        rectangle = result.get("constrained_review_rectangle", {})
        measured = rectangle.get("sides", {}).get(name, {}).get("measured_line")
        raw = result.get("raw_evidence_geometry", {})
        side = raw.get("sides", {}).get(name)
        if measured:
            return measured, measured
        if side is None:
            # Older callers can still provide pre-rectangle evidence directly.
            side = result.get("boundary_measurements", {}).get(name)
        if side is None:
            return None, None
        selected = next(
            (
                m
                for m in side.get("candidate_modes", [])
                if m.get("mode_id") == side.get("selected_mode_id")
                and m.get("source") == "MEASURED_RAW"
                and m.get("mode_id") not in side.get("rejected_mode_ids", [])
                and not m.get("ownership_rejected")
            ),
            None,
        )
        return selected or side, selected

    def _strip(self, normal, rho, span, outward_sign):
        tangent = np.array([-normal[1], normal[0]])
        width = self.config["roi"]["deck_patch_width_m"]
        probe = self.config["boundary"]["side_probe_m"]
        corners = [
            normal * (rho + outward_sign * distance) + tangent * along
            for distance in (-probe, width + probe)
            for along in span
        ]
        corners = np.asarray(corners)
        ids = self.measurement.spatial.query_indexes([*corners.min(0), *corners.max(0)])
        ids = ids[owned_measurement_mask(self.measurement)[ids]]
        points = self.measurement.raw_uvz[ids]
        along = points[:, :2] @ tangent
        distance = outward_sign * (points[:, :2] @ normal - rho)
        keep = (along >= span[0]) & (along <= span[1])
        return ids[keep], points[keep], distance[keep], along[keep]

    def collect(self, result):
        sources = []
        minimum = self.config["boundary"]["profile_min_points"]
        votes = self.config["boundary"]["profile_min_sections"]
        tolerance = self.config["geometry"]["plane_inlier_m"]
        width = self.config["roi"]["deck_patch_width_m"]
        drop = self.config["roi"]["opening_drop_m"]
        for name, _, sign in SIDE_LAYOUT:
            reference, measured = self._reference(result, name)
            if reference is None:
                continue
            normal = np.asarray(reference["normal_uv"], dtype=float)
            rho = reference.get("rho", reference.get("final_rho"))
            edges = physical_sections(
                reference["support_span"],
                self.config["geometry"]["refine_voxel_m"],
                self.config,
            )
            if len(edges) <= votes:
                continue
            self.audit["probed_side_count"] += 1
            candidates = []
            for section_number, span in enumerate(zip(edges[:-1], edges[1:])):
                self.audit["probed_section_count"] += 1
                ids, points, distance, _ = self._strip(normal, rho, span, sign)
                near = (distance >= 0) & (distance <= width)
                inside = (distance < 0) & (
                    distance >= -self.config["boundary"]["side_probe_m"]
                )
                far = (distance > width) & (
                    distance <= width + self.config["boundary"]["side_probe_m"]
                )
                segment_id = (
                    result["opening_identity"]
                    + ":"
                    + name
                    + ":HEIGHT_SECTION:"
                    + str(section_number)
                )
                planes = self.plane_fitter(
                    points[near],
                    self.measurement.raw_point_indexes[ids[near]],
                    segment_id,
                    "UNCLASSIFIED_SURFACE",
                )
                summary = dict(
                    side=name,
                    segment_id=segment_id,
                    near_point_count=int(near.sum()),
                    inside_point_count=int(inside.sum()),
                    far_point_count=int(far.sum()),
                    plane_candidate_count=len(planes),
                    qualified_roles=[],
                )
                self.audit["section_summaries"].append(summary)
                for plane in planes:
                    n = np.asarray(plane["normal_uvz"])
                    predicted = -(points[:, :2] @ n[:2] + plane["offset_m"]) / n[2]
                    below = predicted - points[:, 2]
                    opening_drop = bool(
                        inside.sum() >= minimum
                        and np.quantile(below[inside], 0.25) >= drop
                    )
                    edge_top = (
                        near & (abs(distance) <= tolerance) & (abs(below) <= tolerance)
                    )
                    direct_edge = (
                        int(edge_top.sum())
                        >= self.config["boundary"]["line_min_points"]
                    )
                    lip_above_deck = bool(
                        far.sum() >= minimum and np.quantile(below[far], 0.25) >= drop
                    )
                    continuous_deck = bool(
                        far.sum() >= minimum
                        and np.quantile(abs(below[far]), 0.75) <= tolerance
                    )
                    wall_connection = False
                    if (
                        measured
                        and measured.get("independent_opening_role_gain")
                        and measured.get("source_point_indexes")
                    ):
                        wall_ids = np.isin(
                            self.measurement.raw_point_indexes[ids],
                            measured["source_point_indexes"],
                        )
                        wall_connection = (
                            int(np.count_nonzero(wall_ids & (abs(below) <= tolerance)))
                            >= self.config["boundary"]["line_min_points"]
                        )
                    rim = (opening_drop and (direct_edge or lip_above_deck)) or (
                        wall_connection and lip_above_deck
                    )
                    deck = opening_drop and continuous_deck and not rim
                    if not rim and not deck:
                        self.audit["rejected_sections"].append(
                            dict(
                                side=name,
                                segment_id=segment_id,
                                reason="NO_ROLE_QUALIFIED_RIM_OR_DECK_CONNECTION",
                                opening_drop=opening_drop,
                                edge_top_point_count=int(edge_top.sum()),
                                lip_above_deck=lip_above_deck,
                                continuous_deck=continuous_deck,
                            )
                        )
                        continue
                    role = "RIM_TOP" if rim else "TRUE_DECK"
                    summary["qualified_roles"].append(role)
                    self.audit["local_rim_candidates_before_persistence"] += int(rim)
                    plane.update(
                        plane_role=role,
                        role_confirmed=True,
                        source_roles=[role],
                        side_id=name,
                        physical_role_evidence=dict(
                            opening_drop=opening_drop,
                            edge_top_point_count=int(edge_top.sum()),
                            lip_above_deck=lip_above_deck,
                            current_opening_face_connection=wall_connection,
                            continuous_deck=continuous_deck,
                            source="CURRENT_OWNED_SIDE_STRIP_WITH_INSIDE_OUTSIDE_ROLE",
                        ),
                    )
                    candidates.append(plane)
            for role in ("RIM_TOP", "TRUE_DECK"):
                eligible = [p for p in candidates if p["plane_role"] == role]
                segments = {p["segment_id"] for p in eligible}
                if len(segments) < votes:
                    continue
                for plane in eligible:
                    plane.update(
                        tier="LOCAL_RIM_PLANE" if role == "RIM_TOP" else "DECK_PLANE",
                        source_segment_ids=sorted(segments),
                        segment_vote_count=len(segments),
                        longitudinal_persistence="CURRENT_INDEPENDENT_SIDE_SECTIONS",
                    )
                    sources.append(plane)
                # Fit the same-role union using the existing plane capability.
                # Only actual accepted Raw indexes are reused; no synthetic
                # support is created between independently observed sections.
                ids = np.unique(
                    [i for p in eligible for i in p["source_point_indexes"]]
                )
                order = np.argsort(self.measurement.raw_point_indexes)
                positions = np.searchsorted(
                    self.measurement.raw_point_indexes[order], ids
                )
                local = order[positions]
                merged = self.plane_fitter(
                    self.measurement.raw_uvz[local],
                    ids,
                    result["opening_identity"] + ":" + name + ":PERSISTENT_RIM",
                    role,
                )
                for plane in merged:
                    source_ids = set(plane["source_point_indexes"])
                    supporting = [
                        p
                        for p in eligible
                        if len(source_ids.intersection(p["source_point_indexes"]))
                        >= self.config["boundary"]["line_min_points"]
                    ]
                    support_segments = {p["segment_id"] for p in supporting}
                    if len(support_segments) < votes:
                        continue
                    plane.update(
                        tier=(
                            "SAME_SIDE_RIM_PLANE"
                            if role == "RIM_TOP"
                            else "VESSEL_DECK_REFERENCE"
                        ),
                        role_confirmed=True,
                        source_roles=[role],
                        side_id=name,
                        source_segment_ids=sorted(support_segments),
                        segment_vote_count=len(support_segments),
                        independent_segment_evidence=[
                            dict(
                                segment_id=p["segment_id"],
                                source_point_indexes=p["source_point_indexes"],
                            )
                            for p in supporting
                        ],
                        longitudinal_persistence="CURRENT_INDEPENDENT_SIDE_SECTIONS",
                    )
                    sources.append(plane)
            self.audit["role_qualified_candidate_count"] = len(sources)
        self.audit["rim_candidate_count"] = sum(
            p["plane_role"] == "RIM_TOP" for p in sources
        )
        self.audit["deck_candidate_count"] = sum(
            p["plane_role"] == "TRUE_DECK" for p in sources
        )
        self.audit["role_qualified_candidate_count"] = len(sources)
        self.audit["source_point_indexes_sha256"] = hashlib.sha256(
            np.asarray(
                sorted({i for p in sources for i in p["source_point_indexes"]}),
                dtype="<i8",
            ).tobytes()
        ).hexdigest()
        return sources
