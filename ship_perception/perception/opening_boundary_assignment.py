"""Assign bounded Raw measurements to an existing opening, without recall.

A geometric step can belong to a quay or another object. Its inner material
must link to frozen current-opening support before that step moves this
opening's review side. Context returns never acquire Core ownership here.
"""

import copy
import math
import numpy as np
from scipy.spatial import cKDTree

from .measurement import review_normal_cache
from .opening_rim_source import (
    OpeningRimSource,
    member_mask,
    support_fingerprint,
    point_fingerprint,
)
from .raw_face_refinement import measured_faces, face_position, primary_core_attachment
from .sections import physical_sections


class OpeningBoundaryAssignment:
    def __init__(self, measurement, config):
        self.measurement, self.config = measurement, config
        self.connectivity = OpeningRimSource(
            measurement, config, None, self._nonvertical
        )
        self.raw = measurement.raw_uvz
        self.owned = self.connectivity.owned
        self.original = measurement.raw_point_indexes
        self._masks = {}
        self._no_height_barriers = np.zeros(len(self.raw), dtype=bool)

    def _nonvertical(self, indexes):
        normals, _ = review_normal_cache(self.measurement, self.config, context=True)
        canonical = normals.input_to_canonical[np.asarray(indexes, dtype=np.int64)]
        normals._estimate(canonical)
        return normals.valid[canonical] & (
            abs(normals.normals[canonical, 2])
            > math.sin(math.radians(self.config["geometry"]["normal_angle_deg"]))
        )

    def masks(self, region):
        if region.region_id not in self._masks:
            members = member_mask(self.raw, region)
            other = np.zeros(len(self.raw), dtype=bool)
            for candidate in getattr(self.measurement.local, "opening_identities", ()):
                if (
                    candidate.vessel_id != region.vessel_id
                    or candidate.region_id == region.region_id
                    or not candidate.identity_role.startswith("CONFIRMED")
                ):
                    continue
                mask = member_mask(self.raw, candidate)
                if mask is not None:
                    other |= mask
            self._masks[region.region_id] = members, other
        return self._masks[region.region_id]

    def profile(self, region, row):
        """Keep the Raw observation separate from its current-opening assignment.

        Only actual pair endpoints enter this graph. The measured outside
        return level bounds lower material; it is never a deck or corner Z.
        No subsequent height fit, final polygon, or historical output is read.
        """
        members, other = self.masks(region)
        proof = dict(
            state="UNASSIGNED_CURRENT_OPENING",
            opening_identity=region.region_id,
            frozen_member_support_sha256=support_fingerprint(region),
            supported_section_count=0,
            sections=[],
            core_promoted=False,
            corner_z_source=False,
            steel_certified=False,
        )
        if members is None:
            proof["reason"] = "FROZEN_CURRENT_OPENING_SUPPORT_UNAVAILABLE"
            return proof
        excluded = set((row.get("fit") or {}).get("outlier_section_indexes", []))
        seen_sections = set()
        for section_index, section in enumerate(row.get("sections", [])):
            self.connectivity.checkpoint()
            if section_index in excluded:
                proof["sections"].append(
                    dict(
                        section_index=section_index,
                        supported=False,
                        reason="FIT_OUTLIER_NOT_CURRENT_MEASUREMENT_SUPPORT",
                    )
                )
                continue
            trace = section.get("source_pair_provenance", {})
            pairs = trace.get("pairs", [])
            inside = sorted({i for pair in pairs for i in pair["inside_point_indexes"]})
            outside = sorted(
                {i for pair in pairs for i in pair["outside_point_indexes"]}
            )
            reference = trace.get("outside_surface_median_z")
            if not inside or not outside or reference is None:
                proof["sections"].append(
                    dict(
                        section_index=section_index,
                        supported=False,
                        reason="NO_RAW_PAIR_TRACE",
                    )
                )
                continue
            section_key = (trace["normal_axis"], tuple(trace["section_span_m"]))
            if section_key in seen_sections:
                proof["sections"].append(
                    dict(
                        section_index=section_index,
                        supported=False,
                        reason="REPEATED_PHYSICAL_SECTION_NO_INDEPENDENT_VOTE",
                    )
                )
                continue
            seen_sections.add(section_key)
            # This trace now has a decision consumer. Do not leave its original
            # diagnostic-only flags claiming it was unused by the detector.
            trace.update(
                decision_input=True,
                used_for_detection=True,
                decision_consumer="CURRENT_OPENING_BOUNDARY_ASSIGNMENT",
            )
            axis, inward = trace["normal_axis"], trace["inward_sign"]
            normal = np.eye(2)[axis]
            tangent = np.array([-normal[1], normal[0]])
            span = trace["section_span_m"]
            span = span if axis == 0 else [-span[1], -span[0]]
            rho, sign = trace["anchor_rho_m"], -inward
            upper = self.raw[self.connectivity.positions(outside)]
            bins = np.floor(
                (upper[:, :2] @ tangent - span[0])
                / self.config["boundary"]["max_support_gap_m"]
            ).astype(int)
            distance = sign * (upper[:, :2] @ normal - rho)
            patch = dict(
                normal=normal,
                tangent=tangent,
                rho=rho,
                sign=sign,
                span=span,
                inner={
                    int(b): float(distance[bins == b].min()) for b in np.unique(bins)
                },
                plane=dict(normal_uvz=[0.0, 0.0, 1.0], offset_m=-reference),
            )
            material = self.connectivity._low_attachment(
                patch,
                members,
                other,
                self._no_height_barriers,
                retain_attached=True,
            )
            attached = material.pop("attached_original_point_indexes", [])
            links = []
            if attached:
                targets = self.raw[self.connectivity.positions(inside)]
                q = self.raw[self.connectivity.positions(attached)]
                distances, nearest = cKDTree(q).query(targets, workers=1)
                live = distances <= self.config["boundary"]["max_support_gap_m"]
                links = [
                    dict(
                        inside_point_index=int(inside[i]),
                        attached_low_point_index=int(attached[nearest[i]]),
                        distance_m=float(distances[i]),
                    )
                    for i in np.flatnonzero(live)
                ]
            supported = len(links) >= self.config["boundary"]["line_min_points"]
            proof["sections"].append(
                dict(
                    section_index=section_index,
                    span=span,
                    supported=supported,
                    reference_z=reference,
                    reference_role="MEASURED_PROFILE_LEVEL_NOT_DECK_OR_CORNER_Z",
                    attached_inside_endpoint_count=len(links),
                    links=links,
                    low_material_proof=material,
                )
            )
            proof["supported_section_count"] += int(supported)
        if (
            proof["supported_section_count"]
            >= self.config["boundary"]["profile_min_sections"]
        ):
            proof["state"] = "ASSIGNED_CURRENT_OPENING"
        return proof

    def complete_faces(self, region, name, axis, position, span):
        """Investigate complete faces in the existing side ROI, not coarse peaks.

        A small observed Core patch can attach an independently qualified full
        face. Four independent sections must have this opening's material on
        the inner side; positive material outside disqualifies an internal cut.
        This proves a review measurement assignment, never steel certification.
        """
        members, other = self.masks(region)
        audit = dict(side=name, opening_identity=region.region_id, faces=[])
        if members is None:
            audit["reason"] = "FROZEN_CURRENT_OPENING_SUPPORT_UNAVAILABLE"
            return [], audit
        normals, _ = review_normal_cache(self.measurement, self.config, context=True)
        edges = physical_sections(
            span, self.config["geometry"]["refine_voxel_m"], self.config
        )
        spans = list(zip(edges[:-1], edges[1:]))
        faces = measured_faces(
            normals,
            axis,
            region.conservative_bounds[position],
            span,
            spans,
            self.config,
        )
        modes = []
        inward = 1 if name.endswith("0") else -1
        center = (
            region.conservative_bounds[axis] + region.conservative_bounds[axis + 2]
        ) / 2
        for face in faces:
            indexes = np.asarray(face["point_indexes"], dtype=np.int64)
            points = self.raw[indexes]
            original = self.original[indexes]
            own = self.owned[indexes]
            core = primary_core_attachment(
                points[own], original[own], axis, self.config
            )
            a, b = np.asarray(face["a_xy"]), np.asarray(face["b_xy"])
            tangent = (b - a) / np.linalg.norm(b - a)
            normal = np.array([tangent[1], -tangent[0]])
            if normal[axis] < 0:
                normal = -normal
            tangent = np.array([-normal[1], normal[0]])
            rho = float(normal @ a)
            distance = inward * (self.raw[:, :2] @ normal - rho)
            sections, intervals = [], []
            for lo, hi in spans:
                along = (self.raw[:, 1 - axis] >= lo) & (self.raw[:, 1 - axis] <= hi)
                probe = self.config["boundary"]["side_probe_m"]
                inside = along & (distance >= 0) & (distance <= probe)
                outside = along & (distance < 0) & (distance >= -probe)
                observed = points[
                    (points[:, 1 - axis] >= lo) & (points[:, 1 - axis] <= hi)
                ]
                count = int(np.count_nonzero(inside & self.owned & members & ~other))
                outer_count = int(np.count_nonzero(outside & self.owned & members))
                sections.append(
                    dict(
                        span=[float(lo), float(hi)],
                        inside_owned_member_points=count,
                        outside_owned_member_points=outer_count,
                        face_point_count=len(observed),
                    )
                )
                if len(observed) >= self.config["boundary"]["line_min_points"]:
                    values = observed[:, :2] @ tangent
                    intervals.append([float(values.min()), float(values.max())])
            minimum = self.config["boundary"]["profile_min_points"]
            supported = sum(
                s["inside_owned_member_points"] >= minimum
                and s["face_point_count"] >= self.config["boundary"]["line_min_points"]
                for s in sections
            )
            cuts_interior = (
                sum(s["outside_owned_member_points"] >= minimum for s in sections)
                >= self.config["boundary"]["profile_min_sections"]
            )
            eligible = bool(
                core["supported"]
                and supported >= self.config["boundary"]["profile_min_sections"]
                and not cuts_interior
                and (face_position(face, sum(span) / 2) - center) * inward < 0
            )
            assignment = dict(
                state=(
                    "ASSIGNED_CURRENT_OPENING"
                    if eligible
                    else "UNASSIGNED_CURRENT_OPENING"
                ),
                core_attachment=core,
                sections=sections,
                supported_section_count=supported,
                cuts_positive_opening_material=cuts_interior,
                frozen_member_support_sha256=support_fingerprint(region),
                core_promoted=False,
                steel_certified=False,
            )
            identifier = "OPENING_RAWFACE:" + point_fingerprint(original)
            audit["faces"].append(
                dict(
                    mode_id=identifier,
                    normal_uv=normal.tolist(),
                    rho=rho,
                    source_point_indexes=original.tolist(),
                    assignment=assignment,
                )
            )
            if not eligible:
                continue
            modes.append(
                dict(
                    mode_id=identifier,
                    source="MEASURED_RAW",
                    state="MEASURED_WEAK",
                    normal_uv=normal.tolist(),
                    rho=rho,
                    support_span=[
                        min(s[0] for s in intervals),
                        max(s[1] for s in intervals),
                    ],
                    measured_intervals=intervals,
                    uncertainty_m=max(
                        self.config["geometry"]["plane_inlier_m"],
                        float(np.quantile(abs(points[:, :2] @ normal - rho), 0.95)),
                    ),
                    independent_raw_sections=supported,
                    physical_role_improved=True,
                    independent_opening_role_gain=True,
                    opening_assignment_state=assignment["state"],
                    opening_assignment=assignment,
                    physical_role="CURRENT_PRIMARY_PATCH_ON_COMPLETE_FACE_WITH_ONE_SIDED_OWNED_OPENING_MATERIAL",
                    raw_support_ids=[identifier],
                    source_point_indexes=original.tolist(),
                    steel_certified=False,
                )
            )
        return modes, audit

    def corroborate_primary_faces(self, primary_modes, complete_modes):
        """Attach same-face role evidence without changing a measured Raw line.

        Shared original returns and the existing full-face line tolerance are
        required. A nearby different physical mode remains a competing mode;
        it is never averaged with or substituted for the primary measurement.
        """
        output = copy.deepcopy(primary_modes)
        consumed = set()
        for measured in output:
            indexes = set(measured.get("source_point_indexes", []))
            normal = np.asarray(measured["normal_uv"], dtype=float)
            for complete in complete_modes:
                shared = indexes.intersection(complete["source_point_indexes"])
                if len(shared) < self.config["boundary"]["line_min_points"]:
                    continue
                if float(normal @ complete["normal_uv"]) < math.cos(
                    math.radians(self.config["boundary"]["merge_angle_deg"])
                ):
                    continue
                points = self.raw[
                    self.connectivity.positions(complete["source_point_indexes"])
                ]
                maximum = float(np.max(abs(points[:, :2] @ normal - measured["rho"])))
                if maximum > self.config["boundary"]["line_inlier_m"]:
                    continue
                proof = copy.deepcopy(complete["opening_assignment"])
                proof.update(
                    corroborating_face_mode_id=complete["mode_id"],
                    shared_original_point_count=len(shared),
                    shared_original_point_index_sha256=point_fingerprint(shared),
                    complete_face_max_distance_to_measured_line_m=maximum,
                )
                measured.update(
                    opening_assignment_state="ASSIGNED_CURRENT_OPENING",
                    opening_assignment=proof,
                    independent_opening_role_gain=True,
                    physical_role_gain_basis="OBSERVED_SAME_FACE_PRIMARY_PATCH_AND_ONE_SIDED_OPENING_MATERIAL",
                )
                measured.setdefault("same_face_corroborations", []).append(
                    copy.deepcopy(complete)
                )
                consumed.add(complete["mode_id"])
        output.extend(
            copy.deepcopy(m) for m in complete_modes if m["mode_id"] not in consumed
        )
        return output
