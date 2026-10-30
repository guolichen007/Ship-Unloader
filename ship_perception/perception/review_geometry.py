"""Complete review geometry from established owned opening identities only.

Review geometry, this-frame measurement and steel certification are separate.
No recall, historic coordinates, or rectangle Cartesian products occur here.
"""

import copy
import hashlib
import math
import numpy as np

from .candidate_union import canonical
from .review_frame import line_points, line_displacements, line_coordinate_at

SIDES = (("U0", 0, 0, -1), ("U1", 0, 2, 1), ("V0", 1, 1, -1), ("V1", 1, 3, 1))
INTERSECTIONS = (("U0", "V0"), ("U1", "V0"), ("U1", "V1"), ("U0", "V1"))
SOURCE_PRIORITY = {
    name: i
    for i, name in enumerate(
        (
            "MEASURED_RAW",
            "TOPOLOGY_SUPPORTED",
            "S4_FAMILY_SUPPORTED",
            "CONSERVATIVE_REGION_BOUND",
            "UNOBSERVED_CONSERVATIVE",
        )
    )
}


def infer_fourth_corner(corners_uv):
    """Infer only the missing corner of three independently supported corners."""
    if len(corners_uv) != 3 or not set(corners_uv).issubset(range(4)):
        return None
    missing = next(i for i in range(4) if i not in corners_uv)
    point = (
        np.asarray(corners_uv[(missing - 1) % 4])
        + np.asarray(corners_uv[(missing + 1) % 4])
        - np.asarray(corners_uv[(missing + 2) % 4])
    )
    return dict(
        corner_index=missing,
        point_uv=point.tolist(),
        corner_state="GEOMETRICALLY_INFERRED",
        observed=False,
        steel_certified=False,
    )


class ReviewGeometryEligibility:
    @staticmethod
    def assess(region, axis_status="AXIS_ESTIMATED"):
        role = region.identity_role
        member = region.identity_evidence.get("member_support_cells_rc", [])
        has_direction = axis_status == "AXIS_ESTIMATED" and any(
            s.state in ("SUPPORTING", "STRONG")
            for s in region.side_topology_support.values()
        )
        has_extent = bool(member or region.component_bounds)
        eligible = role in (
            "CONFIRMED_INDEPENDENT_OPENING",
            "CONFIRMED_CONNECTED_OPENING",
        )
        return dict(
            eligible=bool(eligible and has_direction and has_extent),
            identity_role=role,
            stable_longitudinal_constraint=has_direction,
            independent_transverse_extent=has_extent,
            reason=(
                "OWNED_CURRENT_OPENING_TOPOLOGY"
                if eligible and has_direction and has_extent
                else "COMPOUND_OR_UNASSIGNED_OR_OPEN_ENVELOPE"
            ),
        )


def geometry_consistency(sides, config):
    points = []
    failures = []
    for a, b in INTERSECTIONS:
        rows = [sides[a], sides[b]]
        normals = np.asarray([r["normal_uv"] for r in rows], float)
        if abs(np.linalg.det(normals)) < np.finfo(float).eps:
            return dict(valid=False, reason="ILL_CONDITIONED_CORNER", polygon_uv=None)
        p = np.linalg.solve(normals, np.asarray([r["final_rho"] for r in rows]))
        points.append(p)
        for r in rows:
            direction = np.array([-r["normal_uv"][1], r["normal_uv"][0]])
            along = float(p @ direction)
            lo, hi = sorted(r["support_span"])
            extra = max(0.0, lo - along, along - hi)
            uncertainty = r["uncertainty_m"]
            if (
                uncertainty is None
                or extra > config["boundary"]["corner_join_m"] + 2 * uncertainty
            ):
                failures.append("ILL_CONDITIONED_CORNER")
    p = np.asarray(points)
    e = np.roll(p, -1, axis=0) - p
    cross = np.cross(e, np.roll(e, -1, axis=0))
    area = float(np.cross(p, np.roll(p, -1, axis=0)).sum() / 2)
    if not np.isfinite(p).all() or area <= 0 or not (cross > 0).all():
        failures.append("NONCONVEX_OR_SELF_INTERSECTING_OR_INVALID_ORDER")
    angles = []
    for i in range(4):
        a, b = p[(i - 1) % 4] - p[i], p[(i + 1) % 4] - p[i]
        den = np.linalg.norm(a) * np.linalg.norm(b)
        angle = (
            math.degrees(math.acos(float(np.clip(a @ b / den, -1, 1)))) if den else 0.0
        )
        angles.append(angle)
        if not 85.0 <= angle <= 95.0:
            failures.append("REVIEW_CORNER_ANGLE_OUTSIDE_85_95")
    return dict(
        valid=not failures,
        reason=failures[0] if failures else "VALID_REVIEW_GEOMETRY",
        failures=failures,
        polygon_uv=p.tolist(),
        corner_angles_deg=angles,
        area_m2=area,
    )


class ReviewGeometryAssembler:
    def __init__(self, config):
        self.config = config
        self.assembly_operation_count = 0
        self.four_line_cartesian_enumeration_count = 0

    def _prior(self, region, name, axis, pos, sign):
        box = region.conservative_bounds
        normal = np.eye(2)[axis]
        direction = np.array([-normal[1], normal[0]])
        endpoints = np.array([[box[0], box[1]], [box[2], box[3]]]) @ direction
        uncertainty = self.config["geometry"]["coarse_voxel_m"] / 2
        return dict(
            side_id=name,
            final_rho=float(box[pos]),
            final_state="CONSERVATIVE_PRIOR",
            final_source="CONSERVATIVE_REGION_BOUND",
            normal_uv=normal.tolist(),
            support_span=sorted(endpoints.tolist()),
            rho_before=float(box[pos]),
            rho_after=float(box[pos]),
            movement_m=0.0,
            candidate_modes=[],
            selected_mode_id=None,
            rejected_mode_ids=[],
            movement_reason="CURRENT_REGION_CONSERVATIVE_BOUND",
            guard_result="KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY",
            raw_support_ids=[],
            s4_family_ids=[],
            topology_support_ids=list(region.source_proposal_ids),
            height_support_ids=list(
                region.identity_evidence.get("member_height_nodes", [])
            ),
            uncertainty_m=float(uncertainty),
            independent_geometry_support=False,
            axis=axis,
            position=pos,
            outward_sign=sign,
            comparison_points_uv=(
                normal * box[pos] + np.asarray(sorted(endpoints))[:, None] * direction
            ).tolist(),
            coordinate_frame="VESSEL_REVIEW_UV",
            rho_contract="dot(point_review,normal_uv)=final_rho",
        )

    def _apply_mode(self, side, mode):
        self.assembly_operation_count += 1
        side["candidate_modes"].append(copy.deepcopy(mode))
        priority = SOURCE_PRIORITY[mode["source"]]
        old_priority = SOURCE_PRIORITY[side["final_source"]]
        if priority > old_priority:
            side["rejected_mode_ids"].append(mode["mode_id"])
            return
        same = [m for m in side["candidate_modes"] if m["source"] == mode["source"]]
        if len(same) > 1 and any(
            np.max(np.abs(line_displacements(m, mode, side["comparison_points_uv"])))
            > self.config["boundary"]["corner_join_m"]
            for m in same
        ):
            if priority == 0:
                prior = side.get("conservative_review_line", {})
                side["final_state"] = "MULTIMODE"
                side["final_source"] = prior.get(
                    "final_source", "CONSERVATIVE_REGION_BOUND"
                )
                side["final_rho"] = prior.get("final_rho", side["rho_before"])
                side["normal_uv"] = prior.get(
                    "normal_uv", np.eye(2)[side["axis"]].tolist()
                )
                side["support_span"] = prior.get("support_span", side["support_span"])
                side["uncertainty_m"] = prior.get(
                    "uncertainty_m", side["uncertainty_m"]
                )
                side["independent_geometry_support"] = False
                side["selected_mode_id"] = None
                side["movement_reason"] = "EQUAL_PRIORITY_RAW_MODE_CONFLICT"
                side["guard_result"] = "KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY"
            return
        prior = side.get(
            "conservative_review_line",
            dict(
                normal_uv=np.eye(2)[side["axis"]].tolist(), final_rho=side["rho_before"]
            ),
        )
        displacements = line_displacements(prior, mode, side["comparison_points_uv"])
        move = float(displacements[np.argmax(abs(displacements))])
        inward = float(max(0.0, np.max(-side["outward_sign"] * displacements)))
        gain = mode.get("independent_raw_sections", 0) >= self.config["boundary"][
            "profile_min_sections"
        ] and mode.get("physical_role_improved", False)
        if (
            inward > self.config["boundary"]["corner_join_m"]
            or abs(move) > self.config["roi"]["boundary_search_m"] / 2
        ) and not gain:
            side["rejected_mode_ids"].append(mode["mode_id"])
            side["movement_reason"] = "LARGE_MOVE_REQUIRES_STRUCTURAL_CORROBORATION"
            return
        side.update(
            final_rho=float(mode["rho"]),
            normal_uv=list(mode["normal_uv"]),
            support_span=list(mode["support_span"]),
            final_state=mode["state"],
            final_source=mode["source"],
            rho_after=float(mode["rho"]),
            movement_m=float(move),
            endpoint_displacements_m=displacements.tolist(),
            movement_definition="DISPLACEMENT_ALONG_PREVIOUS_NORMAL_AT_FIXED_SUPPORT_STATIONS",
            uncertainty_m=float(mode["uncertainty_m"]),
            selected_mode_id=mode["mode_id"],
            movement_reason=(
                "CURRENT_INDEPENDENT_RAW_GAIN"
                if gain
                else "CURRENT_SUPPORTED_REVIEW_LINE"
            ),
            guard_result="MOVEMENT_SUPPORTED" if gain else "SMALL_SUPPORTED_MOVE",
            independent_geometry_support=priority == 0,
            raw_support_ids=list(mode.get("raw_support_ids", [])),
            s4_family_ids=list(mode.get("s4_family_ids", [])),
        )

    def assemble(self, region, axes, bridge=None, topology_sides=None):
        bridge = bridge or {}
        eligibility = ReviewGeometryEligibility.assess(
            region,
            bridge.get(
                "review_axis_status",
                bridge.get("structural_axis", {}).get("status", "AXIS_ESTIMATED"),
            ),
        )
        if not eligibility["eligible"]:
            return None, eligibility
        sides = {
            name: self._prior(region, name, axis, pos, sign)
            for name, axis, pos, sign in SIDES
        }
        for name, modes in (topology_sides or {}).items():
            for mode in modes:
                self._apply_mode(sides[name], mode)
        topology_prior = copy.deepcopy(sides)
        for family in bridge.get("side_families", []):
            self.assembly_operation_count += 1
            for name, axis, pos, sign in SIDES:
                self.assembly_operation_count += 1
                if family["direction_axis"] == axis:
                    continue
                midpoint = np.asarray(region.conservative_bounds[:2]) + np.asarray(
                    region.conservative_bounds[2:]
                )
                midpoint /= 2
                midpoint[axis] = region.conservative_bounds[pos]
                normal = np.asarray(family["normal_uv"])
                rho = float(family["rho"])
                if normal[axis] < 0:
                    normal = -normal
                    rho = -rho
                if (
                    abs(normal @ midpoint - rho)
                    > self.config["roi"]["support_search_m"]
                ):
                    continue
                direction = np.array([-normal[1], normal[0]])
                source_direction = np.asarray(family["direction_uv"])
                span = np.asarray(family["support_span"]) * float(
                    direction @ source_direction
                )
                mode = dict(
                    mode_id=family["family_id"],
                    rho=rho,
                    normal_uv=normal.tolist(),
                    support_span=sorted(span.tolist()),
                    source="S4_FAMILY_SUPPORTED",
                    state="TOPOLOGY_SUPPORTED",
                    uncertainty_m=family["rho_uncertainty"],
                    s4_family_ids=[family["family_id"]],
                )
                # Multiple S4 nominations remain candidates; only a unique
                # boundary mode can replace the current conservative line.
                sides[name]["candidate_modes"].append(mode)
        for side in sides.values():
            modes = side.pop("candidate_modes")
            side["candidate_modes"] = []
            if (
                modes
                and np.ptp(
                    [
                        line_coordinate_at(
                            m,
                            np.mean(side["comparison_points_uv"], axis=0),
                            side["axis"],
                        )
                        for m in modes
                    ]
                )
                <= self.config["boundary"]["corner_join_m"]
            ):
                self._apply_mode(side, modes[0])
                side["candidate_modes"] = modes
            else:
                side["candidate_modes"] = modes
                if modes:
                    side["movement_reason"] = "MULTIPLE_S4_MODES_PRIOR_PRESERVED"
        result = self._result(region, axes, sides, fallback_sides=topology_prior)
        for side in result["sides"].values():
            side["region_rho_prior"] = side["rho_before"]
            side["rho_before"] = side["final_rho"]
            side["conservative_review_line"] = {
                k: copy.deepcopy(side[k])
                for k in [
                    "final_rho",
                    "final_source",
                    "final_state",
                    "normal_uv",
                    "support_span",
                    "uncertainty_m",
                ]
            }
            side["comparison_points_uv"] = line_points(side).tolist()
        result["conservative_review_sides"] = copy.deepcopy(result["sides"])
        return result, eligibility

    def update(self, region, result, modes, axes):
        sides = copy.deepcopy(result["sides"])
        for name, candidates in modes.items():
            for mode in candidates:
                self._apply_mode(sides[name], mode)
        return self._result(
            region,
            axes,
            sides,
            fallback_sides=result.get("conservative_review_sides", result["sides"]),
        )

    def _result(self, region, axes, sides, fallback_sides=None):
        measurement_evidence = copy.deepcopy(sides)
        check = geometry_consistency(sides, self.config)
        self.assembly_operation_count += 12
        failed_checks = []
        if not check["valid"]:
            failed_checks.append(copy.deepcopy(check))
            # Evidence remains available even when its heterogeneous line
            # intersections are pathological. Never discard the opening.
            for name, axis, pos, sign in SIDES:
                previous = sides[name]
                fallback = (
                    copy.deepcopy(fallback_sides[name])
                    if fallback_sides
                    else self._prior(region, name, axis, pos, sign)
                )
                fallback.update(
                    candidate_modes=previous["candidate_modes"],
                    rejected_mode_ids=previous["rejected_mode_ids"],
                    guard_result=check["reason"],
                    movement_reason="CONSERVATIVE_GEOMETRY_FALLBACK",
                )
                sides[name] = fallback
            check = geometry_consistency(sides, self.config)
        if not check["valid"] and fallback_sides:
            failed_checks.append(copy.deepcopy(check))
            # A topology line can itself have insufficient support extent.
            # The independently established region boundary is the final
            # conservative fallback; never reuse an invalid topology prior.
            for name, axis, pos, sign in SIDES:
                previous = sides[name]
                fallback = self._prior(region, name, axis, pos, sign)
                fallback.update(
                    candidate_modes=previous["candidate_modes"],
                    rejected_mode_ids=previous["rejected_mode_ids"],
                    guard_result=check["reason"],
                    movement_reason="REGION_GEOMETRY_FALLBACK",
                )
                sides[name] = fallback
            check = geometry_consistency(sides, self.config)
        if not check["valid"]:
            raise ValueError(
                "STOP_A_LEGAL_OPENING_REVIEW_GEOMETRY_INVALID:" + region.region_id
            )
        check["rejected_geometry_checks"] = failed_checks
        polygon = np.asarray(check["polygon_uv"])
        xy = polygon @ np.asarray(axes)
        coverage = copy.deepcopy(sides)
        for name, _, _, sign in SIDES:
            coverage[name]["final_rho"] += sign * sides[name]["uncertainty_m"]
        cover = geometry_consistency(coverage, self.config)
        cover_xy = (
            (np.asarray(cover["polygon_uv"]) @ np.asarray(axes)).tolist()
            if cover["valid"]
            else xy.tolist()
        )
        for side in sides.values():
            side.update(
                rho_before_m=side["rho_before"],
                rho_after_m=side["final_rho"],
                boundary_uncertainty_m=side["uncertainty_m"],
                grid_phase=dict(status="NOT_STEEL_CERTIFICATION"),
            )
        corners = [
            dict(
                corner_id="P" + str(i + 1),
                xy_state=(
                    "MEASURED"
                    if all(sides[n]["independent_geometry_support"] for n in pair)
                    else "ESTIMATED"
                ),
                xy_source="LINE_INTERSECTION",
                corner_state="LINE_INTERSECTION",
                uncertainty_m=max(sides[n]["uncertainty_m"] for n in pair),
                observed=all(sides[n]["independent_geometry_support"] for n in pair),
                steel_certified=False,
            )
            for i, pair in enumerate(INTERSECTIONS)
        ]
        return dict(
            opening_identity=region.region_id,
            hatch_id="H" + hashlib.sha256(region.region_id.encode()).hexdigest()[:8],
            vessel_hypothesis_id=region.vessel_id,
            status="UNRESOLVED",
            candidate_type="HATCH_REVIEW_CANDIDATE",
            individual_hatch_count=1,
            topology_status=region.identity_role,
            geometry_status="COMPLETE_REVIEW_GEOMETRY",
            certification_status="UNRESOLVED",
            geometry_role="COMPLETE_REVIEW_NOT_CERTIFIED_STEEL",
            steel_edge_certified=False,
            boundary_uv=polygon.tolist(),
            boundary_xy=xy.tolist(),
            best_estimate_boundary_xy=xy.tolist(),
            review_polygon_xy=xy.tolist(),
            review_coverage_boundary_xy=cover_xy,
            review_coverage_polygon_xy=cover_xy,
            sides=sides,
            corners=corners,
            per_side_source={k: v["final_source"] for k, v in sides.items()},
            boundary_measurements=measurement_evidence,
            conservative_review_sides=copy.deepcopy(fallback_sides or sides),
            per_side_uncertainty={k: v["uncertainty_m"] for k, v in sides.items()},
            source_proposal_ids=list(region.source_proposal_ids),
            vessel_local_axes=np.asarray(axes).tolist(),
            independently_supported_sides=[
                k for k, v in sides.items() if v["independent_geometry_support"]
            ],
            review_support_side_count=sum(
                s["independent_geometry_support"] for s in sides.values()
            ),
            undercoverage_risk=any(
                not s["independent_geometry_support"] for s in sides.values()
            ),
            coverage_status="CURRENT_MEMBER_SUPPORT_REVIEW_NOT_STEEL",
            geometry_consistency=check,
            review_assembly=dict(
                assembly_operation_count=self.assembly_operation_count,
                four_line_cartesian_enumeration_count=0,
            ),
        )
