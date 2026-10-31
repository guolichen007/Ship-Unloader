"""Five-parameter review rectangles fitted to existing, owned side evidence.

The measured lines remain immutable. This model has no recall, historical
geometry, size prior, corner snapping or rectangle enumeration capability.
"""

import copy
import math
import numpy as np
from scipy.optimize import minimize

SIDE_LAYOUT = (("U0", 0, -1), ("U1", 0, 1), ("V0", 1, -1), ("V1", 1, 1))
INCIDENT = {"U0": (0, 3), "U1": (1, 2), "V0": (0, 1), "V1": (3, 2)}


def rectangle_audit(points):
    """Floating-point shape integrity, independent of physical accuracy."""
    p = np.asarray(points, dtype=float)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        return dict(rectangularity_gate_pass=False, reason="INVALID_RECTANGLE_POINTS")
    edges = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 0):
        return dict(rectangularity_gate_pass=False, reason="DEGENERATE_RECTANGLE")
    unit = edges / lengths[:, None]
    angles = np.degrees(
        np.arccos(np.clip(np.sum(-np.roll(unit, 1, axis=0) * unit, axis=1), -1, 1))
    )
    parallel = [
        math.atan2(
            abs(float(np.cross(unit[i], unit[i + 2]))),
            abs(float(unit[i] @ unit[i + 2])),
        )
        for i in range(2)
    ]
    diagonal = [float(np.linalg.norm(p[2] - p[0])), float(np.linalg.norm(p[3] - p[1]))]
    closure = float(np.linalg.norm((p[0] - p[1]) + (p[2] - p[3])))
    midpoint = closure / 2
    orthogonal = float(np.max(abs(np.sum(unit * np.roll(unit, -1, axis=0), axis=1))))
    row = dict(
        max_right_angle_error_deg=float(np.max(abs(angles - 90))),
        parallel_error_12_34_deg=math.degrees(parallel[0]),
        parallel_error_23_41_deg=math.degrees(parallel[1]),
        max_parallel_error_deg=math.degrees(max(parallel)),
        orthogonality_abs_dot=orthogonal,
        opposite_length_error_long_m=abs(float(lengths[0] - lengths[2])),
        opposite_length_error_short_m=abs(float(lengths[1] - lengths[3])),
        diagonal_13_m=diagonal[0],
        diagonal_24_m=diagonal[1],
        diagonal_length_error_m=abs(diagonal[0] - diagonal[1]),
        diagonal_midpoint_error_m=midpoint,
        parallelogram_closure_error_m=closure,
        polygon_area_m2=float(
            np.cross(p - p[0], np.roll(p, -1, axis=0) - p[0]).sum() / 2
        ),
    )
    row.update(
        {
            "side_length_" + name + "_m": float(value)
            for name, value in zip(("12", "23", "34", "41"), lengths)
        }
    )
    row.update(
        {
            "corner_angle_p" + str(i + 1) + "_deg": float(value)
            for i, value in enumerate(angles)
        }
    )
    row["rectangularity_gate_pass"] = bool(
        orthogonal <= 1e-10
        and max(parallel) <= 1e-10
        and midpoint <= 1e-8 * max(1.0, *diagonal)
        and row["opposite_length_error_long_m"]
        <= 1e-8 * max(1.0, lengths[0], lengths[2])
        and row["opposite_length_error_short_m"]
        <= 1e-8 * max(1.0, lengths[1], lengths[3])
        and row["polygon_area_m2"] > 0
    )
    return row


def line_points(line):
    normal = np.asarray(line["normal_uv"], dtype=float)
    normal = normal / np.linalg.norm(normal)
    rho = float(line.get("rho", line.get("final_rho")))
    tangent = np.array([-normal[1], normal[0]])
    # Evaluation stations describe the fitted line; they are not new Raw points.
    span = line.get("support_span", [0.0, 0.0])
    return rho * normal + np.linspace(min(span), max(span), 3)[:, None] * tangent


def selected_measurement(side):
    if side.get("final_state") == "MULTIMODE":
        return None
    return next(
        (
            m
            for m in side.get("candidate_modes", [])
            if m["mode_id"] == side.get("selected_mode_id")
            and m["source"] == "MEASURED_RAW"
            and m["mode_id"] not in side.get("rejected_mode_ids", [])
            and not m.get("ownership_rejected")
        ),
        None,
    )


def evidence_group(name, side, config):
    measured = selected_measurement(side)
    line = measured or side
    selected = next(
        (
            m
            for m in side.get("candidate_modes", [])
            if m["mode_id"] == side.get("selected_mode_id")
        ),
        {},
    )
    if measured:
        rank, source = 0, "MEASURED_RAW"
    elif selected.get("boundary_role_evidence") or any(
        "BEV_RECTILINEAR_PROVIDER" in value
        for value in side.get("topology_support_ids", [])
    ):
        rank, source = 1, "BEV_RASTER_SUPPORTED"
    elif side.get("s4_family_ids"):
        rank, source = 2, "S4_SUPPORTED"
    elif side.get("height_support_ids"):
        rank, source = 3, "HEIGHT_TOPOLOGY_SUPPORTED"
    else:
        rank, source = 4, "CONSERVATIVE_REGION_BOUND"
    sigma = max(
        config["geometry"]["plane_inlier_m"],
        float(line.get("uncertainty_m") or config["geometry"]["coarse_voxel_m"] / 2),
    )
    return dict(
        side=name,
        line=copy.deepcopy(line),
        points=line_points(line),
        sigma_m=sigma,
        source=source,
        source_rank=rank,
        source_weight=2.0**-rank,
        independent_opening_role_gain=bool(
            selected.get("independent_opening_role_gain")
        ),
        measured=copy.deepcopy(measured),
    )


def separator_constraints(region, bridge):
    """Consume frozen ownership; never create, move or connect separators."""
    result = []
    for value in region.identity_evidence.get(
        "side_ownership_constraints", {}
    ).values():
        for key in ("lower_u", "upper_u"):
            if key in value:
                result.append(
                    dict(
                        bound=key,
                        value=float(value[key]),
                        source="EXISTING_SIDE_OWNERSHIP",
                        evidence=copy.deepcopy(value),
                    )
                )
    members = region.identity_evidence.get("member_height_nodes", [])
    for part in bridge.get("node_partition_audit", []):
        sides = {c["side"] for c in part["children"] if c["node_id"] in members}
        if sides == {0}:
            result.append(
                dict(
                    bound="upper_u",
                    value=float(part["separator_faces"][0]),
                    source="CURRENT_S4_PARTITION_AUDIT",
                )
            )
        if sides == {1}:
            result.append(
                dict(
                    bound="lower_u",
                    value=float(part["separator_faces"][1]),
                    source="CURRENT_S4_PARTITION_AUDIT",
                )
            )
    return result


class ConstrainedRectangleSolver:
    def __init__(self, config):
        self.config = config

    def solve(self, region, sides, bridge=None):
        groups = [
            evidence_group(name, sides[name], self.config) for name, _, _ in SIDE_LAYOUT
        ]
        box = np.asarray(region.conservative_bounds, dtype=float)
        origin = (box[:2] + box[2:]) / 2
        scale = max(float(np.linalg.norm(box[2:] - box[:2])), 1.0)
        angles, weights = [], []
        for (_, axis, _), group in zip(SIDE_LAYOUT, groups):
            normal = np.asarray(group["line"]["normal_uv"])
            angles.append(math.atan2(normal[1], normal[0]) - axis * math.pi / 2)
            span = np.linalg.norm(group["points"][-1] - group["points"][0])
            weights.append(
                group["source_weight"]
                * max(span, self.config["geometry"]["plane_inlier_m"]) ** 2
                / group["sigma_m"] ** 2
            )
        theta = float(
            np.angle(np.sum(np.asarray(weights) * np.exp(2j * np.asarray(angles)))) / 2
        )
        # Start within the orientation consensus of protected conservative
        # spans when they exist. Starting at an incompatible Raw angle can
        # leave SLSQP outside the feasible coverage cone. This initialization
        # is current-evidence based and commutes with rigid transformations;
        # theta remains a joint unknown in the actual objective.
        protected = []
        for (name, axis, _), group in zip(SIDE_LAYOUT, groups):
            if group["measured"] is None and not group["independent_opening_role_gain"]:
                normal = np.asarray(sides[name]["normal_uv"])
                angle = math.atan2(normal[1], normal[0]) - axis * math.pi / 2
                span = np.linalg.norm(
                    line_points(sides[name])[-1] - line_points(sides[name])[0]
                )
                protected.append(
                    (angle, max(span, self.config["geometry"]["plane_inlier_m"]) ** 2)
                )
        if protected:
            theta = float(
                np.angle(
                    sum(weight * np.exp(2j * angle) for angle, weight in protected)
                )
                / 2
            )
        eu = np.array([math.cos(theta), math.sin(theta)])
        ev = np.array([-eu[1], eu[0]])
        coordinates = [
            float(np.mean((g["points"] - origin) @ (eu if axis == 0 else ev)))
            for (_, axis, _), g in zip(SIDE_LAYOUT, groups)
        ]
        for i, ((name, axis, sign), group) in enumerate(zip(SIDE_LAYOUT, groups)):
            if group["independent_opening_role_gain"]:
                continue
            candidate = eu if axis == 0 else ev
            den = float(candidate @ sides[name]["normal_uv"])
            if den <= 0:
                continue
            projected = (line_points(sides[name]) - origin) @ candidate
            inward = (
                self.config["boundary"]["corner_join_m"]
                if group["measured"] is not None
                else 0.0
            )
            outward = self.config["roi"]["boundary_search_m"] / 2
            lower = float(np.max(projected - (outward if sign < 0 else inward) * den))
            upper = float(np.min(projected + (inward if sign < 0 else outward) * den))
            if lower <= upper:
                coordinates[i] = float(np.clip(coordinates[i], lower, upper))
        u0, u1, v0, v1 = coordinates
        if u1 <= u0 or v1 <= v0:
            raise ValueError("RECTANGLE_SIDE_ORDER_CONFLICT:" + region.region_id)
        initial = np.array(
            [
                theta,
                (u0 + u1) / (2 * scale),
                (v0 + v1) / (2 * scale),
                math.log((u1 - u0) / scale),
                math.log((v1 - v0) / scale),
            ]
        )

        def model(x):
            a = np.array([math.cos(x[0]), math.sin(x[0])])
            b = np.array([-a[1], a[0]])
            length, width = scale * np.exp(x[3:5])
            center = origin + scale * (x[1] * a + x[2] * b)
            polygon = np.array(
                [
                    center - length / 2 * a - width / 2 * b,
                    center + length / 2 * a - width / 2 * b,
                    center + length / 2 * a + width / 2 * b,
                    center - length / 2 * a + width / 2 * b,
                ]
            )
            rhos = [
                float(center @ a - length / 2),
                float(center @ a + length / 2),
                float(center @ b - width / 2),
                float(center @ b + width / 2),
            ]
            return a, b, length, width, center, polygon, rhos

        def residuals(x):
            a, b, length, width, _, _, _ = model(x)
            values = []
            jacobian = []
            for (_, axis, sign), group in zip(SIDE_LAYOUT, groups):
                normal, derivative = (a, b) if axis == 0 else (b, -a)
                size = length if axis == 0 else width
                points = group["points"] - origin
                weight = (
                    math.sqrt(group["source_weight"] / len(points)) / group["sigma_m"]
                )
                values.extend(
                    weight * (points @ normal - scale * x[1 + axis] - sign * size / 2)
                )
                rows = np.zeros((len(points), 5))
                rows[:, 0] = points @ derivative
                rows[:, 1 + axis] = -scale
                rows[:, 3 + axis] = -sign * size / 2
                jacobian.extend(rows * weight)
            return np.asarray(values), np.asarray(jacobian)

        def objective(x):
            r, jac = residuals(x)
            root = np.sqrt(1 + r * r)
            return float(np.mean(2 * (root - 1))), jac.T @ (2 * r / root) / len(r)

        ownership = separator_constraints(region, bridge or {})

        def constraint_values_and_jacobian(x):
            a, b, length, width, _, polygon, _ = model(x)
            values, derivatives = [], []
            corner_jacobian = np.zeros((4, 2, 5))
            for i, (su, sv) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
                corner_jacobian[i, :, 0] = (
                    scale * (x[1] * b - x[2] * a)
                    + su * length / 2 * b
                    - sv * width / 2 * a
                )
                corner_jacobian[i, :, 1] = scale * a
                corner_jacobian[i, :, 2] = scale * b
                corner_jacobian[i, :, 3] = su * length / 2 * a
                corner_jacobian[i, :, 4] = sv * width / 2 * b
            for value in ownership:
                sign = 1 if value["bound"] == "lower_u" else -1
                values.extend(sign * (polygon[:, 0] - value["value"]))
                derivatives.extend(sign * corner_jacobian[:, 0])
            for index, ((name, axis, sign), group) in enumerate(
                zip(SIDE_LAYOUT, groups)
            ):
                if group["independent_opening_role_gain"]:
                    continue
                previous = sides[name]
                normal = np.asarray(previous["normal_uv"])
                candidate = a if axis == 0 else b
                candidate_derivative = b if axis == 0 else -a
                den = float(candidate @ normal)
                if den <= 0:
                    values.append(-1.0)
                    derivatives.append(np.zeros(5))
                    continue
                points = line_points(previous) - origin
                size = length if axis == 0 else width
                numerator = scale * x[1 + axis] + sign * size / 2 - points @ candidate
                displacement = numerator / den
                jacobian = np.zeros((len(points), 5))
                jacobian[:, 0] = (
                    -points @ candidate_derivative * den
                    - numerator * (candidate_derivative @ normal)
                ) / (den * den)
                jacobian[:, 1 + axis] = scale / den
                jacobian[:, 3 + axis] = sign * size / (2 * den)
                # Existing conservative coverage and large-move protection,
                # now applied during the joint solve rather than after snap.
                # A model constraint supplies no new inward evidence. Keep
                # conservative/unobserved support stations covered; only an
                # actual measured side may use the existing small-move gate.
                allowed_inward = (
                    self.config["boundary"]["corner_join_m"]
                    if group["measured"] is not None
                    else 0.0
                )
                values.extend(allowed_inward + sign * displacement)
                derivatives.extend(sign * jacobian)
                values.extend(
                    self.config["roi"]["boundary_search_m"] / 2 - sign * displacement
                )
                derivatives.extend(-sign * jacobian)
            return (
                (np.asarray(values), np.asarray(derivatives))
                if values
                else (np.array([1.0]), np.zeros((1, 5)))
            )

        def inequalities(x):
            return constraint_values_and_jacobian(x)[0]

        result = minimize(
            objective,
            initial,
            jac=True,
            method="SLSQP",
            bounds=[
                (theta - math.pi / 2, theta + math.pi / 2),
                (None, None),
                (None, None),
                (-30, 30),
                (-30, 30),
            ],
            constraints=[
                dict(
                    type="ineq",
                    fun=lambda x: inequalities(x) / scale,
                    jac=lambda x: constraint_values_and_jacobian(x)[1] / scale,
                )
            ],
            options=dict(ftol=1e-10, maxiter=256),
        )
        a, b, length, width, center, polygon, rhos = model(result.x)
        audit = rectangle_audit(polygon)
        numeric_tolerance = 1e-8 * scale
        feasible = bool(np.min(inequalities(result.x)) >= -numeric_tolerance)
        if not feasible or not audit["rectangularity_gate_pass"]:
            raise ValueError("RECTANGLE_CONSTRAINED_SOLVER_INVALID:" + region.region_id)
        output_sides = {}
        conflicts = []
        for i, ((name, axis, _), group) in enumerate(zip(SIDE_LAYOUT, groups)):
            normal = a if axis == 0 else b
            tangent = np.array([-normal[1], normal[0]])
            evidence = group["line"]
            old_normal = np.asarray(evidence["normal_uv"])
            denominator = abs(float(normal @ old_normal))
            shifts = (rhos[i] - group["points"] @ normal) / max(
                denominator, np.finfo(float).eps
            )
            angle_delta = math.degrees(
                math.atan2(
                    abs(float(np.cross(old_normal, normal))),
                    abs(float(old_normal @ normal)),
                )
            )
            shift = float(np.max(abs(shifts)))
            conflict = bool(
                group["measured"] and shift > group["sigma_m"] + numeric_tolerance
            )
            if conflict:
                conflicts.append(name)
            measured_line = copy.deepcopy(group["measured"])
            output_sides[name] = dict(
                normal_uv=normal.tolist(),
                final_rho=rhos[i],
                support_span=sorted((polygon[list(INCIDENT[name])] @ tangent).tolist()),
                measured_line=measured_line,
                measured_angle=(
                    math.atan2(old_normal[1], old_normal[0]) if measured_line else None
                ),
                measured_rho=(
                    evidence.get("rho", evidence.get("final_rho"))
                    if measured_line
                    else None
                ),
                constrained_review_line=dict(
                    normal_uv=normal.tolist(),
                    rho=rhos[i],
                    coordinate_frame="VESSEL_REVIEW_UV",
                ),
                constrained_angle=math.atan2(normal[1], normal[0]),
                constrained_rho=rhos[i],
                constraint_shift_m=shift,
                constraint_endpoint_shifts_m=shifts.tolist(),
                constraint_angle_delta_deg=angle_delta,
                rectangle_constraint_status=(
                    "RECTANGLE_CONSTRAINT_CONFLICT"
                    if conflict
                    else (
                        "WITHIN_CURRENT_EVIDENCE_UNCERTAINTY"
                        if shift <= group["sigma_m"] + numeric_tolerance
                        else "CONSERVATIVE_REVIEW_CONSTRAINT"
                    )
                ),
                input_evidence_source=group["source"],
                input_source_rank=group["source_rank"],
                evidence_uncertainty_m=group["sigma_m"],
            )
        return dict(
            model="FIVE_DOF_ORTHOGONAL_RECTANGLE",
            coordinate_frame="VESSEL_REVIEW_UV",
            theta_rad=float(result.x[0]),
            theta_deg=math.degrees(result.x[0]),
            center_uv=center.tolist(),
            length_m=float(length),
            width_m=float(width),
            u0=rhos[0],
            u1=rhos[1],
            v0=rhos[2],
            v1=rhos[3],
            polygon_uv=polygon.tolist(),
            sides=output_sides,
            rectangularity=audit,
            constraint_conflict_sides=conflicts,
            boundary_measurement_status=(
                "UNRESOLVED" if conflicts or not result.success else "PARTIAL"
            ),
            steel_certified=False,
            solver=dict(
                success=bool(result.success),
                status=int(result.status),
                message=str(result.message),
                iterations=int(result.nit),
                objective=float(result.fun),
                loss="SOFT_L1_CURRENT_UNCERTAINTY_NORMALIZED",
                source_rank_weights=[1, 0.5, 0.25, 0.125, 0.0625],
                operation_count=int(result.nfev)
                * sum(len(g["points"]) for g in groups),
                four_line_cartesian_enumeration_count=0,
                geometry_input="EXISTING_SAME_IDENTITY_SIDE_LINES_AND_OBSERVED_INTERVALS_NOT_PREVIOUS_POLYGON",
                origin_uv=origin.tolist(),
            ),
            separator_constraints=ownership,
            minimum_guard_margin_m=float(np.min(inequalities(result.x))),
        )


def constrain_review_geometry(region, hatch, config, bridge=None):
    """Final review model, separate from every original measurement and guard."""
    rectangle = ConstrainedRectangleSolver(config).solve(region, hatch["sides"], bridge)
    result = copy.deepcopy(hatch)
    result["raw_evidence_geometry"] = {
        key: copy.deepcopy(hatch[key])
        for key in (
            "boundary_uv",
            "boundary_xy",
            "sides",
            "corners",
            "geometry_consistency",
            "boundary_measurements",
        )
    }
    axes = np.asarray(hatch["vessel_local_axes"])
    xy = np.asarray(rectangle["polygon_uv"]) @ axes
    rectangle["polygon_xy"] = xy.tolist()
    result["constrained_review_rectangle"] = rectangle
    for name, axis, sign in SIDE_LAYOUT:
        side = result["sides"][name]
        constrained = rectangle["sides"][name]
        side["evidence_source_before_rectangle"] = side["final_source"]
        side["evidence_state_before_rectangle"] = side["final_state"]
        side.update(copy.deepcopy(constrained))
        side.update(
            final_source="CONSTRAINED_REVIEW_RECTANGLE",
            final_state="GEOMETRY_CONSTRAINED",
            independent_geometry_support=False,
            measured_intervals=[],
            measured_interval=[],
            conservative_extension_interval=[constrained["support_span"]],
            uncertainty_m=max(side["uncertainty_m"], constrained["constraint_shift_m"]),
            movement_reason="JOINT_FIVE_DOF_RECTANGLE_CONSTRAINT",
            guard_result=constrained["rectangle_constraint_status"],
        )
        # The actual Raw tangent intervals remain on measured_line and in the
        # untouched boundary_measurements; they are never relabeled as model
        # observations after the shared normal has changed.
        side["boundary_uncertainty_m"] = side["uncertainty_m"]
    for key in ("boundary_xy", "best_estimate_boundary_xy", "review_polygon_xy"):
        result[key] = xy.tolist()
    result["boundary_uv"] = copy.deepcopy(rectangle["polygon_uv"])
    theta = rectangle["theta_rad"]
    eu = np.array([math.cos(theta), math.sin(theta)])
    ev = np.array([-eu[1], eu[0]])
    values = {
        name: rectangle[name.lower()] + sign * result["sides"][name]["uncertainty_m"]
        for name, _, sign in SIDE_LAYOUT
    }
    coverage = (
        np.array(
            [
                values["U0"] * eu + values["V0"] * ev,
                values["U1"] * eu + values["V0"] * ev,
                values["U1"] * eu + values["V1"] * ev,
                values["U0"] * eu + values["V1"] * ev,
            ]
        )
        @ axes
    )
    result["review_coverage_boundary_xy"] = coverage.tolist()
    result["review_coverage_polygon_xy"] = coverage.tolist()
    intersections = (("U0", "V0"), ("U1", "V0"), ("U1", "V1"), ("U0", "V1"))
    for corner, pair in zip(result["corners"], intersections):
        corner.update(
            xy_state="ESTIMATED",
            xy_source="CONSTRAINED_RECTANGLE_SOLVER",
            corner_state="MODEL_CONSTRAINED_NOT_OBSERVED",
            observed=False,
            steel_certified=False,
            uncertainty_m=max(result["sides"][n]["uncertainty_m"] for n in pair),
        )
    result["geometry_consistency"] = dict(
        valid=rectangle["rectangularity"]["rectangularity_gate_pass"],
        reason="VALID_CONSTRAINED_REVIEW_RECTANGLE",
        polygon_uv=rectangle["polygon_uv"],
        corner_angles_deg=[
            rectangle["rectangularity"]["corner_angle_p" + str(i) + "_deg"]
            for i in range(1, 5)
        ],
        area_m2=rectangle["rectangularity"]["polygon_area_m2"],
        rectangularity=rectangle["rectangularity"],
    )
    result.update(
        boundary_measurement_status=rectangle["boundary_measurement_status"],
        rectangle_constraint_conflict_count=len(rectangle["constraint_conflict_sides"]),
        position_review_status="NOT_EVALUATED_RUNTIME",
        calibration_scope="XY_RECTANGLE_REVIEW",
        z_scope="Z_UNOBSERVED_UNLESS_SEPARATELY_OBSERVED",
        independently_supported_sides=[],
        review_support_side_count=0,
        undercoverage_risk=True,
        per_side_source={
            name: side["final_source"] for name, side in result["sides"].items()
        },
        per_side_uncertainty={
            name: side["uncertainty_m"] for name, side in result["sides"].items()
        },
    )
    result["review_assembly"]["rectangle_solver_operation_count"] = rectangle["solver"][
        "operation_count"
    ]
    return result
