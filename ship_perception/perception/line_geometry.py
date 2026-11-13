"""Intersections of measured side lines in the immutable vessel UV basis."""

import numpy as np


CORNER_PAIRS = (("U0", "V0"), ("U1", "V0"), ("U1", "V1"), ("U0", "V1"))
SIDE_SPECS = (("U0", 0, 0, 1), ("U1", 0, 2, -1), ("V0", 1, 1, 1), ("V1", 1, 3, -1))


def guard_line_move(
    before, rho, slope, reference, span, sign, sections, independent, config
):
    supported = sections >= config["boundary"]["profile_min_sections"] and independent
    proposed = rho + slope * (np.asarray(span) - reference)
    maximum_inward = float(np.max(sign * (proposed - before)))
    maximum_move = float(np.max(abs(proposed - before)))
    if maximum_move > config["boundary"]["corner_join_m"] and not supported:
        return before, 0.0, "KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY", True
    return (
        rho,
        slope,
        "RAW_GEOMETRY_REVIEW" if supported else "PARTIAL_RAW_SUPPORT",
        not supported,
    )


def line_value(line, along):
    return line["rho_at_reference_m"] + line["slope"] * (
        np.asarray(along) - line["reference_along_m"]
    )


def _fit_positions(fit, along):
    return [
        fit["rho_at_reference_m"] + slope * (along - fit["reference_along_m"])
        for slope in (fit["slope"], fit["low_slope"], fit["high_slope"])
    ]


def guarded_intersections(sides, guard_bounds, config):
    """Recheck actual endpoints; each unsupported side can be restored once."""
    for _ in range(len(SIDE_SPECS) + 1):
        best, _, _ = intersect_final_sides(sides, config)
        restored = False
        for name, axis, position, sign in SIDE_SPECS:
            side = sides[name]
            indexes = [i for i, pair in enumerate(CORNER_PAIRS) if name in pair]
            along = best[indexes, 1 - axis]
            values = line_value(side["line"], along)
            alternatives = [line_value(side["proposed_line"], along)]
            for run in side["phase_runs"]:
                fit = run["fit"]
                if fit["status"] == "CONSENSUS":
                    alternatives.extend(_fit_positions(fit, along))
            for mode in side["opening_role_reanchor"].get("measured_modes", []):
                fit = mode.get("fit")
                alternatives.extend(
                    _fit_positions(fit, along) if fit else [np.full(2, mode["rho"])]
                )
            side["boundary_uncertainty_m"] = max(
                side["boundary_uncertainty_m"],
                float(np.max(abs(np.asarray(alternatives) - values)))
                + config["geometry"]["plane_inlier_m"],
            )
            inward = float(np.max(sign * (values - guard_bounds[position])))
            side["side_max_endpoint_inward_move_m"] = inward
            if (
                np.max(abs(values - guard_bounds[position]))
                > config["boundary"]["corner_join_m"]
                and not side["independent_geometry_support"]
            ):
                side["boundary_uncertainty_m"] = max(
                    side["boundary_uncertainty_m"],
                    float(np.max(abs(values - guard_bounds[position]))),
                )
                side["line"].update(
                    slope=0.0,
                    rho_at_reference_m=guard_bounds[position],
                    source="CONSERVATIVE_AXIS_LINE",
                )
                side.update(
                    rho_after_m=guard_bounds[position],
                    side_inward_move_m=0.0,
                    status="KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY",
                    undercoverage_risk=True,
                )
                restored = True
        if not restored:
            break
    # Updated uncertainty must also pass through the final intersection matrix.
    return intersect_final_sides(sides, config)


def intersect_final_sides(sides, config):
    corners, radii, conditions, crossing_angles = [], [], [], []
    for u_name, v_name in CORNER_PAIRS:
        u, v = sides[u_name]["line"], sides[v_name]["line"]
        matrix = np.array([[1.0, -u["slope"]], [-v["slope"], 1.0]])
        if not np.isfinite(matrix).all():
            raise ValueError("INVALID_FINAL_SIDE_LINE")
        cosine = abs(np.dot(matrix[0], matrix[1])) / np.prod(
            np.linalg.norm(matrix, axis=1)
        )
        angle = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
        if angle < config["boundary"]["min_intersection_angle_deg"]:
            raise ValueError("UNSTABLE_FINAL_LINE_INTERSECTION")
        # Solve around a nearby origin, avoiding large world-coordinate
        # intercept cancellation. rho uncertainty is in each normal axis.
        origin = np.array([u["rho_at_reference_m"], v["rho_at_reference_m"]])
        offset = np.array(
            [
                u["slope"] * (origin[1] - u["reference_along_m"]),
                v["slope"] * (origin[0] - v["reference_along_m"]),
            ]
        )
        inverse = np.linalg.inv(matrix)
        corners.append(origin + inverse @ offset)
        radii.append(
            abs(inverse)
            @ np.array(
                [
                    sides[u_name]["boundary_uncertainty_m"],
                    sides[v_name]["boundary_uncertainty_m"],
                ]
            )
        )
        conditions.append(float(np.linalg.cond(matrix)))
        crossing_angles.append(angle)
    corners, radii = np.asarray(corners), np.asarray(radii)
    edges = np.roll(corners, -1, axis=0) - corners
    crosses = (
        edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    if not np.isfinite(corners).all() or np.any(crosses <= 0):
        raise ValueError("DEGENERATE_FINAL_BOUNDARY")
    angles = []
    for i in range(4):
        a, b = -edges[i - 1], edges[i]
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    parallel = []
    for i in (0, 1):
        a, b = edges[i], edges[i + 2]
        cosine = abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
        parallel.append(float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))))
    worst = int(np.argmax(abs(np.asarray(angles) - 90.0)))
    metrics = dict(
        orthogonality_deg=angles[worst],
        orthogonality_max_error_deg=abs(angles[worst] - 90.0),
        corner_angles_deg=angles,
        parallelism_deg=max(parallel),
        opposite_side_parallelism_deg=parallel,
        rectangle_constraint_applied=False,
        intersection_stable=True,
        intersection_min_angle_deg=min(crossing_angles),
        intersection_condition_numbers=conditions,
    )
    return corners, radii, metrics
