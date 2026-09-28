"""Offline R2G refinement of existing R2F hatch rectangles.

This stage never generates a hatch proposal. Raw XYZ and current-scan 3D
observations refine one shared ship axis and the four opening-side line rhos.
"""

import itertools
import math
from collections import defaultdict

import numpy as np

from .boundary_topology import _interval_coverage, _observed_intervals
from .raw_topview_evidence import build_topview, raster_channels, raster_modes
from .rectangle_fusion import _opening_relation, _structural_modes


FORBIDDEN_ROLES = frozenset(("INTERIOR_SURFACE", "ROLE_AMBIGUOUS_OVERFLOW"))
SIDE_SPECS = (("U0", 0, 1), ("U1", 0, -1), ("V0", 1, 1), ("V1", 1, -1))


def _angle_delta_deg(angle, origin):
    return (math.degrees(angle - origin) + 45.0) % 90.0 - 45.0


def _weighted_median(values, weights):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2)])


def _edge_votes(edges, forbidden, initial_angle):
    by_node = defaultdict(list)
    for edge in edges:
        if edge.get("segment_id") in forbidden:
            continue
        delta = np.asarray(edge["b_raw"][:2]) - np.asarray(edge["a_raw"][:2])
        length = float(np.linalg.norm(delta))
        if length <= 0:
            continue
        angle = math.atan2(delta[1], delta[0])
        weight = min(float(edge["observed_support_length"]), 10.0)
        by_node[edge["node_id"]].append(
            (_angle_delta_deg(angle, initial_angle), weight)
        )
    votes = []
    contribution = {}
    for node_id, rows in by_node.items():
        total = sum(weight for _, weight in rows)
        if total <= 0:
            continue
        contribution[node_id] = 1.0
        votes.extend((angle, weight / total, node_id) for angle, weight in rows)
    return votes, contribution


def _raw_transition_votes(points, axes, rectangles, config):
    """Fit measured raster transition strips around existing sides only."""
    grid = build_topview(points, axes, config)["fine"]
    masks = raster_channels(grid, config)
    margin = config["boundary"]["corner_join_m"]
    votes = []
    for hatch in rectangles:
        axial = np.asarray(hatch["polygon_xy"]) @ np.asarray(axes).T
        box = (
            axial[:, 0].min(),
            axial[:, 0].max(),
            axial[:, 1].min(),
            axial[:, 1].max(),
        )
        for side_id, normal_axis, _ in SIDE_SPECS:
            rho = box[(0, 1, 2, 3)[("U0", "U1", "V0", "V1").index(side_id)]]
            along_span = (box[2], box[3]) if normal_axis == 0 else (box[0], box[1])
            mask = masks[normal_axis]["HEIGHT_TRANSITION"]
            rows, cols = np.nonzero(mask)
            normal = (
                grid.x0 + (cols + 1) * grid.cell_m
                if normal_axis == 0
                else grid.y0 + (rows + 1) * grid.cell_m
            )
            along = (
                grid.y0 + (rows + 0.5) * grid.cell_m
                if normal_axis == 0
                else grid.x0 + (cols + 0.5) * grid.cell_m
            )
            selected = (
                (np.abs(normal - rho) <= margin)
                & (along >= along_span[0])
                & (along <= along_span[1])
            )
            if selected.sum() < config["boundary"]["line_min_points"]:
                continue
            along, normal = along[selected], normal[selected]
            if np.ptp(along) < config["boundary"]["line_min_length_m"]:
                continue
            slope = float(np.polyfit(along, normal, 1)[0])
            # U sides have tangent V, V sides have tangent U.
            delta = math.degrees(math.atan(slope)) * (1 if normal_axis == 1 else -1)
            votes.append((delta, min(float(np.ptp(along)), 10.0), hatch["hatch_id"]))
    return votes


def diagnose_axes(points, segments, edges, rectangles, initial_axes, config):
    basis = np.asarray(initial_axes, dtype=float)
    if basis.shape != (2, 2) or not np.allclose(basis @ basis.T, np.eye(2), atol=1e-5):
        raise ValueError("INVALID_INITIAL_AXES")
    initial_angle = math.atan2(basis[0, 1], basis[0, 0])
    forbidden = {
        row["segment_id"]
        for row in segments
        if row.get("reference_role") in FORBIDDEN_ROLES
    }
    edge_votes, contributions = _edge_votes(edges, forbidden, initial_angle)
    raw_votes = _raw_transition_votes(points, basis, rectangles, config)
    tolerance = config["boundary"]["merge_angle_deg"]
    valid_edges = [(d, w) for d, w, _ in edge_votes if abs(d) <= tolerance]
    valid_raw = [(d, w) for d, w, _ in raw_votes if abs(d) <= tolerance]

    def consensus(rows):
        if not rows:
            return None
        values, weights = zip(*rows)
        return _weighted_median(values, weights)

    structural_delta = consensus(valid_edges)
    raw_delta = consensus(valid_raw)
    total_weight = sum(weight for _, weight, _ in edge_votes)
    if (
        raw_delta is not None
        and structural_delta is not None
        and abs(structural_delta - raw_delta) > tolerance / 2
    ):
        # A few long but wrong 3D lines may form a competing node. Keep the
        # per-node caps and identify the structural mode corroborated by the
        # independent RawXYZ transition orientation.
        corroborated = [
            (d, w) for d, w, _ in edge_votes if abs(d - raw_delta) <= tolerance / 2
        ]
        if sum(weight for _, weight in corroborated) >= 0.5 * total_weight:
            valid_edges = corroborated
            structural_delta = consensus(valid_edges)
    inlier_fraction = (
        sum(weight for _, weight in valid_edges) / total_weight if total_weight else 0.0
    )
    dispersion = (
        math.sqrt(
            sum(
                weight * (delta - structural_delta) ** 2
                for delta, weight in valid_edges
            )
            / sum(weight for _, weight in valid_edges)
        )
        if valid_edges
        else None
    )
    source_consistency = (
        raw_delta is None
        or structural_delta is None
        or abs(raw_delta - structural_delta) <= tolerance
    )
    if structural_delta is None or inlier_fraction < 0.5 or not source_consistency:
        state = "AXIS_AMBIGUOUS"
    elif dispersion is not None and dispersion > tolerance:
        state = "AXIS_AMBIGUOUS"
    elif (
        raw_delta is not None
        and abs(structural_delta) > config["frame"]["aligned_angle_max_deg"]
        and abs(raw_delta) > config["frame"]["aligned_angle_max_deg"]
        and structural_delta * raw_delta > 0
        and abs(structural_delta - raw_delta) <= tolerance / 2
    ):
        state = "AXIS_DRIFT_SUSPECTED"
    else:
        state = "AXIS_STABLE"
    delta = 0.0
    if state == "AXIS_DRIFT_SUSPECTED":
        delta = _weighted_median((structural_delta, raw_delta), (1.0, 1.0))
        delta = float(np.clip(delta, -tolerance, tolerance))
    refined_angle = initial_angle + math.radians(delta)
    axis_u = np.asarray((math.cos(refined_angle), math.sin(refined_angle)))
    axes = np.asarray((axis_u, (-axis_u[1], axis_u[0])))
    diagnostics = dict(
        initial_axis_angle_deg=math.degrees(initial_angle),
        refined_axis_angle_deg=math.degrees(refined_angle),
        axis_delta_deg=delta,
        profile_face_edge_orientation_consensus_deg=structural_delta,
        narrow_structure_orientation_consensus_deg=raw_delta,
        per_node_axis_contribution=contributions,
        axis_inlier_fraction=inlier_fraction,
        axis_dispersion_deg=dispersion,
        source_consistency=bool(source_consistency),
        status=state,
        structural_vote_count=len(edge_votes),
        raw_transition_vote_count=len(raw_votes),
    )
    return axes, diagnostics


def _coverage(values, span, config):
    return float(
        _interval_coverage(
            _observed_intervals(values, config["boundary"]["max_support_gap_m"]), *span
        )
    )


def _mode_profile(modes, observations, normal_axis, rho, along_span, config):
    tolerance = config["geometry"]["plane_inlier_m"]
    nearby = [
        mode
        for mode in modes
        if abs(mode["rho"] - rho) <= tolerance
        and mode["along_max"] >= along_span[0]
        and mode["along_min"] <= along_span[1]
    ]
    channels = {}
    for channel in (
        "OBSERVED_PROFILE_BREAK",
        "OBSERVED_3D_FACE",
        "HEIGHT_TRANSITION",
        "DENSITY_TRANSITION",
        "VERTICAL_SPAN_RIDGE",
    ):
        values = [
            value
            for mode in nearby
            if channel in mode["evidence_types"]
            for value in mode["along_values"]
            if along_span[0] <= value <= along_span[1]
        ]
        channels[channel] = _coverage(values, along_span, config)
    for channel in ("OBSERVED_PROFILE_BREAK", "OBSERVED_3D_FACE"):
        measured = [
            row["xy"][1 - normal_axis]
            for row in observations
            if row["reference_role"] not in FORBIDDEN_ROLES
            and row["evidence_type"] == channel
            and abs(row["xy"][normal_axis] - rho) <= config["boundary"]["line_inlier_m"]
            and along_span[0] <= row["xy"][1 - normal_axis] <= along_span[1]
        ]
        channels[channel] = max(
            channels[channel], _coverage(measured, along_span, config)
        )
    structural = max(channels["OBSERVED_PROFILE_BREAK"], channels["OBSERVED_3D_FACE"])
    raster = max(
        channels["HEIGHT_TRANSITION"],
        channels["DENSITY_TRANSITION"],
        channels["VERTICAL_SPAN_RIDGE"],
    )
    return channels, structural, raster


def _raw_edge_rho(axial_points, normal_axis, rho, along_span, interior_sign, config):
    """Continuous midpoint of paired surface returns across a measured step."""
    normal = axial_points[:, normal_axis]
    along = axial_points[:, 1 - normal_axis]
    z = axial_points[:, 2]
    width = config["geometry"]["candidate_voxel_m"]
    selected = (
        (np.abs(normal - rho) <= width)
        & (along >= along_span[0])
        & (along <= along_span[1])
    )
    if selected.sum() < config["boundary"]["line_min_points"]:
        return None, 0
    normal, along, z = normal[selected], along[selected], z[selected]
    signed = interior_sign * (normal - rho)
    # A raw return can lie exactly on a raster interface. It still belongs to
    # the outside surface and is needed for a sub-voxel paired midpoint.
    outside = z[signed <= 0]
    inside = z[signed > 0]
    if min(len(outside), len(inside)) < config["boundary"]["line_min_points"]:
        return None, int(selected.sum())
    outside_z, inside_z = float(np.median(outside)), float(np.median(inside))
    if abs(outside_z - inside_z) < config["boundary"]["profile_min_drop_m"]:
        return None, int(selected.sum())
    # Pair nearest unlike returns within each measured along-line interval.
    interval = config["boundary"]["max_support_gap_m"]
    bins = np.floor((along - along_span[0]) / interval).astype(int)
    midpoints = []
    for index in np.unique(bins):
        in_bin = bins == index
        high = in_bin & (np.abs(z - outside_z) < np.abs(z - inside_z))
        low = in_bin & ~high
        if not high.any() or not low.any():
            continue
        outer = signed[high]
        inner = signed[low]
        outer = outer[outer <= 0]
        inner = inner[inner >= 0]
        if len(outer) and len(inner):
            midpoints.append(rho + interior_sign * (outer.max() + inner.min()) / 2)
    # Local windows are shorter than a whole side; require several paired
    # along-line sections without imposing the full-side point count.
    if len(midpoints) < min(
        config["boundary"]["line_min_points"],
        config["boundary"]["profile_min_sections"],
    ):
        return None, int(selected.sum())
    return float(np.median(midpoints)), int(selected.sum())


def _continuous_rho(axial_points, observations, normal_axis, rho, span, sign, config):
    tolerance = config["boundary"]["line_inlier_m"]
    positions = [
        row["xy"]
        for row in observations
        if row["reference_role"] not in FORBIDDEN_ROLES
        and abs(row["xy"][normal_axis] - rho) <= tolerance
        and span[0] <= row["xy"][1 - normal_axis] <= span[1]
    ]
    structural_rho = None
    if len(positions) >= config["boundary"]["line_min_points"]:
        structural_rho = float(np.median(np.asarray(positions)[:, normal_axis]))
    raw_rho, raw_count = _raw_edge_rho(
        axial_points, normal_axis, rho, span, sign, config
    )
    if (
        structural_rho is not None
        and raw_rho is not None
        and abs(structural_rho - raw_rho) <= config["boundary"]["corner_join_m"]
    ):
        refined = float(np.median((structural_rho, raw_rho)))
    elif structural_rho is not None:
        refined = structural_rho
    elif raw_rho is not None:
        refined = raw_rho
    else:
        refined = float(rho)
    uncertainty = max(
        config["geometry"]["plane_inlier_m"],
        abs(
            (structural_rho if structural_rho is not None else refined)
            - (raw_rho if raw_rho is not None else refined)
        ),
    )
    return refined, float(uncertainty), raw_count, raw_rho is not None


def _side_candidates(
    side_id,
    normal_axis,
    sign,
    before,
    span,
    modes,
    grid,
    axial_points,
    observations,
    config,
):
    search = config["roi"]["boundary_search_m"]
    forbidden_initial_count = sum(
        row["reference_role"] in FORBIDDEN_ROLES
        and abs(row["xy"][normal_axis] - before) <= config["boundary"]["corner_join_m"]
        and span[0] <= row["xy"][1 - normal_axis] <= span[1]
        for row in observations
    )
    nearby = [
        mode
        for mode in modes
        if abs(mode["rho"] - before) <= search
        and mode["along_max"] >= span[0]
        and mode["along_min"] <= span[1]
    ]
    # The proposal stage already fixed the hatch. Reuse its eight-peak search
    # budget while also keeping the eight nearest distinct physical rims.
    ranked = sorted(nearby, key=lambda row: -row["support_count"])[:8]
    ranked += sorted(nearby, key=lambda row: abs(row["rho"] - before))[:8]
    # Precision uses the un-deduplicated modes. Only numerical near-duplicates
    # within plane-inlier tolerance are merged; distinct physical rims survive.
    values = [float(before)]
    for mode in sorted(ranked, key=lambda row: -row["support_count"]):
        if all(
            abs(mode["rho"] - previous) > config["geometry"]["plane_inlier_m"]
            for previous in values
        ):
            values.append(float(mode["rho"]))
    measured_rhos = sorted(
        row["xy"][normal_axis]
        for row in observations
        if row["reference_role"] not in FORBIDDEN_ROLES
        and abs(row["xy"][normal_axis] - before) <= search
        and span[0] <= row["xy"][1 - normal_axis] <= span[1]
    )
    if measured_rhos:
        clusters = [[measured_rhos[0]]]
        for observed_rho in measured_rhos[1:]:
            if observed_rho - clusters[-1][-1] > config["geometry"]["plane_inlier_m"]:
                clusters.append([])
            clusters[-1].append(observed_rho)
        for cluster in clusters:
            if len(cluster) >= config["boundary"]["line_min_points"]:
                peak = float(np.median(cluster))
                if all(
                    abs(peak - previous) > config["geometry"]["plane_inlier_m"]
                    for previous in values
                ):
                    values.append(peak)
    rows = []
    for rho in values:
        channels, structural, raster = _mode_profile(
            modes, observations, normal_axis, rho, span, config
        )
        relation = _opening_relation(grid, normal_axis, rho, span, sign, config)
        forbidden_count = sum(
            row["reference_role"] in FORBIDDEN_ROLES
            and abs(row["xy"][normal_axis] - rho)
            <= 2 * config["boundary"]["line_inlier_m"]
            and span[0] <= row["xy"][1 - normal_axis] <= span[1]
            for row in observations
        )
        refined, uncertainty, raw_count, raw_measured = _continuous_rho(
            axial_points, observations, normal_axis, rho, span, sign, config
        )
        supported = config["roi"]["min_enclosure_ratio"]
        multi = structural >= supported and raster >= supported
        height = channels["HEIGHT_TRANSITION"]
        face = channels["OBSERVED_3D_FACE"]
        structure_strip = max(
            face, channels["VERTICAL_SPAN_RIDGE"], channels["DENSITY_TRANSITION"]
        )
        if (
            max(forbidden_count, forbidden_initial_count)
            >= config["boundary"]["line_min_points"]
        ):
            role = "DISTANT_STRUCTURE"
        elif (
            abs(rho - before) > search
            or relation["far_support_fraction"] < config["roi"]["min_enclosure_ratio"]
        ):
            role = "DISTANT_STRUCTURE"
        elif (
            height >= supported
            and relation["canonical_inner"]
            and (structural >= supported or structure_strip >= supported)
        ):
            role = "INNER_EDGE"
        elif structural >= supported and relation["status"] == "OPENING_LOWER":
            role = "INNER_EDGE"
        elif (
            structural == 0
            and height > 0
            and structure_strip == 0
            and sign * (rho - before) > config["boundary"]["corner_join_m"]
        ):
            role = "CARGO_BOUNDARY"
        else:
            role = "AMBIGUOUS"
        rows.append(
            dict(
                side_id=side_id,
                rho_grid=float(rho),
                rho_refined=refined,
                refinement_delta_m=float(refined - rho),
                role=role,
                evidence_types=[
                    name for name, coverage in channels.items() if coverage > 0
                ],
                profile_break_coverage=channels["OBSERVED_PROFILE_BREAK"],
                face_coverage=face,
                height_transition_coverage=height,
                density_transition_coverage=channels["DENSITY_TRANSITION"],
                vertical_span_coverage=channels["VERTICAL_SPAN_RIDGE"],
                structure_strip_support=structure_strip,
                structural_coverage=structural,
                raster_coverage=raster,
                opening_side_relation=relation,
                raw_point_support=raw_count,
                raw_boundary_measured=raw_measured,
                uncertainty_m=uncertainty,
                forbidden_observation_count=forbidden_count,
            )
        )
    rows.sort(
        key=lambda row: (
            row["role"] == "INNER_EDGE",
            row["structural_coverage"] >= supported
            and row["raster_coverage"] >= supported,
            max(row["structural_coverage"], row["raster_coverage"]),
            row["opening_side_relation"]["status"] == "OPENING_LOWER",
            -row["uncertainty_m"],
            -abs(row["rho_refined"] - before),
        ),
        reverse=True,
    )
    # Always retain the initial line, even when its evidence is ambiguous.
    chosen = rows[:6]
    initial = next(row for row in rows if row["rho_grid"] == before)
    if initial not in chosen:
        chosen.append(initial)
    return chosen, rows


def _joint_score(sides, before, config, shared_width_target=None):
    rho = [row["rho_refined"] for row in sides]
    if min(rho[1] - rho[0], rho[3] - rho[2]) <= 2 * config["boundary"]["side_probe_m"]:
        return None
    if any(row["role"] in ("DISTANT_STRUCTURE", "CARGO_BOUNDARY") for row in sides):
        return None
    supported = config["roi"]["min_enclosure_ratio"]
    inferred = sum(
        max(row["structural_coverage"], row["raster_coverage"]) == 0 for row in sides
    )
    if inferred > config["boundary"]["max_inferred_edges"]:
        return None
    strengths = [
        max(row["structural_coverage"], row["raster_coverage"]) for row in sides
    ]
    return (
        sum(row["role"] == "INNER_EDGE" for row in sides),
        sum(
            row["structural_coverage"] >= supported
            and row["raster_coverage"] >= supported
            for row in sides
        ),
        round(min(strengths), 5),
        4 - inferred,
        sum(row["opening_side_relation"]["canonical_inner"] for row in sides),
        round(
            sum(
                row["structural_coverage"]
                for row in sides
                if row["structural_coverage"] >= supported
            ),
            5,
        ),
        round(sum(row["raster_coverage"] for row in sides), 5),
        -round(sum(row["uncertainty_m"] for row in sides), 5),
        (
            -round(abs((rho[3] - rho[2]) - shared_width_target), 5)
            if shared_width_target is not None
            and any(
                row["role"] != "INNER_EDGE" or row["structural_coverage"] < supported
                for row in sides[2:]
            )
            else 0.0
        ),
        -round(
            sum(
                abs(row["rho_refined"] - original)
                for row, original in zip(sides, before)
            ),
            5,
        ),
    )


def _ship_width_consensus(rectangles, axes, config):
    """Use only strong, mutually agreeing hatches from this scan."""
    supported = config["roi"]["min_enclosure_ratio"]
    strong_widths = []
    for hatch in rectangles:
        sides = {row["side_id"]: row for row in hatch["sides"]}
        if not all(
            sides[key]["structural_coverage"] >= supported
            and sides[key]["raster_coverage"] >= supported
            for key in ("V0", "V1")
        ):
            continue
        axial = np.asarray(hatch["polygon_xy"]) @ axes.T
        strong_widths.append(float(np.ptp(axial, axis=0)[1]))
    target = float(np.median(strong_widths)) if len(strong_widths) >= 2 else None
    mad = (
        float(np.median(np.abs(np.asarray(strong_widths) - target)))
        if target is not None
        else None
    )
    if mad is None or mad > config["boundary"]["corner_join_m"]:
        target = None
    return target, strong_widths, mad


def _local_rho_consensus(
    candidates,
    modes,
    observations,
    grid,
    axial_points,
    normal_axis,
    sign,
    before,
    span,
    config,
):
    """Measure a side in separated middle-body sections before moving it."""
    length = span[1] - span[0]
    trim = min(config["roi"]["support_search_m"], length / 4)
    start, end = span[0] + trim, span[1] - trim
    window_m = config["boundary"]["profile_half_length_m"]
    # Non-overlapping windows are independent evidence along the side.
    stride_m = max(window_m, config["boundary"]["line_min_length_m"])
    centers = np.arange(start + window_m / 2, end - window_m / 2 + 1e-9, stride_m)
    samples = []
    supported = config["roi"]["min_enclosure_ratio"]
    for center in centers:
        local_span = (float(center - window_m / 2), float(center + window_m / 2))
        possibilities = []
        for candidate in candidates:
            if candidate["role"] in ("DISTANT_STRUCTURE", "CARGO_BOUNDARY"):
                continue
            rho = candidate["rho_grid"]
            channels, structural, raster = _mode_profile(
                modes, observations, normal_axis, rho, local_span, config
            )
            if max(structural, raster) < supported:
                continue
            relation = _opening_relation(
                grid, normal_axis, rho, local_span, sign, config
            )
            score = (
                structural >= supported and raster >= supported,
                channels["HEIGHT_TRANSITION"] >= supported,
                relation["canonical_inner"],
                max(structural, raster),
                -abs(rho - before),
            )
            possibilities.append((score, rho, structural, raster))
        if not possibilities:
            samples.append(
                dict(along_center_m=float(center), status="REJECT_NO_LOCAL_BOUNDARY")
            )
            continue
        _, rho, structural, raster = max(possibilities, key=lambda row: row[0])
        refined, _, raw_count, raw_measured = _continuous_rho(
            axial_points, observations, normal_axis, rho, local_span, sign, config
        )
        samples.append(
            dict(
                along_center_m=float(center),
                rho_grid=float(rho),
                rho_measured=float(refined),
                structural_coverage=float(structural),
                raster_coverage=float(raster),
                raw_point_support=raw_count,
                raw_boundary_measured=raw_measured,
                status="LOCAL_MEASUREMENT",
            )
        )
    measured = [
        row["rho_measured"] for row in samples if row["status"] == "LOCAL_MEASUREMENT"
    ]
    if measured:
        median = float(np.median(measured))
        mad = float(np.median(np.abs(np.asarray(measured) - median)))
        threshold = max(config["boundary"]["corner_join_m"], 3 * mad)
        for row in samples:
            if row["status"] == "LOCAL_MEASUREMENT":
                row["status"] = (
                    "ACCEPTED"
                    if abs(row["rho_measured"] - median) <= threshold
                    else "REJECT_OUTLIER"
                )
        accepted = [row for row in samples if row["status"] == "ACCEPTED"]
        if accepted:
            median = float(np.median([row["rho_measured"] for row in accepted]))
            mad = float(
                np.median(
                    np.abs(
                        np.asarray([row["rho_measured"] for row in accepted]) - median
                    )
                )
            )
    else:
        median, mad, accepted = None, None, []
    enough = (
        len(accepted) >= config["boundary"]["profile_min_sections"]
        and mad is not None
        and mad <= config["boundary"]["corner_join_m"]
    )
    return dict(
        local_rho_samples=samples,
        accepted_rho_samples=[row for row in samples if row["status"] == "ACCEPTED"],
        rejected_rho_samples=[
            row for row in samples if row["status"].startswith("REJECT")
        ],
        rho_median=median,
        rho_mad=mad,
        independent_along_support_bins=len(accepted),
        along_support_distribution=[row["along_center_m"] for row in accepted],
        rho_dispersion=mad,
        confident=bool(enough),
    )


def refine(points, segments, edges, initial, config):
    """Return exactly one refined record per existing R2F hatch, never more."""
    rectangles = initial["rectangles"]
    initial_axes = np.asarray(initial["axes"], dtype=float)
    axes, diagnostics = diagnose_axes(
        points, segments, edges, rectangles, initial_axes, config
    )
    grids = build_topview(points, axes, config)
    structure, observations, _ = _structural_modes(segments, axes, config)
    raw = raster_modes(grids["fine"], config)
    modes = {
        axis: structure[axis] + [{**row, "kind": "RASTER"} for row in raw[axis]]
        for axis in (0, 1)
    }
    axial_points = np.column_stack(
        (np.asarray(points)[:, :2] @ axes.T, np.asarray(points)[:, 2])
    )
    axial_observations = [
        {**row, "xy": (np.asarray(row["xy"]) @ axes.T).tolist()} for row in observations
    ]
    initial_widths = [
        float(np.ptp(np.asarray(hatch["polygon_xy"]) @ axes.T, axis=0)[1])
        for hatch in rectangles
    ]
    shared_width_target, strong_widths, width_mad = _ship_width_consensus(
        rectangles, axes, config
    )
    results = []
    for hatch in rectangles:
        polygon_before = np.asarray(hatch["polygon_xy"], dtype=float)
        axial = polygon_before @ axes.T
        before = (
            float(axial[:, 0].min()),
            float(axial[:, 0].max()),
            float(axial[:, 1].min()),
            float(axial[:, 1].max()),
        )
        by_side = []
        profiles = {}
        for side_id, normal_axis, sign in SIDE_SPECS:
            index = ("U0", "U1", "V0", "V1").index(side_id)
            span = (
                (before[2], before[3]) if normal_axis == 0 else (before[0], before[1])
            )
            shortlist, full_profile = _side_candidates(
                side_id,
                normal_axis,
                sign,
                before[index],
                span,
                modes[normal_axis],
                grids["fine"],
                axial_points,
                axial_observations,
                config,
            )
            by_side.append(shortlist)
            profiles[side_id] = full_profile
        ranked = []
        for combination in itertools.product(*by_side):
            score = _joint_score(combination, before, config, shared_width_target)
            if score is not None:
                ranked.append((score, combination))
        if ranked:
            _, selected = max(ranked, key=lambda item: item[0])
        else:
            selected = tuple(
                next(row for row in rows if row["rho_grid"] == original)
                for rows, original in zip(by_side, before)
            )
        refined_rows = []
        for index, (side_id, normal_axis, sign) in enumerate(SIDE_SPECS):
            row = selected[index]
            original = before[index]
            span = (
                (before[2], before[3]) if normal_axis == 0 else (before[0], before[1])
            )
            local = _local_rho_consensus(
                profiles[side_id],
                modes[normal_axis],
                axial_observations,
                grids["fine"],
                axial_points,
                normal_axis,
                sign,
                original,
                span,
                config,
            )
            nearby_baseline = [
                candidate
                for candidate in profiles[side_id]
                if abs(candidate["rho_grid"] - original)
                <= config["boundary"]["corner_join_m"]
            ]
            baseline_strength = max(
                (
                    max(candidate["structural_coverage"], candidate["raster_coverage"])
                    for candidate in nearby_baseline
                ),
                default=0.0,
            )
            chosen_strength = max(row["structural_coverage"], row["raster_coverage"])
            large_move = (
                abs(row["rho_refined"] - original) > config["boundary"]["corner_join_m"]
            )
            meaningful_gain = (
                chosen_strength - baseline_strength
                >= config["roi"]["min_enclosure_ratio"]
            )
            corroborated = (
                local["confident"]
                and local["rho_median"] is not None
                and abs(local["rho_median"] - row["rho_refined"])
                <= config["boundary"]["corner_join_m"]
            )
            accepted = local["accepted_rho_samples"]
            repeated_raw_boundary = (
                corroborated
                and row["role"] == "AMBIGUOUS"
                and row["height_transition_coverage"]
                >= config["roi"]["min_enclosure_ratio"]
                and row["opening_side_relation"]["canonical_inner"]
                and abs(local["rho_median"] - original)
                <= config["boundary"]["corner_join_m"]
                and len(accepted) >= config["boundary"]["profile_min_sections"]
                and all(sample["raw_boundary_measured"] for sample in accepted)
            )
            if repeated_raw_boundary:
                row = {**row, "role": "INNER_EDGE"}
            if (
                row["role"] != "INNER_EDGE"
                or not corroborated
                or (large_move and not meaningful_gain)
            ):
                row = {
                    **row,
                    "rho_refined": original,
                    "uncertainty_m": max(
                        row["uncertainty_m"], abs(row["rho_refined"] - original)
                    ),
                    "refinement_status": "REFINEMENT_NOT_CONFIDENT_KEEP_R2F_SIDE",
                }
            else:
                row = {
                    **row,
                    "rho_refined": float(local["rho_median"]),
                    "refinement_status": "MULTISECTION_REFINED",
                }
            refined_rows.append({**row, "local_consensus": local})
        selected = tuple(refined_rows)
        after = [row["rho_refined"] for row in selected]
        if not ranked:
            status = "NO_SAFE_REFINEMENT"
        elif all(
            row["refinement_status"] == "MULTISECTION_REFINED" for row in selected
        ):
            status = "READY_FOR_MANUAL_REVIEW"
        else:
            status = "REVIEW_REQUIRED"
        corners_axial = np.asarray(
            (
                (after[0], after[2]),
                (after[1], after[2]),
                (after[1], after[3]),
                (after[0], after[3]),
            )
        )
        polygon_after = corners_axial @ axes
        sides = {}
        for row, original in zip(selected, before):
            sides[row["side_id"]] = dict(
                rho_before=original,
                rho_grid=row["rho_grid"],
                rho_after=row["rho_refined"],
                rho_final=row["rho_refined"],
                delta_m=float(row["rho_refined"] - original),
                role=row["role"],
                evidence_types=row["evidence_types"],
                structural_coverage=row["structural_coverage"],
                raster_coverage=row["raster_coverage"],
                uncertainty_m=row["uncertainty_m"],
                raw_point_support=row["raw_point_support"],
                raw_boundary_measured=row["raw_boundary_measured"],
                refinement_status=row["refinement_status"],
                **row["local_consensus"],
            )
        results.append(
            dict(
                hatch_id=hatch["hatch_id"],
                axis_initial_deg=diagnostics["initial_axis_angle_deg"],
                axis_refined_deg=diagnostics["refined_axis_angle_deg"],
                axis_delta_deg=diagnostics["axis_delta_deg"],
                polygon_before=polygon_before.tolist(),
                polygon_after=polygon_after.tolist(),
                sides=sides,
                side_refinement_profiles=profiles,
                geometry_constrained_side_count=sum(
                    max(row["structural_coverage"], row["raster_coverage"]) == 0
                    for row in selected
                ),
                status=status,
                coordinate_frame="RAW_INPUT_FRAME",
            )
        )
    return dict(
        schema="ship_perception.v15.r1.rectangle_refinement.1",
        status="READY_FOR_MANUAL_REVIEW" if results else "NO_CONFIRMED_HATCH",
        static_calibration_status="UNVERIFIED_NO_INDEPENDENT_INNER_EDGE_GOLDEN",
        axis_diagnostics=diagnostics,
        axes=axes.tolist(),
        raster_dedup_precision_bypass=True,
        area_bias_removed=True,
        rectangle_count_before=len(rectangles),
        rectangle_count_after=len(results),
        shared_hatch_width_diagnostic=dict(
            initial_widths_m=initial_widths,
            target_m=shared_width_target,
            strong_reference_widths_m=strong_widths,
            strong_reference_mad_m=width_mad,
            refined_widths_m=[
                float(np.ptp(np.asarray(row["polygon_after"]) @ axes.T, axis=0)[1])
                for row in results
            ],
            role="WEAK_TIE_BREAK_ONLY",
        ),
        rectangles=results,
    )
