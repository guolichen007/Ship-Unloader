"""Offline Raw3D and RawXYZ top-view fusion for provisional hatch rectangles.

The solver accepts any number of rectangles in a search ROI. It never reads
review images, expected scene counts, annotation polygons or hatch dimensions.
"""

import numpy as np

from .boundary_topology import (
    _candidate_modes,
    _observations,
    _observed_intervals,
    _interval_coverage,
    structural_axes,
)
from .raw_topview_evidence import build_topview, raster_modes


def _extent(box, axes):
    x0, y0, x1, y1 = box
    corners = np.asarray(((x0, y0), (x0, y1), (x1, y0), (x1, y1)))
    return [
        (float((corners @ axis).min()), float((corners @ axis).max())) for axis in axes
    ]


def _structural_modes(segments, axes, config):
    observations, executed = _observations(segments)
    raw = _candidate_modes(observations, axes, config)
    modes = {0: [], 1: []}
    for (node_id, direction_axis), rows in raw.items():
        normal_axis = 1 - direction_axis
        for row in rows:
            types = sorted(
                {
                    observations[index]["evidence_type"]
                    for index in row["observation_indexes"]
                }
            )
            modes[normal_axis].append(
                {
                    **row,
                    "node_id": node_id,
                    "evidence_types": types,
                    "kind": "STRUCTURAL",
                }
            )
    return modes, observations, executed


def _deduplicate_raster_modes(rows, config):
    """Suppress near-identical sampled rho peaks, retaining separate rims."""
    selected = []
    distance = config["geometry"]["candidate_voxel_m"] * 2
    for row in sorted(rows, key=lambda item: -item["support_count"]):
        if any(
            abs(row["rho"] - prior["rho"]) <= distance
            and min(row["along_max"], prior["along_max"])
            >= max(row["along_min"], prior["along_min"])
            for prior in selected
        ):
            continue
        selected.append({**row, "kind": "RASTER", "node_id": None})
    return selected


def _side_support(modes, rho, along_span, config):
    """Accumulate measured support near a proposed line without filling gaps."""
    tolerance = config["boundary"]["corner_join_m"]
    nearby = [
        mode
        for mode in modes
        if abs(mode["rho"] - rho) <= tolerance
        and mode["along_max"] >= along_span[0]
        and mode["along_min"] <= along_span[1]
    ]
    structural = [mode for mode in nearby if mode["kind"] == "STRUCTURAL"]
    raster = [mode for mode in nearby if mode["kind"] == "RASTER"]
    structural_values = [
        value
        for mode in structural
        for value in mode["along_values"]
        if along_span[0] <= value <= along_span[1]
    ]
    raster_values = [
        value
        for mode in raster
        for value in mode["along_values"]
        if along_span[0] <= value <= along_span[1]
    ]
    gap = config["boundary"]["max_support_gap_m"]
    structural_intervals = _observed_intervals(structural_values, gap)
    raster_intervals = _observed_intervals(raster_values, gap)
    structural_coverage = _interval_coverage(structural_intervals, *along_span)
    raster_coverage = _interval_coverage(raster_intervals, *along_span)
    minimum_length = config["boundary"]["line_min_length_m"]
    if structural_coverage * (along_span[1] - along_span[0]) < minimum_length:
        structural = []
        structural_values = []
        structural_intervals = []
        structural_coverage = 0.0
    if raster_coverage * (along_span[1] - along_span[0]) < minimum_length:
        raster = []
        raster_values = []
        raster_intervals = []
        raster_coverage = 0.0
    nearby = structural + raster
    types = sorted({name for mode in nearby for name in mode["evidence_types"]})
    channel_coverage = {}
    for channel in types:
        values = [
            value
            for mode in nearby
            if channel in mode["evidence_types"]
            for value in mode["along_values"]
            if along_span[0] <= value <= along_span[1]
        ]
        channel_coverage[channel] = float(
            _interval_coverage(_observed_intervals(values, gap), *along_span)
        )
    if structural and raster:
        level = "MULTI_EVIDENCE"
    elif structural:
        level = "OBSERVED_STRUCTURAL"
    elif raster:
        level = "OBSERVED_RASTER_BOUNDARY"
    else:
        level = "UNRESOLVED"
    return dict(
        rho=float(rho),
        evidence_level=level,
        evidence_types=types,
        structural_coverage=float(structural_coverage),
        raster_coverage=float(raster_coverage),
        channel_coverage=channel_coverage,
        structural_intervals=structural_intervals,
        raster_intervals=raster_intervals,
        structural_observation_ids=sorted(
            {item for mode in structural for item in mode["observation_ids"]}
        ),
        structural_node_ids=sorted({mode["node_id"] for mode in structural}),
        raster_cell_count=len(raster_values),
    )


def _opening_relation(grid, normal_axis, rho, along_span, interior_sign, config):
    """Compare measured cells inside/outside a side; cargo can make it unknown."""
    along = np.arange(along_span[0], along_span[1], grid.cell_m)
    if not len(along):
        return dict(
            status="UNKNOWN",
            support_fraction=0.0,
            far_support_fraction=0.0,
            outside_minus_inside_m=None,
            far_difference_m=None,
            canonical_inner=False,
        )

    def contrast(step):
        samples = []
        for normal in (rho + interior_sign * step, rho - interior_sign * step):
            if normal_axis == 0:
                u, v = np.full(len(along), normal), along
            else:
                u, v = along, np.full(len(along), normal)
            col = np.floor((u - grid.x0) / grid.cell_m).astype(int)
            row = np.floor((v - grid.y0) / grid.cell_m).astype(int)
            valid = (
                (row >= 0)
                & (row < grid.count.shape[0])
                & (col >= 0)
                & (col < grid.count.shape[1])
            )
            values = np.full(len(along), np.nan)
            values[valid] = grid.median[row[valid], col[valid]]
            samples.append(values)
        paired = np.isfinite(samples[0]) & np.isfinite(samples[1])
        if not paired.any():
            return None, 0.0
        return float(np.median(samples[1][paired] - samples[0][paired])), float(
            paired.mean()
        )

    difference, support = contrast(config["boundary"]["side_probe_m"])
    far_difference, far_support = contrast(config["roi"]["support_search_m"] / 2)
    if difference is None:
        return dict(
            status="UNKNOWN",
            support_fraction=0.0,
            far_support_fraction=far_support,
            outside_minus_inside_m=None,
            far_difference_m=far_difference,
            canonical_inner=False,
        )
    threshold = config["boundary"]["profile_min_drop_m"]
    status = (
        "OPENING_LOWER"
        if difference >= threshold
        else (
            "INTERIOR_HIGH_OR_CARGO" if difference <= -threshold else "HEIGHT_AMBIGUOUS"
        )
    )
    return dict(
        status=status,
        support_fraction=support,
        far_support_fraction=far_support,
        outside_minus_inside_m=difference,
        far_difference_m=far_difference,
        canonical_inner=(
            difference >= threshold
            and far_difference is not None
            and far_difference >= threshold
        ),
    )


def _candidate_rectangle(
    u0, u1, v0, v1, axes, modes, grid, config, node_id, forbidden_axial
):
    # Both side probes must fit inside a resolved opening. This is an evidence
    # resolution condition, not a hatch dimension or aspect-ratio prior.
    if min(u1 - u0, v1 - v0) <= 2 * config["boundary"]["side_probe_m"]:
        return None
    for mode in modes[0]:
        if not (
            u0 + config["roi"]["support_search_m"]
            < mode["rho"]
            < u1 - config["roi"]["support_search_m"]
        ):
            continue
        inside = [value for value in mode["along_values"] if v0 <= value <= v1]
        if mode["kind"] == "STRUCTURAL":
            internal = _observed_intervals(
                inside, config["boundary"]["max_support_gap_m"]
            )
            if (
                _interval_coverage(internal, v0, v1)
                >= config["roi"]["min_enclosure_ratio"]
            ):
                return None
        else:
            measured_fraction = min(1.0, len(inside) * mode["cell_m"] / (v1 - v0))
            if measured_fraction < 2 * config["roi"]["min_enclosure_ratio"]:
                continue
            # A partial 3D face can independently identify a transverse
            # crossbeam even when its observed span is too short to satisfy
            # the ordinary full-side coverage gate.
            structural_crossbeam = any(
                abs(other["rho"] - mode["rho"]) <= config["boundary"]["corner_join_m"]
                and len([value for value in other["along_values"] if v0 <= value <= v1])
                >= config["boundary"]["line_min_points"]
                for other in modes[0]
                if other["kind"] == "STRUCTURAL"
            )
            if structural_crossbeam:
                return None
            relation = _opening_relation(grid, 0, mode["rho"], (v0, v1), 1, config)
            near = relation["outside_minus_inside_m"]
            far = relation["far_difference_m"]
            threshold = config["boundary"]["profile_min_drop_m"]
            if (
                near is not None
                and far is not None
                and near * far > 0
                and min(abs(near), abs(far)) >= threshold
                and min(relation["support_fraction"], relation["far_support_fraction"])
                >= config["roi"]["min_enclosure_ratio"]
            ):
                return None
    side_specs = (
        ("U0", 0, u0, (v0, v1), 1),
        ("U1", 0, u1, (v0, v1), -1),
        ("V0", 1, v0, (u0, u1), 1),
        ("V1", 1, v1, (u0, u1), -1),
    )
    sides = []
    conflicts = 0
    for side_id, normal_axis, rho, span, sign in side_specs:
        support = _side_support(modes[normal_axis], rho, span, config)
        support["side_id"] = side_id
        support["opening_relation"] = _opening_relation(
            grid, normal_axis, rho, span, sign, config
        )
        support["opening_side_status"] = (
            "CANONICAL_INNER_EDGE_CANDIDATE"
            if support["opening_relation"]["canonical_inner"]
            else "OPENING_SIDE_UNRESOLVED"
        )
        if len(forbidden_axial):
            on_line = np.abs(forbidden_axial[:, normal_axis] - rho) <= (
                2 * config["boundary"]["line_inlier_m"]
            )
            on_line &= (forbidden_axial[:, 1 - normal_axis] >= span[0]) & (
                forbidden_axial[:, 1 - normal_axis] <= span[1]
            )
            if int(on_line.sum()) >= config["boundary"]["line_min_points"]:
                support["role_conflict"] = "FORBIDDEN_INTERIOR_OR_OVERFLOW_REFERENCE"
                support["opening_side_status"] = "ROLE_AMBIGUOUS"
                conflicts += 1
        sides.append(support)
    if conflicts:
        return None
    observed = sum(side["evidence_level"] != "UNRESOLVED" for side in sides)
    structural = sum(side["structural_coverage"] > 0 for side in sides)
    if observed < 3 or structural < 1:
        return None
    weak = sum(side["evidence_level"] == "UNRESOLVED" for side in sides)
    if weak > config["boundary"]["max_inferred_edges"]:
        return None
    for side in sides:
        if side["evidence_level"] == "UNRESOLVED":
            side["evidence_level"] = "GEOMETRY_CONSTRAINED"
    corners = [
        u * axes[0] + v * axes[1] for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
    ]
    relation = sum(
        side["opening_relation"]["status"] == "OPENING_LOWER" for side in sides
    )
    canonical = sum(side["opening_relation"]["canonical_inner"] for side in sides)
    opening_strength = sum(
        max(0.0, side["opening_relation"]["outside_minus_inside_m"] or 0.0)
        for side in sides
    )
    dual_return = sum(
        side["opening_relation"]["far_support_fraction"]
        >= config["roi"]["min_enclosure_ratio"]
        for side in sides
    )
    adverse = sum(
        side["opening_relation"]["status"] == "INTERIOR_HIGH_OR_CARGO" for side in sides
    )
    multi = sum(side["evidence_level"] == "MULTI_EVIDENCE" for side in sides)
    raster = sum(side["raster_coverage"] for side in sides)
    structural_coverage = sum(side["structural_coverage"] for side in sides)
    side_strength = [
        max(side["structural_coverage"], side["raster_coverage"]) for side in sides
    ]
    strong = sum(
        value >= config["roi"]["min_enclosure_ratio"] for value in side_strength
    )
    majority_supported = sum(value >= 0.5 for value in side_strength)
    if strong < 4 or majority_supported < 3 or dual_return < 4:
        return None
    local = sum(node_id in side["structural_node_ids"] for side in sides)
    # Height is only a ranking cue. A cargo-filled opening with reversed
    # height contrast remains eligible when its structural evidence survives.
    score = (
        strong,
        dual_return,
        round((u1 - u0) * (v1 - v0), 3),
        canonical,
        round(min(side_strength), 3),
        relation - adverse,
        multi,
        structural,
        round(structural_coverage, 3),
        local,
        round(raster, 3),
        -weak,
    )
    status = (
        "PROVISIONAL_RECTANGLE"
        if canonical >= 3 and multi >= 3 and not conflicts
        else "REVIEW_REQUIRED_RECTANGLE"
    )
    return dict(
        node_id=node_id,
        status=status,
        polygon_xy=[point.tolist() for point in corners],
        corners=[
            dict(corner_xy=point.tolist(), corner_source="GEOMETRIC_LINE_INTERSECTION")
            for point in corners
        ],
        sides=sides,
        geometry_constrained_side_count=weak,
        canonical_inner_edge_count=canonical,
        opening_strength_m=float(opening_strength),
        role_conflict_side_count=conflicts,
        area_m2=float((u1 - u0) * (v1 - v0)),
        score=score,
        coordinate_frame="RAW_INPUT_FRAME",
    )


def _distinct(values, distance):
    result = []
    for value in values:
        if all(abs(value - other) > distance for other in result):
            result.append(value)
    return result


def _spread_modes(rows, span, rank):
    """Bound search work while retaining rho peaks throughout a search ROI."""
    groups = [[] for _ in range(8)]
    width = max(span[1] - span[0], 1e-9)
    for row in rows:
        index = min(7, max(0, int(8 * (row["rho"] - span[0]) / width)))
        groups[index].append(row)
    return [
        row for group in groups for row in sorted(group, key=rank, reverse=True)[:3]
    ]


def solve(points, segments, edges, height_regions, config):
    """Fuse 3D lines and RawXYZ rasters; return reviewable provisional polygons."""
    forbidden = {
        row["segment_id"]
        for row in segments
        if row.get("reference_role") in ("INTERIOR_SURFACE", "ROLE_AMBIGUOUS_OVERFLOW")
    }
    axes = structural_axes(
        [edge for edge in edges if edge.get("segment_id") not in forbidden],
        2 * config["boundary"]["merge_angle_deg"],
    )
    if axes is None:
        return dict(
            status="STRUCTURAL_AXES_UNRESOLVED",
            rectangles=[],
            raster_modes={0: [], 1: []},
        )
    grids = build_topview(points, axes, config)
    structure, observations, executed = _structural_modes(segments, axes, config)
    forbidden_xy = np.asarray(
        [
            row["xy"]
            for row in observations
            if row["reference_role"] in ("INTERIOR_SURFACE", "ROLE_AMBIGUOUS_OVERFLOW")
        ],
        dtype=float,
    ).reshape(-1, 2)
    forbidden_axial = forbidden_xy @ np.asarray(axes).T
    raw_raster = raster_modes(grids["fine"], config)
    raster = {
        axis: _deduplicate_raster_modes(raw_raster[axis], config) for axis in (0, 1)
    }
    modes = {axis: structure[axis] + raster[axis] for axis in (0, 1)}
    proposals = []
    margin = config["roi"]["support_search_m"]
    tolerance = config["boundary"]["corner_join_m"]
    for node in height_regions:
        u_span, v_span = _extent(node["bbox_xy"], axes)
        structural_v = [
            mode
            for mode in structure[1]
            if v_span[0] - margin <= mode["rho"] <= v_span[1] + margin
            and mode["along_max"] >= u_span[0]
            and mode["along_min"] <= u_span[1]
        ]
        # Opposite structural boundaries anchor the opening role. This is a
        # provisional solver: raster-only boxes never become steel truth.
        structural_values = _distinct(
            [
                row["rho"]
                for row in sorted(structural_v, key=lambda row: -row["support_count"])
            ],
            tolerance,
        )[:8]
        raster_v = [
            mode
            for mode in raster[1]
            if v_span[0] - margin <= mode["rho"] <= v_span[1] + margin
            and mode["along_max"] >= u_span[0]
            and mode["along_min"] <= u_span[1]
        ]
        ranked_v = sorted(raster_v, key=lambda row: -row["support_count"])
        v_values = _distinct(
            structural_values + [row["rho"] for row in ranked_v[:8]], tolerance
        )
        u_modes = [
            mode
            for mode in modes[0]
            if u_span[0] - margin <= mode["rho"] <= u_span[1] + margin
            and mode["along_max"] >= v_span[0] - margin
            and mode["along_min"] <= v_span[1] + margin
        ]
        u_modes.sort(key=lambda row: -row["support_count"])
        u_modes = _spread_modes(u_modes, u_span, lambda row: row["support_count"])
        for i, va in enumerate(v_values):
            for vb in v_values[i + 1 :]:
                v0, v1 = sorted((va, vb))
                if not v_span[0] - margin <= (v0 + v1) / 2 <= v_span[1] + margin:
                    continue
                ranked_u = _spread_modes(
                    u_modes,
                    u_span,
                    lambda row: sum(v0 <= value <= v1 for value in row["along_values"]),
                )
                u_values = _distinct([row["rho"] for row in ranked_u], tolerance)
                for j, ua in enumerate(u_values):
                    for ub in u_values[j + 1 :]:
                        u0, u1 = sorted((ua, ub))
                        if (
                            not u_span[0] - margin
                            <= (u0 + u1) / 2
                            <= u_span[1] + margin
                        ):
                            continue
                        candidate = _candidate_rectangle(
                            u0,
                            u1,
                            v0,
                            v1,
                            axes,
                            modes,
                            grids["fine"],
                            config,
                            node["node_id"],
                            forbidden_axial,
                        )
                        if candidate is not None:
                            proposals.append(candidate)
    proposals.sort(key=lambda row: row["score"], reverse=True)
    accepted = []
    overlap_limit = config["roi"]["min_enclosure_ratio"]
    for candidate in proposals:
        polygon = np.asarray(candidate["polygon_xy"])
        projected = polygon @ np.asarray(axes).T
        box = (
            float(projected[:, 0].min()),
            float(projected[:, 1].min()),
            float(projected[:, 0].max()),
            float(projected[:, 1].max()),
        )
        area = (box[2] - box[0]) * (box[3] - box[1])
        duplicate = False
        for prior_index, prior in enumerate(accepted):
            old = np.asarray(prior["polygon_xy"]) @ np.asarray(axes).T
            old_box = (
                float(old[:, 0].min()),
                float(old[:, 1].min()),
                float(old[:, 0].max()),
                float(old[:, 1].max()),
            )
            intersection = max(
                0.0, min(box[2], old_box[2]) - max(box[0], old_box[0])
            ) * max(0.0, min(box[3], old_box[3]) - max(box[1], old_box[1]))
            old_area = (old_box[2] - old_box[0]) * (old_box[3] - old_box[1])
            overlap_u = min(box[2], old_box[2]) - max(box[0], old_box[0])
            overlap_v = min(box[3], old_box[3]) - max(box[1], old_box[1])
            gap_v = max(0.0, max(box[1], old_box[1]) - min(box[3], old_box[3]))
            shared_u = overlap_u >= overlap_limit * min(
                box[2] - box[0], old_box[2] - old_box[0]
            )
            # A zero-width gap between two proposals with a common long rim
            # describes one opening and a neighboring rim strip. Independent
            # openings need a measured separator between them.
            touches = shared_u and gap_v <= config["boundary"]["side_probe_m"]
            substantial_overlap = (
                overlap_u > config["boundary"]["side_probe_m"]
                and overlap_v > config["boundary"]["side_probe_m"]
            )
            if (
                intersection / min(area, old_area) >= overlap_limit
                or touches
                or substantial_overlap
            ):
                # Parallel inner and outer rims can yield nearly identical
                # boxes. If exactly one side moves within the measured search
                # width, retain the one with stronger opening-side evidence.
                old_coordinates = (old_box[0], old_box[2], old_box[1], old_box[3])
                coordinates = (box[0], box[2], box[1], box[3])
                offsets = [abs(a - b) for a, b in zip(old_coordinates, coordinates)]
                rim_variant = (
                    sum(
                        value <= config["boundary"]["corner_join_m"]
                        for value in offsets
                    )
                    >= 3
                    and max(offsets) <= config["roi"]["support_search_m"]
                )
                if rim_variant and (
                    candidate["canonical_inner_edge_count"]
                    > prior["canonical_inner_edge_count"]
                    or (
                        candidate["canonical_inner_edge_count"]
                        == prior["canonical_inner_edge_count"]
                        and candidate["opening_strength_m"]
                        > prior["opening_strength_m"]
                        + config["boundary"]["profile_min_drop_m"]
                    )
                ):
                    accepted[prior_index] = candidate
                duplicate = True
                break
        if not duplicate:
            candidate["hatch_id"] = "H%03d" % len(accepted)
            accepted.append(candidate)
    # Relative area only resolves weak fragments left beside stronger complete
    # enclosures. A genuinely small opening with two canonical observed sides
    # stays eligible; no absolute hatch dimensions enter the solver.
    dominant_area = max((row["area_m2"] for row in accepted), default=0.0)
    fragments = []
    retained = []
    for candidate in accepted:
        elevated_enclosure = all(
            side["opening_relation"]["status"] == "INTERIOR_HIGH_OR_CARGO"
            for side in candidate["sides"]
        ) and any(
            other is not candidate and other["canonical_inner_edge_count"] >= 2
            for other in accepted
        )
        if elevated_enclosure:
            # A raised roof beside a measured opening has the same rectangular
            # topology as a cargo-filled hatch. Preserve the geometry for
            # review, but do not claim its opening role from shape alone.
            candidate["status"] = "REVIEW_ONLY_ROLE_AMBIGUOUS"
            candidate["reject_reason"] = "ELEVATED_ENCLOSURE_BESIDE_MEASURED_OPENING"
            candidate["hatch_id"] = "F%03d" % len(fragments)
            fragments.append(candidate)
        elif (
            candidate["area_m2"] < config["roi"]["min_enclosure_ratio"] * dominant_area
            and sum(side["structural_coverage"] > 0 for side in candidate["sides"]) < 4
            and candidate["canonical_inner_edge_count"] < 4
        ):
            candidate["status"] = "REVIEW_ONLY_FRAGMENT"
            candidate["reject_reason"] = "WEAK_SMALL_FRAGMENT_BESIDE_COMPLETE_ENCLOSURE"
            candidate["hatch_id"] = "F%03d" % len(fragments)
            fragments.append(candidate)
        else:
            candidate["hatch_id"] = "H%03d" % len(retained)
            retained.append(candidate)
    accepted = retained
    used_ids = {
        item
        for candidate in accepted
        for side in candidate["sides"]
        for item in side["structural_observation_ids"]
    }
    mode_ids = {
        item
        for axis_modes in structure.values()
        for mode in axis_modes
        for item in mode["observation_ids"]
    }
    for observation in observations:
        if observation["observation_id"] in used_ids:
            observation["status"] = "USED_IN_RECTANGLE_CANDIDATE"
            observation["reject_reason"] = None
        elif observation["observation_id"] in mode_ids:
            observation["reject_reason"] = "REJECT_RECTANGLE_TOPOLOGY"
    rejected = sum(row["status"] == "REJECTED" for row in observations)
    accounting = len(used_ids) + rejected == len(observations) and all(
        (row["status"] == "USED_IN_RECTANGLE_CANDIDATE")
        != (row["reject_reason"] is not None)
        for row in observations
    )
    provisional = sum(row["status"] == "PROVISIONAL_RECTANGLE" for row in accepted)
    return dict(
        status=(
            "PROVISIONAL"
            if provisional
            else "REVIEW_REQUIRED" if accepted else "NO_CONFIRMED_HATCH"
        ),
        static_calibration_status="UNVERIFIED_NO_INDEPENDENT_INNER_EDGE_GOLDEN",
        axes=[axis.tolist() for axis in axes],
        raw_topview_channels=list(
            ("median_z", "point_count", "vertical_span", "valid_mask")
        ),
        raster_mode_count={axis: len(raster[axis]) for axis in (0, 1)},
        structural_mode_count={axis: len(structure[axis]) for axis in (0, 1)},
        profile_executed_segments=executed,
        observation_count=len(observations),
        used_count=len(used_ids),
        rejected_count=rejected,
        observation_accounting_pass=accounting,
        observations=observations,
        canonical_inner_edge_count=sum(
            row["canonical_inner_edge_count"] for row in accepted
        ),
        provisional_rectangle_count=provisional,
        rectangles=accepted,
        fragment_hypotheses=fragments,
    )
