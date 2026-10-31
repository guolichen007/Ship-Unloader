"""Existing bounded height/Raw side capabilities, isolated from the B5 solver.

Only established OpeningRegion sides enter here. No hatch recall, identity,
rectangle combinations, historical geometry or steel certification occurs.
"""

import hashlib
import numpy as np
from scipy import ndimage
from scipy.stats import theilslopes
from ship_perception.r1_static.vessel_height_topology import (
    _relative_levels,
    _line_fraction,
)
from ship_perception.r1_static.height_rim_review import (
    _section_transition,
    _guard_review_displacement,
)
from .sections import physical_sections
from .review_geometry import SIDES
from .review_frame import line_points, line_displacements, line_coordinate_at
from types import SimpleNamespace


def local_height_modes(grid, box, config):
    """Find contour lines repeatedly observed at nearby XY over height levels.

    A repeated threshold crossing is topology evidence only. It cannot certify
    a steel edge or a fully measured fourth corner.
    """
    levels, topology = _relative_levels(grid, box, config)
    if not levels:
        return dict(
            levels_m=[],
            topology=topology,
            modes={0: [], 1: []},
            persistent_corners=[],
            footprint_edges={0: [], 1: []},
        )
    u0, v0, u1, v1 = box
    margin = config["roi"]["support_search_m"]
    minimum_fraction = config["roi"]["min_enclosure_ratio"] / 2
    observations = {0: [], 1: []}
    for level_index, level in enumerate(levels):
        mask = grid.occupancy & (
            grid.median >= level if topology == "CARGO_FILLED" else grid.median <= level
        )
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool))
        for axis in (0, 1):
            if axis == 0:
                edge = (
                    (mask[:, 1:] != mask[:, :-1])
                    & grid.occupancy[:, 1:]
                    & grid.occupancy[:, :-1]
                )
            else:
                edge = (
                    (mask[1:, :] != mask[:-1, :])
                    & grid.occupancy[1:, :]
                    & grid.occupancy[:-1, :]
                )
            origin = grid.x0 if axis == 0 else grid.y0
            count = edge.shape[1 if axis == 0 else 0]
            normal_band = (
                (u0 - margin, u1 + margin) if axis == 0 else (v0 - margin, v1 + margin)
            )
            along_span = (v0, v1) if axis == 0 else (u0, u1)
            for index in range(count):
                rho = origin + (index + 1) * grid.cell_m
                if not normal_band[0] <= rho <= normal_band[1]:
                    continue
                coverage, along_hits = _line_fraction(
                    edge, grid, axis, index, along_span
                )
                if coverage >= minimum_fraction:
                    observations[axis].append(
                        dict(
                            rho=float(rho),
                            level_index=level_index,
                            level_m=float(level),
                            coverage=coverage,
                            along_values=along_hits,
                        )
                    )
    modes = {0: [], 1: []}
    tolerance = config["boundary"]["corner_join_m"]
    for axis, rows in observations.items():
        groups = []
        for row in sorted(rows, key=lambda item: (-item["coverage"], item["rho"])):
            matching = next(
                (
                    group
                    for group in groups
                    if abs(group[0]["rho"] - row["rho"]) <= tolerance
                ),
                None,
            )
            if matching is None:
                groups.append([row])
            else:
                matching.append(row)
        for group in groups:
            distinct = {row["level_index"] for row in group}
            if len(distinct) < 2:
                continue
            locations = np.array([row["rho"] for row in group])
            position = float(np.median(locations))
            dispersion = float(np.median(np.abs(locations - position)))
            if dispersion > tolerance:
                continue
            modes[axis].append(
                dict(
                    normal_axis=axis,
                    rho=position,
                    level_count=len(distinct),
                    support_level_indices=sorted(distinct),
                    support_levels_m=sorted({row["level_m"] for row in group}),
                    position_mad_m=dispersion,
                    coverage=float(max((row["coverage"] for row in group))),
                    along_min=float(
                        min((value for row in group for value in row["along_values"]))
                    ),
                    along_max=float(
                        max((value for row in group for value in row["along_values"]))
                    ),
                    evidence_type="HEIGHT_PERSISTENT_EDGE_NOT_STEEL",
                )
            )
        modes[axis].sort(key=lambda item: item["rho"])
    footprint_edges = {0: [], 1: []}
    for axis in (0, 1):
        edge = (
            grid.occupancy[:, 1:] != grid.occupancy[:, :-1]
            if axis == 0
            else grid.occupancy[1:, :] != grid.occupancy[:-1, :]
        )
        origin = grid.x0 if axis == 0 else grid.y0
        count = edge.shape[1 if axis == 0 else 0]
        normal_band = (
            (u0 - margin, u1 + margin) if axis == 0 else (v0 - margin, v1 + margin)
        )
        along_span = (v0, v1) if axis == 0 else (u0, u1)
        for index in range(count):
            rho = origin + (index + 1) * grid.cell_m
            if not normal_band[0] <= rho <= normal_band[1]:
                continue
            coverage, _ = _line_fraction(edge, grid, axis, index, along_span)
            if coverage >= minimum_fraction:
                footprint_edges[axis].append(
                    dict(
                        rho=float(rho),
                        coverage=coverage,
                        evidence_type="VESSEL_SUPPORT_LIMIT_NOT_STEEL",
                    )
                )
    return dict(
        levels_m=levels,
        topology=topology,
        modes=modes,
        persistent_corners=[],
        footprint_edges=footprint_edges,
    )


def _mode_near(rho, modes, tolerance):
    return min(
        (mode for mode in modes if abs(mode["rho"] - rho) <= tolerance),
        key=lambda mode: abs(mode["rho"] - rho),
        default=None,
    )


def _merge_candidates(primary, height, current_bounds, config):
    """Keep measured parallel modes separate and attach nearby height evidence."""
    tolerance = config["boundary"]["line_inlier_m"]
    rows = []
    for source in primary:
        if source.get("provider_scope") == "CONTEXT_ONLY":
            continue
        if min(source.get("primary_raw3d", 0), source.get("primary_bev", 0)) <= 0:
            continue
        if not any((abs(source["rho"] - row["rho"]) <= tolerance for row in rows)):
            rows.append(dict(source))
    for source in height:
        row = _mode_near(source["rho"], rows, tolerance)
        if row is None:
            rows.append(
                dict(
                    rho=source["rho"],
                    primary_raw3d=0.0,
                    primary_bev=0.0,
                    provider_ids=[],
                    source_vessel_ids=[],
                    source_support_ids=[],
                    provider_scope="HEIGHT_TOPOLOGY_ONLY",
                )
            )
    for rho in current_bounds:
        if _mode_near(rho, rows, tolerance) is None:
            rows.append(
                dict(
                    rho=float(rho),
                    primary_raw3d=0.0,
                    primary_bev=0.0,
                    provider_ids=[],
                    source_vessel_ids=[],
                    source_support_ids=[],
                    provider_scope="CURRENT_REGION_BOUND_ONLY",
                )
            )
    proximity = config["boundary"]["corner_join_m"]
    for row in rows:
        match = _mode_near(row["rho"], height, proximity)
        row["height_level_count"] = match["level_count"] if match else 0
        row["height_rho_m"] = match["rho"] if match else None
        row["height_position_mad_m"] = match["position_mad_m"] if match else None
        row["height_distance_m"] = abs(row["rho"] - match["rho"]) if match else None
        row["height_coverage"] = match["coverage"] if match else 0.0
        row["height_support_levels_m"] = match["support_levels_m"] if match else []
    return sorted(rows, key=lambda row: row["rho"])


def _strip_relation(grid, rho, span, axis, inside_sign, config):
    """Compare adjacent measured strips along the full proposed side."""
    u, v = grid.centers()
    normal, along = (u, v) if axis == 0 else (v, u)
    probe = config["boundary"]["side_probe_m"]
    width = config["roi"]["support_search_m"] / 2
    along_margin = config["boundary"]["corner_join_m"]
    segment = (along >= span[0] + along_margin) & (along <= span[1] - along_margin)
    inside = (
        segment
        & (inside_sign * (normal - rho) >= probe)
        & (inside_sign * (normal - rho) <= width)
    )
    outside = (
        segment
        & (-inside_sign * (normal - rho) >= probe)
        & (-inside_sign * (normal - rho) <= width)
    )
    valid_inside = inside & grid.occupancy
    valid_outside = outside & grid.occupancy
    inside_fraction = float(valid_inside.sum() / max(1, inside.sum()))
    outside_fraction = float(valid_outside.sum() / max(1, outside.sum()))
    inside_z = (
        float(np.median(grid.median[valid_inside])) if valid_inside.any() else None
    )
    outside_z = (
        float(np.median(grid.median[valid_outside])) if valid_outside.any() else None
    )
    contrast = (
        abs(inside_z - outside_z)
        if inside_z is not None and outside_z is not None
        else 0.0
    )
    lower_interior = (
        min(inside_fraction, outside_fraction) >= config["roi"]["min_enclosure_ratio"]
        and inside_z is not None
        and (outside_z is not None)
        and (outside_z - inside_z >= config["roi"]["opening_drop_m"])
    )
    return dict(
        inside_occupancy_fraction=inside_fraction,
        outside_occupancy_fraction=outside_fraction,
        inside_median_z_m=inside_z,
        outside_median_z_m=outside_z,
        boundary_asymmetry=float(abs(inside_fraction - outside_fraction)),
        height_contrast_m=float(contrast),
        opening_height_transition=bool(lower_interior),
        opening_one_sided=inside_fraction >= config["roi"]["min_enclosure_ratio"]
        and outside_fraction <= inside_fraction / 2,
        interior_ridge=not lower_interior
        and min(inside_fraction, outside_fraction)
        >= config["roi"]["min_enclosure_ratio"]
        and (
            abs(inside_fraction - outside_fraction)
            < config["roi"]["min_enclosure_ratio"] / 2
        ),
    )


def _section_consensus(grid, rho, span, axis, inside_sign, config):
    """Measure the same physical transition in independent occupied sections."""
    edges = np.linspace(span[0], span[1], 6)
    sections = []
    for start, end in zip(edges[:-1], edges[1:]):
        margin = config["boundary"]["corner_join_m"]
        relation = _strip_relation(
            grid, rho, (start - margin, end + margin), axis, inside_sign, config
        )
        if (
            max(
                relation["inside_occupancy_fraction"],
                relation["outside_occupancy_fraction"],
            )
            < config["roi"]["min_enclosure_ratio"]
        ):
            continue
        sections.append(dict(start_m=float(start), end_m=float(end), **relation))
    transitions = sum(
        (
            row["opening_one_sided"] or row["opening_height_transition"]
            for row in sections
        )
    )
    ridges = sum((row["interior_ridge"] for row in sections))
    return dict(
        section_count=len(sections),
        transition_count=transitions,
        internal_ridge_section_count=ridges,
        left_endpoint_support=bool(
            sections
            and (
                sections[0]["opening_one_sided"]
                or sections[0]["opening_height_transition"]
            )
        ),
        right_endpoint_support=bool(
            sections
            and (
                sections[-1]["opening_one_sided"]
                or sections[-1]["opening_height_transition"]
            )
        ),
        sections=sections,
    )


def _side_rank(row, relation, old_rho, config, side):
    """Lexicographic evidence; no area, width or larger-rho reward."""
    minimum = config["roi"]["min_enclosure_ratio"]
    dual = min(row["primary_raw3d"], row["primary_bev"])
    height = row["height_level_count"] >= 2
    footprint = row.get("support_limit_validated", False)
    viable = (
        height
        or footprint
        or (
            relation["opening_one_sided"]
            and (not relation["interior_ridge"])
            and (dual >= minimum)
        )
    )
    if (
        side == "V0"
        and relation["interior_ridge"]
        and (abs(row["rho"] - old_rho) > config["boundary"]["corner_join_m"])
    ):
        viable = False
    if side == "V0":
        boundary_class = 2 if footprint else 1 if row.get("first_inward_dual") else 0
    else:
        boundary_class = 2 if height else 1 if dual >= minimum else 0
    return (
        int(viable),
        boundary_class,
        row.get("section_consensus", {}).get("transition_count", 0),
        -row.get("section_consensus", {}).get("internal_ridge_section_count", 0),
        int(relation["opening_one_sided"]),
        int(height),
        int(dual >= minimum),
        int(footprint),
        (
            -round(abs(row["height_rho_m"] - old_rho), 3)
            if side == "V1" and height
            else 0.0
        ),
        (
            -round(row.get("height_distance_m") or 0.0, 3)
            if side == "V1" and height
            else 0.0
        ),
        round(relation["boundary_asymmetry"], 3),
        -round(abs(row["rho"] - old_rho), 3),
        round(min(dual, 1.0), 3),
        round(row["height_coverage"], 3),
    )


def _enrich_side(
    rows, grid, span, axis, sign, old_rho, config, side, support_limit=None
):
    result = []
    for source in rows:
        row = dict(source)
        minimum = config["roi"]["min_enclosure_ratio"]
        tolerance = config["boundary"]["corner_join_m"]
        row["support_limit_rho"] = support_limit
        row["support_limit_validated"] = bool(
            side == "V0"
            and support_limit is not None
            and (abs(row["rho"] - old_rho) <= tolerance)
            and (abs(row["rho"] - support_limit) <= tolerance)
        )
        row["first_inward_dual"] = bool(
            side == "V0"
            and support_limit is not None
            and (
                support_limit
                <= row["rho"]
                <= support_limit + config["roi"]["support_search_m"] / 4
            )
            and (min(row["primary_raw3d"], row["primary_bev"]) >= minimum)
        )
        row["opening_relation"] = _strip_relation(
            grid, row["rho"], span, axis, sign, config
        )
        row["section_consensus"] = _section_consensus(
            grid, row["rho"], span, axis, sign, config
        )
        row["evidence_rank"] = list(
            _side_rank(row, row["opening_relation"], old_rho, config, side)
        )
        result.append(row)
    return sorted(result, key=lambda row: tuple(row["evidence_rank"]), reverse=True)


def compile_constraints(cache, regions, config, previous):
    """Finite existing-side work; source priorities are applied by the assembler.

    S4 and B1 estimates are untouched. A review direction constraint is only
    inferred from current qualified upper-contour samples when S4 has not
    already supplied that longitudinal constraint.
    """
    grid = cache.context_grid("fine")
    x, y = grid.axis_centers()
    transposed = SimpleNamespace(
        cell_m=grid.cell_m,
        count=grid.count.T,
        median=grid.median.T,
        axis_centers=lambda: (y, x),
    )
    evidence = {}
    slopes = []
    for region in regions:
        if not region.identity_role.startswith("CONFIRMED"):
            continue
        box = region.conservative_bounds
        height = local_height_modes(grid, box, config)
        candidates = cache.side_candidates(box)
        sides = {}
        for name, axis, pos, outward in SIDES:
            span = (box[1 - axis], box[3 - axis])
            local = [
                m
                for m in height["modes"][axis]
                if abs(m["rho"] - box[pos]) <= config["roi"]["support_search_m"]
                and outward * (m["rho"] - (box[axis] + box[axis + 2]) / 2) > 0
            ]
            rows = _merge_candidates(candidates[name], local, [box[pos]], config)
            ranked = _enrich_side(
                rows, grid, span, axis, -outward, box[pos], config, name
            )
            minimum = config["boundary"]["profile_min_sections"]
            qualified = [
                r
                for r in ranked
                if r["evidence_rank"][0]
                and r["section_consensus"]["transition_count"] >= minimum
                and not r["opening_relation"]["interior_ridge"]
            ]
            seeds = [box[pos]] + [r["rho"] for r in qualified]
            details = []
            for seed in sorted(set(seeds)):
                detail = _section_transition(
                    transposed if axis == 0 else grid,
                    span,
                    seed,
                    "V" + name[-1],
                    config,
                    section_edges=physical_sections(span, grid.cell_m, config),
                )
                detail["anchor_m"] = seed
                details.append(detail)
            accepted = [r for r in details if r["accepted"]]
            selected = max(
                accepted,
                key=lambda r: (
                    r["median_contrast_m"],
                    len(r["samples"]),
                    -r["dispersion_m"],
                ),
                default=None,
            )
            prior = previous[region.region_id].get(name, [])
            old_rho = prior[-1]["rho"] if prior else box[pos]
            old_role = _section_consensus(grid, old_rho, span, axis, -outward, config)
            band = [m["rho"] for m in prior]
            for m in prior:
                band.extend(m.get("candidate_band_rhos", []))
            if not band:
                band = [box[pos]]
            height_matches = any(
                min(band) - config["boundary"]["corner_join_m"]
                <= m["rho"]
                <= max(band) + config["boundary"]["corner_join_m"]
                for m in local
            )
            measured_topology = any(m["state"] == "TOPOLOGY_SUPPORTED" for m in prior)
            preserve = bool(
                measured_topology
                and height_matches
                and (old_role["transition_count"] >= minimum or axis == 1)
            )
            if (
                axis == 0
                and qualified
                and abs(qualified[0]["rho"] - old_rho)
                > config["boundary"]["corner_join_m"]
            ):
                preserve = False
            # A current one-sided primary footprint remains a valid extent
            # without a two-sided height contour; it is not a cargo/deck role.
            if (
                measured_topology
                and axis == 1
                and old_role["transition_count"] >= minimum
                and any(z["opening_one_sided"] for z in old_role["sections"])
            ):
                preserve = True
            sides[name] = dict(
                selected=selected,
                alternatives=details,
                qualified_role_rows=qualified,
                prior_role=old_role,
                preserve_prior=preserve,
                height_topology=height["topology"],
            )
        upper = sides["V1"]
        if upper["selected"] and not upper["preserve_prior"]:
            slopes.append(upper["selected"]["slope_deg"])
        evidence[region.region_id] = sides
    # Same semantics as the existing height-rim review: one current contour
    # direction for conservative review lines, never a rewrite of S4 axes or
    # an independent Raw measurement. All original samples stay in provenance.
    angle = np.radians(float(np.median(slopes))) if slopes else 0.0
    basis = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    output = {}
    for region in regions:
        if region.region_id not in evidence:
            continue
        box = region.conservative_bounds
        result = {name: list(rows) for name, rows in previous[region.region_id].items()}
        for name, axis, pos, outward in SIDES:
            item = evidence[region.region_id][name]
            if item["preserve_prior"]:
                continue
            detail = item["selected"]
            role = item["qualified_role_rows"]
            if detail is None and not role:
                continue
            span = (box[1 - axis], box[3 - axis])
            if detail is not None:
                samples = np.asarray(detail["samples"])
                points = samples[:, :2] if axis == 1 else samples[:, [1, 0]]
                normal = basis[axis]
                rho = float(np.median(points @ normal))
                sigma = max(
                    config["geometry"]["plane_inlier_m"], detail["dispersion_m"]
                )
                count = len(points)
                role_gain = bool(role)
                station = np.median(points, axis=0)
                matched_families = []
                if not role_gain:
                    for f in getattr(cache, "s4_topology_bridge", {}).get(
                        "side_families", []
                    ):
                        if (
                            f["family_id"]
                            not in region.side_topology_support[name].provider_ids
                        ):
                            continue
                        tangent = np.array([-normal[1], normal[0]])
                        family_points = line_points(
                            dict(f, support_span=f["support_span"])
                        )
                        lo = max(
                            float(np.min(points @ tangent)),
                            float(np.min(family_points @ tangent)),
                        )
                        hi = min(
                            float(np.max(points @ tangent)),
                            float(np.max(family_points @ tangent)),
                        )
                        if hi <= lo:
                            continue
                        stations = rho * normal + np.array([lo, hi])[:, None] * tangent
                        delta = line_displacements(
                            dict(normal_uv=normal.tolist(), rho=rho), f, stations
                        )
                        if (
                            np.max(np.abs(delta))
                            <= config["boundary"]["corner_join_m"]
                            + f["rho_uncertainty"]
                        ):
                            matched_families.append(f["family_id"])
                role_gain |= bool(matched_families)
                item["corroborating_assigned_s4_families"] = matched_families
            else:
                selected = role[0]
                normal = np.eye(2)[axis]
                rho = selected["rho"]
                sigma = max(
                    config["geometry"]["plane_inlier_m"],
                    selected.get("height_position_mad_m") or 0,
                )
                points = np.zeros((2, 2))
                points[:, axis] = rho
                points[:, 1 - axis] = span
                count = selected["section_consensus"]["transition_count"]
                role_gain = True
            if (
                region.identity_evidence.get("nomination", {}).get("polarity") == "LOW"
                and axis == 0
                and detail is None
            ):
                role_gain = False
            old = np.eye(2)[axis] * box[pos]
            displacement = rho - normal @ old
            old_points = np.zeros((2, 2))
            old_points[:, axis] = box[pos]
            old_points[:, 1 - axis] = span
            prior_line = dict(normal_uv=np.eye(2)[axis].tolist(), rho=box[pos])
            candidate_line = dict(normal_uv=normal.tolist(), rho=rho)
            inward = float(
                max(
                    0.0,
                    np.max(
                        -outward
                        * line_displacements(prior_line, candidate_line, old_points)
                    ),
                )
            )
            if inward > config["boundary"]["corner_join_m"] and not role_gain:
                # Retain direction evidence without applying the rejected
                # inward position. The conservative line encloses both ends
                # of the original complete side; it is explicitly inferred.
                old_points = np.zeros((2, 2))
                old_points[:, axis] = box[pos]
                old_points[:, 1 - axis] = span
                rho = float((np.max if outward > 0 else np.min)(old_points @ normal))
                state = "CONSERVATIVE_PRIOR"
                reason = "RAW_DIRECTION_RETAINED_POSITION_GUARD_PRESERVES_COVERAGE"
            else:
                state = "TOPOLOGY_SUPPORTED"
                reason = "CURRENT_RAW_SECTIONS_AND_INDEPENDENT_LOCAL_SIDE_ROLE"
            tangent = np.array([-normal[1], normal[0]])
            identity = hashlib.sha256(
                repr((region.region_id, name, rho, detail)).encode()
            ).hexdigest()
            separator_reject = False
            for partition in getattr(cache, "s4_topology_bridge", {}).get(
                "node_partition_audit", []
            ):
                membership = set(
                    region.identity_evidence.get("member_height_nodes", [])
                )
                labels = {
                    c["side"]
                    for c in partition["children"]
                    if c["node_id"] in membership
                }
                if axis != 0 or len(labels) != 1:
                    continue
                which = next(iter(labels))
                candidate_points = np.zeros((2, 2))
                candidate_points[:, 1] = span
                candidate_points[:, 0] = (
                    rho - normal[1] * candidate_points[:, 1]
                ) / normal[0]
                if (
                    which == 0
                    and name == "U1"
                    and candidate_points[:, 0].max() > partition["separator_faces"][0]
                ) or (
                    which == 1
                    and name == "U0"
                    and candidate_points[:, 0].min() < partition["separator_faces"][1]
                ):
                    separator_reject = True
            if separator_reject:
                item["geometry_guard"] = "CURRENT_SEPARATOR_PRESERVED"
                continue
            result[name] = [
                dict(
                    mode_id="LOCAL_ROLE:" + identity,
                    source="TOPOLOGY_SUPPORTED",
                    state=state,
                    rho=rho,
                    normal_uv=normal.tolist(),
                    support_span=[
                        float(np.min(points @ tangent)),
                        float(np.max(points @ tangent)),
                    ],
                    uncertainty_m=sigma,
                    independent_raw_sections=count,
                    physical_role_improved=role_gain,
                    independent_opening_role_gain=role_gain,
                    physical_role="CURRENT_BOUNDED_OPENING_CONTOUR_REVIEW_NOT_STEEL",
                    topology_support_ids=sorted(
                        {pid for r in role for pid in r["provider_ids"]}
                    ),
                    height_support_ids=["CURRENT_RAW_GRID:" + identity],
                    raw_support_ids=[],
                    boundary_role_evidence=item,
                    movement_basis=reason,
                    review_direction_basis="CURRENT_UPPER_CONTOUR_CONSERVATIVE_INFERENCE_NOT_S4_ESTIMATE",
                    review_direction_delta_deg=float(np.degrees(angle)),
                )
            ]
        output[region.region_id] = result
    cache.review_side_capability_audit = evidence
    return output
