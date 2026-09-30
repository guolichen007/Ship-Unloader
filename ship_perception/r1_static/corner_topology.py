"""B5 vessel-scoped corner graph and conservative rectangular opening solver.

The output is a static review hypothesis. Three independently supported
corners define the fourth geometrically, but do not make it observed steel.
"""

from itertools import product

import numpy as np

from .opening_semantics import _box
from .r2h_b5_corner_forensics import _observations, _side_modes, audit_lineage
from .vessel_height_topology import build_vessel_scoped_topview, persistent_height_modes


SIDES = ("U0", "U1", "V0", "V1")


def _mode_near(rho, modes, tolerance):
    return min((mode for mode in modes if abs(mode["rho"] - rho) <= tolerance),
               key=lambda mode: abs(mode["rho"] - rho), default=None)


def _merge_candidates(primary, height, baseline, config):
    """Keep measured parallel modes separate and attach nearby height evidence."""
    tolerance = config["boundary"]["line_inlier_m"]
    rows = []
    for source in primary:
        if source.get("provider_scope") == "CONTEXT_ONLY":
            continue
        if min(source.get("primary_raw3d", 0), source.get("primary_bev", 0)) <= 0:
            continue
        if not any(abs(source["rho"] - row["rho"]) <= tolerance for row in rows):
            rows.append(dict(source))
    for source in height:
        row = _mode_near(source["rho"], rows, tolerance)
        if row is None:
            rows.append(dict(rho=source["rho"], primary_raw3d=0.0,
                             primary_bev=0.0, provider_ids=[],
                             source_vessel_ids=[], source_support_ids=[],
                             provider_scope="HEIGHT_TOPOLOGY_ONLY"))
    for rho in baseline:
        if _mode_near(rho, rows, tolerance) is None:
            rows.append(dict(rho=float(rho), primary_raw3d=0.0,
                             primary_bev=0.0, provider_ids=[],
                             source_vessel_ids=[], source_support_ids=[],
                             provider_scope="B4_BASELINE_ONLY"))
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
    inside = segment & (inside_sign * (normal - rho) >= probe) & (
        inside_sign * (normal - rho) <= width)
    outside = segment & (-inside_sign * (normal - rho) >= probe) & (
        -inside_sign * (normal - rho) <= width)
    valid_inside = inside & grid.occupancy
    valid_outside = outside & grid.occupancy
    inside_fraction = float(valid_inside.sum() / max(1, inside.sum()))
    outside_fraction = float(valid_outside.sum() / max(1, outside.sum()))
    inside_z = float(np.median(grid.median[valid_inside])) if valid_inside.any() else None
    outside_z = float(np.median(grid.median[valid_outside])) if valid_outside.any() else None
    contrast = abs(inside_z - outside_z) if inside_z is not None and outside_z is not None else 0.0
    return dict(
        inside_occupancy_fraction=inside_fraction,
        outside_occupancy_fraction=outside_fraction,
        inside_median_z_m=inside_z,
        outside_median_z_m=outside_z,
        boundary_asymmetry=float(abs(inside_fraction - outside_fraction)),
        height_contrast_m=float(contrast),
        opening_one_sided=(inside_fraction >= config["roi"]["min_enclosure_ratio"] and
                           outside_fraction <= inside_fraction / 2),
        interior_ridge=(min(inside_fraction, outside_fraction) >=
                        config["roi"]["min_enclosure_ratio"] and
                        abs(inside_fraction - outside_fraction) <
                        config["roi"]["min_enclosure_ratio"] / 2),
    )


def _section_consensus(grid, rho, span, axis, inside_sign, config):
    """Measure the same physical transition in independent occupied sections."""
    edges = np.linspace(span[0], span[1], 6)
    sections = []
    for start, end in zip(edges[:-1], edges[1:]):
        # _strip_relation omits the corner margin, so extend this local span.
        margin = config["boundary"]["corner_join_m"]
        relation = _strip_relation(grid, rho, (start - margin, end + margin),
                                   axis, inside_sign, config)
        if max(relation["inside_occupancy_fraction"],
               relation["outside_occupancy_fraction"]) < config["roi"]["min_enclosure_ratio"]:
            continue
        sections.append(dict(start_m=float(start), end_m=float(end), **relation))
    transitions = sum(row["opening_one_sided"] for row in sections)
    ridges = sum(row["interior_ridge"] for row in sections)
    return dict(section_count=len(sections), transition_count=transitions,
                internal_ridge_section_count=ridges,
                left_endpoint_support=bool(sections and sections[0]["opening_one_sided"]),
                right_endpoint_support=bool(sections and sections[-1]["opening_one_sided"]),
                sections=sections)


def _side_rank(row, relation, old_rho, config, side):
    """Lexicographic evidence; no area, width or larger-rho reward."""
    minimum = config["roi"]["min_enclosure_ratio"]
    dual = min(row["primary_raw3d"], row["primary_bev"])
    height = row["height_level_count"] >= 2
    footprint = row.get("support_limit_validated", False)
    # Height topology is independent of B2 provider ownership. A contour with
    # continuous cargo on both sides needs a unique spatial family; several
    # plausible upper families are withheld by _upper_multimode_conflict.
    viable = (height or footprint or
              (relation["opening_one_sided"] and not relation["interior_ridge"] and
               dual >= minimum))
    if (side == "V0" and relation["interior_ridge"] and
            abs(row["rho"] - old_rho) > config["boundary"]["corner_join_m"]):
        # A loaded hold can show a persistent cargo contour on both sides.
        # Without an opening/outside transition it cannot move the lower rim.
        viable = False
    if side == "V0":
        # The lower support limit distinguishes the vessel from the shore.
        # An old measured side may be retained when independent B1 topology
        # locates the same boundary; otherwise use the first inward dual mode.
        boundary_class = 2 if footprint else 1 if row.get("first_inward_dual") else 0
    else:
        # An occupied-on-both-sides persistent contour defeats outer-hull
        # lines and internal cargo ridges, even when the old box was low.
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
        -round(abs(row["height_rho_m"] - old_rho), 3)
        if side == "V1" and height else 0.0,
        -round(row.get("height_distance_m") or 0.0, 3)
        if side == "V1" and height else 0.0,
        round(relation["boundary_asymmetry"], 3),
        -round(abs(row["rho"] - old_rho), 3),
        round(min(dual, 1.0), 3),
        round(row["height_coverage"], 3),
    )


def _enrich_side(rows, grid, span, axis, sign, old_rho, config, side, support_limit=None):
    result = []
    for source in rows:
        row = dict(source)
        minimum = config["roi"]["min_enclosure_ratio"]
        tolerance = config["boundary"]["corner_join_m"]
        row["support_limit_rho"] = support_limit
        row["support_limit_validated"] = bool(
            side == "V0" and support_limit is not None and
            abs(row["rho"] - old_rho) <= tolerance and
            abs(row["rho"] - support_limit) <= tolerance)
        row["first_inward_dual"] = bool(
            side == "V0" and support_limit is not None and
            support_limit <= row["rho"] <= support_limit + config["roi"]["support_search_m"] / 4 and
            min(row["primary_raw3d"], row["primary_bev"]) >= minimum)
        row["opening_relation"] = _strip_relation(grid, row["rho"], span, axis, sign, config)
        row["section_consensus"] = _section_consensus(
            grid, row["rho"], span, axis, sign, config)
        row["evidence_rank"] = list(_side_rank(
            row, row["opening_relation"], old_rho, config, side))
        result.append(row)
    return sorted(result, key=lambda row: tuple(row["evidence_rank"]), reverse=True)


def _line_reaches(observations, axis, rho, along, provider, config):
    tolerance = config["boundary"]["corner_join_m"]
    return max((row["coverage"] for row in observations if
                row["normal_axis"] == axis and row["source_scope"] == "SAME_VESSEL" and
                row["provider_type"] == provider and abs(row["rho"] - rho) <= tolerance and
                row["along"][0] - tolerance <= along <= row["along"][1] + tolerance),
               default=0.0)


def _local_u_persistence(grid, u0, v0, lower_row, config):
    """Locate a repeated U contour ending beside a transverse support edge."""
    u, v = grid.centers()
    width = config["roi"]["support_search_m"] / 2
    probe = config["boundary"]["side_probe_m"]
    sign = 1 if lower_row else -1
    strip = (sign * (v - v0) >= probe) & (sign * (v - v0) <= width)
    left = strip & (u >= u0 - width) & (u <= u0 - probe) & grid.occupancy
    right = strip & (u >= u0 + probe) & (u <= u0 + width) & grid.occupancy
    if min(int(left.sum()), int(right.sum())) < config["roi"]["min_deck_support_cells"]:
        return dict(level_count=0, position_median=None, position_mad_m=None)
    left_z = float(np.median(grid.median[left]))
    right_z = float(np.median(grid.median[right]))
    if abs(left_z - right_z) < config["roi"]["opening_drop_m"]:
        return dict(level_count=0, position_median=None, position_mad_m=None)
    cols = np.flatnonzero((u[0] >= u0 - width) & (u[0] <= u0 + width))
    rows = np.flatnonzero(strip[:, 0])
    positions = []
    for fraction in (0.25, 0.5, 0.75):
        level = left_z + fraction * (right_z - left_z)
        # Per-column occupancy prevents missing cells from masquerading as a
        # contour. The same XY transition must recur across relative levels.
        column = []
        for col in cols:
            valid = grid.occupancy[rows, col]
            if valid.sum() >= max(1, len(rows) * config["roi"]["min_enclosure_ratio"]):
                column.append(float(np.median(grid.median[rows[valid], col])))
            else:
                column.append(np.nan)
        values = np.asarray(column)
        live = np.isfinite(values[:-1]) & np.isfinite(values[1:])
        crossings = np.flatnonzero(live & ((values[:-1] >= level) != (values[1:] >= level)))
        if len(crossings):
            interfaces = grid.x0 + (cols[crossings] + 1) * grid.cell_m
            position = min(interfaces, key=lambda value: abs(value - u0))
            if abs(position - u0) <= config["boundary"]["corner_join_m"]:
                positions.append(float(position))
    if not positions:
        return dict(level_count=0, position_median=None, position_mad_m=None)
    position = float(np.median(positions))
    return dict(level_count=len(positions), position_median=position,
                position_mad_m=float(np.median(np.abs(np.asarray(positions) - position))))


def _corner(u_mode, v_mode, observations, grid, inside_index, config,
            persistent_corners=()):
    u0, v0 = u_mode["rho"], v_mode["rho"]
    raw = "RAW3D_STRUCTURAL_PROVIDER"
    bev = "BEV_RECTILINEAR_PROVIDER"
    u_raw = _line_reaches(observations, 0, u0, v0, raw, config)
    u_bev = _line_reaches(observations, 0, u0, v0, bev, config)
    v_raw = _line_reaches(observations, 1, v0, u0, raw, config)
    v_bev = _line_reaches(observations, 1, v0, u0, bev, config)
    # The inside quadrant must contain measured vessel-owned points. An outer
    # hull or quay L with two empty outside quadrants is not a Hatch corner.
    u, v = grid.centers()
    width = config["roi"]["support_search_m"] / 2
    signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    coverage = []
    height = []
    for su, sv in signs:
        window = ((su * (u - u0) > 0) & (su * (u - u0) <= width) &
                  (sv * (v - v0) > 0) & (sv * (v - v0) <= width))
        owned = window & grid.occupancy
        coverage.append(float(owned.sum() / max(1, window.sum())))
        height.append(float(np.median(grid.median[owned])) if owned.any() else None)
    inside = coverage[inside_index]
    outer_conflict = sum(value < config["roi"]["min_enclosure_ratio"] / 4
                         for value in coverage) >= 2
    structural_u = min(u_raw, u_bev)
    structural_v = min(v_raw, v_bev)
    height_u = u_mode["height_level_count"] >= 2
    height_v = v_mode["height_level_count"] >= 2
    footprint_v = v_mode.get("support_limit_validated", False)
    persistent = min((row for row in persistent_corners if
                      np.linalg.norm(np.asarray(row["corner_xy"]) - [u0, v0]) <=
                      config["boundary"]["corner_join_m"] * 2),
                     key=lambda row: np.linalg.norm(np.asarray(row["corner_xy"]) -
                                                    [u0, v0]), default=None)
    local_u = _local_u_persistence(grid, u0, v0, inside_index in (2, 3), config)
    minimum = config["roi"]["min_enclosure_ratio"]
    measured = min(structural_u, structural_v) >= minimum
    topology = (height_u or structural_u >= minimum or local_u["level_count"] >= 2 or
                (u_raw >= minimum and local_u["level_count"] >= 1)) and (
        height_v or footprint_v or structural_v >= minimum)
    # A missing outside quadrant can also be B1's shore split. It prevents a
    # measured-steel claim, but does not erase independently measured topology.
    supported = inside >= minimum / 2 and topology
    role = ("MEASURED_INTERSECTION_CANDIDATE" if measured and not outer_conflict else
            "HEIGHT_TOPOLOGY_CORNER" if supported and persistent is not None and
            not outer_conflict else
            "FOOTPRINT_CONTEXT_CORNER" if supported and footprint_v and
            (not outer_conflict or local_u["level_count"] >= 2 or
             (u_raw >= minimum and local_u["level_count"] >= 1)) else
            "UNRESOLVED_CORNER")
    return dict(
        corner_xy=[float(u0), float(v0)],
        role=role,
        inside_quadrant=inside_index,
        vessel_quadrant_occupancy=coverage,
        median_z_quadrants=height,
        outer_hull_conflict=outer_conflict,
        u_raw3d=u_raw, u_bev=u_bev, v_raw3d=v_raw, v_bev=v_bev,
        u_height_levels=u_mode["height_level_count"],
        v_height_levels=v_mode["height_level_count"],
        footprint_context_corner=footprint_v,
        persistent_height_corner_id=(persistent["contour_ids"] if persistent else []),
        persistent_height_corner_level_count=(persistent["level_count"] if persistent else 0),
        local_u_persistence=local_u,
        position_mad_m=max(u_mode.get("height_position_mad_m") or 0.0,
                           v_mode.get("height_position_mad_m") or 0.0),
    )


def solve_three_corners(corners):
    """Three supported orthogonal corners determine one geometric fourth."""
    valid = [row for row in corners if row["role"] in (
        "MEASURED_INTERSECTION_CANDIDATE", "HEIGHT_TOPOLOGY_CORNER",
        "FOOTPRINT_CONTEXT_CORNER")]
    positions = {tuple(row["corner_xy"]) for row in valid}
    if len(positions) < 3:
        return dict(state="UNRESOLVED", inferred_corner=None)
    measured = sum(row["role"] == "MEASURED_INTERSECTION_CANDIDATE" for row in valid)
    independently_supported = sum(
        row["role"] in ("MEASURED_INTERSECTION_CANDIDATE", "HEIGHT_TOPOLOGY_CORNER") or
        (row["role"] == "FOOTPRINT_CONTEXT_CORNER" and
         row.get("local_u_persistence", {}).get("level_count", 0) >= 2 and
         row.get("v_height_levels", 0) >= 2)
        for row in valid)
    # Footprint alone never closes a Hatch. A real Raw3D/BEV intersection is
    # still required when local height topology supplies the other corners.
    if independently_supported < 2 or (measured == 0 and all(
            row["role"] == "FOOTPRINT_CONTEXT_CORNER" for row in valid)):
        return dict(state="UNRESOLVED", inferred_corner=None)
    x_values = sorted({xy[0] for xy in positions})
    y_values = sorted({xy[1] for xy in positions})
    if len(x_values) != 2 or len(y_values) != 2:
        return dict(state="UNRESOLVED", inferred_corner=None)
    missing = [(x, y) for x, y in product(x_values, y_values) if (x, y) not in positions]
    if len(missing) == 1:
        return dict(state="PARTIAL_HATCH_3C", inferred_corner=list(missing[0]),
                    inferred_role="INFERRED_GEOMETRIC_CORNER")
    return dict(state="PARTIAL_HATCH_4C_REVIEW", inferred_corner=None)


def _cell_corners(left, right, lower, upper, observations, grid, config):
    return [
        _corner(left, lower, observations, grid, 2, config),
        _corner(right, lower, observations, grid, 3, config),
        _corner(right, upper, observations, grid, 0, config),
        _corner(left, upper, observations, grid, 1, config),
    ]


def _side_provenance(mode, config):
    minimum = config["roi"]["min_enclosure_ratio"]
    dual = min(mode["primary_raw3d"], mode["primary_bev"])
    if dual >= minimum and mode["height_level_count"] >= 2:
        return "RAW3D_BEV_PLUS_HEIGHT_TOPOLOGY"
    if mode.get("support_limit_validated"):
        return "VESSEL_FOOTPRINT_CONTEXT_EDGE_NOT_STEEL"
    if dual >= minimum:
        return "RAW3D_BEV_EDGE_CANDIDATE"
    if mode["height_level_count"] >= 2:
        return "HEIGHT_TOPOLOGY_SUPPORTED_EDGE"
    return "GEOMETRY_CONSTRAINED_EDGE"


def _pair_rank(lower, upper, cells):
    unique = {tuple(c["corner_xy"]): c for cell in cells for c in cell}
    supported = sum(c["role"] != "UNRESOLVED_CORNER" for c in unique.values())
    measured = sum(c["role"] == "MEASURED_INTERSECTION_CANDIDATE"
                   for c in unique.values())
    min_cell = min(sum(c["role"] != "UNRESOLVED_CORNER" for c in cell)
                   for cell in cells)
    if min_cell < 3:
        return None
    # Rank physical edge evidence before fourth-corner completeness. Three
    # independently supported corners geometrically close the rectangle.
    return (tuple(lower["evidence_rank"]), tuple(upper["evidence_rank"]),
            min_cell, supported, measured)


def _upper_multimode_conflict(height_modes, upper_candidates, config):
    """Fail closed when loaded cargo supports several unseparated top rims."""
    persistent = sorted(row["rho"] for row in height_modes if row["level_count"] >= 2)
    distinct = []
    for rho in persistent:
        if not distinct or rho - distinct[-1] > config["boundary"]["corner_join_m"]:
            distinct.append(rho)
    viable = [row for row in upper_candidates if row["evidence_rank"][0] and
              row["height_level_count"] >= 2]
    return bool(len(distinct) >= 2 and viable and all(
        row["opening_relation"]["interior_ridge"] for row in viable))


def _section_rho_modes(rows, side, config):
    """Independently choose local rho evidence, then report median and MAD."""
    active = [row for row in rows if row["evidence_rank"][0]]
    count = max((len(row["section_consensus"]["sections"]) for row in active), default=0)
    selected = []
    for index in range(count):
        candidates = [row for row in active if len(row["section_consensus"]["sections"]) > index]
        if not candidates:
            continue
        def quality(row):
            local = row["section_consensus"]["sections"][index]
            structural = min(row["primary_raw3d"], row["primary_bev"])
            if side == "V0":
                return (int(local["opening_one_sided"]),
                        round(local["boundary_asymmetry"], 3), structural)
            return (int(row["height_level_count"] >= 2),
                    -int(local["interior_ridge"]),
                    round(local["boundary_asymmetry"], 3), structural)
        winner = max(candidates, key=quality)
        selected.append(float(winner["rho"]))
    if not selected:
        return dict(section_rho_modes=[], section_count=0,
                    section_rho_median=None, section_rho_mad=None)
    median = float(np.median(selected))
    return dict(section_rho_modes=selected, section_count=len(selected),
                section_rho_median=median,
                section_rho_mad=float(np.median(np.abs(np.asarray(selected) - median))))


def arbitrate_unresolved_side(side, side_id, vessel_id, config,
                            neighboring_opening=False):
    """Reject a later large inward move supported by mixed context and a weak transition.

    The older edge still remains a review candidate, never certified steel.
    Tiny B4 adjustments are preserved because raster registration alone can
    account for them. The rule uses physical evidence, not a scene identifier.
    """
    older = float(side["rho_before"])
    later = float(side["rho_after"])
    join = config["boundary"]["corner_join_m"]
    record = dict(r2g_rho=older, b4_rho=later, final_rho=later,
                  final_stage="B4", rejection_reason=None,
                  evidence_stronger_than_replaced_stage=None)
    if abs(later - older) <= config["roi"]["support_search_m"] / 4:
        return record
    # A high-contrast line at the end of a single vessel can be its outer
    # hull. A measured gap to another opening is independent end evidence.
    if side_id not in ("U0", "U1") or not neighboring_opening:
        return record
    if vessel_id not in side.get("selected_source_vessel_ids", []):
        return record
    if set(side.get("selected_source_vessel_ids", [])) <= {vessel_id}:
        return record
    if side.get("previous_role") != "INNER_EDGE":
        return record
    modes = side.get("candidate_modes", [])
    prior = min(modes, key=lambda row: abs(row["rho"] - older), default=None)
    chosen = min(modes, key=lambda row: abs(row["rho"] - later), default=None)
    if prior is None or chosen is None or abs(prior["rho"] - older) > join:
        return record
    providers = prior.get("provider_ids", [])
    own_raw = any(value.startswith(vessel_id + ":RAW3D_STRUCTURAL_PROVIDER:")
                  for value in providers)
    own_bev = any(value.startswith(vessel_id + ":BEV_RECTILINEAR_PROVIDER:")
                  for value in providers)
    minimum = config["roi"]["min_enclosure_ratio"]
    prior_contrast = prior.get("opening_relation", {}).get("outside_minus_inside_m")
    chosen_contrast = chosen.get("opening_relation", {}).get("outside_minus_inside_m")
    if (own_raw and own_bev and
            min(prior.get("raw3d_coverage", 0), prior.get("bev_coverage", 0)) >= minimum and
            prior.get("opening_relation", {}).get("canonical_inner") and
            prior_contrast is not None and chosen_contrast is not None and
            prior_contrast >= chosen_contrast + config["roi"]["opening_drop_m"]):
        record.update(final_rho=older, final_stage="R2G",
                      rejection_reason="B4_MIXED_CONTEXT_WEAKER_OPENING_END",
                      evidence_stronger_than_replaced_stage=True,
                      r2g_candidate_rho=prior["rho"],
                      r2g_opening_contrast_m=prior_contrast,
                      b4_opening_contrast_m=chosen_contrast)
    return record


def _has_neighboring_opening(side_id, rho_before, current_box, hatch_id,
                            opening_context, config):
    """Check adjacent cells across all same-vessel groups, not just one recursion leaf."""
    if side_id not in ("U0", "U1"):
        return False
    join = config["boundary"]["corner_join_m"]
    reach = config["roi"]["support_search_m"]
    for other_id, other in opening_context:
        if other_id == hatch_id or min(other[3], current_box[3]) - max(
                other[1], current_box[1]) < join:
            continue
        gap = (other[0] - rho_before if side_id == "U1" else
               rho_before - other[2])
        if 0 < gap <= reach:
            return True
    return False


def solve_vessel_openings(points, b2, b4, private_indexes, vessel_id, config,
                          _opening_context=None):
    """Solve shared U chain and V rows using vessel-owned height and B2 lines."""
    hatches = [row for row in b4["rectangles"] if row.get("vessel_hypothesis_id") == vessel_id
               and row.get("polygon_after") is not None]
    if not hatches:
        return dict(vessel_hypothesis_id=vessel_id, rectangles=[], status="NO_B4_OPENING")
    axes = np.asarray(hatches[0]["axes"])
    if _opening_context is None:
        _opening_context = [(row["hatch_id"], _box(row["polygon_after"], axes))
                            for row in hatches]
    # A common U station is a separator only when two opening cells actually
    # touch. Distant hatches (notably 8-22) keep independent U stations and V
    # rows; pairing consecutive stations across their gap creates a false cell.
    ordered_hatches = sorted(hatches, key=lambda row: _box(row["polygon_after"], axes)[0])
    groups = []
    join = config["boundary"]["corner_join_m"]
    for hatch in ordered_hatches:
        box = _box(hatch["polygon_after"], axes)
        if groups:
            previous = _box(groups[-1][-1]["polygon_after"], axes)
            vertical_overlap = min(previous[3], box[3]) - max(previous[1], box[1])
            if (abs(previous[2] - box[0]) <= join and
                    vertical_overlap >= join):
                groups[-1].append(hatch)
                continue
        groups.append([hatch])
    if len(groups) > 1:
        children = [solve_vessel_openings(
            points, b2, dict(rectangles=group), private_indexes, vessel_id, config,
            _opening_context=_opening_context)
            for group in groups]
        result = dict(children[0])
        result["status"] = "INDEPENDENT_OPENING_GROUPS"
        result["independent_groups"] = children
        result["rectangles"] = [row for child in children for row in child["rectangles"]]
        result["rejected_external_points"] = children[0]["rejected_external_points"]
        lineage = audit_lineage(dict(rectangles=[
            dict(hatch, polygon_before=hatch["polygon_after"])
            for hatch in hatches]), b2, config)
        assignments = {row["hatch_id"]: row for row in
                       lineage["one_to_one_forensic_assignment"]}
        for row in result["rectangles"]:
            row["hatch_hypothesis_id"] = assignments[row["hatch_id"]]["proposal_id"]
            row["lineage_status"] = assignments[row["hatch_id"]]["status"]
        result["lineage"] = lineage
        return result
    scoped = build_vessel_scoped_topview(points, private_indexes, axes, config)
    fine = scoped["grids"]["fine"]
    bounds = [_box(row["polygon_after"], axes) for row in hatches]
    envelope = [min(box[0] for box in bounds), min(box[1] for box in bounds),
                max(box[2] for box in bounds), max(box[3] for box in bounds)]
    height = persistent_height_modes(fine, envelope, config)
    margin = config["roi"]["support_search_m"]
    roi = [envelope[0] - margin, envelope[1] - margin,
           envelope[2] + margin, envelope[3] + margin]
    observations = _observations(b2, vessel_id, axes, roi)
    stations = sorted(value for box in bounds for value in (box[0], box[2]))
    station_groups = []
    for value in stations:
        if station_groups and abs(value - np.median(station_groups[-1])) <= join:
            station_groups[-1].append(value)
        else:
            station_groups.append([value])
    u_stations = [float(np.median(group)) for group in station_groups]
    if len(u_stations) != len(hatches) + 1:
        raise ValueError("OPENING_U_STATION_TOPOLOGY_AMBIGUOUS")
    u_chain = []
    for old in u_stations:
        nearby = _side_modes(observations, 0, (envelope[1], envelope[3]),
                             (old - margin, old + margin), config)
        sources = _merge_candidates(nearby, height["modes"][0], [old], config)
        local = [row for row in sources if abs(row["rho"] - old) <= margin and
                 min(row["primary_raw3d"], row["primary_bev"]) >=
                 config["roi"]["min_enclosure_ratio"]]
        chosen = min(local, key=lambda row: abs(row["rho"] - old)) if local else min(
            sources, key=lambda row: abs(row["rho"] - old))
        u_chain.append(chosen)
    # A multi-hatch cell graph shares exactly the same measured transverse
    # separator station, even if two old B4 polygons listed it separately.
    span_u = (envelope[0], envelope[2])
    midpoint = (envelope[1] + envelope[3]) / 2
    candidates = {}
    for side, band, sign in (("V0", (roi[1], midpoint), 1),
                             ("V1", (midpoint, roi[3]), -1)):
        old = float(np.median([box[1 if side == "V0" else 3] for box in bounds]))
        primary = _side_modes(observations, 1, span_u, band, config)
        # Recover measured modes near the old side even if the global top-16
        # shortlist is crowded by parallel cargo bands.
        primary += _side_modes(observations, 1, span_u, (old - margin, old + margin), config)
        merged = _merge_candidates(primary, height["modes"][1], [old], config)
        merged = [row for row in merged if band[0] <= row["rho"] <= band[1]]
        if side == "V0":
            # The lower footprint is often the outer hull. A persistent cargo
            # contour there is still not a steel measurement (7-16). This
            # refinement may move the opening inward; a large outward move
            # requires an independent edge review and remains at B4 here.
            merged = [row for row in merged if
                      row["rho"] >= old - config["boundary"]["corner_join_m"]]
        support_limit = None
        if side == "V0":
            support_rows = [row for row in height["footprint_edges"][1]
                            if old - margin <= row["rho"] <= old + margin]
            if support_rows:
                support_limit = min(row["rho"] for row in support_rows)
        candidates[side] = _enrich_side(
            merged, fine, span_u, 1, sign, old, config, side, support_limit)
    pairs = []
    corner_cache = {}

    def cached_corner(u_mode, v_mode, inside_index):
        key = (u_mode["rho"], v_mode["rho"], inside_index)
        if key not in corner_cache:
            corner_cache[key] = _corner(
                u_mode, v_mode, observations, fine, inside_index, config,
                height["persistent_corners"])
        return corner_cache[key]

    for lower, upper in product(candidates["V0"], candidates["V1"]):
        if lower["rho"] >= upper["rho"] or not lower["evidence_rank"][0] or not upper["evidence_rank"][0]:
            continue
        cells = [[cached_corner(left, lower, 2), cached_corner(right, lower, 3),
                  cached_corner(right, upper, 0), cached_corner(left, upper, 1)]
                 for left, right in zip(u_chain, u_chain[1:])]
        score = _pair_rank(lower, upper, cells)
        if score is None:
            continue
        pairs.append((score, lower, upper, cells))
    pairs.sort(key=lambda row: row[0], reverse=True)
    blocker = ("MULTIPLE_PERSISTENT_UPPER_RIM_FAMILIES" if
               _upper_multimode_conflict(height["modes"][1], candidates["V1"], config)
               else None)
    chosen = pairs[0] if pairs and blocker is None else None
    ordered = sorted(zip(hatches, bounds), key=lambda pair: pair[1][0])
    outputs = []
    for index, (hatch, old_box) in enumerate(ordered):
        if chosen is None:
            lower = upper = None
            corners = []
            geometry = dict(state="UNRESOLVED", inferred_corner=None)
        else:
            _, lower, upper, cells = chosen
            corners = cells[index]
            geometry = solve_three_corners(corners)
        arbitration = {}
        if geometry["state"] == "UNRESOLVED":
            bounds_after = list(old_box)
            for side_id, position in zip(SIDES, (0, 2, 1, 3)):
                older_rho = float(hatch["sides"][side_id]["rho_before"])
                nearby_gap = _has_neighboring_opening(
                    side_id, older_rho, old_box, hatch["hatch_id"],
                    _opening_context, config)
                decision = arbitrate_unresolved_side(
                    hatch["sides"][side_id], side_id, vessel_id, config,
                    neighboring_opening=nearby_gap)
                arbitration[side_id] = decision
                bounds_after[position] = decision["final_rho"]
        else:
            bounds_after = [u_chain[index]["rho"], lower["rho"],
                            u_chain[index + 1]["rho"], upper["rho"]]
            arbitration = {side_id: dict(
                r2g_rho=float(hatch["sides"][side_id]["rho_before"]),
                b4_rho=float(hatch["sides"][side_id]["rho_after"]),
                final_rho=float(bounds_after[position]), final_stage="B5",
                rejection_reason=None, evidence_stronger_than_replaced_stage=True)
                for side_id, position in zip(SIDES, (0, 2, 1, 3))}
        polygon = (np.asarray([[bounds_after[0], bounds_after[1]],
                               [bounds_after[2], bounds_after[1]],
                               [bounds_after[2], bounds_after[3]],
                               [bounds_after[0], bounds_after[3]]]) @ axes).tolist()
        side_modes = dict(U0=u_chain[index], U1=u_chain[index+1], V0=lower, V1=upper)
        outputs.append(dict(
            hatch_id=hatch["hatch_id"], vessel_hypothesis_id=vessel_id,
            hatch_hypothesis_id=None,
            polygon_before=hatch["polygon_after"], polygon_after=polygon,
            status=geometry["state"], inferred_corner=geometry["inferred_corner"],
            corners=corners,
            side_stage_arbitration=arbitration,
            sides={key: dict(rho_before=old_box[0 if key == "U0" else
                                               2 if key == "U1" else
                                               1 if key == "V0" else 3],
                             rho_after=bounds_after[0 if key == "U0" else
                                                   2 if key == "U1" else
                                                   1 if key == "V0" else 3],
                             role=(("R2G_RESTORED_REVIEW" if arbitration[key]["final_stage"] == "R2G"
                                    else "B4_UNCHANGED_UNRESOLVED") if geometry["state"] ==
                                   "UNRESOLVED" else
                                   _side_provenance(side_modes[key], config) if
                                   side_modes[key] else "GEOMETRY_CONSTRAINED_EDGE"),
                             selected_source_vessel_ids=[vessel_id] if
                             geometry["state"] != "UNRESOLVED" and side_modes[key] and
                             min(side_modes[key]["primary_raw3d"],side_modes[key]["primary_bev"])
                             >= config["roi"]["min_enclosure_ratio"] else [],
                             height_level_count=side_modes[key]["height_level_count"] if
                             geometry["state"] != "UNRESOLVED" and side_modes[key] else 0)
                   for key in SIDES},
        ))
    # Lineage is assigned after the independent opening cells exist.
    lineage = audit_lineage(dict(rectangles=[dict(hatch, polygon_before=hatch["polygon_after"])
                                            for hatch in hatches]), b2, config)
    assignments = {row["hatch_id"]: row for row in lineage["one_to_one_forensic_assignment"]}
    for row in outputs:
        row["hatch_hypothesis_id"] = assignments[row["hatch_id"]]["proposal_id"]
        row["lineage_status"] = assignments[row["hatch_id"]]["status"]
    return dict(
        vessel_hypothesis_id=vessel_id,
        point_selection=scoped["point_selection"],
        vessel_shell_reference_z_m=scoped["vessel_shell_reference_z_m"],
        vessel_shell_z_mad_m=scoped["vessel_shell_z_mad_m"],
        vessel_footprint_cells=scoped["vessel_footprint_cells"],
        b1_owned_point_count=scoped["b1_owned_point_count"],
        restored_inside_footprint_points=scoped["restored_inside_footprint_points"],
        rejected_external_points=scoped["rejected_external_points"],
        height_topology=height,
        u_boundary_chain=u_chain,
        v0_candidates=candidates["V0"], v1_candidates=candidates["V1"],
        section_rho_consensus={side: _section_rho_modes(candidates[side], side, config)
                               for side in ("V0", "V1")},
        pair_candidates=[dict(v0=row[1]["rho"], v1=row[2]["rho"],
                              score=str(row[0]),
                              supported_unique_corners=len({tuple(c["corner_xy"])
                                                            for cell in row[3] for c in cell
                                                            if c["role"] != "UNRESOLVED_CORNER"}),
                              measured_unique_corners=len({tuple(c["corner_xy"])
                                                           for cell in row[3] for c in cell
                                                           if c["role"] ==
                                                           "MEASURED_INTERSECTION_CANDIDATE"}))
                         for row in pairs[:10]],
        selected_pair=(dict(v0=chosen[1]["rho"],v1=chosen[2]["rho"],
                            score=str(chosen[0])) if chosen else None),
        selection_blocker=blocker,
        shared_separator=(u_chain[1]["rho"] if len(u_chain) == 3 else None),
        lineage=lineage,
        rectangles=outputs,
    )
