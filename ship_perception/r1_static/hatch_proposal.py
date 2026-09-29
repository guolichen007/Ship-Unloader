"""R2H-B2 static, per-vessel Hatch proposal evidence and conservative states.

This is a recall layer. Provider lines are observations, never steel labels;
only four supported sides with an opening-role observation get a provisional
polygon. The existing R2F/R2G calibration path remains a separate gate.
"""

import math

import numpy as np
from scipy import ndimage

from .height_grid import R1HeightGrid
from .heightmap_topology import detect_regions
from .raw_topview_evidence import build_topview, raster_modes
from .scene_vessel import infer_scene_vessels


def _world_line(axes, normal_axis, rho, start, end):
    a = (rho, start) if normal_axis == 0 else (start, rho)
    b = (rho, end) if normal_axis == 0 else (end, rho)
    basis = np.asarray(axes)
    return [list(np.asarray(a) @ basis), list(np.asarray(b) @ basis)]


def _measured_intervals(values, gap, cell):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        return []
    result, start, previous = [], float(values[0]), float(values[0])
    for value in values[1:]:
        value = float(value)
        if value - previous > gap:
            result.append([start - cell / 2, previous + cell / 2])
            start = value
        previous = value
    result.append([start - cell / 2, previous + cell / 2])
    return result


def _record(vessel, kind, index, geometry_type, source_geometry, interval,
            rho_or_region, coverage, uncertainty, evidence_types, conflicts=()):
    vessel_id = vessel["vessel_hypothesis_id"]
    return dict(provider_id="%s:%s:%05d" % (vessel_id, kind, index),
                provider_type=kind, vessel_hypothesis_id=vessel_id,
                source_support_ids=vessel["source_support_ids"],
                observation_id="%s:%s:STATIC:%05d" % (vessel_id, kind, index),
                frame_id="STATIC_SINGLE_FRAME", sensor_id="UNKNOWN",
                coordinate_frame="RAW_INPUT_FRAME", geometry_type=geometry_type,
                source_geometry=source_geometry,
                local_axis_family=vessel["local_axes"],
                support_interval=interval, rho_or_region=rho_or_region,
                coverage=float(coverage), uncertainty_m=float(uncertainty),
                evidence_types=list(evidence_types), conflicts=list(conflicts),
                role="UNKNOWN")


def _height_provider(points, vessel, config, research):
    """Run the frozen Height algorithm only on this vessel's Raw XYZ."""
    trace = {}
    result, _ = detect_regions(points, config, research, forensic_trace=trace)
    axes = np.asarray(vessel["local_axes"])
    evidence = []
    for region in result["regions"]:
        bbox = region["bbox_xy"]
        corners = np.asarray(((bbox[0], bbox[1]), (bbox[0], bbox[3]),
                              (bbox[2], bbox[1]), (bbox[2], bbox[3]))) @ axes.T
        axial = [float(corners[:, 0].min()), float(corners[:, 1].min()),
                 float(corners[:, 0].max()), float(corners[:, 1].max())]
        evidence.append(_record(
            vessel, "HEIGHT_BASIN_PROVIDER", len(evidence), "REGION_XY",
            dict(bbox_xy=bbox, axial_bounds=axial, source_node=region["source_node"]),
            None, dict(region_id=region["region_id"]), region["occupancy"],
            config["geometry"]["coarse_voxel_m"], ["HEIGHT_BASIN"],
            ["HEIGHT_IS_PROPOSAL_NOT_STEEL"]))
    dropped = []
    for row in trace.get("dropped_by_global_relative_score", []):
        dropped.append(dict(**row, vessel_hypothesis_id=vessel["vessel_hypothesis_id"],
                            reason="WITHIN_VESSEL_RELATIVE_SCORE_FILTER"))
    return evidence, dict(region_count=result["region_count"],
                          dropped_within_vessel=dropped)


def _bev_provider(points, vessel, config):
    """Measured Raw XYZ raster transitions; no Height input or region ROI."""
    axes = vessel["local_axes"]
    grid = build_topview(points, axes, config)["fine"]
    modes = raster_modes(grid, config)
    evidence = []
    for normal_axis in (0, 1):
        for mode in modes[normal_axis]:
            interval = [mode["along_min"], mode["along_max"]]
            length = max(interval[1] - interval[0], grid.cell_m)
            evidence.append(_record(
                vessel, "BEV_RECTILINEAR_PROVIDER", len(evidence),
                "AXIAL_LINE_INTERVAL",
                dict(endpoints_xy=_world_line(axes, normal_axis, mode["rho"], *interval),
                     channel_counts=mode["channel_counts"],
                     observed_intervals=_measured_intervals(
                         mode["along_values"], config["boundary"]["max_support_gap_m"],
                         grid.cell_m)), interval,
                dict(normal_axis=normal_axis, rho=mode["rho"]),
                min(1.0, mode["support_count"] * grid.cell_m / length),
                grid.cell_m / 2, mode["evidence_types"]))
    return evidence, grid


def _raw3d_provider(points, vessel, config):
    """Direct Raw XYZ section profiles and vertical spans in this local axis.

    This reads neither Height regions nor BEV provider results. A profile break
    or tall cell is structural evidence with unknown physical role until the
    opening association examines its interior and surroundings.
    """
    axes = np.asarray(vessel["local_axes"])
    axial = np.column_stack((np.asarray(points[:, :2]) @ axes.T, points[:, 2]))
    grid = R1HeightGrid.from_points(axial, config["geometry"]["coarse_voxel_m"])
    valid = grid.occupancy
    span = grid.q90 - grid.q10
    evidence = []
    raw_xy = axial[:, :2]

    def verified_vertical_face(normal_axis, rho, interval):
        normal = raw_xy[:, normal_axis]
        along = raw_xy[:, 1 - normal_axis]
        band = max(config["geometry"]["refine_voxel_m"],
                   config["boundary"]["profile_step_m"])
        mask = (np.abs(normal - rho) <= band) & (along >= interval[0]) & (along <= interval[1])
        face_points = axial[mask]
        if len(face_points) < config["geometry"]["min_plane_points"]:
            return False
        if np.ptp(face_points[:, 2]) < config["boundary"]["face_vertical_span_min_m"]:
            return False
        if len(face_points) > 3000:
            face_points = face_points[np.linspace(0, len(face_points)-1, 3000).astype(int)]
        _, _, vectors = np.linalg.svd(face_points - face_points.mean(axis=0),
                                      full_matrices=False)
        return bool(abs(vectors[-1, 2]) <= math.sin(math.radians(
            config["geometry"]["normal_angle_deg"])))
    for normal_axis in (0, 1):
        if normal_axis == 0:
            near, far = (slice(None), slice(None, -1)), (slice(None), slice(1, None))
            origin, along_origin = grid.x0, grid.y0
        else:
            near, far = (slice(None, -1), slice(None)), (slice(1, None), slice(None))
            origin, along_origin = grid.y0, grid.x0
        paired = valid[near] & valid[far]
        profile = paired & (np.abs(grid.median[near] - grid.median[far]) >=
                            config["boundary"]["profile_min_drop_m"])
        face = paired & (np.maximum(span[near], span[far]) >=
                         config["boundary"]["face_vertical_span_min_m"])
        extent = profile.shape[1 if normal_axis == 0 else 0]
        for normal_index in range(extent):
            profile_line = profile[:, normal_index] if normal_axis == 0 else profile[normal_index]
            face_line = face[:, normal_index] if normal_axis == 0 else face[normal_index]
            observed = profile_line | face_line
            if int(observed.sum()) * grid.cell_m < config["boundary"]["line_min_length_m"]:
                continue
            max_gap = max(1, int(math.ceil(config["boundary"]["max_support_gap_m"] /
                                           grid.cell_m)))
            joined = ndimage.binary_closing(observed, structure=np.ones(max_gap, bool)) | observed
            labels, count = ndimage.label(joined)
            rho = origin + (normal_index + 1) * grid.cell_m
            for label_id in range(1, count + 1):
                indexes = np.flatnonzero(observed & (labels == label_id))
                if len(indexes) * grid.cell_m < config["boundary"]["line_min_length_m"]:
                    continue
                interval = [along_origin + (float(indexes.min()) + .5) * grid.cell_m,
                            along_origin + (float(indexes.max()) + .5) * grid.cell_m]
                types = []
                if np.any(profile_line[indexes]):
                    types.append("RAW3D_PROFILE_BREAK")
                if np.any(face_line[indexes]):
                    types.append("RAW3D_VERTICAL_SPAN")
                    if verified_vertical_face(normal_axis, rho, interval):
                        types.append("RAW3D_VERIFIED_VERTICAL_FACE")
                observed_along = along_origin + (indexes + .5) * grid.cell_m
                evidence.append(_record(
                    vessel, "RAW3D_STRUCTURAL_PROVIDER", len(evidence),
                    "AXIAL_LINE_INTERVAL",
                    dict(endpoints_xy=_world_line(axes, normal_axis, rho, *interval),
                         raw_section_count=int(len(indexes)),
                         observed_intervals=_measured_intervals(
                             observed_along, config["boundary"]["max_support_gap_m"],
                             grid.cell_m)), interval,
                    dict(normal_axis=normal_axis, rho=float(rho)),
                    min(1.0, len(indexes) * grid.cell_m /
                        max(interval[1] - interval[0], grid.cell_m)),
                    grid.cell_m / 2, types))
    return evidence


def _axial_extent(points, axes):
    xy = np.asarray(points[:, :2]) @ np.asarray(axes).T
    return [float(xy[:, 0].min()), float(xy[:, 1].min()),
            float(xy[:, 0].max()), float(xy[:, 1].max())]


def _seed_lines(lines, normal_axis, extent, config):
    """Thin only the *search seeds*; all physical lines stay in provenance."""
    rows = [row for row in lines if row["rho_or_region"]["normal_axis"] == normal_axis]
    rows.sort(key=lambda row: (row["coverage"] *
                               (row["support_interval"][1] - row["support_interval"][0]),
                               row["coverage"]), reverse=True)
    selected = []
    distance = config["roi"]["support_search_m"] / 2
    for row in rows:
        rho = row["rho_or_region"]["rho"]
        if not extent[normal_axis] <= rho <= extent[normal_axis + 2]:
            continue
        if any(abs(rho - prior["rho_or_region"]["rho"]) < distance and
               min(row["support_interval"][1], prior["support_interval"][1]) >=
               max(row["support_interval"][0], prior["support_interval"][0])
               for prior in selected):
            continue
        selected.append(row)
    return selected


def _association_seeds(evidence, extent, config):
    lines = [row for row in evidence if row["geometry_type"] == "AXIAL_LINE_INTERVAL"]
    seeds = []
    for row in evidence:
        if row["geometry_type"] == "REGION_XY":
            seeds.append(dict(bounds_axial=row["source_geometry"]["axial_bounds"],
                              source_provider_ids=[row["provider_id"]],
                              seed_type="HEIGHT_REGION", anchor_axis=None))
    for normal_axis in (0, 1):
        selected = _seed_lines(lines, normal_axis, extent, config)
        for index, first in enumerate(selected):
            for second in selected[index + 1:]:
                rho_a, rho_b = (first["rho_or_region"]["rho"],
                                second["rho_or_region"]["rho"])
                if abs(rho_a - rho_b) <= 2 * config["boundary"]["side_probe_m"]:
                    continue
                along = [max(first["support_interval"][0], second["support_interval"][0]),
                         min(first["support_interval"][1], second["support_interval"][1])]
                if along[1] - along[0] < config["boundary"]["line_min_length_m"]:
                    continue
                pair = sorted((rho_a, rho_b))
                bounds = [pair[0], along[0], pair[1], along[1]] if normal_axis == 0 else [
                    along[0], pair[0], along[1], pair[1]]
                seeds.append(dict(bounds_axial=bounds,
                                  source_provider_ids=[first["provider_id"], second["provider_id"]],
                                  seed_type="MEASURED_PARALLEL_PAIR",
                                  anchor_axis=normal_axis))
    return seeds


def _coverage(intervals, start, end):
    clipped = sorted((max(start, a), min(end, b)) for a, b in intervals
                     if b > start and a < end)
    if not clipped or end <= start:
        return 0.0
    length, lo, hi = 0.0, *clipped[0]
    for a, b in clipped[1:]:
        if a > hi:
            length += hi - lo
            lo, hi = a, b
        else:
            hi = max(hi, b)
    return float(min(1.0, (length + hi - lo) / (end - start)))


def _side_evidence(bounds, lines, config, seed):
    u0, v0, u1, v1 = bounds
    specs = (("U0", 0, u0, v0, v1), ("U1", 0, u1, v0, v1),
             ("V0", 1, v0, u0, u1), ("V1", 1, v1, u0, u1))
    result = {}
    for name, normal_axis, rho, start, end in specs:
        nearby = [row for row in lines
                  if row["rho_or_region"]["normal_axis"] == normal_axis
                  and abs(row["rho_or_region"]["rho"] - rho) <=
                  config["boundary"]["corner_join_m"]
                  and row["support_interval"][1] > start
                  and row["support_interval"][0] < end]
        coverage = _coverage([interval for row in nearby
                              for interval in row["source_geometry"]["observed_intervals"]],
                             start, end)
        state = ("MEASURED_STRONG" if coverage >= config["roi"]["min_enclosure_ratio"]
                 else "MEASURED_WEAK" if nearby else
                 "GEOMETRY_CONSTRAINED" if seed["seed_type"] == "HEIGHT_REGION" else
                 "MISSING" if normal_axis != seed["anchor_axis"] else
                 "GEOMETRY_CONSTRAINED")
        result[name] = dict(state=state, rho=float(rho), coverage=coverage,
                            supporting_provider_ids=[row["provider_id"] for row in nearby],
                            evidence_types=sorted({item for row in nearby
                                                   for item in row["evidence_types"]}),
                            uncertainty_m=min((row["uncertainty_m"] for row in nearby),
                                              default=None))
    return result


def _patch_median(grid, bounds):
    u0, v0, u1, v1 = bounds
    col0 = max(0, int(math.floor((u0 - grid.x0) / grid.cell_m)))
    col1 = min(grid.count.shape[1], int(math.ceil((u1 - grid.x0) / grid.cell_m)))
    row0 = max(0, int(math.floor((v0 - grid.y0) / grid.cell_m)))
    row1 = min(grid.count.shape[0], int(math.ceil((v1 - grid.y0) / grid.cell_m)))
    cells = grid.median[row0:row1, col0:col1]
    values = cells[np.isfinite(cells)]
    return float(np.median(values)) if len(values) else None


def _opening_evidence(grid, bounds, config):
    u0, v0, u1, v1 = bounds
    step = config["boundary"]["side_probe_m"]
    inside = _patch_median(grid, [u0 + step, v0 + step, u1 - step, v1 - step])
    outside = [_patch_median(grid, row) for row in
               ([u0-step, v0, u0, v1], [u1, v0, u1+step, v1],
                [u0, v0-step, u1, v0], [u0, v1, u1, v1+step])]
    far_step = config["roi"]["support_search_m"] / 2
    far = [_patch_median(grid, row) for row in
           ([u0-step-far_step, v0, u0-far_step, v1],
            [u1+far_step, v0, u1+step+far_step, v1],
            [u0, v0-step-far_step, u1, v0-far_step],
            [u0, v1+far_step, u1, v1+step+far_step])]
    observed = [value for value in outside if value is not None]
    contrast = float(np.median(observed) - inside) if inside is not None and observed else None
    threshold = config["roi"]["opening_drop_m"]
    if contrast is None:
        status = "UNKNOWN"
    elif contrast >= threshold:
        status = "OPENING_LOWER"
    elif contrast <= -threshold:
        status = "RAISED_INTERIOR_OR_CARGO"
    else:
        status = "HEIGHT_AMBIGUOUS"
    separator_like = False
    if inside is not None:
        for a, b in ((0, 1), (2, 3)):
            if all(outside[index] is not None and far[index] is not None and
                   inside - outside[index] >= threshold and
                   inside - far[index] >= threshold for index in (a, b)):
                separator_like = True
    return dict(status=status, outside_minus_inside_m=contrast,
                outside_side_support_count=len(observed), inside_median_z=inside,
                outside_side_median_z=outside, far_side_median_z=far,
                raised_strip_with_lower_regions_both_sides=separator_like)


def _evaluate_seed(seed, lines, grid, vessel, config, verified_rectangles):
    bounds = seed["bounds_axial"]
    u0, v0, u1, v1 = bounds
    if min(u1-u0, v1-v0) <= 2 * config["boundary"]["side_probe_m"]:
        return None
    sides = _side_evidence(bounds, lines, config, seed)
    strong = sum(row["state"] == "MEASURED_STRONG" for row in sides.values())
    weak = sum(row["state"] == "MEASURED_WEAK" for row in sides.values())
    opening = _opening_evidence(grid, bounds, config)
    providers = sorted(set(seed["source_provider_ids"] + [item for side in sides.values()
                                                           for item in side["supporting_provider_ids"]]))
    conflicts = []
    if opening["status"] == "RAISED_INTERIOR_OR_CARGO":
        conflicts.append("RAISED_INTERIOR_ROLE_AMBIGUOUS" if
                         opening["raised_strip_with_lower_regions_both_sides"] else
                         "CARGO_OR_STRUCTURE_HEIGHT_AMBIGUOUS")
    # Raw height and profile breaks can be cargo or hull surfaces. A complete
    # provisional polygon needs the existing stricter R2F steel-role gate as
    # independent confirmation. This association never creates an edge.
    verified_match = next((row for row in verified_rectangles
                           if row.get("status") == "PROVISIONAL_RECTANGLE"
                           and row.get("source_stage") == "R2F"
                           and row.get("vessel_hypothesis_id") ==
                           vessel["vessel_hypothesis_id"]
                           and _iou(bounds, row["bounds_axial"]) >=
                           1 - config["roi"]["min_enclosure_ratio"]), None)
    if strong == 4 and verified_match is None:
        conflicts.append("STEEL_ROLE_NOT_INDEPENDENTLY_VERIFIED")
    if conflicts:
        state = ("HATCH_PROPOSAL" if strong >= 2
                 and "RAISED_INTERIOR_ROLE_AMBIGUOUS" not in conflicts
                 else "UNRESOLVED")
    elif strong == 4 and opening["status"] == "OPENING_LOWER" and verified_match is not None:
        state = "PROVISIONAL_HATCH"
    elif strong >= 3 and opening["status"] == "OPENING_LOWER":
        state = "PARTIAL_HATCH"
    elif strong >= 2 or (seed["seed_type"] == "HEIGHT_REGION" and
                         opening["status"] == "OPENING_LOWER"):
        state = "HATCH_PROPOSAL"
    else:
        state = "UNRESOLVED"
    axes = np.asarray(vessel["local_axes"])
    polygon = None
    if state == "PROVISIONAL_HATCH":
        polygon = [(np.asarray(corner) @ axes).tolist() for corner in
                   ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
    return dict(hatch_hypothesis_id=None,
                vessel_hypothesis_id=vessel["vessel_hypothesis_id"],
                supporting_provider_ids=providers, source_support_ids=vessel["source_support_ids"],
                bounds_axial=[float(value) for value in bounds], polygon_xy=polygon,
                side_evidence=sides, opening_evidence=opening,
                role="UNKNOWN" if conflicts else "HATCH_OPENING" if state == "PROVISIONAL_HATCH"
                else "UNKNOWN", role_conflicts=conflicts,
                uncertainty_m=max((row["uncertainty_m"] or config["boundary"]["corner_join_m"]
                                   for row in sides.values())),
                measured_strong_side_count=strong, measured_weak_side_count=weak,
                state=state, seed_type=seed["seed_type"],
                verified_rectangle_id=verified_match["rectangle_id"] if verified_match else None)


def _overlap(a, b):
    x0, y0, x1, y1 = a
    u0, v0, u1, v1 = b
    intersection = max(0, min(x1, u1)-max(x0, u0)) * max(0, min(y1, v1)-max(y0, v0))
    area_a, area_b = (x1-x0)*(y1-y0), (u1-u0)*(v1-v0)
    return intersection / max(min(area_a, area_b), 1e-9)


def _iou(a, b):
    x0, y0, x1, y1 = a
    u0, v0, u1, v1 = b
    intersection = max(0, min(x1, u1)-max(x0, u0)) * max(0, min(y1, v1)-max(y0, v0))
    area_a, area_b = (x1-x0)*(y1-y0), (u1-u0)*(v1-v0)
    return intersection / max(area_a + area_b - intersection, 1e-9)


def propose_for_vessel(points, vessel, config, research, *, height_override=None,
                       verified_rectangles=()):
    """Independent providers first; association and state come afterwards."""
    points = np.asarray(points, dtype=np.float32)
    if vessel["classification_status"] not in ("VESSEL_HYPOTHESIS", "UNRESOLVED"):
        raise ValueError("INVALID_VESSEL_HYPOTHESIS_STATUS")
    if height_override is None:
        height, height_summary = _height_provider(points, vessel, config, research)
    else:
        height, height_summary = list(height_override), dict(region_count=len(height_override),
                                                              dropped_within_vessel=[])
    bev, grid = _bev_provider(points, vessel, config)
    raw3d = _raw3d_provider(points, vessel, config)
    evidence = height + bev + raw3d
    extent = _axial_extent(points, vessel["local_axes"])
    seeds = _association_seeds(evidence, extent, config)
    lines = [row for row in evidence if row["geometry_type"] == "AXIAL_LINE_INTERVAL"]
    candidates = [_evaluate_seed(seed, lines, grid, vessel, config,
                                 verified_rectangles) for seed in seeds]
    candidates = [row for row in candidates if row is not None]
    if vessel["classification_status"] == "UNRESOLVED":
        for row in candidates:
            row["state"] = "UNRESOLVED"
            row["role"] = "UNKNOWN"
            row["polygon_xy"] = None
            row["verified_rectangle_id"] = None
            row["role_conflicts"] = sorted(set(row["role_conflicts"] +
                                                ["VESSEL_ROLE_UNRESOLVED"]))
    priority = {"PROVISIONAL_HATCH": 3, "PARTIAL_HATCH": 2,
                "HATCH_PROPOSAL": 1, "UNRESOLVED": 0}
    candidates.sort(key=lambda row: (priority[row["state"]], row["measured_strong_side_count"],
                                     min(side["coverage"] for side in row["side_evidence"].values()),
                                     len(row["supporting_provider_ids"]),
                                     -row["uncertainty_m"]), reverse=True)
    accepted, suppressed = [], []
    overlap_limit = config["roi"]["min_enclosure_ratio"]

    def same_local_hypothesis(first, second):
        # A raised transverse strip is a competing physical role, even when
        # its small bounds sit inside a larger opening proposal.
        separator = "RAISED_INTERIOR_ROLE_AMBIGUOUS"
        if ((separator in first["role_conflicts"]) !=
                (separator in second["role_conflicts"])):
            return False
        a, b = first["bounds_axial"], second["bounds_axial"]
        close_center = all(abs((a[index] + a[index+2] - b[index] - b[index+2]) / 2)
                           <= config["roi"]["support_search_m"] for index in (0, 1))
        return (_iou(a, b) >= 1 - overlap_limit or
                (close_center and _overlap(a, b) >= overlap_limit))

    for candidate in candidates:
        duplicate = next((row for row in accepted if
                          same_local_hypothesis(candidate, row)), None)
        if duplicate is not None:
            suppressed.append(dict(reason="WITHIN_VESSEL_GEOMETRIC_OVERLAP",
                                   vessel_hypothesis_id=vessel["vessel_hypothesis_id"],
                                   source_provider_ids=candidate["supporting_provider_ids"],
                                   kept_hatch_hypothesis_id=duplicate["hatch_hypothesis_id"]))
            continue
        candidate["hatch_hypothesis_id"] = "%s:H%04d" % (
            vessel["vessel_hypothesis_id"], len(accepted))
        accepted.append(candidate)
    if not accepted and evidence:
        accepted.append(dict(hatch_hypothesis_id="%s:H0000" % vessel["vessel_hypothesis_id"],
                             vessel_hypothesis_id=vessel["vessel_hypothesis_id"],
                             supporting_provider_ids=[row["provider_id"] for row in evidence],
                             source_support_ids=vessel["source_support_ids"],
                             bounds_axial=None, polygon_xy=None, side_evidence=None,
                             opening_evidence=dict(status="UNKNOWN"), role="UNKNOWN",
                             role_conflicts=["NO_RECTILINEAR_ASSOCIATION"],
                             uncertainty_m=None, measured_strong_side_count=0,
                             measured_weak_side_count=0, state="UNRESOLVED",
                             seed_type="UNASSOCIATED_EVIDENCE"))
    return dict(vessel_hypothesis_id=vessel["vessel_hypothesis_id"],
                vessel_classification_status=vessel["classification_status"],
                provider_summary=dict(height=height_summary, bev_count=len(bev),
                                      raw3d_count=len(raw3d)),
                provider_evidence=evidence, hatch_hypotheses=accepted,
                association_candidate_count=len(candidates),
                suppressed_associations=suppressed)


def discover_scene_hatches(points, config, research):
    """Scope all evidence, association and suppression by B1 vessel identity."""
    scene, auxiliary = infer_scene_vessels(points, config, research)
    per_vessel, skipped = [], []
    for vessel in scene["vessel_hypotheses"]:
        if vessel["point_count"] < (research["min_points_per_coarse_cell"] *
                                    config["roi"]["min_deck_support_cells"]):
            skipped.append(dict(vessel_hypothesis_id=vessel["vessel_hypothesis_id"],
                                reason="INSUFFICIENT_RAW_SUPPORT_FOR_PROVIDERS"))
            continue
        indexes = auxiliary["vessel_point_indexes"][vessel["vessel_hypothesis_id"]]
        per_vessel.append(propose_for_vessel(np.asarray(points)[indexes], vessel,
                                               config, research))
    return dict(schema_version="ship_perception.v15r.hatch_proposal.1",
                coordinate_frame="RAW_INPUT_FRAME", frame_id="STATIC_SINGLE_FRAME",
                sensor_id="UNKNOWN", target_vessel_status="NOT_SELECTED",
                scene=scene, per_vessel=per_vessel,
                skipped_hypotheses=skipped)
