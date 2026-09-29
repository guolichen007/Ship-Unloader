"""Static Raw XYZ scene supports and reviewable vessel hypotheses.

Connected components, rectilinear seams, and axis families are evidence, not
semantic labels. No target vessel is selected here, and no hatch is confirmed.
"""

import math

import numpy as np
from scipy import ndimage, signal

from .height_grid import R1HeightGrid
from .raw_topview_evidence import raster_channels


def _axis(xy, gradients=None, cell_indexes=None, minimum_coherence=0.35):
    """Major occupied-cell direction; each support gets its own estimate."""
    xy = np.asarray(xy, dtype=float)
    center = xy.mean(axis=0)
    covariance = (xy - center).T @ (xy - center) / max(len(xy), 1)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    if gradients is not None and cell_indexes is not None:
        phase_grid, magnitude = gradients
        rows, cols = cell_indexes
        unique = np.unique(rows * phase_grid.shape[1] + cols)
        phase_values = phase_grid.flat[unique]
        weights = magnitude.flat[unique]
        phase = np.sum(weights * phase_values)
        coherence = float(abs(phase) / max(weights.sum(), 1e-9))
        if coherence >= minimum_coherence:
            angle = float(np.angle(phase) / 4)
            structural = np.array([math.cos(angle), math.sin(angle)])
            if abs(structural @ direction) < abs(np.array([-structural[1], structural[0]]) @ direction):
                structural = np.array([-structural[1], structural[0]])
            direction = structural
    if direction[0] < 0 or (abs(direction[0]) < 1e-9 and direction[1] < 0):
        direction = -direction
    cross = np.array([-direction[1], direction[0]])
    dispersion = float(math.degrees(math.atan2(math.sqrt(max(values.min(), 0)),
                                                math.sqrt(max(values.max(), 1e-9)))))
    if gradients is not None and cell_indexes is not None and coherence >= minimum_coherence:
        residual = np.angle(phase_values * np.exp(-4j * angle)) / 4
        dispersion = float(np.degrees(np.sqrt(np.average(residual * residual,
                                                        weights=np.maximum(weights, 1e-9)))))
    return np.asarray((direction, cross)), dispersion


def _raw_gradients(grid, valid):
    """Fourfold phase makes observed orthogonal rims vote for one axis family."""
    weighted = np.where(valid, grid.median, 0.0)
    norm = np.maximum(ndimage.gaussian_filter(valid.astype(float), 1), 1e-9)
    surface = ndimage.gaussian_filter(weighted, 1) / norm
    gx = ndimage.sobel(surface, axis=1)
    gy = ndimage.sobel(surface, axis=0)
    magnitude = np.hypot(gx, gy)
    phase = np.exp(4j * np.arctan2(gy, gx))
    magnitude[~valid] = 0
    return phase, magnitude


def _angle_difference(a, b):
    return abs((math.degrees(math.atan2(a[1], a[0]) - math.atan2(b[1], b[0]))
                + 90) % 180 - 90)


def _bbox(xy):
    return [float(xy[:, 0].min()), float(xy[:, 1].min()),
            float(xy[:, 0].max()), float(xy[:, 1].max())]


def _component_seeds(mask, minimum_cells):
    """A thin bridge may join two bodies; eroded cores are only split seeds."""
    opened = ndimage.binary_opening(mask, structure=np.ones((3, 3), bool))
    labels, count = ndimage.label(opened, structure=np.ones((3, 3), bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    live = [index for index in range(1, count + 1) if sizes[index] >= minimum_cells]
    if len(live) <= 1:
        return [mask]
    seeds = np.where(np.isin(labels, live), labels, 0)
    nearest = ndimage.distance_transform_edt(seeds == 0, return_distances=False,
                                              return_indices=True)
    assigned = seeds[tuple(nearest)]
    return [mask & (assigned == index) for index in live]


def _seams(points, axes, config, research):
    """Measured full-length gaps or height troughs between broad body cores."""
    cell = config["geometry"]["coarse_voxel_m"]
    axial = np.column_stack((points[:, :2] @ axes.T, points[:, 2]))
    grid = R1HeightGrid.from_points(axial, cell)
    occupied = grid.occupancy
    if occupied.shape[0] < 3 or occupied.shape[1] < 3:
        return [], dict(rectilinear_support=0.0, opening_contrast_m=0.0)
    rows, cols = np.nonzero(occupied)
    if not len(rows):
        return [], dict(rectilinear_support=0.0, opening_contrast_m=0.0)
    lo, hi = np.quantile(cols, (0.15, 0.85))
    central = (np.arange(occupied.shape[1]) >= lo) & (np.arange(occupied.shape[1]) <= hi)
    if central.sum() < 2:
        central[:] = True
    columns = int(central.sum())
    height = np.full(occupied.shape[0], np.nan)
    for index in range(len(height)):
        values = grid.median[index, central & occupied[index]]
        if len(values) >= max(2, research["min_points_per_coarse_cell"] // 2):
            height[index] = float(np.median(values))
    good = np.flatnonzero(np.isfinite(height))
    if len(good) < 3:
        return [], dict(rectilinear_support=0.0, opening_contrast_m=0.0)
    profile = np.interp(np.arange(len(height)), good, height[good])
    sigma = max(1.0, config["roi"]["support_connectivity_m"] / cell)
    smooth = ndimage.gaussian_filter1d(profile, sigma)
    separation = max(2, int(math.ceil(config["roi"]["support_search_m"] / cell)))
    peaks, _ = signal.find_peaks(smooth, prominence=config["roi"]["opening_drop_m"],
                                 distance=separation)
    # A partial vessel can have a peak on the crop boundary; adding endpoints
    # is only for seam search, never for geometric edge confirmation.
    if len(peaks) and smooth[0] > smooth[peaks[0]]:
        peaks = np.r_[0, peaks]
    if len(peaks) and smooth[-1] > smooth[peaks[-1]]:
        peaks = np.r_[peaks, len(smooth) - 1]
    channels = raster_channels(grid, config)[1]
    transitions = channels["HEIGHT_TRANSITION"][:, central]
    line_coverage = transitions.sum(axis=1) / max(columns, 1)
    distances = ndimage.distance_transform_edt(occupied)
    distance_profile = ndimage.gaussian_filter1d(distances[:, central].mean(axis=1), sigma)
    distance_peaks, _ = signal.find_peaks(
        distance_profile, prominence=config["roi"]["min_deck_span_m"] / cell / 2,
        distance=separation)
    candidates = []
    for a, b in zip(peaks, peaks[1:]):
        valley = a + int(np.argmin(smooth[a:b + 1]))
        contrast = min(smooth[a], smooth[b]) - smooth[valley]
        radius = max(1, int(math.ceil(config["roi"]["support_connectivity_m"] / cell)))
        edge_coverage = float(np.max(line_coverage[max(0, valley - radius):
                                                     min(len(line_coverage), valley + radius + 1)]))
        if (contrast >= config["roi"]["opening_drop_m"] and
                edge_coverage >= config["roi"]["min_enclosure_ratio"]):
            candidates.append((valley, "HEIGHT_TROUGH", float(contrast), edge_coverage))
    for a, b in zip(distance_peaks, distance_peaks[1:]):
        valley = a + int(np.argmin(distance_profile[a:b + 1]))
        if distance_profile[valley] <= config["roi"]["min_enclosure_ratio"] * min(
                distance_profile[a], distance_profile[b]):
            candidates.append((valley, "SPATIAL_NECK", float(min(
                distance_profile[a], distance_profile[b]) - distance_profile[valley]), 0.0))
    # Nearby measurements describe one physical seam, not multiple vessels.
    candidates.sort(key=lambda row: (-row[2], row[0]))
    chosen = []
    for row in candidates:
        if all(abs(row[0] - prior[0]) * cell > config["roi"]["support_search_m"] / 2
               for prior in chosen):
            chosen.append(row)
    chosen.sort(key=lambda row: row[0])
    return [(float(grid.y0 + (row[0] + 0.5) * cell), row[1], row[2], row[3])
            for row in chosen], dict(
                rectilinear_support=float(np.max(line_coverage)),
                opening_contrast_m=float(np.nanmax(smooth) - np.nanmin(smooth)))


def _merge_collinear(groups, points, config):
    """Rejoin occluded pieces along U; overlapping parallel bodies stay apart."""
    tolerance = config["boundary"]["merge_angle_deg"]
    gap_limit = config["roi"]["support_search_m"]
    overlap_min = config["roi"]["min_enclosure_ratio"]
    groups = list(groups)
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                if _angle_difference(a["axes"][0], b["axes"][0]) > tolerance:
                    continue
                axis = a["axes"]
                pa = points[a["point_indexes"], :2] @ axis.T
                pb = points[b["point_indexes"], :2] @ axis.T
                ua, ub = (pa[:, 0].min(), pa[:, 0].max()), (pb[:, 0].min(), pb[:, 0].max())
                va, vb = (pa[:, 1].min(), pa[:, 1].max()), (pb[:, 1].min(), pb[:, 1].max())
                along_gap = max(ua[0], ub[0]) - min(ua[1], ub[1])
                cross_overlap = min(va[1], vb[1]) - max(va[0], vb[0])
                if not (0 < along_gap <= gap_limit and
                        cross_overlap / max(min(va[1]-va[0], vb[1]-vb[0]), 1e-9)
                        >= overlap_min):
                    continue
                merged = {**a, "point_indexes": np.r_[a["point_indexes"], b["point_indexes"]],
                          "support_ids": sorted(set(a["support_ids"] + b["support_ids"])),
                          "component_ids": sorted(set(a["component_ids"] + b["component_ids"])),
                          "reason": "COLLINEAR_OCCLUSION_MERGE"}
                merged["axes"], merged["axis_dispersion_deg"] = _axis(
                    points[merged["point_indexes"], :2])
                groups[i] = merged
                groups.pop(j)
                changed = True
                break
            if changed:
                break
    return groups


def infer_scene_vessels(points, config, research):
    """Return 0..N conservative hypotheses and private point-index provenance."""
    points = np.asarray(points, dtype=np.float32)
    cell = config["geometry"]["coarse_voxel_m"]
    grid = R1HeightGrid.from_points(points, cell)
    # Support recall uses occupancy; the height evidence stage applies its
    # own density gate. A sparse but coherent vessel must not disappear here.
    observed = grid.occupancy
    valid = ndimage.binary_closing(observed, structure=np.ones((3, 3), bool)) | observed
    gradients = _raw_gradients(grid, observed)
    labels, count = ndimage.label(valid, structure=np.ones((3, 3), bool))
    point_rows, point_cols = grid.cell_indices(points[:, :2])
    point_component = labels[point_rows, point_cols]
    minimum_cells = config["roi"]["min_deck_support_cells"]
    supports, groups = [], []
    for component_id in range(1, count + 1):
        mask = labels == component_id
        pieces = _component_seeds(mask, minimum_cells)
        for piece in pieces:
            rows, cols = np.nonzero(piece)
            if not len(rows):
                continue
            xy = np.column_stack((grid.x0 + (cols + .5) * cell,
                                  grid.y0 + (rows + .5) * cell))
            indexes = np.flatnonzero((point_component == component_id) &
                                     piece[point_rows, point_cols])
            support_id = "S%04d" % len(supports)
            axes, dispersion = _axis(xy, gradients, (rows, cols),
                                     config["roi"]["min_enclosure_ratio"])
            support = dict(support_id=support_id, component_ids=[component_id],
                           bbox_xy=_bbox(xy), point_count=int(len(indexes)),
                           support_area_m2=float(len(rows) * cell * cell),
                           spatial_centroid=xy.mean(axis=0).tolist(),
                           candidate_axis_families=[axes[0].tolist()],
                           rectilinear_support=None, status="SCENE_SUPPORT",
                           reason="ERODED_CORE_SPLIT" if len(pieces) > 1
                           else "CONNECTED_COMPONENT_CORE")
            supports.append(support)
            if len(rows) < minimum_cells or len(indexes) < minimum_cells:
                support["status"] = "UNRESOLVED"
                support["reason"] = "INSUFFICIENT_SPATIAL_SUPPORT"
                continue
            local_points = points[indexes]
            seams, evidence = _seams(local_points, axes, config, research)
            support["rectilinear_support"] = evidence["rectilinear_support"]
            projected = local_points[:, :2] @ axes[1]
            limits = [-np.inf] + [seam[0] for seam in seams] + [np.inf]
            for lo, hi in zip(limits, limits[1:]):
                member = indexes[(projected >= lo) & (projected < hi)]
                if len(member) < minimum_cells:
                    continue
                cells = np.unique(point_rows[member] * grid.count.shape[1] + point_cols[member])
                member_axis, member_dispersion = _axis(
                    points[member, :2], gradients,
                    (cells // grid.count.shape[1], cells % grid.count.shape[1]),
                    config["roi"]["min_enclosure_ratio"])
                groups.append(dict(point_indexes=member, support_ids=[support_id],
                                   component_ids=[component_id], axes=member_axis,
                                   axis_dispersion_deg=member_dispersion,
                                   split_seams=[dict(rho=seam[0], evidence=seam[1],
                                                     contrast=seam[2], coverage=seam[3])
                                                for seam in seams],
                                   reason="MEASURED_CROSS_AXIS_SEAM" if seams
                                   else "SPATIAL_CORE"))
    groups = _merge_collinear(groups, points, config)
    hypotheses, point_provenance = [], {}
    for group in groups:
        indexes = np.unique(group["point_indexes"])
        xy = points[indexes, :2]
        cells = np.unique(point_rows[indexes] * grid.count.shape[1] + point_cols[indexes])
        axes, dispersion = _axis(
            xy, gradients, (cells // grid.count.shape[1], cells % grid.count.shape[1]),
            config["roi"]["min_enclosure_ratio"])
        axial = xy @ axes.T
        robust_extent = np.percentile(axial, (5, 95), axis=0)
        length, width = robust_extent[1] - robust_extent[0]
        _, evidence = _seams(points[indexes], axes, config, research)
        # No selection of a target vessel. A flat elongated wharf remains
        # unresolved; a varying interior plus long measured structure is a
        # vessel hypothesis for further Hatch discovery, not a confirmed ship.
        rectilinear = evidence["rectilinear_support"]
        contrast = evidence["opening_contrast_m"]
        plausible = (length >= config["roi"]["min_deck_span_m"] and
                     width >= 2 * config["roi"]["support_search_m"] and
                     length / max(width, cell) >= 1 / config["roi"]["min_enclosure_ratio"] and
                     contrast >= config["roi"]["opening_drop_m"] and
                     rectilinear >= config["roi"]["min_enclosure_ratio"])
        status = "VESSEL_HYPOTHESIS" if plausible else "UNRESOLVED"
        reason = group["reason"] if plausible else "VESSEL_ROLE_EVIDENCE_INSUFFICIENT"
        hypothesis_id = "V%04d" % len(hypotheses)
        hypotheses.append(dict(vessel_hypothesis_id=hypothesis_id,
                               source_support_ids=sorted(set(group["support_ids"])),
                               component_ids=group["component_ids"], bbox_xy=_bbox(xy),
                               point_count=int(len(indexes)), local_axes=axes.tolist(),
                               axis_confidence=float(max(0.0, 1.0 - dispersion / 45.0)),
                               axis_dispersion_deg=dispersion,
                               rectilinear_support=rectilinear,
                               opening_contrast_m=contrast,
                               classification_status=status, reason=reason,
                               split_seams=group["split_seams"]))
        point_provenance[hypothesis_id] = indexes
    return dict(schema_version="ship_perception.v15r.scene_vessel.1",
                coordinate_frame="RAW_INPUT_FRAME",
                frame_id="STATIC_SINGLE_FRAME", sensor_id="UNKNOWN",
                target_vessel_status="NOT_SELECTED",
                raw_connected_component_count=int(count),
                scene_support_candidate_count=len(supports),
                vessel_hypothesis_count=len(hypotheses),
                scene_support_candidates=supports,
                vessel_hypotheses=hypotheses), dict(
                    vessel_point_indexes=point_provenance, grid=grid, valid=valid)
