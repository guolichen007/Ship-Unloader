"""Vessel-owned height evidence and persistent axis-aligned contour fragments.

Height contours measure opening topology, not steel. B1 point provenance is
used before rasterization; no point is removed because its Z differs from deck.
"""

import numpy as np
from scipy import ndimage

from .raw_topview_evidence import build_topview


def build_vessel_scoped_topview(points, point_indexes, axes, config):
    """Recover all Raw Z returns inside the B1-owned vessel XY footprint."""
    points = np.asarray(points)
    indexes = np.unique(np.asarray(point_indexes, dtype=np.int64))
    if not len(indexes) or indexes.min() < 0 or indexes.max() >= len(points):
        raise ValueError("INVALID_VESSEL_POINT_PROVENANCE")
    owned = points[indexes]
    owned_grids = build_topview(owned, axes, config)
    footprint_grid = owned_grids["fine"]
    footprint = footprint_grid.occupancy
    # Fill only small raster holes; do not join the shore or another vessel.
    footprint = footprint | ndimage.binary_closing(
        footprint, structure=np.ones((3, 3), bool))
    axial_xy = points[:, :2] @ np.asarray(axes).T
    row, col = footprint_grid.cell_indices(axial_xy)
    in_bounds = ((row >= 0) & (row < footprint.shape[0]) &
                 (col >= 0) & (col < footprint.shape[1]))
    selected = np.zeros(len(points), dtype=bool)
    selected[in_bounds] = footprint[row[in_bounds], col[in_bounds]]
    # The B1 support is authoritative even on a raster boundary.
    selected[indexes] = True
    recovered = points[selected]
    grids = build_topview(recovered, axes, config)
    # This reference is diagnostic only. It never filters hold/cargo points.
    shell_reference = float(np.median(owned[:, 2]))
    return dict(
        points=recovered,
        selected_point_mask=selected,
        grids=grids,
        vessel_shell_reference_z_m=shell_reference,
        vessel_shell_z_mad_m=float(np.median(np.abs(owned[:, 2] - shell_reference))),
        vessel_footprint_cells=int(footprint.sum()),
        b1_owned_point_count=int(len(indexes)),
        restored_inside_footprint_points=int(selected.sum() - len(indexes)),
        rejected_external_points=int(len(points) - selected.sum()),
        point_selection="B1_VESSEL_XY_FOOTPRINT_RESTORE_ALL_RAW_Z",
    )


def _relative_levels(grid, box, config):
    """Use local interior/ring height families; invariant to a global Z shift."""
    u, v = grid.centers()
    u0, v0, u1, v1 = box
    margin = config["roi"]["support_search_m"]
    inside = (
        (u >= u0 + margin / 2)
        & (u <= u1 - margin / 2)
        & (v >= v0 + margin / 2)
        & (v <= v1 - margin / 2)
        & grid.occupancy
    )
    outer = (
        (u >= u0 - margin)
        & (u <= u1 + margin)
        & (v >= v0 - margin)
        & (v <= v1 + margin)
        & ~((u >= u0) & (u <= u1) & (v >= v0) & (v <= v1))
        & grid.occupancy
    )
    minimum = config["roi"]["min_deck_support_cells"]
    if min(int(inside.sum()), int(outer.sum())) < minimum:
        return [], "INSUFFICIENT_LOCAL_HEIGHT_REFERENCE"
    interior_z = float(np.median(grid.median[inside]))
    ring_values = grid.median[outer]
    # A neighboring hatch or a cabin can occupy much of the ring. For a
    # raised cargo interior, the lower local shell family is the useful
    # reference; for a low basin use the higher shell family. This changes
    # only contour levels, never point ownership or an edge role.
    ring_median = float(np.median(ring_values))
    exterior_z = float(np.quantile(ring_values, 0.2 if interior_z > ring_median else 0.8))
    difference = interior_z - exterior_z
    if abs(difference) < config["roi"]["opening_drop_m"]:
        return [], "HEIGHT_REFERENCE_AMBIGUOUS"
    # Fractions of the measured relative contrast are level choices, not
    # absolute vessel, cargo, or quay heights.
    levels = (exterior_z + difference * np.linspace(0.1, 0.5, 5)).tolist()
    return levels, "CARGO_FILLED" if difference > 0 else "LOW_BASIN"


def _line_fraction(edge, grid, normal_axis, rho_index, along_span):
    coordinate = grid.y0 if normal_axis == 0 else grid.x0
    indices = np.arange(edge.shape[0 if normal_axis == 0 else 1])
    along = coordinate + (indices + 0.5) * grid.cell_m
    selected = (along >= along_span[0]) & (along <= along_span[1])
    if not selected.any():
        return 0.0, []
    values = edge[:, rho_index] if normal_axis == 0 else edge[rho_index, :]
    hits = along[selected & values]
    return float(len(hits) / selected.sum()), hits.tolist()


def persistent_height_modes(grid, box, config):
    """Find contour lines repeatedly observed at nearby XY over height levels.

    A repeated threshold crossing is topology evidence only. It cannot certify
    a steel edge or a fully measured fourth corner.
    """
    levels, topology = _relative_levels(grid, box, config)
    if not levels:
        return dict(levels_m=[], topology=topology, modes={0: [], 1: []},
                    persistent_corners=[],
                    footprint_edges={0: [], 1: []})
    u0, v0, u1, v1 = box
    margin = config["roi"]["support_search_m"]
    minimum_fraction = config["roi"]["min_enclosure_ratio"] / 2
    observations = {0: [], 1: []}
    for level_index, level in enumerate(levels):
        mask = grid.occupancy & (
            (grid.median >= level) if topology == "CARGO_FILLED" else (grid.median <= level)
        )
        # Only fill individual raster gaps. The mask never joins separated
        # ships because the input has already been scoped by B1 provenance.
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool))
        for axis in (0, 1):
            if axis == 0:
                edge = (mask[:, 1:] != mask[:, :-1]) & grid.occupancy[:, 1:] & grid.occupancy[:, :-1]
            else:
                edge = (mask[1:, :] != mask[:-1, :]) & grid.occupancy[1:, :] & grid.occupancy[:-1, :]
            origin = grid.x0 if axis == 0 else grid.y0
            count = edge.shape[1 if axis == 0 else 0]
            normal_band = (u0 - margin, u1 + margin) if axis == 0 else (v0 - margin, v1 + margin)
            along_span = (v0, v1) if axis == 0 else (u0, u1)
            for index in range(count):
                rho = origin + (index + 1) * grid.cell_m
                if not normal_band[0] <= rho <= normal_band[1]:
                    continue
                coverage, along_hits = _line_fraction(edge, grid, axis, index, along_span)
                if coverage >= minimum_fraction:
                    observations[axis].append(dict(
                        rho=float(rho), level_index=level_index, level_m=float(level),
                        coverage=coverage, along_values=along_hits,
                    ))
    modes = {0: [], 1: []}
    tolerance = config["boundary"]["corner_join_m"]
    for axis, rows in observations.items():
        groups = []
        for row in sorted(rows, key=lambda item: (-item["coverage"], item["rho"])):
            matching = next((group for group in groups if abs(
                group[0]["rho"] - row["rho"]) <= tolerance), None)
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
            modes[axis].append(dict(
                normal_axis=axis,
                rho=position,
                level_count=len(distinct),
                support_level_indices=sorted(distinct),
                support_levels_m=sorted({row["level_m"] for row in group}),
                position_mad_m=dispersion,
                coverage=float(max(row["coverage"] for row in group)),
                along_min=float(min(value for row in group for value in row["along_values"])),
                along_max=float(max(value for row in group for value in row["along_values"])),
                evidence_type="HEIGHT_PERSISTENT_EDGE_NOT_STEEL",
            ))
        modes[axis].sort(key=lambda item: item["rho"])
    # A support limit is useful for locating the ship-side opening region, but
    # it is explicitly separate from a two-sided measured height contour.
    footprint_edges = {0: [], 1: []}
    for axis in (0, 1):
        edge = ((grid.occupancy[:, 1:] != grid.occupancy[:, :-1]) if axis == 0 else
                (grid.occupancy[1:, :] != grid.occupancy[:-1, :]))
        origin = grid.x0 if axis == 0 else grid.y0
        count = edge.shape[1 if axis == 0 else 0]
        normal_band = (u0 - margin, u1 + margin) if axis == 0 else (v0 - margin, v1 + margin)
        along_span = (v0, v1) if axis == 0 else (u0, u1)
        for index in range(count):
            rho = origin + (index + 1) * grid.cell_m
            if not normal_band[0] <= rho <= normal_band[1]:
                continue
            coverage, _ = _line_fraction(edge, grid, axis, index, along_span)
            if coverage >= minimum_fraction:
                footprint_edges[axis].append(dict(
                    rho=float(rho), coverage=coverage,
                    evidence_type="VESSEL_SUPPORT_LIMIT_NOT_STEEL",
                ))
    corners = persistent_height_corners(grid, levels, topology, modes, config)
    return dict(levels_m=levels, topology=topology, modes=modes,
                persistent_corners=corners, footprint_edges=footprint_edges)


def persistent_height_corners(grid, levels, topology, modes, config):
    """Actively find L transitions shared by perpendicular contours at >=2 levels.

    These are measured height topology positions, never certified steel corners.
    Neither a B4 rectangle nor its existing corners enters this provider.
    """
    if not levels:
        return []
    u, v = grid.centers()
    probe = max(grid.cell_m * 2, config["boundary"]["side_probe_m"])
    join = config["boundary"]["corner_join_m"] * 2
    signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    result = []
    for u_line in modes[0]:
        for v_line in modes[1]:
            u0, v0 = u_line["rho"], v_line["rho"]
            if not (u_line["along_min"] - join <= v0 <= u_line["along_max"] + join and
                    v_line["along_min"] - join <= u0 <= v_line["along_max"] + join):
                continue
            quadrant_z = []
            for su, sv in signs:
                cells = ((su * (u - u0) > 0) & (su * (u - u0) <= probe) &
                         (sv * (v - v0) > 0) & (sv * (v - v0) <= probe) &
                         grid.occupancy)
                quadrant_z.append(float(np.median(grid.median[cells])) if cells.any() else None)
            shared_levels = set(u_line["support_level_indices"]) & set(
                v_line["support_level_indices"])
            supporting = []
            for index in sorted(shared_levels):
                if any(value is None for value in quadrant_z):
                    continue
                binary = [value >= levels[index] for value in quadrant_z]
                if sum(binary) in (1, 3):
                    supporting.append(index)
            if len(supporting) < 2:
                continue
            result.append(dict(
                corner_xy=[float(u0), float(v0)],
                level_count=len(supporting),
                position_median_xy=[float(u0), float(v0)],
                position_mad_m=max(u_line["position_mad_m"], v_line["position_mad_m"]),
                support_levels_m=[float(levels[index]) for index in supporting],
                contour_ids=[f"L{index}:U{u0:.3f}:V{v0:.3f}" for index in supporting],
                evidence_type="PERSISTENT_HEIGHT_CORNER_NOT_STEEL",
                median_z_quadrants=quadrant_z,
                topology=topology,
            ))
    return result
