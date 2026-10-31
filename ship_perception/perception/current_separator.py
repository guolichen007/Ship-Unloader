"""Current B2 grouping and measured separator primitives, without the B5 solver.

The two compatibility functions retain the existing implementation verbatim.
No final rectangle, old output, scene name, or independent annotation is read.
"""

import numpy as np
from scipy import ndimage
from .side_evidence import (
    project_provider_observations as _observations,
    bounded_side_modes as _side_modes,
)


def _initial_seed_groups(proposals, config):
    """Group overlapping B2 partial openings, preserving separate basins."""
    eligible = [
        row
        for row in proposals
        if row.get("state") == "PARTIAL_HATCH"
        and row.get("measured_strong_side_count", 0) >= 3
        and row.get("opening_evidence", {}).get("status") == "OPENING_LOWER"
        and not row.get("role_conflicts")
        and row.get("bounds_axial")
    ]
    groups = []
    margin = config["boundary"]["corner_join_m"]
    for row in eligible:
        box = row["bounds_axial"]
        matched = []
        for index, group in enumerate(groups):
            if any(
                min(box[2], other["bounds_axial"][2]) + margin
                >= max(box[0], other["bounds_axial"][0])
                and min(box[3], other["bounds_axial"][3]) + margin
                >= max(box[1], other["bounds_axial"][1])
                for other in group
            ):
                matched.append(index)
        if matched:
            merged = [row]
            for index in reversed(matched):
                merged.extend(groups.pop(index))
            groups.append(merged)
        else:
            groups.append([row])

    def seed_quality(row):
        box = row["bounds_axial"]
        width, height = box[2] - box[0], box[3] - box[1]
        return (
            min(width, height),
            width * height,
            row.get("measured_strong_side_count", 0),
        )

    return [(max(group, key=seed_quality), group) for group in groups]


def _separator_split(seed, b2, vessel, grid, config):
    """Propose two cells only around two separately measured separator faces."""
    box = seed["bounds_axial"]
    margin = config["roi"]["support_search_m"]
    roi = [box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin]
    observations = _observations(
        b2, vessel["vessel_hypothesis_id"], np.asarray(vessel["local_axes"]), roi
    )
    modes = _side_modes(
        observations, 0, (box[1], box[3]), (box[0] + margin, box[2] - margin), config
    )
    minimum = config["roi"]["min_enclosure_ratio"]
    rhos = sorted(
        row["rho"]
        for row in modes
        if min(row["primary_raw3d"], row["primary_bev"]) >= minimum
    )
    families = []
    for rho in rhos:
        if (
            families
            and rho - families[-1][-1] <= config["boundary"]["corner_join_m"] * 2
        ):
            families[-1].append(rho)
        else:
            families.append([rho])
    candidates = [
        (left, right)
        for left, right in zip(families[:-1], families[1:])
        if right[0] - left[-1] <= margin
        and box[0] + margin < np.median(left) < np.median(right) < box[2] - margin
    ]
    # Dense B2 line sampling can fill a whole transverse beam with nearly
    # identical scores. Recover its two *measured* faces from a Raw XYZ count
    # ridge, then snap each face to a same-vessel Raw3D+BEV mode.
    u, v = grid.centers()
    along = (v[:, 0] >= box[1] + margin / 4) & (v[:, 0] <= box[3] - margin / 4)
    axial = (u[0] >= box[0] + margin) & (u[0] <= box[2] - margin)
    if along.any() and axial.sum() >= config["roi"]["min_opening_cells"]:
        density = np.mean(grid.count[along], axis=0)
        reference = float(np.median(density[axial]))
        mad = float(np.median(np.abs(density[axial] - reference)))
        high = axial & (density > reference + 3 * mad)
        high = ndimage.binary_closing(high, structure=np.ones(3, bool)) & axial
        labels, count = ndimage.label(high)
        for label in range(1, count + 1):
            columns = np.flatnonzero(labels == label)
            if len(columns) < 2 or len(columns) * grid.cell_m > margin:
                continue
            left_edge = u[0, columns[0]] - grid.cell_m / 2
            right_edge = u[0, columns[-1]] + grid.cell_m / 2
            tolerance = config["boundary"]["corner_join_m"] * 2
            left_mode = min(
                (rho for rho in rhos if abs(rho - left_edge) <= tolerance),
                key=lambda rho: abs(rho - left_edge),
                default=None,
            )
            right_mode = min(
                (rho for rho in rhos if abs(rho - right_edge) <= tolerance),
                key=lambda rho: abs(rho - right_edge),
                default=None,
            )
            if (
                left_mode is not None
                and right_mode is not None
                and left_mode < right_mode
            ):
                candidates.append(([left_mode], [right_mode]))
    if not candidates:
        if len(rhos) >= 2 and max(rhos) - min(rhos) > margin / 4:
            return [box], {"unresolved_transverse_band_m": [min(rhos), max(rhos)]}
        return [box], None
    left, right = min(
        candidates,
        key=lambda pair: abs(
            (np.median(pair[0]) + np.median(pair[1])) / 2 - (box[0] + box[2]) / 2
        ),
    )
    first, second = float(np.median(left)), float(np.median(right))
    return [[box[0], box[1], first, box[3]], [second, box[1], box[2], box[3]]], [
        first,
        second,
    ]
