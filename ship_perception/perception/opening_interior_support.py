"""Spatial ownership is independent of material surface continuity."""

import numpy as np
from scipy.ndimage import distance_transform_edt
from .local_rim_evidence import digest


def signed_clearance(xy, polygon):
    p = np.asarray(polygon, float)
    edges = np.roll(p, -1, axis=0) - p
    sign = 1.0 if np.cross(p - p[0], np.roll(p, -1, axis=0) - p[0]).sum() >= 0 else -1.0
    return np.stack(
        [sign * np.cross(e, xy - a) / np.linalg.norm(e) for a, e in zip(p, edges)],
        axis=1,
    )


def spatial_interior_core(points, members, indexes, polygon, config):
    """Use equal occupied-cell votes, not a single Z plane or point density.

    The protected subset is selected before side proposals, from deep connected
    current ownership. All remaining original members remain explicitly present
    as boundary uncertainty. No subset is a steel contact certificate.
    """
    mask = np.zeros(len(points), bool)
    minimum = config["roi"]["min_opening_cells"]
    step = config["geometry"]["coarse_voxel_m"]
    origin = np.asarray(polygon).min(0)
    ids = np.flatnonzero(members)
    audit = dict(
        state="SPATIAL_CORE_UNOBSERVABLE",
        original_count=len(ids),
        original_digest=digest(indexes[ids]),
        original_mask_modified=False,
        spatial_core_count=0,
        material_continuity_asserted=False,
        steel_certified=False,
    )
    if not len(ids):
        return mask, audit
    cells = np.floor(
        (points[ids, :2] - origin) / step + 64 * np.finfo(float).eps
    ).astype(np.int64)
    keys, inv, count = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    centers = origin + (keys + 0.5) * step
    clearance = signed_clearance(centers, polygon).min(1)
    tile_origin = keys.min(0)
    shape = keys.max(0) - tile_origin + 3
    occupancy = np.zeros(tuple(shape), bool)
    at = keys - tile_origin + 1
    occupancy[at[:, 0], at[:, 1]] = True
    inward_distance = distance_transform_edt(occupancy) * step - step / 2
    clearance = np.minimum(clearance, inward_distance[at[:, 0], at[:, 1]])
    active = {
        tuple(k): i for i, k in enumerate(keys) if clearance[i] > 0 and count[i] > 0
    }
    components = []
    seen = set()
    for seed in sorted(active):
        if seed in seen:
            continue
        todo = [seed]
        seen.add(seed)
        group = []
        while todo:
            k = todo.pop()
            group.append(active[k])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (k[0] + dx, k[1] + dy)
                    if q in active and q not in seen:
                        seen.add(q)
                        todo.append(q)
        components.append(group)
    groups = [g for g in components if len(g) >= minimum]
    if not groups:
        return mask, dict(
            audit,
            occupied_cell_count=len(keys),
            component_sizes=[len(g) for g in components],
        )
    # Deepest supported component, then physical area, never a material Z label.
    group = max(groups, key=lambda g: (float(np.max(clearance[g])), len(g)))
    depth = max(step, float(np.median(clearance[group])))
    core_tiles = np.asarray(group)[clearance[group] >= depth]
    if len(core_tiles) < minimum:
        return mask, dict(audit, core_cell_count=len(core_tiles))
    mask[ids[np.isin(inv, core_tiles)]] = True
    mask &= signed_clearance(points[:, :2], polygon).min(1) >= step / 2
    if not mask.any():
        return mask, dict(audit, reason="NO_RAW_POINT_INSIDE_PROTECTED_TILES")
    anchor = np.median(points[mask, :2], axis=0)
    return mask, dict(
        audit,
        state="SPATIAL_INTERIOR_CORE",
        anchor_uv=anchor.tolist(),
        spatial_core_count=int(mask.sum()),
        spatial_core_digest=digest(indexes[mask]),
        spatial_core_cell_count=len(core_tiles),
        connected_cell_count=len(group),
        clearance_m=depth,
        boundary_uncertain_count=int(np.count_nonzero(members & ~mask)),
        boundary_uncertain_digest=digest(indexes[members & ~mask]),
        height_quantiles_m=np.quantile(
            points[mask, 2], [0.01, 0.25, 0.5, 0.75, 0.99]
        ).tolist(),
        basis="CURRENT_FROZEN_CELL_2D_CONNECTIVITY_AND_SIGNED_CLEARANCE;ALL_Z_MODES_RETAINED",
    )
