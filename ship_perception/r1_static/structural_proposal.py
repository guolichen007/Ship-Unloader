"""Local ring contrast at two frozen V1.5 resolutions; no global deck."""
import numpy as np
from scipy import ndimage

from .height_grid import R1HeightGrid
from .model import Proposal


def _overlap(a, b):
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = x * y
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(min(area_a, area_b), 1e-9)


def _one_scale(points, config, cell_m):
    grid = R1HeightGrid.from_points(points, cell_m)
    valid = grid.occupancy
    high = np.where(valid, grid.q90, 0.0)
    # An annulus excludes the seed's immediate neighborhood. Both radii come
    # from the existing support-search distance, not scan-specific dimensions.
    radius = config["roi"]["support_search_m"]
    inner = max(1, int(round(radius / cell_m)))
    outer = max(inner + 1, 2 * inner)
    def box_mean(size):
        area = float(size * size)
        total = ndimage.uniform_filter(high, size=size, mode="constant") * area
        count = ndimage.uniform_filter(valid.astype(float), size=size, mode="constant") * area
        return total, count
    out_sum, out_count = box_mean(2 * outer + 1)
    in_sum, in_count = box_mean(2 * inner + 1)
    ring_count = out_count - in_count
    ring_height = (out_sum - in_sum) / np.maximum(ring_count, 1.0)
    drop = ring_height - grid.q10
    roi = config["roi"]
    seed = valid & (ring_count >= roi["min_deck_support_cells"]) & (drop > roi["opening_drop_m"])
    seed &= drop < roi["max_opening_depth_m"]
    labels, number = ndimage.label(seed, structure=np.ones((3, 3), dtype=np.uint8))
    rows = []
    for label_id, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        local = labels[sl] == label_id
        cells = int(local.sum())
        area = cells * cell_m * cell_m
        if cells < roi["min_opening_cells"] or not roi["opening_min_area_m2"] <= area <= roi["opening_max_area_m2"]:
            continue
        rr, cc = sl
        bbox = (grid.x0 + cc.start * cell_m, grid.y0 + rr.start * cell_m,
                grid.x0 + cc.stop * cell_m, grid.y0 + rr.stop * cell_m)
        rows.append((bbox, cells, float(np.mean(drop[sl][local])), cell_m))
    return rows, grid


def propose(points, config):
    sizes = (config["geometry"]["coarse_voxel_m"], config["geometry"]["candidate_voxel_m"])
    rows = []
    grids = []
    for size in dict.fromkeys(sizes):
        found, grid = _one_scale(points, config, size)
        rows.extend(found)
        grids.append(grid)
    # A coarse component may contain several openings. Consolidate proposal
    # envelopes across scales, but never equate one envelope with one hatch.
    groups = []
    for row in sorted(rows, key=lambda r: (-r[1] * r[2], r[0])):
        matches = [group for group in groups if any(_overlap(row[0], item[0]) >= 0.35 for item in group)]
        if not matches:
            groups.append([row])
        else:
            matches[0].append(row)
            for extra in matches[1:]:
                matches[0].extend(extra)
                groups.remove(extra)
    proposals = []
    for index, group in enumerate(groups):
        box = (min(r[0][0] for r in group), min(r[0][1] for r in group),
               max(r[0][2] for r in group), max(r[0][3] for r in group))
        proposals.append(Proposal("p%03d" % index, box, sum(r[1] for r in group),
                                  float(np.average([r[2] for r in group], weights=[r[1] for r in group])),
                                  tuple(sorted({r[3] for r in group}))))
    return proposals, grids
