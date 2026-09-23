"""Split one structural proposal into any number of measured lower-return regions."""
import numpy as np
from scipy import ndimage

from .height_grid import R1HeightGrid
from .model import Opening


def find_openings(points, proposal, plane, config):
    x0, y0, x1, y1 = proposal.bbox_xy
    margin = config["geometry"]["candidate_voxel_m"]
    xy = points[:, :2]
    keep = (xy[:, 0] >= x0 - margin) & (xy[:, 0] <= x1 + margin)
    keep &= (xy[:, 1] >= y0 - margin) & (xy[:, 1] <= y1 + margin)
    local = points[keep]
    if not len(local):
        return [], None
    cell_m = config["geometry"]["candidate_voxel_m"]
    grid = R1HeightGrid.from_points(local, cell_m)
    xx, yy = grid.centers()
    normal, offset = plane
    lower = xx * normal[0] + yy * normal[1] + grid.q10 * normal[2] + offset
    drop = -lower
    roi = config["roi"]
    # Sparse static scans may place one return in a 15 cm cell. Reject isolated
    # noise by connected physical area below, not by a per-cell point count.
    mask = grid.occupancy.copy()
    mask &= (drop > roi["opening_drop_m"]) & (drop < roi["max_opening_depth_m"])
    mask &= (xx >= x0 - margin) & (xx <= x1 + margin) & (yy >= y0 - margin) & (yy <= y1 + margin)
    # Close one empty lattice cell between measured returns. The mask is only
    # an opening seed; downstream sides still require raw 3D observations.
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=np.uint8))
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    openings = []
    for label_id, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        component = labels[sl] == label_id
        cells = int(component.sum())
        area = cells * cell_m * cell_m
        if cells < roi["min_opening_cells"] or not roi["opening_min_area_m2"] <= area <= roi["opening_max_area_m2"]:
            continue
        rr, cc = sl
        bbox = (grid.x0 + cc.start * cell_m, grid.y0 + rr.start * cell_m,
                grid.x0 + cc.stop * cell_m, grid.y0 + rr.stop * cell_m)
        # Closing may bridge empty grid cells; only measured returns have a
        # physical drop and may contribute to its reported mean.
        observed = component & grid.occupancy[sl]
        openings.append(Opening("%s-o%03d" % (proposal.proposal_id, len(openings)), proposal.proposal_id,
                                bbox, cells, float(area), float(np.mean(drop[sl][observed]))))
    return openings, grid
