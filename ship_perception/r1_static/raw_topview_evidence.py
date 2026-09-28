"""Raw XYZ top-view evidence in the current ship axes.

The rasters are measurements, not rendered PNG inputs or steel-edge labels.
Every transition keeps its measured support positions for later topology checks.
"""

import math

import numpy as np
from scipy import ndimage

from .height_grid import R1HeightGrid


CHANNELS = ("HEIGHT_TRANSITION", "DENSITY_TRANSITION", "VERTICAL_SPAN_RIDGE")


def build_topview(points, axes, config):
    """Build frozen coarse/fine grids after rotating Raw XYZ into ship axes."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("INVALID_RAW_POINTS")
    if not np.isfinite(points).all():
        raise ValueError("NONFINITE_RAW_POINTS")
    basis = np.asarray(axes, dtype=float)
    if basis.shape != (2, 2) or not np.allclose(basis @ basis.T, np.eye(2), atol=1e-5):
        raise ValueError("INVALID_SHIP_AXES")
    axial = np.column_stack((points[:, :2] @ basis.T, points[:, 2]))
    return {
        name: R1HeightGrid.from_points(axial, config["geometry"][key])
        for name, key in (("fine", "candidate_voxel_m"), ("coarse", "coarse_voxel_m"))
    }


def raster_channels(grid, config):
    """Return measured XY transition masks for each direction and channel."""
    valid = grid.occupancy
    span = grid.maximum - grid.minimum
    result = {}
    for normal_axis in (0, 1):
        # Grid columns are ship U and rows are ship V. The edge coordinate is
        # the interface between adjacent occupied cells, never a cell center.
        if normal_axis == 0:
            left = (slice(None), slice(None, -1))
            right = (slice(None), slice(1, None))
        else:
            left = (slice(None, -1), slice(None))
            right = (slice(1, None), slice(None))
        both = valid[left] & valid[right]
        height_delta = np.abs(grid.median[left] - grid.median[right])
        density_delta = np.abs(np.log1p(grid.count[left]) - np.log1p(grid.count[right]))
        span_delta = np.abs(span[left] - span[right])
        result[normal_axis] = {
            "HEIGHT_TRANSITION": both
            & (height_delta >= config["boundary"]["profile_min_drop_m"]),
            "DENSITY_TRANSITION": both
            & (density_delta >= math.log(2.0))
            & (
                np.maximum(grid.count[left], grid.count[right])
                >= config["geometry"]["outlier_min_points"]
            ),
            "VERTICAL_SPAN_RIDGE": both
            & (
                np.maximum(span[left], span[right])
                >= config["boundary"]["face_vertical_span_min_m"]
            )
            & (span_delta >= config["boundary"]["profile_min_drop_m"]),
        }
    return result


def raster_modes(grid, config):
    """Find axis-aligned measured line fragments without expected hatch size."""
    channels = raster_channels(grid, config)
    minimum_cells = max(
        2, int(math.ceil(config["boundary"]["line_min_length_m"] / grid.cell_m))
    )
    modes = {0: [], 1: []}
    for normal_axis in (0, 1):
        shape = channels[normal_axis][CHANNELS[0]].shape
        for normal_index in range(shape[1 if normal_axis == 0 else 0]):
            row = {}
            for channel in CHANNELS:
                mask = channels[normal_axis][channel]
                values = (
                    mask[:, normal_index] if normal_axis == 0 else mask[normal_index, :]
                )
                row[channel] = np.flatnonzero(values)
            union = np.unique(np.concatenate(tuple(row.values())))
            if len(union) < minimum_cells:
                continue
            # Preserve disjoint observed spans. Closing is only used to decide
            # which fragments share one candidate rho; it adds no support.
            occupied = np.zeros(shape[0 if normal_axis == 0 else 1], dtype=bool)
            occupied[union] = True
            gap_cells = max(
                1, int(math.ceil(config["roi"]["support_search_m"] / grid.cell_m))
            )
            joined = ndimage.binary_closing(
                occupied, structure=np.ones(gap_cells), border_value=0
            )
            labels, count = ndimage.label(joined | occupied)
            origin_normal = grid.x0 if normal_axis == 0 else grid.y0
            origin_along = grid.y0 if normal_axis == 0 else grid.x0
            rho = origin_normal + (normal_index + 1) * grid.cell_m
            for label_id in range(1, count + 1):
                support = union[labels[union] == label_id]
                if len(support) < minimum_cells:
                    continue
                types = [
                    channel
                    for channel, ids in row.items()
                    if np.intersect1d(ids, support).size
                ]
                modes[normal_axis].append(
                    {
                        "rho": float(rho),
                        "along_values": (
                            origin_along + (support + 0.5) * grid.cell_m
                        ).tolist(),
                        "along_min": float(
                            origin_along + (support.min() + 0.5) * grid.cell_m
                        ),
                        "along_max": float(
                            origin_along + (support.max() + 0.5) * grid.cell_m
                        ),
                        "support_count": int(len(support)),
                        "evidence_types": types,
                        "channel_counts": {
                            channel: int(np.intersect1d(ids, support).size)
                            for channel, ids in row.items()
                        },
                        "cell_m": grid.cell_m,
                    }
                )
    return modes
