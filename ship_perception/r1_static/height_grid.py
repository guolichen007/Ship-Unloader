"""Deterministic height quantiles on the raw XY lattice."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class R1HeightGrid:
    x0: float
    y0: float
    cell_m: float
    count: np.ndarray
    q10: np.ndarray
    median: np.ndarray
    q90: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def occupancy(self):
        return self.count > 0

    def centers(self):
        rows, cols = np.indices(self.count.shape)
        return self.x0 + (cols + 0.5) * self.cell_m, self.y0 + (rows + 0.5) * self.cell_m

    def cell_indices(self, xy):
        xy = np.asarray(xy)
        col = np.floor((xy[:, 0] - self.x0) / self.cell_m).astype(np.int64)
        row = np.floor((xy[:, 1] - self.y0) / self.cell_m).astype(np.int64)
        return row, col

    @classmethod
    def from_points(cls, points, cell_m):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
            raise ValueError("INVALID_RAW_POINTS")
        if not np.isfinite(cell_m) or cell_m <= 0:
            raise ValueError("INVALID_CELL_SIZE")
        # Anchor the lattice to the observed cloud, so translating raw XYZ
        # cannot change proposal decisions through an absolute-world grid phase.
        x0 = float(points[:, 0].min())
        y0 = float(points[:, 1].min())
        col = np.floor((points[:, 0] - x0) / cell_m).astype(np.int64)
        row = np.floor((points[:, 1] - y0) / cell_m).astype(np.int64)
        nx, ny = int(col.max()) + 1, int(row.max()) + 1
        if nx * ny > 10000000:
            raise ValueError("HEIGHT_GRID_EXTENT_TOO_LARGE")
        key = row * nx + col
        order = np.lexsort((points[:, 2], key))
        sorted_key, z = key[order], points[order, 2]
        unique, starts, counts = np.unique(sorted_key, return_index=True, return_counts=True)
        shape = (ny, nx)
        count = np.zeros(shape, dtype=np.int32)
        count.flat[unique] = counts
        arrays = []
        for quantile in (0, 0.1, 0.5, 0.9, 1):
            values = np.full(shape, np.nan, dtype=np.float64)
            index = starts + np.floor((counts - 1) * quantile).astype(np.int64)
            values.flat[unique] = z[index]
            arrays.append(values)
        return cls(x0, y0, float(cell_m), count, arrays[1], arrays[2], arrays[3], arrays[0], arrays[4])
