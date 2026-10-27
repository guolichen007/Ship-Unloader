"""Lazy exact Raw quantiles on a shared vessel-local four-phase lattice."""

from collections import OrderedDict
import numpy as np
from ship_perception.r1_static.height_grid import R1HeightGrid


class RasterPhaseMeasurementCache:
    TILE_CELLS = 64  # Storage partition only; never changes physical resolution.

    def __init__(self, spatial, config, counters, enabled=True):
        self.spatial, self.config, self.counters = spatial, config, counters
        self.enabled = enabled
        self.cell = config["geometry"]["refine_voxel_m"]
        self.origin = spatial.xyz[:, :2].min(0) - self.cell
        self.tiles = OrderedDict()
        self.bytes = 0

    def _tile(self, column, row, phase):
        key = (column, row, tuple(phase))
        if self.enabled and key in self.tiles:
            self.tiles.move_to_end(key)
            self.counters["raster_tile_cache_hit"] = (
                self.counters.get("raster_tile_cache_hit", 0) + 1
            )
            return self.tiles[key]
        size = self.TILE_CELLS
        origin = self.origin + np.asarray(phase) * self.cell
        lower = np.array([column, row]) * size
        lo, hi = origin + lower * self.cell, origin + (lower + size) * self.cell
        points = self.spatial.query(
            [*np.nextafter(lo, -np.inf), *np.nextafter(hi, np.inf)]
        )
        cells = np.floor((points[:, :2] - origin) / self.cell).astype(np.int64) - lower
        keep = ((cells >= 0) & (cells < size)).all(1)
        cells, points = cells[keep], points[keep]
        count = np.zeros((size, size), dtype=np.int32)
        values = [np.full((size, size), np.nan) for _ in range(5)]
        if len(points):
            keys = cells[:, 1] * size + cells[:, 0]
            order = np.lexsort((points[:, 2], keys))
            unique, starts, counts = np.unique(
                keys[order], return_index=True, return_counts=True
            )
            count.flat[unique] = counts
            z = points[order, 2]
            for out, quantile in zip(values, (0.1, 0.5, 0.9, 0.0, 1.0)):
                out.flat[unique] = z[
                    starts + np.floor((counts - 1) * quantile).astype(np.int64)
                ]
        tile = (count, *values)
        self.counters["raster_build_count"] = (
            self.counters.get("raster_build_count", 0) + 1
        )
        self.counters["raster_tile_build_count"] = (
            self.counters.get("raster_tile_build_count", 0) + 1
        )
        byte_count = sum(a.nbytes for a in tile)
        while self.tiles and self.bytes + byte_count > 64 * 1024 * 1024:
            _, old = self.tiles.popitem(last=False)
            self.bytes -= sum(a.nbytes for a in old)
        if self.enabled:
            self.tiles[key] = tile
            self.bytes += byte_count
        return tile

    def grid(self, box, margin, phase):
        origin = self.origin + np.asarray(phase) * self.cell
        low = np.floor((np.asarray(box[:2]) - margin - origin) / self.cell).astype(int)
        high = (
            np.ceil((np.asarray(box[2:]) + margin - origin) / self.cell).astype(int) + 1
        )
        nx, ny = high - low
        if nx <= 0 or ny <= 0 or nx * ny > 10000000:
            raise ValueError("HEIGHT_GRID_EXTENT_TOO_LARGE")
        arrays = [np.zeros((ny, nx), dtype=np.int32)] + [
            np.full((ny, nx), np.nan) for _ in range(5)
        ]
        size = self.TILE_CELLS
        for row in range(low[1] // size, (high[1] - 1) // size + 1):
            for col in range(low[0] // size, (high[0] - 1) // size + 1):
                tile_low = np.array([col, row]) * size
                tile_box = np.r_[
                    origin + tile_low * self.cell,
                    origin + (tile_low + size) * self.cell,
                ]
                if not (
                    any(
                        tile_box[0] <= edge + margin and tile_box[2] >= edge - margin
                        for edge in (box[0], box[2])
                    )
                    or any(
                        tile_box[1] <= edge + margin and tile_box[3] >= edge - margin
                        for edge in (box[1], box[3])
                    )
                ):
                    continue
                tile = self._tile(col, row, phase)
                begin, end = np.maximum(low, tile_low), np.minimum(
                    high, tile_low + size
                )
                target = (
                    slice(begin[1] - low[1], end[1] - low[1]),
                    slice(begin[0] - low[0], end[0] - low[0]),
                )
                source = (
                    slice(begin[1] - tile_low[1], end[1] - tile_low[1]),
                    slice(begin[0] - tile_low[0], end[0] - tile_low[0]),
                )
                for out, raw in zip(arrays, tile):
                    out[target] = raw[source]
        self.counters["raster_phase_view_count"] = (
            self.counters.get("raster_phase_view_count", 0) + 1
        )
        local_origin = origin + low * self.cell
        return R1HeightGrid(*local_origin, self.cell, *arrays)
