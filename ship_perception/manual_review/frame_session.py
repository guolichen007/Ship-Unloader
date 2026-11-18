"""Build a bound view and inclusive Raw index once per frame instance."""

import hashlib
from pathlib import Path
import struct
from time import perf_counter
from uuid import uuid4
import zlib

import numpy as np

from ship_perception.io.pcd_source import read_pcd
from ship_perception.perception.measurement import RawSpatialIndex
from ship_perception.r1_static.height_grid import R1HeightGrid

from .contract import (
    ContractError,
    FLAGS,
    Projection,
    RAW_ROWS,
    VERSION,
    check_id,
    digest,
    parse_json,
    points4,
    rectangle_axes,
)

from .side_candidates import CONFIG

MAX_CELLS = 10_000_000
MAX_GRID_BYTES = 700_000_000
MAX_POINTS = 12_000_000
MAX_WIRE_BYTES = 256 * 1024**2
DECODER = "ship_perception.io.pcd_source.read_pcd/finite-v1"


def projection_for(xyz, cell_m):
    if not np.isfinite(cell_m) or not 0.05 <= cell_m <= 5:
        raise ContractError("CELL_SIZE_RANGE")
    xy = xyz[:, :2].astype(np.float64)
    origin = xy.min(axis=0)
    if np.max(np.abs(xy)) > 1e7:
        raise ContractError("XY_PRECISION_LIMIT")
    nx, ny = (
        np.floor((xy.max(axis=0) - origin) / cell_m).astype(np.int64) + 1
    ).tolist()
    if (
        nx * ny > MAX_CELLS
        or max(nx, ny) > 16384
        or nx * ny * 63 + len(xyz) * 48 > MAX_GRID_BYTES
    ):
        raise ContractError("GRID_BUDGET_CHOOSE_COARSER")
    return Projection(*origin.tolist(), float(cell_m), nx, ny)


def png_rgba(rgba):
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    ny, nx, _ = rgba.shape
    rows = np.zeros((ny, 1 + nx * 4), dtype=np.uint8)
    rows[:, 1:] = rgba.reshape(ny, -1)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", nx, ny, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 3))
        + chunk(b"IEND", b"")
    )


def layers_for(grid):
    occupied = np.flipud(grid.occupancy)
    density = np.log1p(np.flipud(grid.count)[occupied])
    median = np.flipud(grid.median)[occupied]
    rgba = np.zeros((*occupied.shape, 4), dtype=np.uint8)
    rgba[occupied, 3] = 255
    images = {}
    for name, values in (("height", median), ("density", density), ("mask", None)):
        if values is None:
            rgba[occupied, :3] = (76, 185, 136)
        else:
            low, high = np.percentile(values, [2, 98])
            color = (
                np.clip((values - low) / max(high - low, 1e-12), 0, 1) * 215 + 40
            ).astype(np.uint8)
            rgba[occupied, :3] = np.column_stack((color, color, color))
        images[name] = png_rgba(rgba)
    return images


class FrameSession:
    def __init__(self, path, cell_m=0.1, frame_id="pcd", overlay=None):
        started = perf_counter()
        path = Path(path).resolve()
        if path.suffix.lower() != ".pcd" or not path.is_file():
            raise ContractError("PCD_REQUIRED")
        if path.stat().st_size > MAX_WIRE_BYTES:
            raise ContractError("PCD_WIRE_BUDGET")
        if not isinstance(frame_id, str) or not 0 < len(frame_id) <= 128:
            raise ContractError("FRAME_ID_LENGTH")
        self.frame = read_pcd(path, frame_id, max_points=MAX_POINTS)
        with path.open("rb") as stream:
            observed = hashlib.sha256()
            for block in iter(lambda: stream.read(1024**2), b""):
                observed.update(block)
        if observed.hexdigest() != self.frame.provenance["wire_sha256"]:
            raise ContractError("INPUT_CHANGED_DURING_DECODE")
        read_end = perf_counter()
        self.projection = p = projection_for(self.frame.xyz, cell_m)
        grid = R1HeightGrid.from_points(
            self.frame.xyz.astype(np.float64), p.s, origin_xy=(p.x0, p.y0)
        )
        self.layers = layers_for(grid)
        map_end = perf_counter()
        self.index = RawSpatialIndex(self.frame.xyz)
        self.manifest = {
            **FLAGS,
            "schema": "manual_map.1",
            "session_id": uuid4().hex,
            "frame_instance_id": uuid4().hex,
            "view_id": uuid4().hex,
            "frame_id": self.frame.frame_id,
            "stamp": self.frame.stamp,
            "source_count": self.frame.source_point_count,
            "finite_count": len(self.frame.xyz),
            "dtype": self.frame.xyz.dtype.str,
            "wire_sha256": observed.hexdigest(),
            "semantic_sha256": self.frame.sha256,
            "decoder": DECODER,
            "raw_rows": RAW_ROWS,
            "units": "m",
            "version": VERSION,
            "nx": p.nx,
            "ny": p.ny,
            "x0": p.x0,
            "y0": p.y0,
            "s": p.s,
            "pixel_to_xy": [[p.s, 0, p.x0], [0, -p.s, p.y0 + p.ny * p.s], [0, 0, 1]],
            "pixel_rule": "continuous_edges_u_right_v_down_centers_half",
            "array_rule": "row0_low_y_half_open_cells",
            "image_rule": "row0_high_y_empty_transparent",
            "client_xy_tolerance_m": 2e-7,
            "matching_config": CONFIG.audit(),
            "layers": {
                k: hashlib.sha256(v).hexdigest() for k, v in self.layers.items()
            },
            "config": {
                "cell_m": p.s,
                "max_cells": MAX_CELLS,
                "max_grid_bytes": MAX_GRID_BYTES,
            },
        }
        self.targets = {}
        self.overlay_path = None
        self.manifest["overlay_status"] = "DISABLED"
        if overlay is not None:
            try:
                self._read_overlay(Path(overlay).resolve())
            except (ValueError, OSError, KeyError, TypeError):
                self.manifest["overlay_status"] = "INVALID_OR_WRONG_FRAME"
        self.manifest["manifest_sha256"] = digest(self.manifest)
        self.timings = dict(
            pcd_read_s=read_end - started,
            map_s=map_end - read_end,
            index_and_bind_s=perf_counter() - map_end,
        )

    def _read_overlay(self, path):
        if path.stat().st_size > 1024**2:
            raise ContractError("OVERLAY_SIZE")
        data = path.read_bytes()
        overlay = parse_json(data)
        if overlay["schema"] != "manual_auto_overlay.1" or any(
            overlay[k] != self.manifest[k]
            for k in ("wire_sha256", "semantic_sha256", "frame_id")
        ):
            raise ContractError("OVERLAY_IDENTITY")
        targets = {}
        if not isinstance(overlay["targets"], list) or len(overlay["targets"]) > 128:
            raise ContractError("OVERLAY_TARGETS")
        for row in overlay["targets"]:
            key = check_id(row["opening_id"])
            vessel = check_id(row["vessel_id"])
            xy = points4(row["xy"])
            rectangle_axes(xy)
            if key in targets:
                raise ContractError("DUPLICATE_TARGET")
            targets[key] = {"vessel_id": vessel, "xy": xy.tolist()}
        self.targets = targets
        self.overlay_path = path
        self.manifest.update(
            overlay_status="DISPLAY_ONLY",
            overlay_sha256=hashlib.sha256(data).hexdigest(),
        )
