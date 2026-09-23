"""Dependency-free PNG and point-exact, region-colored PLY for height basins."""
from pathlib import Path
import struct
import zlib

import numpy as np

from .visualization import GRAY, write_colored_ply


PALETTE = ((238, 77, 92), (36, 190, 220), (255, 188, 53),
           (123, 214, 85), (177, 116, 238), (255, 110, 192),
           (80, 140, 250), (223, 220, 75))

GLYPHS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "r": ("000", "110", "101", "100", "100"),
    "-": ("000", "000", "111", "000", "000"),
    ".": ("000", "000", "000", "000", "010"),
    ",": ("000", "000", "000", "010", "100"),
    "[": ("110", "100", "100", "100", "110"),
    "]": ("011", "001", "001", "001", "011"),
    " ": ("000", "000", "000", "000", "000"),
}


def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))


def _save_png(path, rgb):
    """Write 8-bit RGB PNG using only the standard library and NumPy."""
    pixels = np.ascontiguousarray(rgb, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("INVALID_RGB_IMAGE")
    height, width, _ = pixels.shape
    scanlines = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    data = (b"\x89PNG\r\n\x1a\n" +
            _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            _chunk(b"IDAT", zlib.compress(scanlines, 6)) + _chunk(b"IEND", b""))
    Path(path).write_bytes(data)


def _draw_text(canvas, text, x, y, color, scale=2):
    for character in text:
        glyph = GLYPHS.get(character, GLYPHS[" "])
        for row, pattern in enumerate(glyph):
            for col, bit in enumerate(pattern):
                if bit == "1":
                    x0, y0 = x + col * scale, y + row * scale
                    canvas[y0:y0 + scale, x0:x0 + scale] = color
        x += 4 * scale


def _draw_line(canvas, start, end, color, width=2):
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    count = max(2, int(np.ceil(np.max(np.abs(b - a)))) + 1)
    coords = np.rint(np.linspace(a, b, count)).astype(int)
    for dx in range(-(width // 2), width - width // 2):
        for dy in range(-(width // 2), width - width // 2):
            x, y = coords[:, 0] + dx, coords[:, 1] + dy
            inside = (x >= 0) & (x < canvas.shape[1]) & (y >= 0) & (y < canvas.shape[0])
            canvas[y[inside], x[inside]] = color


def _surface_image(grid, valid, scale):
    surface = grid.median
    if np.any(valid):
        low, high = np.percentile(surface[valid], (5, 95))
    else:
        low, high = 0.0, 1.0
    fraction = np.clip((np.nan_to_num(surface, nan=low) - low) /
                       max(high - low, 1e-9), 0, 1)
    rgb = np.empty(surface.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = 35 + (220 * fraction).astype(np.uint8)
    rgb[..., 1] = 29 + (198 * np.sqrt(fraction)).astype(np.uint8)
    rgb[..., 2] = 92 + (86 * (1 - fraction)).astype(np.uint8)
    rgb[~valid] = (15, 18, 27)
    return np.repeat(np.repeat(np.flipud(rgb), scale, axis=0), scale, axis=1)


def _pixel(grid, point, scale):
    x = (point[0] - grid.x0) / grid.cell_m * scale
    y = (grid.count.shape[0] - (point[1] - grid.y0) / grid.cell_m) * scale
    return float(x), float(y)


def _edge_points(corners, step_m):
    rows = []
    for start, end in ((0, 1), (1, 2), (2, 3), (3, 0),
                       (4, 5), (5, 6), (6, 7), (7, 4),
                       (0, 4), (1, 5), (2, 6), (3, 7)):
        a, b = np.asarray(corners[start], dtype=float), np.asarray(corners[end], dtype=float)
        count = max(2, int(np.ceil(np.linalg.norm(b - a) / step_m)) + 1)
        rows.append(np.linspace(a, b, count))
    return np.vstack(rows)


def render_regions(directory, points, result, auxiliary, config):
    """Every colored raw point belongs to its recorded coarse region mask."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    grid, valid, masks = auxiliary["coarse"], auxiliary["valid"], auxiliary["masks"]
    scale = 8
    surface = _surface_image(grid, valid, scale)
    _save_png(directory / "height_surface.png", surface)
    levels = result["levels_m"]
    level_image = np.full(grid.count.shape + (3,), (15, 18, 27), dtype=np.uint8)
    if levels:
        bins = np.searchsorted(np.asarray(levels),
                               np.nan_to_num(grid.median, nan=levels[0]))
        for index in range(len(levels) + 1):
            level_image[valid & (bins == index)] = PALETTE[index % len(PALETTE)]
    level_image = np.repeat(np.repeat(np.flipud(level_image), scale, axis=0), scale, axis=1)
    level_legend = np.full((level_image.shape[0] + 28, level_image.shape[1], 3),
                           (15, 18, 27), dtype=np.uint8)
    level_legend[28:] = level_image
    for index, level in enumerate(levels):
        left = 12 + index * 112
        if left + 100 > level_legend.shape[1]:
            break
        level_legend[8:20, left:left + 12] = PALETTE[(index + 1) % len(PALETTE)]
        _draw_text(level_legend, "%.1f" % level, left + 18, 9, (245, 245, 245))
    _save_png(directory / "height_levels.png", level_legend)

    raw_colors = np.tile(np.asarray(GRAY, dtype=np.uint8), (len(points), 1))
    rows, cols = grid.cell_indices(points[:, :2])
    in_grid = ((rows >= 0) & (rows < grid.count.shape[0]) &
               (cols >= 0) & (cols < grid.count.shape[1]))
    overlay = surface.copy()
    box_cloud, box_colors = [], []
    ownership = {}
    assigned = np.zeros(len(points), dtype=bool)
    for index, (record, mask) in enumerate(zip(result["regions"], masks)):
        color = PALETTE[index % len(PALETTE)]
        record["color_rgb"] = list(color)
        owned = np.zeros(len(points), dtype=bool)
        owned[in_grid] = mask[rows[in_grid], cols[in_grid]]
        if np.any(owned & assigned):
            raise ValueError("OVERLAPPING_HEIGHTMAP_REGION_OWNERSHIP")
        assigned |= owned
        raw_colors[owned] = color
        record["raw_colored_point_count"] = int(owned.sum())
        ids = np.flatnonzero(owned)
        ownership[record["region_id"]] = ids
        write_colored_ply(directory / "regions" / (record["region_id"] + "_points.ply"),
                          points[ids],
                          np.tile(np.asarray(color, dtype=np.uint8), (len(ids), 1)))
        region_pixels = np.repeat(np.repeat(np.flipud(mask), scale, axis=0), scale, axis=1)
        overlay[region_pixels] = ((overlay[region_pixels].astype(float) +
                                   np.asarray(color)) / 2).astype(np.uint8)
        corners = record["calibration_box_raw"]
        wire = _edge_points(corners, config["geometry"]["coarse_voxel_m"] / 2)
        box_cloud.append(wire)
        box_colors.append(np.tile(np.asarray(color, dtype=np.uint8), (len(wire), 1)))
    for record in result["regions"]:
        corners = [_pixel(grid, point, scale) for point in record["calibration_box_xy"]]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            _draw_line(overlay, a, b, record["color_rgb"], width=3)
        _draw_text(overlay, record["region_id"], int(corners[0][0]),
                   int(corners[0][1]), (255, 255, 255))
        if record["topology_review"] == "POSSIBLE_MERGE":
            for alternative in record["alternate_subregions"]:
                x0, y0, x1, y1 = alternative["bbox_xy"]
                box = [_pixel(grid, point, scale) for point in
                       ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
                for a, b in zip(box, box[1:] + box[:1]):
                    _draw_line(overlay, a, b, (255, 255, 255), width=1)
    header_height = 28 * max(1, len(result["regions"]))
    annotated = np.full((overlay.shape[0] + header_height, overlay.shape[1], 3),
                        (15, 18, 27), dtype=np.uint8)
    annotated[header_height:] = overlay
    for index, record in enumerate(result["regions"]):
        y = 7 + 28 * index
        annotated[y:y + 14, 12:26] = record["color_rgb"]
        x0, y0, x1, y1 = record["bbox_xy"]
        label = "%s [%.1f,%.1f]-[%.1f,%.1f]" % (record["region_id"], x0, y0, x1, y1)
        _draw_text(annotated, label, 34, y + 2, (245, 245, 245))
    _save_png(directory / "hatch_regions.png", annotated)
    frame = np.vstack(box_cloud) if box_cloud else np.empty((0, 3))
    frame_colors = np.vstack(box_colors) if box_colors else np.empty((0, 3), dtype=np.uint8)
    write_colored_ply(directory / "hatch_boxes.ply", frame, frame_colors)
    write_colored_ply(directory / "hatch_regions.ply", np.vstack((points, frame)),
                      np.vstack((raw_colors, frame_colors)))
    np.savez_compressed(directory / "region_point_ownership.npz", **ownership)
    return dict(raw_vertex_count=int(len(points)), box_vertex_count=int(len(frame)),
                ownership_npz="region_point_ownership.npz",
                color_semantics="RAW_XY_IN_SELECTED_COARSE_BASIN; BOX_VERTICES_ARE_PROVISIONAL")
