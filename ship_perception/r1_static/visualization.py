"""Dependency-free binary RGB PLY outputs for CloudCompare review."""
from pathlib import Path

import numpy as np


GRAY = (145, 145, 145)
GREEN = (45, 205, 85)
PURPLE = (170, 65, 210)
RED = (230, 55, 55)
BLUE = (45, 105, 235)
CYAN = (35, 215, 220)
WHITE = (255, 255, 255)
YELLOW = (250, 220, 40)
ORANGE = (245, 145, 35)


def write_colored_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or not np.isfinite(xyz).all():
        raise ValueError("INVALID_RGB_PLY_POINTS")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = np.empty(len(xyz), dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                              ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
    for index, field in enumerate(("x", "y", "z")):
        cloud[field] = xyz[:, index]
    for index, field in enumerate(("red", "green", "blue")):
        cloud[field] = rgb[:, index]
    header = ("ply\nformat binary_little_endian 1.0\n"
              "element vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n") % len(xyz)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(cloud.tobytes())
    return dict(vertex_count=int(len(xyz)), path=str(path))


def _segment(a, b, spacing):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    count = max(2, int(np.ceil(np.linalg.norm(b - a) / spacing)) + 1)
    return np.linspace(a, b, count)


def _markers(result, spacing):
    xyz, rgb = [], []
    for hatch in result["hatches"]:
        for edge in hatch["boundaries"]:
            color = CYAN if "OBSERVED_3D_FACE" in edge["evidence_type"] else BLUE
            segment = _segment(edge["a_raw"], edge["b_raw"], spacing)
            xyz.append(segment)
            rgb.append(np.tile(np.array(color, dtype=np.uint8), (len(segment), 1)))
        for corner in hatch["polygon_raw"] or []:
            offsets = np.array(((0, 0, 0), (spacing, 0, 0), (-spacing, 0, 0),
                                (0, spacing, 0), (0, -spacing, 0), (0, 0, spacing)))
            xyz.append(np.asarray(corner) + offsets)
            rgb.append(np.tile(np.array(WHITE, dtype=np.uint8), (len(offsets), 1)))
        center = np.asarray(hatch["center_raw"])
        xyz.append(np.asarray([center, center + [0, 0, spacing], center - [0, 0, spacing]]))
        rgb.append(np.tile(np.array(YELLOW, dtype=np.uint8), (3, 1)))
    if not xyz:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    return np.vstack(xyz), np.vstack(rgb)


def write_debug_clouds(directory, points, proposals, deck_supports, result, config):
    directory = Path(directory)
    xyz = np.asarray(points, dtype=np.float32)
    gray = np.tile(np.array(GRAY, dtype=np.uint8), (len(xyz), 1))
    proposal_color = gray.copy()
    for proposal in proposals:
        x0, y0, x1, y1 = proposal.bbox_xy
        mask = ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) &
                (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1))
        proposal_color[mask] = PURPLE
    deck_color = gray.copy()
    for proposal, deck, plane in deck_supports:
        if plane is None:
            continue
        x0, y0, x1, y1 = proposal.bbox_xy
        band = config["roi"]["support_search_m"]
        ring = ((xyz[:, 0] >= x0 - band) & (xyz[:, 0] <= x1 + band) &
                (xyz[:, 1] >= y0 - band) & (xyz[:, 1] <= y1 + band))
        ring &= ~((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) &
                  (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1))
        residual = np.abs(xyz @ plane[0] + plane[1])
        deck_color[ring & (residual <= config["geometry"]["plane_inlier_m"])] = GREEN
    scene_color = proposal_color.copy()
    scene_color[np.all(deck_color == np.array(GREEN), axis=1)] = GREEN
    for hatch in result["hatches"]:
        x0, y0, x1, y1 = hatch["extent"]
        mask = ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) &
                (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1))
        deck = hatch["local_deck"]
        if deck["normal_raw"] is not None:
            normal = np.asarray(deck["normal_raw"])
            relative = xyz @ normal + deck["offset"]
            mask &= relative < -config["roi"]["opening_drop_m"]
        scene_color[mask] = RED if hatch["status"] == "COMPLETE_OBSERVED" else ORANGE
    markers, marker_rgb = _markers(result, config["geometry"]["refine_voxel_m"])
    all_points = np.vstack((xyz, markers))
    all_color = np.vstack((scene_color, marker_rgb))
    return {
        "scene_debug.ply": write_colored_ply(directory / "scene_debug.ply", all_points, all_color),
        "proposal_debug.ply": write_colored_ply(directory / "proposal_debug.ply", xyz, proposal_color),
        "local_deck_debug.ply": write_colored_ply(directory / "local_deck_debug.ply", xyz, deck_color),
        "boundary_debug.ply": write_colored_ply(directory / "boundary_debug.ply", np.vstack((xyz, markers)),
                                                 np.vstack((gray, marker_rgb))),
    }
