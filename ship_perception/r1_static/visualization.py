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


def seed_point_ids(points, proposals):
    """Return precisely the raw points whose source grid cells were seeds."""
    xy = np.asarray(points)[:, :2]
    selected = np.zeros(len(xy), dtype=bool)
    for proposal in proposals:
        for component in proposal.seed_components:
            cells = np.asarray(component.cells_rc, dtype=np.int64).reshape(-1, 2)
            if not len(cells):
                continue
            x0, y0, x1, y1 = component.bbox_xy
            candidate = np.flatnonzero((xy[:, 0] >= x0) & (xy[:, 0] < x1) &
                                       (xy[:, 1] >= y0) & (xy[:, 1] < y1))
            if not len(candidate):
                continue
            row = np.floor((xy[candidate, 1] - component.origin_xy[1]) / component.cell_m).astype(np.int64)
            col = np.floor((xy[candidate, 0] - component.origin_xy[0]) / component.cell_m).astype(np.int64)
            low = cells.min(axis=0)
            high = cells.max(axis=0)
            width = int(high[1] - low[1] + 1)
            valid = ((row >= low[0]) & (row <= high[0]) &
                     (col >= low[1]) & (col <= high[1]))
            raw_keys = (row[valid] - low[0]) * width + col[valid] - low[1]
            seed_keys = (cells[:, 0] - low[0]) * width + cells[:, 1] - low[1]
            selected[candidate[valid]] |= np.isin(raw_keys, seed_keys)
    return np.flatnonzero(selected)


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
        if hatch["status"] == "PARTIAL":
            offsets = np.array(((spacing, 0, 0), (-spacing, 0, 0),
                                (0, spacing, 0), (0, -spacing, 0)))
            xyz.append(center + offsets)
            rgb.append(np.tile(np.array(ORANGE, dtype=np.uint8), (len(offsets), 1)))
    if not xyz:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    return np.vstack(xyz), np.vstack(rgb)


def write_debug_clouds(directory, points, proposals, deck_supports, result, config):
    directory = Path(directory)
    xyz = np.asarray(points, dtype=np.float32)
    gray = np.tile(np.array(GRAY, dtype=np.uint8), (len(xyz), 1))
    proposal_color = gray.copy()
    seed_ids = seed_point_ids(xyz, proposals)
    proposal_color[seed_ids] = PURPLE
    deck_color = gray.copy()
    for proposal, deck, plane, ownership in deck_supports:
        if plane is None:
            continue
        deck_color[ownership["accepted_raw_ids"]] = GREEN
    scene_color = proposal_color.copy()
    scene_color[np.all(deck_color == np.array(GREEN), axis=1)] = GREEN
    markers, marker_rgb = _markers(result, config["geometry"]["refine_voxel_m"])
    all_points = np.vstack((xyz, markers))
    all_color = np.vstack((scene_color, marker_rgb))
    frames = []
    for proposal in proposals:
        x0, y0, x1, y1 = proposal.bbox_xy
        own_seed_ids = seed_point_ids(xyz, [proposal])
        z = float(np.median(xyz[own_seed_ids, 2])) if len(own_seed_ids) else float(np.median(xyz[:, 2]))
        corners = np.array(((x0, y0, z), (x1, y0, z),
                            (x1, y1, z), (x0, y1, z)))
        frames.extend(_segment(a, b, config["geometry"]["coarse_voxel_m"])
                      for a, b in zip(corners, np.roll(corners, -1, axis=0)))
    envelope_xyz = np.vstack(frames) if frames else np.empty((0, 3))
    envelope_rgb = np.tile(np.array(PURPLE, dtype=np.uint8), (len(envelope_xyz), 1))
    return {
        "scene_debug.ply": write_colored_ply(directory / "scene_debug.ply", all_points, all_color),
        "proposal_debug.ply": write_colored_ply(directory / "proposal_debug.ply", xyz, proposal_color),
        "proposal_seed_debug.ply": write_colored_ply(directory / "proposal_seed_debug.ply", xyz, proposal_color),
        "proposal_envelope_debug.ply": write_colored_ply(directory / "proposal_envelope_debug.ply",
                                                         envelope_xyz, envelope_rgb),
        "local_deck_debug.ply": write_colored_ply(directory / "local_deck_debug.ply", xyz, deck_color),
        "boundary_debug.ply": write_colored_ply(directory / "boundary_debug.ply", np.vstack((xyz, markers)),
                                                 np.vstack((gray, marker_rgb))),
    }


def write_candidate_debug(directory, points, deck_supports):
    """Persist exact proposal-plane ownership in Work, outside formal result JSON."""
    directory = Path(directory) / "candidate_plane_debug"
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for proposal, _, _, ownership in deck_supports:
        candidates = ownership["candidate_plane_raw_ids"]
        arrays = {"candidate_%02d" % index: ids for index, ids in enumerate(candidates)}
        arrays.update({"competitor_%02d" % candidate_index: ids
                       for candidate_index, ids in zip(ownership.get("competitor_candidate_indexes", []),
                                                       ownership["competitor_raw_ids"])})
        target = directory / ("%s_ownership.npz" % proposal.proposal_id)
        np.savez_compressed(target, **arrays)
        previews = []
        for index, ids in enumerate(candidates[:2]):
            name = "%s_candidate_plane_%02d.ply" % (proposal.proposal_id, index)
            write_colored_ply(directory / name, points[ids],
                              np.tile(np.array(RED if index else GREEN, dtype=np.uint8), (len(ids), 1)))
            previews.append(name)
        artifacts[proposal.proposal_id] = dict(ownership_npz=str(target), preview_ply=previews,
                                               candidate_raw_counts=[int(len(ids)) for ids in candidates],
                                               competitor_candidate_indexes=ownership.get("competitor_candidate_indexes", []))
    return artifacts
