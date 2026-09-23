"""Dependency-free binary RGB PLY outputs for CloudCompare review."""
from collections import defaultdict
import json
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


def component_point_ids(points, component):
    """Map one recorded seed cell set back to raw point IDs."""
    xy = np.asarray(points)[:, :2]
    cells = np.asarray(component.cells_rc, dtype=np.int64).reshape(-1, 2)
    if not len(cells):
        return np.empty(0, dtype=np.int64)
    x0, y0, x1, y1 = component.bbox_xy
    candidate = np.flatnonzero((xy[:, 0] >= x0) & (xy[:, 0] < x1) &
                               (xy[:, 1] >= y0) & (xy[:, 1] < y1))
    if not len(candidate):
        return np.empty(0, dtype=np.int64)
    row = np.floor((xy[candidate, 1] - component.origin_xy[1]) / component.cell_m).astype(np.int64)
    col = np.floor((xy[candidate, 0] - component.origin_xy[0]) / component.cell_m).astype(np.int64)
    low = cells.min(axis=0)
    high = cells.max(axis=0)
    width = int(high[1] - low[1] + 1)
    valid = ((row >= low[0]) & (row <= high[0]) &
             (col >= low[1]) & (col <= high[1]))
    raw_keys = (row[valid] - low[0]) * width + col[valid] - low[1]
    seed_keys = (cells[:, 0] - low[0]) * width + cells[:, 1] - low[1]
    return candidate[valid][np.isin(raw_keys, seed_keys)]


def seed_point_ids(points, proposals):
    """Return precisely the raw points whose source grid cells were seeds."""
    selected = np.zeros(len(points), dtype=bool)
    for proposal in proposals:
        for component in proposal.seed_components:
            selected[component_point_ids(points, component)] = True
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


def write_debug_clouds(directory, points, proposals, deck_supports, result, config,
                       accepted_support_ids=None):
    directory = Path(directory)
    xyz = np.asarray(points, dtype=np.float32)
    gray = np.tile(np.array(GRAY, dtype=np.uint8), (len(xyz), 1))
    proposal_color = gray.copy()
    seed_ids = seed_point_ids(xyz, proposals)
    proposal_color[seed_ids] = PURPLE
    deck_color = gray.copy()
    if accepted_support_ids is None:
        for proposal, deck, plane, ownership in deck_supports:
            if plane is not None:
                deck_color[ownership["accepted_raw_ids"]] = GREEN
    else:
        deck_color[accepted_support_ids] = GREEN
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


def write_perimeter_artifacts(directory, points, forensics, config):
    """Write actual seed/segment/deck ownership and all observed rough boundaries."""
    directory = Path(directory)
    xyz = np.asarray(points, dtype=np.float32)
    palette = (BLUE, CYAN, PURPLE, ORANGE)
    spacing = config["geometry"]["refine_voxel_m"]
    groups = defaultdict(lambda: dict(seed_ids=[], support_ids=[], perimeter_xyz=[],
                                      perimeter_rgb=[], boundary_xyz=[], boundary_rgb=[],
                                      records=[]))
    all_seed_ids, all_support_ids = [], []
    segment_dir = directory / "segment_ownership"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segment_count = 0
    for item in forensics:
        seed = item["seed"]
        group = groups[seed.proposal_id]
        seed_ids = component_point_ids(xyz, seed.component)
        group["seed_ids"].append(seed_ids)
        all_seed_ids.append(seed_ids)
        z = float(np.median(xyz[seed_ids, 2])) if len(seed_ids) else float(np.median(xyz[:, 2]))
        if seed.contour_xy:
            contour = np.asarray(seed.contour_xy, dtype=float)
            contour_xyz = np.column_stack((contour, np.full(len(contour), z)))
            group["perimeter_xyz"].append(contour_xyz)
            group["perimeter_rgb"].append(
                np.tile(np.array(WHITE, dtype=np.uint8), (len(contour_xyz), 1)))
        center = np.array(((seed.bbox_xy[0] + seed.bbox_xy[2]) / 2,
                           (seed.bbox_xy[1] + seed.bbox_xy[3]) / 2, z))
        group["perimeter_xyz"].append(center.reshape(1, 3))
        group["perimeter_rgb"].append(np.array([YELLOW], dtype=np.uint8))
        segment_records = []
        for index, segment in enumerate(item["segments"]):
            segment_count += 1
            deck, _, ownership = item["segment_decks"][segment.segment_id]
            support_ids = ownership["selected_raw_ids"]
            group["support_ids"].append(support_ids)
            all_support_ids.append(support_ids)
            start = np.array((*segment.rough_start, z))
            end = np.array((*segment.rough_end, z))
            line = _segment(start, end, config["geometry"]["coarse_voxel_m"])
            color = palette[index % len(palette)]
            group["perimeter_xyz"].append(line)
            group["perimeter_rgb"].append(
                np.tile(np.array(color, dtype=np.uint8), (len(line), 1)))
            midpoint = (start + end) / 2
            tip = midpoint + np.array((*segment.outward_normal, 0.0)) * min(
                config["roi"]["support_connectivity_m"], segment.length_m / 4)
            arrow = _segment(midpoint, tip, spacing)
            group["perimeter_xyz"].append(arrow)
            group["perimeter_rgb"].append(
                np.tile(np.array(ORANGE, dtype=np.uint8), (len(arrow), 1)))
            arrays = dict(strip_raw_ids=ownership["strip_raw_ids"],
                          selected_raw_ids=support_ids)
            arrays.update({"candidate_%02d" % candidate_index: ids
                           for candidate_index, ids in enumerate(ownership["candidate_raw_ids"])})
            np.savez_compressed(segment_dir / ("%s_ownership.npz" % segment.segment_id), **arrays)
            write_colored_ply(segment_dir / ("segment_%s_support.ply" % segment.segment_id),
                              xyz[support_ids],
                              np.tile(np.array(GREEN, dtype=np.uint8), (len(support_ids), 1)))
            for candidate_index, ids in enumerate(ownership["candidate_raw_ids"][:2]):
                write_colored_ply(
                    segment_dir / ("segment_%s_candidate_plane_%02d.ply" %
                                   (segment.segment_id, candidate_index)),
                    xyz[ids],
                    np.tile(np.array(palette[candidate_index], dtype=np.uint8), (len(ids), 1)))
            segment_records.append({**segment.record(), **deck,
                                    "ownership_npz": str(segment_dir /
                                                         ("%s_ownership.npz" % segment.segment_id))})
        for edge in item["boundary"]["boundaries"]:
            line = _segment(edge["a_raw"], edge["b_raw"], spacing)
            color = CYAN if "OBSERVED_3D_FACE" in edge["evidence_type"] else BLUE
            group["boundary_xyz"].append(line)
            group["boundary_rgb"].append(
                np.tile(np.array(color, dtype=np.uint8), (len(line), 1)))
        group["records"].append(dict(**seed.record(), segments=segment_records,
                                     status=item["boundary"]["status"],
                                     observed_boundaries=item["boundary"]["boundaries"]))

    def stacked(rows, dtype=float):
        return np.vstack(rows).astype(dtype) if rows else np.empty((0, 3), dtype=dtype)

    metadata = dict(per_proposal={}, segment_ownership_count=segment_count)
    global_perimeter_xyz, global_perimeter_rgb = [], []
    global_boundary_xyz, global_boundary_rgb = [], []
    review_dir = directory / "per_proposal_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    for proposal_id, group in groups.items():
        seed_ids = np.unique(np.concatenate(group["seed_ids"])) if group["seed_ids"] else np.empty(0, dtype=int)
        support_ids = (np.unique(np.concatenate(group["support_ids"]))
                       if group["support_ids"] else np.empty(0, dtype=int))
        perimeter_xyz = stacked(group["perimeter_xyz"])
        perimeter_rgb = stacked(group["perimeter_rgb"], np.uint8)
        boundary_xyz = stacked(group["boundary_xyz"])
        boundary_rgb = stacked(group["boundary_rgb"], np.uint8)
        targets = {
            "seed": review_dir / ("%s_seed.ply" % proposal_id),
            "perimeter": review_dir / ("%s_perimeter.ply" % proposal_id),
            "segment_decks": review_dir / ("%s_segment_decks.ply" % proposal_id),
            "boundary": review_dir / ("%s_boundary.ply" % proposal_id),
            "json": review_dir / ("%s_perimeter.json" % proposal_id),
        }
        write_colored_ply(targets["seed"], xyz[seed_ids],
                          np.tile(np.array(PURPLE, dtype=np.uint8), (len(seed_ids), 1)))
        write_colored_ply(targets["perimeter"], perimeter_xyz, perimeter_rgb)
        write_colored_ply(targets["segment_decks"], xyz[support_ids],
                          np.tile(np.array(GREEN, dtype=np.uint8), (len(support_ids), 1)))
        write_colored_ply(targets["boundary"], boundary_xyz, boundary_rgb)
        targets["json"].write_text(json.dumps(group["records"], ensure_ascii=False,
                                              indent=2, allow_nan=False) + "\n", encoding="utf-8")
        metadata["per_proposal"][proposal_id] = {key: str(path) for key, path in targets.items()}
        global_perimeter_xyz.extend(group["perimeter_xyz"])
        global_perimeter_rgb.extend(group["perimeter_rgb"])
        global_boundary_xyz.extend(group["boundary_xyz"])
        global_boundary_rgb.extend(group["boundary_rgb"])
    raw_rgb = np.tile(np.array(GRAY, dtype=np.uint8), (len(xyz), 1))
    if all_seed_ids:
        raw_rgb[np.unique(np.concatenate(all_seed_ids))] = PURPLE
    if all_support_ids:
        raw_rgb[np.unique(np.concatenate(all_support_ids))] = GREEN
    combined_xyz = np.vstack((xyz, stacked(global_perimeter_xyz),
                              stacked(global_boundary_xyz)))
    combined_rgb = np.vstack((raw_rgb, stacked(global_perimeter_rgb, np.uint8),
                              stacked(global_boundary_rgb, np.uint8)))
    metadata["perimeter_debug.ply"] = write_colored_ply(
        directory / "perimeter_debug.ply", combined_xyz, combined_rgb)
    return metadata
