"""Research-only bridge from every height node to existing raw-3D boundary evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ship_perception.tools.v15_config_identity import config_hash
from ship_perception.tools.v15_pcd import decode

from .batch import NORMAL6
from .heightmap_batch import RESEARCH_CONFIG
from .heightmap_topology import (
    _height_levels,
    _link_levels,
    _nodes_at_level,
    _vessel_support,
    detect_regions,
)
from .model import OpeningSeed, SeedComponent
from .opening_seed import FovMap, _outer_contour, perimeter_segments
from .perimeter_boundary import refine_perimeter_boundaries
from .perimeter_deck import estimate_segment_deck
from .run import DEFAULT_CONFIG, REPO, _atomic_json, _git_sha, resolve_config
from .visualization import GRAY, write_colored_ply


FIELDS = (
    "scene_id",
    "node_id",
    "level_index",
    "level_m",
    "parent_id",
    "area_m2",
    "fill_ratio",
    "height_drop_m",
    "height_rise_side_count",
    "touches_observation_boundary",
    "perimeter_segment_count",
    "segment_deck_resolved_count",
    "segment_deck_ambiguous_count",
    "observed_profile_edge_count",
    "observed_3d_face_count",
    "combined_profile_face_count",
    "observed_edge_count",
    "total_edge_coverage",
    "mean_edge_coverage",
    "minimum_edge_coverage",
    "observed_support_length_m",
    "fit_residual_p50_m",
    "fit_residual_p95_m",
    "boundary_status",
    "polygon_observed",
    "fov_status",
)
PALETTE = (
    (238, 77, 92),
    (36, 190, 220),
    (255, 188, 53),
    (123, 214, 85),
    (177, 116, 238),
    (255, 110, 192),
    (80, 140, 250),
    (223, 220, 75),
)


def _hierarchy(points, config, research):
    """Rebuild nodes without touching the baseline selector or joining fragments."""
    baseline, aux = detect_regions(points, config, research)
    grid, valid, vessel = aux["coarse"], aux["valid"], aux["vessel"]
    if not np.any(vessel):
        return baseline, aux, []
    levels = _height_levels(grid.median, valid & vessel, research)
    groups = [
        _nodes_at_level(
            grid, grid.median, valid, vessel, level, index, config, research
        )
        for index, level in enumerate(levels)
    ]
    _link_levels(groups)
    nodes = [node for group in groups for node in group]
    expected = [
        (row["node_id"], row["level_m"], row["bbox_xy"])
        for row in baseline["level_hierarchy"]
    ]
    actual = [(node.node_id, node.level_m, list(node.bbox_xy)) for node in nodes]
    if actual != expected:
        raise ValueError("HEIGHT_HIERARCHY_REBUILD_MISMATCH")
    return baseline, aux, nodes


def _node_seeds(node, grid, fov):
    labels, count = ndimage.label(node.mask, structure=np.ones((3, 3), dtype=bool))
    seeds = []
    for component_id in range(1, count + 1):
        cells = np.argwhere(labels == component_id)
        if not len(cells):
            continue
        component = SeedComponent(
            (grid.x0, grid.y0),
            grid.cell_m,
            tuple(map(tuple, cells.tolist())),
            node.bbox_xy,
        )
        contour = _outer_contour(component)
        touches = fov.touches(contour) if contour else False
        seeds.append(
            OpeningSeed(
                "%s-s%03d" % (node.node_id, component_id - 1),
                node.node_id,
                component,
                (component_id,),
                contour,
                (grid.cell_m,),
                (
                    "PARTIAL_FOV"
                    if touches
                    else "FULLY_OBSERVED" if contour else "UNKNOWN_FOV"
                ),
                touches,
            )
        )
    return seeds


def _trace(points, start, end, step_m):
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    count = max(2, int(np.ceil(np.linalg.norm(b - a) / step_m)) + 1)
    return np.linspace(a, b, count)


def _node_audit(points, node, grid, fov, config, tree):
    seeds = _node_seeds(node, grid, fov)
    all_segments, decks, edges, contours = [], [], [], []
    status = []
    for seed in seeds:
        if seed.contour_xy:
            xy = list(seed.contour_xy)
            z = node.dominant_height_m
            contours.extend(
                _trace(points, (*a, z), (*b, z), grid.cell_m / 2)
                for a, b in zip(xy, xy[1:] + xy[:1])
            )
        segments = perimeter_segments(seed, fov, config)
        segment_decks = {
            segment.segment_id: estimate_segment_deck(points, segment, config, tree)
            for segment in segments
        }
        refined = refine_perimeter_boundaries(
            points, seed, segments, segment_decks, config, tree
        )
        all_segments.extend(segments)
        decks.extend(row[0] for row in segment_decks.values())
        edges.extend(refined["boundaries"])
        status.append(refined["status"])
    coverage = [edge["coverage"] for edge in edges]
    residual50 = [edge["fit_residual_p50_m"] for edge in edges]
    residual95 = [edge["fit_residual_p95_m"] for edge in edges]
    crop = bool(
        np.any(node.mask[0])
        or np.any(node.mask[-1])
        or np.any(node.mask[:, 0])
        or np.any(node.mask[:, -1])
    )
    fov_status = (
        "PARTIAL_FOV"
        if crop or any(seed.touches_scan_boundary for seed in seeds)
        else "FULLY_OBSERVED" if seeds else "UNKNOWN_FOV"
    )
    complete = bool(
        len(seeds) == 1
        and status
        and all(row == "COMPLETE_OBSERVED" for row in status)
        and fov_status == "FULLY_OBSERVED"
    )
    boundary_status = (
        "COMPLETE_OBSERVED" if complete else "PARTIAL" if edges else "UNRESOLVED"
    )
    record = dict(
        node_id=node.node_id,
        level_index=node.level_index,
        level_m=node.level_m,
        parent_id=node.parent.node_id if node.parent else None,
        children_ids=[child.node_id for child in node.children],
        bbox_xy=list(node.bbox_xy),
        area_m2=node.area_m2,
        fill_ratio=node.fill_ratio,
        height_drop_m=node.height_drop_m,
        height_rise_side_count=node.height_rise_side_count,
        touches_observation_boundary=fov_status == "PARTIAL_FOV",
        perimeter_segment_count=len(all_segments),
        segment_deck_resolved_count=sum(row["status"] == "RESOLVED" for row in decks),
        segment_deck_ambiguous_count=sum(
            row["status"] == "SEGMENT_DECK_AMBIGUOUS" for row in decks
        ),
        observed_profile_edge_count=sum(
            edge["evidence_type"] == "OBSERVED_PROFILE_BREAK" for edge in edges
        ),
        observed_3d_face_count=sum(
            edge["evidence_type"] == "OBSERVED_3D_FACE" for edge in edges
        ),
        combined_profile_face_count=sum("+" in edge["evidence_type"] for edge in edges),
        observed_edge_count=len(edges),
        total_edge_coverage=float(sum(coverage)),
        mean_edge_coverage=float(np.mean(coverage)) if coverage else 0.0,
        minimum_edge_coverage=float(min(coverage)) if coverage else 0.0,
        observed_support_length_m=float(
            sum(edge["observed_support_length"] for edge in edges)
        ),
        fit_residual_p50_m=float(np.median(residual50)) if residual50 else None,
        fit_residual_p95_m=float(max(residual95)) if residual95 else None,
        boundary_status=boundary_status,
        polygon_observed=complete,
        fov_status=fov_status,
        contour_component_count=len(seeds),
        contours_xy=[list(map(list, seed.contour_xy)) for seed in seeds],
        segment_deck_statuses=[
            dict(
                segment_id=row["segment_id"], status=row["status"], reason=row["reason"]
            )
            for row in decks
        ],
        observed_edges=edges,
    )
    return record, contours


def _write_overlays(directory, points, records, contours, config):
    step = config["geometry"]["coarse_voxel_m"] / 2
    clouds, colors, profiles, faces = [], [], [], []
    for index, (record, contour_parts) in enumerate(zip(records, contours)):
        color = PALETTE[index % len(PALETTE)]
        if contour_parts:
            cloud = np.vstack(contour_parts)
            clouds.append(cloud)
            colors.append(np.tile(color, (len(cloud), 1)))
        for edge in record["observed_edges"]:
            line = _trace(points, edge["a_raw"], edge["b_raw"], step)
            if "OBSERVED_PROFILE_BREAK" in edge["evidence_type"]:
                profiles.append(line)
            if "OBSERVED_3D_FACE" in edge["evidence_type"]:
                faces.append(line)
    empty = np.empty((0, 3))
    contour_cloud = np.vstack(clouds) if clouds else empty
    contour_colors = (
        np.vstack(colors).astype(np.uint8) if colors else empty.astype(np.uint8)
    )
    write_colored_ply(
        directory / "all_node_contours.ply", contour_cloud, contour_colors
    )
    write_colored_ply(
        directory / "node_structure_overlay.ply",
        np.vstack((points, contour_cloud)),
        np.vstack((np.tile(GRAY, (len(points), 1)), contour_colors)).astype(np.uint8),
    )
    for name, lines, color in (
        ("observed_profile_edges.ply", profiles, (45, 105, 235)),
        ("observed_3d_faces.ply", faces, (35, 215, 220)),
    ):
        cloud = np.vstack(lines) if lines else empty
        write_colored_ply(
            directory / name, cloud, np.tile(color, (len(cloud), 1)).astype(np.uint8)
        )


def _baseline_signature(result):
    return [
        (row["source_node"], row["bbox_xy"], row["topology_score"])
        for row in result["regions"]
    ]


def _mask_xy(node, cell_m):
    cells = np.argwhere(node.mask)
    return (cells[:, [1, 0]].astype(float) + 0.5) * cell_m


def _principal_axis(xy):
    centered = xy - xy.mean(axis=0)
    covariance = centered.T @ centered / max(len(xy), 1)
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    if axis[0] < 0:
        axis = -axis
    return axis


def _project_interval(xy, axis):
    projection = xy @ axis
    return float(projection.min()), float(projection.max())


def _interval_overlap(first, second):
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _interval_gap(first, second):
    return max(0.0, max(first[0], second[0]) - min(first[1], second[1]))


def _line_features(record, evidence_token=None):
    rows = []
    for edge in record["observed_edges"]:
        if evidence_token and evidence_token not in edge["evidence_type"]:
            continue
        a, b = np.asarray(edge["a_raw"][:2]), np.asarray(edge["b_raw"][:2])
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length:
            rows.append((direction / length, (a + b) / 2, length))
    return rows


def _line_compatibility(first, second):
    pairs = []
    for u, center_u, length_u in first:
        for v, center_v, length_v in second:
            angle = float(np.degrees(np.arccos(np.clip(abs(u @ v), -1, 1))))
            normal = np.array((-u[1], u[0]))
            offset = float(abs((center_v - center_u) @ normal))
            pairs.append((angle, offset, min(length_u, length_v)))
    if not pairs:
        return None, None
    best = min(pairs, key=lambda row: (row[0], row[1], -row[2]))
    parallel = [row for row in pairs if row[0] <= 15.0]
    return best[0], min((row[1] for row in parallel), default=None)


def _ancestor(first, second):
    cursor = first.parent
    while cursor is not None:
        if cursor is second:
            return True
        cursor = cursor.parent
    return False


def fragment_compatibility(nodes, records, config):
    """Diagnostic cross-level pairs; no mask union or completed bounding box."""
    by_id = {row["node_id"]: row for row in records}
    cell_m = config["geometry"]["coarse_voxel_m"]
    geometry = {node.node_id: _mask_xy(node, cell_m) for node in nodes}
    axes = {node_id: _principal_axis(xy) for node_id, xy in geometry.items()}
    pairs = []
    for index, first in enumerate(nodes):
        for second in nodes[index + 1 :]:
            if first.level_index == second.level_index:
                continue
            a, b = by_id[first.node_id], by_id[second.node_id]
            xy_a, xy_b = geometry[first.node_id], geometry[second.node_id]
            axis_a, axis_b = axes[first.node_id], axes[second.node_id]
            alignment = float(abs(axis_a @ axis_b))
            major_axis = axis_a
            minor_axis = np.array((-major_axis[1], major_axis[0]))
            major_first = _project_interval(xy_a, major_axis)
            major_second = _project_interval(xy_b, major_axis)
            minor_first = _project_interval(xy_a, minor_axis)
            minor_second = _project_interval(xy_b, minor_axis)
            major_overlap = _interval_overlap(major_first, major_second) / max(
                min(major_first[1] - major_first[0], major_second[1] - major_second[0]),
                1e-9,
            )
            minor_gap = _interval_gap(minor_first, minor_second)
            minor_complementary = _interval_overlap(minor_first, minor_second) == 0
            lines_a, lines_b = _line_features(a), _line_features(b)
            direction, line_offset = _line_compatibility(lines_a, lines_b)
            profile_direction, _ = _line_compatibility(
                _line_features(a, "OBSERVED_PROFILE_BREAK"),
                _line_features(b, "OBSERVED_PROFILE_BREAK"),
            )
            face_direction, _ = _line_compatibility(
                _line_features(a, "OBSERVED_3D_FACE"),
                _line_features(b, "OBSERVED_3D_FACE"),
            )
            ancestor = _ancestor(first, second) or _ancestor(second, first)
            overlap_ratio = float(
                np.count_nonzero(first.mask & second.mask)
                / max(
                    min(np.count_nonzero(first.mask), np.count_nonzero(second.mask)), 1
                )
            )
            conflict = bool(overlap_ratio > 0.5 and not ancestor)
            structurally_supported = (
                direction is not None
                and direction <= 15
                and line_offset is not None
                and line_offset <= config["roi"]["support_band_m"]
            )
            if conflict or major_overlap < 0.25 or alignment < 0.85:
                association = "INCOMPATIBLE"
            elif (
                minor_complementary
                and structurally_supported
                and minor_gap <= 2 * config["roi"]["support_search_m"]
                and major_overlap >= 0.5
            ):
                association = "COMPATIBLE"
            else:
                association = "AMBIGUOUS"
            pairs.append(
                dict(
                    node_ids=[first.node_id, second.node_id],
                    major_axis_alignment=alignment,
                    major_axis_overlap_ratio=float(major_overlap),
                    minor_axis_gap_m=float(minor_gap),
                    minor_axis_complementarity=bool(minor_complementary),
                    structural_line_direction_difference_deg=direction,
                    boundary_line_alignment_m=line_offset,
                    profile_evidence_compatibility=(
                        profile_direction is not None and profile_direction <= 15
                    ),
                    face_3d_compatibility=(
                        face_direction is not None and face_direction <= 15
                    ),
                    ancestor_relationship=ancestor,
                    cross_level_relationship=True,
                    spatial_conflict=conflict,
                    association_status=association,
                    boundary_recoverability=(
                        "ASSOCIATED_BUT_BOUNDARY_INCOMPLETE"
                        if association == "COMPATIBLE"
                        else "NOT_ASSOCIABLE" if association == "INCOMPATIBLE" else None
                    ),
                    boundary_recoverability_reason=(
                        "NO_CLOSED_RAW_3D_BOUNDARY_WAS_DEMONSTRATED_FOR_GROUP"
                        if association == "COMPATIBLE"
                        else (
                            "GEOMETRIC_CONFLICT"
                            if association == "INCOMPATIBLE"
                            else "FRAGMENT_ASSOCIATION_UNRESOLVED"
                        )
                    ),
                )
            )
    return dict(
        schema="ship_perception.v15r.fragment_compatibility.1",
        association_is_not_boundary_completion=True,
        pairs=pairs,
    )


def audit_scene(points, config, research, baseline_result):
    result, auxiliary, nodes = _hierarchy(points, config, research)
    if (
        _baseline_signature(result) != _baseline_signature(baseline_result)
        or result["region_count"] != baseline_result["region_count"]
    ):
        raise ValueError("BASELINE004_BEHAVIOR_CHANGED")
    grid = auxiliary["coarse"]
    fov, tree = FovMap(points, config), cKDTree(points[:, :2])
    records, contours = [], []
    selected = {row["source_node"] for row in result["regions"]}
    for node in nodes:
        record, contour = _node_audit(points, node, grid, fov, config, tree)
        record["baseline_selected"] = node.node_id in selected
        records.append(record)
        contours.append(contour)
    return result, records, contours, nodes


def run_batch(run_id, baseline_root, manifest_path, data_root, output_root):
    if not run_id or run_id in (".", "..") or any(c in run_id for c in "/\\:"):
        raise ValueError("INVALID_RUN_ID")
    destination = Path(output_root) / run_id
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("FORENSIC_RUN_ALREADY_EXISTS:" + str(destination))
    config, _ = resolve_config(DEFAULT_CONFIG)
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    research_sha = hashlib.sha256(
        json.dumps(research, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scans = {scan["scan_id"]: scan for scan in manifest["scans"]}
    summary = []
    for scene, scan_id in NORMAL6.items():
        scan = scans[scan_id]
        if scan["split_role"] != "DEVELOPMENT":
            raise ValueError("NON_DEVELOPMENT_INPUT:" + scan_id)
        pcd = Path(data_root) / scan["pcd_path"]
        input_sha = hashlib.sha256(pcd.read_bytes()).hexdigest()
        if input_sha != scan["pcd_sha256"]:
            raise ValueError("INPUT_SHA_MISMATCH:" + scene)
        frozen = json.loads(
            (Path(baseline_root) / scene / "hatch_regions.json").read_text(
                encoding="utf-8"
            )
        )
        if input_sha != frozen["input_sha256"]:
            raise ValueError("BASELINE_INPUT_MISMATCH:" + scene)
        if (
            config_hash(config) != frozen["v15_config_hash"]
            or research_sha != frozen["research_config_sha256"]
        ):
            raise ValueError("BASELINE_CONFIG_MISMATCH:" + scene)
        points, _ = decode(pcd)
        result, records, contours, nodes = audit_scene(points, config, research, frozen)
        directory = destination / scene
        directory.mkdir(parents=True, exist_ok=False)
        _write_overlays(directory, points, records, contours, config)
        _atomic_json(
            directory / "node_structure_audit.json",
            dict(
                schema="ship_perception.v15r.height_structure_audit.1",
                scene_id=scene,
                scan_id=scan_id,
                software_git_sha=_git_sha(),
                input_sha256=input_sha,
                baseline_004_software_git_sha=frozen["software_git_sha"],
                v15_config_hash=frozen["v15_config_hash"],
                research_config_sha256=research_sha,
                baseline_004_behavior_unchanged=True,
                selected_region_signature=_baseline_signature(result),
                node_count=len(nodes),
                nodes=records,
                edge_semantics="RAW_3D_OBSERVED_ONLY; DECK_IS_LOCAL_HORIZONTAL_SUPPORT",
            ),
        )
        _atomic_json(
            directory / "fragment_compatibility.json",
            fragment_compatibility(nodes, records, config),
        )
        summary.extend(
            {
                field: (scene if field == "scene_id" else record.get(field))
                for field in FIELDS
            }
            for record in records
        )
        print(
            "%s: %d nodes, %d observed edges"
            % (scene, len(nodes), sum(row["observed_edge_count"] for row in records)),
            flush=True,
        )
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "normal6_structure_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO / "Ship-Unloader-Work/r1_heightmap/baseline_004_heightmap",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=REPO / "ship_perception/datasets/v15_dataset_manifest.json",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO / "Ship-Unloader-Data/legacy/hold_detector",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "Ship-Unloader-Work/r1_height_structure_forensics",
    )
    args = parser.parse_args()
    run_batch(
        args.run_id,
        args.baseline_root,
        args.dataset_manifest,
        args.data_root,
        args.output_root,
    )


if __name__ == "__main__":
    main()
