"""S4-R1: unlock the existing _profile chain via PlaneFamily reference.

A segment flagged SEGMENT_DECK_AMBIGUOUS by ``estimate_segment_deck`` may still
have a qualified candidate that belongs to a reference-capable plane family
(BROAD_PERIMETER_SUPPORT with a strict cross-segment core). When exactly one
such family explains the segment, we pass the segment's OWN candidate plane
(never the family's global plane) to the existing ``_profile``, keeping the
chain observed-first. Nothing here modifies ``estimate_segment_deck``,
``_profile``, or ``hypothesis_solver``.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ship_perception.tools.v15_config_identity import config_hash
from ship_perception.tools.v15_pcd import decode

from .batch import NORMAL6
from .boundary_refinement import _face_normal_supported, _fit_line, _profile, _raw3
from .family_consensus import (NodeGeometry, assign_family_roles,
                               deterministic_families, normalize_normal)
from .height_structure_forensics import _baseline_signature, _hierarchy, _node_seeds
from .height_structure_forensics_p1 import _offset_from_height
from .heightmap_batch import RESEARCH_CONFIG
from .opening_seed import FovMap, perimeter_segments
from .perimeter_deck import estimate_segment_deck
from .run import DEFAULT_CONFIG, REPO, _atomic_json, _git_sha, resolve_config

REFERENCE_ROLE = "BROAD_PERIMETER_SUPPORT"


def collect_segment_candidates(points, segment, deck, forensic, config, node_id, centroid):
    """Reconstruct qualified candidate planes from estimate_segment_deck output."""
    anchor = (np.asarray(segment.rough_start, dtype=float) +
              np.asarray(segment.rough_end, dtype=float)) / 2
    normals = deck["candidate_plane_normals"]
    heights = deck["candidate_plane_heights"]
    widths = deck["candidate_outward_widths"]
    coverages = deck["candidate_along_coverages"]
    raw_id_lists = forensic["candidate_raw_ids"]
    min_width = config["roi"]["deck_patch_width_m"]
    min_span = config["roi"]["min_deck_span_m"]
    min_fill = config["roi"]["deck_patch_fill_min"]
    tangent = np.asarray(segment.tangent, dtype=float)
    start = np.asarray(segment.rough_start, dtype=float)
    indexes = []
    for index in range(len(normals)):
        normal = normalize_normal(np.asarray(normals[index], dtype=float))
        offset = _offset_from_height(normal, heights[index], anchor)
        ids = np.asarray(raw_id_lists[index], dtype=np.int64)
        raw = points[ids].astype(float)
        along = (raw[:, :2] - start) @ tangent
        span = float(np.max(along) - np.min(along)) if len(ids) else 0.0
        qualified = (widths[index] >= min_width and span >= min_span and
                     coverages[index] >= min_fill)
        if not qualified:
            continue
        indexes.append(dict(
            node_id=node_id, segment_id=segment.segment_id, candidate_index=index,
            normal=normal, offset=float(offset),
            d_centered=float(offset + float(normal @ centroid)),
            support_count=int(len(ids)), outward_width=float(widths[index]),
            along_span=span, along_coverage=float(coverages[index]),
            raw_ids=ids,
        ))
    return indexes


def run_profile_for_segment(points, tree, segment, plane, config):
    """Run the existing _profile/face/line chain for one segment with one plane."""
    boundary = config["boundary"]
    spacing = max(boundary["profile_step_m"],
                  min(boundary["max_support_gap_m"] * 0.8,
                      config["geometry"]["coarse_voxel_m"] / 2))
    start = np.asarray(segment.rough_start, dtype=float)
    tangent = np.asarray(segment.tangent, dtype=float)
    inward = -np.asarray(segment.outward_normal, dtype=float)
    positions = np.arange(spacing / 2, segment.length_m, spacing)
    rows = [_profile(points, tree, start + tangent * position, tangent, inward, plane, config,
                     "%s-p%d" % (segment.segment_id, index))
            for index, position in enumerate(positions)]
    breaks = np.asarray([row["break_position"] for row in rows if row["valid"]],
                        dtype=float).reshape(-1, 2)
    faces = np.asarray([row["face_position"] for row in rows
                        if row["face_position"] is not None], dtype=float).reshape(-1, 2)
    break_line = _fit_line(breaks, config) if len(breaks) else None
    face_line = _fit_line(faces, config) if len(faces) else None
    if face_line is not None and not _face_normal_supported(points, face_line, plane, config):
        face_line = None
    chosen = None
    evidence = None
    if break_line is not None and face_line is not None:
        separation = float(np.linalg.norm(
            (break_line["a_xy"] + break_line["b_xy"] -
             face_line["a_xy"] - face_line["b_xy"]) / 2))
        if separation <= max(boundary["line_inlier_m"], config["roi"]["support_band_m"]):
            chosen = _fit_line(np.vstack((breaks, faces)), config) or break_line
            evidence = "OBSERVED_PROFILE_BREAK+OBSERVED_3D_FACE"
        else:
            chosen, evidence = break_line, "OBSERVED_PROFILE_BREAK"
    elif break_line is not None:
        chosen, evidence = break_line, "OBSERVED_PROFILE_BREAK"
    elif face_line is not None:
        chosen, evidence = face_line, "OBSERVED_3D_FACE"
    edge = None
    if chosen is not None:
        edge = dict(edge_id=segment.segment_id, segment_id=segment.segment_id,
                    a_raw=_raw3(chosen["a_xy"], plane), b_raw=_raw3(chosen["b_xy"], plane),
                    evidence_type=evidence, support_count=chosen["support_count"],
                    observed_support_length=chosen["support_length"],
                    fit_residual_p50_m=chosen["residual_p50"],
                    fit_residual_p95_m=chosen["residual_p95"],
                    visibility="VISIBLE", uncertainty_m=chosen["normal_uncertainty"],
                    coverage=min(1.0, chosen["support_length"] / segment.length_m),
                    reason="RAW_3D_OBSERVED_SUPPORT")
    return edge, rows


def is_reference_capable(family, config):
    return (family["role_hypothesis"] == REFERENCE_ROLE and
            family["unique_segment_votes"] >= 2 and
            family["spatial_connectedness"] >= 0.5 and
            family["role_hypothesis"] != "ROLE_AMBIGUOUS_OVERFLOW")


def _family_rank_key(family, min_distance):
    return (family["unique_segment_votes"], family["unique_node_votes"],
            family["support_boundary_inner_fraction"] + family["support_boundary_outer_fraction"],
            family["support_point_count"], -min_distance)


def resolve_segment_reference(segment_candidate_indexes, family_of, families_by_id,
                              candidates, config):
    reference_candidates = []
    for candidate_index in segment_candidate_indexes:
        family_id = family_of[candidate_index]
        if family_id is None or family_id not in families_by_id:
            continue
        family = families_by_id[family_id]
        if is_reference_capable(family, config):
            reference_candidates.append((candidate_index, family))
    if not reference_candidates:
        return dict(status="NO_LOCAL_REFERENCE_OBSERVATION", family_id=None,
                    candidate_index=None, role=None)
    distinct = {}
    for candidate_index, family in reference_candidates:
        rep = np.asarray(family["representative_normal"], dtype=float)
        distance = abs(candidates[candidate_index]["d_centered"] - family["representative_offset"])
        distinct.setdefault(family["family_id"], []).append((candidate_index, family, distance))
    if len(distinct) == 1:
        family_id = next(iter(distinct))
        candidate_index, family, _ = min(distinct[family_id], key=lambda row: row[2])
        return dict(status="FAMILY_REFERENCE_RESOLVED", family_id=family_id,
                    candidate_index=candidate_index, role=family["role_hypothesis"])
    ranked = []
    for family_id, entries in distinct.items():
        family = entries[0][1]
        min_distance = min(entry[2] for entry in entries)
        ranked.append((_family_rank_key(family, min_distance), family_id, family))
    ranked.sort(key=lambda row: row[0], reverse=True)
    if len(ranked) >= 2 and ranked[0][0] == ranked[1][0]:
        return dict(status="FAMILY_REFERENCE_AMBIGUOUS", family_id=None,
                    candidate_index=None, role=None)
    family_id = ranked[0][1]
    candidate_index, family, _ = min(distinct[family_id], key=lambda row: row[2])
    return dict(status="FAMILY_REFERENCE_RESOLVED", family_id=family_id,
                candidate_index=candidate_index, role=family["role_hypothesis"])


def analyze_scene_s4r1(points, config, research, baseline_result):
    result, auxiliary, nodes = _hierarchy(points, config, research)
    if (_baseline_signature(result) != _baseline_signature(baseline_result)
            or result["region_count"] != baseline_result["region_count"]):
        raise ValueError("BASELINE004_BEHAVIOR_CHANGED")
    grid = auxiliary["coarse"]
    fov = FovMap(points, config)
    tree = cKDTree(points[:, :2])
    centroid = points.mean(axis=0)
    node_geometries = {node.node_id: NodeGeometry(node, grid) for node in nodes}

    candidates = []
    segment_entries = []
    for node in nodes:
        for seed in _node_seeds(node, grid, fov):
            for segment in perimeter_segments(seed, fov, config):
                deck, plane, forensic = estimate_segment_deck(points, segment, config, tree)
                candidate_dicts = collect_segment_candidates(
                    points, segment, deck, forensic, config, node.node_id, centroid)
                start = len(candidates)
                candidates.extend(candidate_dicts)
                segment_entries.append(dict(
                    node_id=node.node_id, seed_id=seed.seed_id, segment=segment,
                    deck=deck, plane=plane, candidate_indexes=list(range(start, len(candidates))),
                ))

    strict_angle = config["frame"]["normal_refine_deg"]
    strict_sep = config["geometry"]["plane_inlier_m"]
    families, family_of = deterministic_families(candidates, strict_angle, strict_sep,
                                                 2 * strict_angle, 2 * strict_sep)
    family_records = assign_family_roles(families, candidates, points, node_geometries, config)
    families_by_id = {record["family_id"]: record for record in family_records}
    candidate_family_id = [("f%03d" % family_of[index] if family_of[index] is not None else None)
                           for index in range(len(candidates))]

    segment_resolutions, edges, narrow_strips = [], [], []
    for entry in segment_entries:
        segment = entry["segment"]
        deck = entry["deck"]
        resolution = resolve_segment_reference(entry["candidate_indexes"], candidate_family_id,
                                               families_by_id, candidates, config)
        profile_before = deck["status"] == "RESOLVED"
        profile_after = profile_before or resolution["status"] == "FAMILY_REFERENCE_RESOLVED"
        edge = None
        if profile_before:
            edge, rows = run_profile_for_segment(points, tree, segment, entry["plane"], config)
        elif resolution["status"] == "FAMILY_REFERENCE_RESOLVED":
            candidate = candidates[resolution["candidate_index"]]
            plane = (candidate["normal"], candidate["offset"])
            edge, rows = run_profile_for_segment(points, tree, segment, plane, config)
        else:
            rows = []
        if edge is not None:
            edge["via_family_reference"] = bool(
                not profile_before and resolution["status"] == "FAMILY_REFERENCE_RESOLVED")
            edges.append(edge)
        segment_resolutions.append(dict(
            node_id=entry["node_id"], segment_id=segment.segment_id,
            original_deck_status=deck["status"], candidate_count=len(entry["candidate_indexes"]),
            reference_family_id=resolution["family_id"],
            reference_resolution_status=(resolution["status"] if deck["status"] == "SEGMENT_DECK_AMBIGUOUS"
                                         else "NOT_APPLICABLE"),
            reference_candidate_index=resolution["candidate_index"],
            reference_role=resolution["role"],
            profile_executed_before=bool(profile_before),
            profile_executed_after=bool(profile_after),
            profile_break_count=int(sum(1 for row in rows if row["valid"])),
            face_count=int(sum(1 for row in rows if row["face_position"] is not None)),
            line_fit_result=(edge["evidence_type"] if edge is not None else None),
        ))
    narrow_strips = [dict(family_id=record["family_id"],
                          representative_normal=record["representative_normal"],
                          representative_offset=record["representative_offset"],
                          member_segment_ids=record["segment_ids"],
                          along_span_m=record["total_along_span"],
                          transverse_width_m=record["median_transverse_width"],
                          spatial_connectedness=record["spatial_connectedness"],
                          boundary_inner_fraction=record["support_boundary_inner_fraction"],
                          boundary_outer_fraction=record["support_boundary_outer_fraction"],
                          semantics="STRUCTURE_TOP_CANDIDATE")
                     for record in family_records
                     if record["role_hypothesis"] == "NARROW_BOUNDARY_STRIP"]
    node_summary = {}
    for record in segment_resolutions:
        summary = node_summary.setdefault(record["node_id"], dict(
            segment_count=0, family_reference_resolved_count=0, profile_executed_after_count=0,
            observed_structural_edge_count=0, narrow_boundary_strip_count=0,
            reference_family_ids=[], cross_node_consensus_count=0))
        summary["segment_count"] += 1
        if record["reference_resolution_status"] == "FAMILY_REFERENCE_RESOLVED":
            summary["family_reference_resolved_count"] += 1
            if record["reference_family_id"] not in summary["reference_family_ids"]:
                summary["reference_family_ids"].append(record["reference_family_id"])
        if record["profile_executed_after"]:
            summary["profile_executed_after_count"] += 1
        if record["line_fit_result"] is not None:
            summary["observed_structural_edge_count"] += 1
    for record in family_records:
        if record["unique_node_votes"] >= 2:
            for node_id in record["node_ids"]:
                if node_id in node_summary:
                    node_summary[node_id]["cross_node_consensus_count"] += 1
    return dict(
        scene=result, candidates=candidates, families=family_records,
        candidate_family_id=candidate_family_id,
        segment_resolutions=segment_resolutions, edges=edges, narrow_strips=narrow_strips,
        node_summary=node_summary,
    )


def run_batch_s4r1(run_id, baseline_root, manifest_path, data_root, output_root):
    if not run_id or run_id in (".", "..") or any(c in run_id for c in "/\\:"):
        raise ValueError("INVALID_RUN_ID")
    destination = Path(output_root) / run_id
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("S4R1_RUN_ALREADY_EXISTS:" + str(destination))
    config, _ = resolve_config(DEFAULT_CONFIG)
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    research_sha = hashlib.sha256(
        json.dumps(research, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scans = {scan["scan_id"]: scan for scan in manifest["scans"]}
    summary_rows = []
    for scene, scan_id in NORMAL6.items():
        scan = scans[scan_id]
        if scan["split_role"] != "DEVELOPMENT":
            raise ValueError("NON_DEVELOPMENT_INPUT:" + scan_id)
        pcd = Path(data_root) / scan["pcd_path"]
        input_sha = hashlib.sha256(pcd.read_bytes()).hexdigest()
        if input_sha != scan["pcd_sha256"]:
            raise ValueError("INPUT_SHA_MISMATCH:" + scene)
        frozen = json.loads(
            (Path(baseline_root) / scene / "hatch_regions.json").read_text(encoding="utf-8"))
        if input_sha != frozen["input_sha256"]:
            raise ValueError("BASELINE_INPUT_MISMATCH:" + scene)
        if (config_hash(config) != frozen["v15_config_hash"]
                or research_sha != frozen["research_config_sha256"]):
            raise ValueError("BASELINE_CONFIG_MISMATCH:" + scene)
        points, _ = decode(pcd)
        analysis = analyze_scene_s4r1(points, config, research, frozen)
        directory = destination / scene
        directory.mkdir(parents=True, exist_ok=False)
        _atomic_json(directory / "plane_families.json", dict(
            schema="ship_perception.v15r.s4r1_plane_families.1", scene_id=scene,
            strict_angle_deg=config["frame"]["normal_refine_deg"],
            strict_sep_m=config["geometry"]["plane_inlier_m"],
            families=analysis["families"],
            semantics="DETERMINISTIC_STRICT_CORE+RELAXED_ATTACHMENT; ROLE_IS_HYPOTHESIS",
        ))
        _atomic_json(directory / "segment_reference_resolution.json", dict(
            schema="ship_perception.v15r.s4r1_reference_resolution.1", scene_id=scene,
            segments=analysis["segment_resolutions"],
        ))
        _atomic_json(directory / "structural_edges.json", dict(
            schema="ship_perception.v15r.s4r1_structural_edges.1", scene_id=scene,
            edges=analysis["edges"],
        ))
        _atomic_json(directory / "narrow_boundary_strips.json", dict(
            schema="ship_perception.v15r.s4r1_narrow_boundary_strips.1", scene_id=scene,
            narrow_boundary_strips=analysis["narrow_strips"],
            semantics="STRUCTURE_TOP_CANDIDATE_NOT_CONFIRMED_COAMING",
        ))
        _atomic_json(directory / "node_structural_evidence.json", dict(
            schema="ship_perception.v15r.s4r1_node_evidence.1", scene_id=scene,
            nodes=analysis["node_summary"],
        ))
        for record in analysis["segment_resolutions"]:
            summary_rows.append(dict(scene_id=scene, **record))
        print("%s: %d segments, %d resolved-by-family, %d edges"
              % (scene, len(analysis["segment_resolutions"]),
                 sum(1 for record in analysis["segment_resolutions"]
                     if record["reference_resolution_status"] == "FAMILY_REFERENCE_RESOLVED"),
                 len(analysis["edges"])), flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene_id", "node_id", "segment_id", "original_deck_status", "candidate_count",
                  "reference_family_id", "reference_resolution_status", "reference_candidate_index",
                  "reference_role", "profile_executed_before", "profile_executed_after",
                  "profile_break_count", "face_count", "line_fit_result"]
    _write_csv(destination / "normal6_s4r1_summary.csv", summary_rows, fieldnames)
    _atomic_json(destination / "batch_summary.json", dict(
        schema="ship_perception.v15r.s4r1_batch.1", run_id=run_id,
        software_git_sha=_git_sha(), scenes={scene: dict() for scene in NORMAL6}))
    return summary_rows


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-root", type=Path,
                        default=REPO / "Ship-Unloader-Work/r1_heightmap/baseline_004_heightmap")
    parser.add_argument("--dataset-manifest", type=Path,
                        default=REPO / "ship_perception/datasets/v15_dataset_manifest.json")
    parser.add_argument("--data-root", type=Path,
                        default=REPO / "Ship-Unloader-Data/legacy/hold_detector")
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "Ship-Unloader-Work/r1_family_reference")
    args = parser.parse_args()
    run_batch_s4r1(args.run_id, args.baseline_root, args.dataset_manifest,
                   args.data_root, args.output_root)


if __name__ == "__main__":
    main()
