"""Research-only P1 forensics: why segment decks are ambiguous and what the
existing profile/face chain actually observes near the real hatch boundary.

This module is never imported by the detector or the structural audit path.
It reuses the detector primitives (``estimate_segment_deck``, ``_profile``,
``_fit_line``, ``_face_normal_supported``) as diagnostic probes on the raw XYZ
without changing their behavior. No annotation is read here; the weak-label
Oracle probe lives in ``height_structure_evaluator_p1``.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ship_perception.tools.v15_config_identity import config_hash
from ship_perception.tools.v15_pcd import decode

from .batch import NORMAL6
from .boundary_refinement import _face_normal_supported, _fit_line
from .height_structure_forensics import _baseline_signature, _hierarchy, _node_seeds
from .heightmap_batch import RESEARCH_CONFIG
from .heightmap_topology import detect_regions
from .local_deck import _plane_angle
from .opening_seed import FovMap, _outer_contour, _simplify_closed, perimeter_segments
from .perimeter_boundary import refine_perimeter_boundaries
from .perimeter_deck import estimate_segment_deck
from .run import DEFAULT_CONFIG, REPO, _atomic_json, _git_sha, resolve_config
from .visualization import write_colored_ply

# Per-candidate palette used for CloudCompare review PLYs.
PALETTE = (
    (36, 190, 220),
    (255, 188, 53),
    (123, 214, 85),
    (177, 116, 238),
    (255, 110, 192),
    (80, 140, 250),
)

TURNING_BINS = [(-180, -135), (-135, -90), (-90, -45), (-45, 0),
                (0, 45), (45, 90), (90, 135), (135, 180)]


def _offset_from_height(normal, height_m, anchor_xy):
    """Reconstruct the plane offset from the anchor height, inverse of
    ``local_deck._anchor_height``."""
    return float(-normal[0] * anchor_xy[0] - normal[1] * anchor_xy[1] -
                 normal[2] * height_m)


def _vertical_connector(points, strip_ids, low_h, high_h, config):
    """Raw returns inside the support strip that bridge the vertical gap."""
    tol = config["geometry"]["plane_inlier_m"]
    if high_h - low_h <= 2 * tol or not len(strip_ids):
        return 0, 0.0
    z = points[strip_ids, 2]
    bridge = (z > low_h + tol) & (z < high_h - tol)
    if not np.any(bridge):
        return 0, 0.0
    zs = z[bridge]
    return int(np.count_nonzero(bridge)), float(np.ptp(zs))


def _classify_pair(features, config):
    """Forensic classification of one qualified plane pair. Never named as a
    product decision; only a hypothesis label for the report."""
    tol = config["geometry"]["plane_inlier_m"]
    if not features["narrow_is_elevated"]:
        return "UNKNOWN_PAIR", "NARROWER_PLANE_NOT_ELEVATED"
    if features["height_difference_m"] < 3 * tol:
        return "UNKNOWN_PAIR", "HEIGHT_DIFFERENCE_BELOW_AMBIGUITY_GATE"
    if features["height_difference_m"] > config["roi"]["max_height_above_deck_m"]:
        return "UNKNOWN_PAIR", "HEIGHT_DIFFERENCE_EXCEEDS_COAMING_RANGE"
    if features["relative_width"] >= 1.0:
        return "UNKNOWN_PAIR", "ELEVATED_PLANE_NOT_NARROWER_THAN_BROAD_PLANE"
    if features["relative_boundary_distance"] >= 1.0:
        return "UNKNOWN_PAIR", "ELEVATED_PLANE_NOT_CLOSER_TO_BOUNDARY"
    if (features["vertical_connector_support_count"] < config["geometry"]["normal_min_points"]
            or features["vertical_connector_span_m"] < config["boundary"]["face_vertical_span_min_m"]):
        return "UNKNOWN_PAIR", "NO_VERTICAL_CONNECTOR_RETURNS_BETWEEN_PLANES"
    return ("DECK_COAMING_LIKE_PAIR",
            "BROAD_LOW_PLANE+NARROW_ELEVATED_ADJACENT_PLANE+VERTICAL_CONNECTOR")


def segment_ambiguity_detail(points, segment, deck, forensic, config, node_id=None):
    """Full plane-candidate and pair detail for one SEGMENT_DECK_AMBIGUOUS."""
    anchor = (np.asarray(segment.rough_start, dtype=float) +
              np.asarray(segment.rough_end, dtype=float)) / 2
    start = np.asarray(segment.rough_start, dtype=float)
    tangent = np.asarray(segment.tangent, dtype=float)
    outward = np.asarray(segment.outward_normal, dtype=float)
    normals = [np.asarray(row, dtype=float) for row in deck["candidate_plane_normals"]]
    heights = deck["candidate_plane_heights"]
    counts = deck["candidate_support_counts"]
    widths = deck["candidate_outward_widths"]
    coverages = deck["candidate_along_coverages"]
    candidate_ids = forensic["candidate_raw_ids"]
    strip_ids = forensic["strip_raw_ids"]
    min_width = config["roi"]["deck_patch_width_m"]
    min_span = config["roi"]["min_deck_span_m"]
    min_fill = config["roi"]["deck_patch_fill_min"]

    scalars, raw_ids_by_candidate = [], []
    for index in range(len(normals)):
        normal = normals[index]
        height = heights[index]
        offset = _offset_from_height(normal, height, anchor)
        ids = candidate_ids[index]
        raw = points[ids].astype(float)
        signed = raw @ normal + offset
        outward_distance = (raw[:, :2] - start) @ outward
        along_distance = (raw[:, :2] - start) @ tangent
        span = float(np.max(along_distance) - np.min(along_distance)) if len(ids) else 0.0
        qualified = bool(widths[index] >= min_width and span >= min_span and
                         coverages[index] >= min_fill)
        scalars.append(dict(
            candidate_index=index,
            normal=[float(value) for value in normal],
            offset=float(offset),
            anchor_height_m=float(height),
            support_count=int(counts[index]),
            outward_width_m=float(widths[index]),
            along_coverage=float(coverages[index]),
            along_span_m=span,
            residual_p50_m=float(np.percentile(np.abs(signed), 50)),
            residual_p95_m=float(np.percentile(np.abs(signed), 95)),
            normal_distance_min_m=float(np.percentile(signed, 0)),
            normal_distance_p50_m=float(np.percentile(signed, 50)),
            normal_distance_p95_m=float(np.percentile(signed, 95)),
            median_distance_to_segment_m=float(np.median(outward_distance)) if len(ids) else None,
            qualified=qualified,
            raw_point_count=int(len(ids)),
        ))
        raw_ids_by_candidate.append(ids)

    qualified_indexes = [row["candidate_index"] for row in scalars if row["qualified"]]
    pairs = []
    for first in range(len(qualified_indexes)):
        for second in range(first + 1, len(qualified_indexes)):
            i, j = qualified_indexes[first], qualified_indexes[second]
            left, right = scalars[i], scalars[j]
            if left["outward_width_m"] >= right["outward_width_m"]:
                broad, narrow = left, right
                broad_ids, narrow_ids = raw_ids_by_candidate[i], raw_ids_by_candidate[j]
            else:
                broad, narrow = right, left
                broad_ids, narrow_ids = raw_ids_by_candidate[j], raw_ids_by_candidate[i]
            height_difference = abs(narrow["anchor_height_m"] - broad["anchor_height_m"])
            low_h = min(narrow["anchor_height_m"], broad["anchor_height_m"])
            high_h = max(narrow["anchor_height_m"], broad["anchor_height_m"])
            vertical_count, vertical_span = _vertical_connector(
                points, strip_ids, low_h, high_h, config)
            broad_set = set(int(value) for value in broad_ids)
            narrow_set = set(int(value) for value in narrow_ids)
            overlap = len(broad_set & narrow_set) / max(min(len(broad_set), len(narrow_set)), 1)
            features = dict(
                height_difference_m=float(height_difference),
                narrow_is_elevated=bool(narrow["anchor_height_m"] > broad["anchor_height_m"]),
                relative_width=float(narrow["outward_width_m"] /
                                     max(broad["outward_width_m"], 1e-9)),
                relative_boundary_distance=(
                    float(narrow["median_distance_to_segment_m"] /
                          max(broad["median_distance_to_segment_m"], 1e-9))
                    if broad["median_distance_to_segment_m"] is not None else None),
                vertical_connector_support_count=vertical_count,
                vertical_connector_span_m=vertical_span,
            )
            classification, classification_reason = _classify_pair(features, config)
            pairs.append(dict(
                pair=[i, j],
                broad_candidate_index=broad["candidate_index"],
                narrow_candidate_index=narrow["candidate_index"],
                broad_height_m=broad["anchor_height_m"],
                narrow_height_m=narrow["anchor_height_m"],
                height_difference_m=float(height_difference),
                normal_angle_difference_deg=float(
                    _plane_angle((np.asarray(left["normal"]), left["offset"]),
                                 (np.asarray(right["normal"]), right["offset"]))),
                broad_width_m=broad["outward_width_m"],
                narrow_width_m=narrow["outward_width_m"],
                relative_width=features["relative_width"],
                broad_support_count=broad["support_count"],
                narrow_support_count=narrow["support_count"],
                relative_support_count=float(narrow["support_count"] /
                                             max(broad["support_count"], 1)),
                broad_boundary_distance_m=broad["median_distance_to_segment_m"],
                narrow_boundary_distance_m=narrow["median_distance_to_segment_m"],
                relative_boundary_distance=features["relative_boundary_distance"],
                raw_support_overlap=float(overlap),
                vertical_connector_support_count=vertical_count,
                vertical_connector_span_m=vertical_span,
                classification=classification,
                classification_reason=classification_reason,
            ))

    return dict(
        node_id=node_id,
        seed_id=segment.opening_seed_id,
        segment_id=segment.segment_id,
        segment_length_m=segment.length_m,
        fov_status=segment.fov_status,
        deck_status=deck["status"],
        deck_reason=deck["reason"],
        support_strip_point_count=int(len(strip_ids)),
        plane_candidate_count=len(scalars),
        qualified_candidate_count=len(qualified_indexes),
        plane_candidates=scalars,
        plane_pairs=pairs,
    ), raw_ids_by_candidate


def segment_lineage(points, segment, deck, plane, profile_rows, config, node_id=None):
    """Per-segment failure lineage: DECK -> profile -> face -> line fit."""
    record = dict(
        node_id=node_id,
        seed_id=segment.opening_seed_id,
        segment_id=segment.segment_id,
        segment_length_m=segment.length_m,
        fov_status=segment.fov_status,
        deck_status=deck["status"],
        deck_reason=deck["reason"],
        deck_plane_candidate_count=deck["plane_candidate_count"],
        profile_executed=False,
        profile_failure_category="PROFILE_NOT_EXECUTED",
        profile_sample_count=0,
        deck_support_count=0,
        lower_support_count=0,
        face_option_count=0,
        break_candidate_count=0,
        profile_reason=None,
        line_fit_result=None,
        face_normal_validation_result="NOT_RUN",
    )
    if deck["status"] != "RESOLVED" or plane is None:
        record["profile_not_executed_reason"] = "DECK_NOT_RESOLVED"
        return record
    record["profile_executed"] = True
    record["profile_sample_count"] = int(sum(row["sample_count"] for row in profile_rows))
    record["deck_support_count"] = int(sum(row["deck_support_count"] for row in profile_rows))
    record["lower_support_count"] = int(sum(row["lower_support_count"] for row in profile_rows))
    record["face_option_count"] = int(sum(1 for row in profile_rows if row["face_position"] is not None))
    record["break_candidate_count"] = int(sum(1 for row in profile_rows if row["valid"]))
    reasons = [row["reason"] for row in profile_rows if row["reason"]]
    record["profile_reason"] = Counter(reasons).most_common(1)[0][0] if reasons else None

    breaks = np.asarray([row["break_position"] for row in profile_rows if row["valid"]],
                        dtype=float).reshape(-1, 2)
    faces = np.asarray([row["face_position"] for row in profile_rows
                        if row["face_position"] is not None], dtype=float).reshape(-1, 2)
    break_line = _fit_line(breaks, config) if len(breaks) else None
    face_line = _fit_line(faces, config) if len(faces) else None
    if face_line is not None:
        face_valid = _face_normal_supported(points, face_line, plane, config)
        record["face_normal_validation_result"] = "PASS" if face_valid else "FAIL"
        if not face_valid:
            face_line = None
    else:
        record["face_normal_validation_result"] = "NOT_RUN"
    if break_line is not None and face_line is not None:
        separation = float(np.linalg.norm(
            (break_line["a_xy"] + break_line["b_xy"] -
             face_line["a_xy"] - face_line["b_xy"]) / 2))
        record["line_fit_result"] = (
            "PROFILE_BREAK+OBSERVED_3D_FACE"
            if separation <= max(config["boundary"]["line_inlier_m"],
                                 config["roi"]["support_band_m"])
            else "OBSERVED_PROFILE_BREAK")
    elif break_line is not None:
        record["line_fit_result"] = "OBSERVED_PROFILE_BREAK"
    elif face_line is not None:
        record["line_fit_result"] = "OBSERVED_3D_FACE"
    if record["line_fit_result"] is None:
        record["profile_failure_category"] = "PROFILE_EXECUTED_BUT_FAILED"
    else:
        record["profile_failure_category"] = "PROFILE_EXECUTED_OK"
    return record


def _turning_angles(vertices):
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 2)
    number = len(vertices)
    if number < 3:
        return []
    angles = []
    for index in range(number):
        previous = vertices[(index - 1) % number] - vertices[index]
        following = vertices[(index + 1) % number] - vertices[index]
        length_prev, length_next = np.linalg.norm(previous), np.linalg.norm(following)
        if length_prev < 1e-9 or length_next < 1e-9:
            continue
        cross = previous[0] * following[1] - previous[1] * following[0]
        dot = previous[0] * following[0] + previous[1] * following[1]
        angles.append(math.degrees(math.atan2(cross, dot)))
    return angles


def _turning_histogram(angles):
    histogram = [0] * len(TURNING_BINS)
    for angle in angles:
        for index, (low, high) in enumerate(TURNING_BINS):
            if low <= angle < high:
                histogram[index] += 1
                break
        else:
            histogram[-1] += 1
    return histogram


def contour_audit(node, seeds, config, fov):
    """Lightweight per-node contour geometry (raw vs simplified perimeter)."""
    raw_counts, simplified_counts, segment_lengths, segment_orientations = [], [], [], []
    turning_angles = []
    segment_count = 0
    for seed in seeds:
        raw_counts.append(len(_outer_contour(seed.component)))
        simplified = _simplify_closed(seed.contour_xy,
                                      config["geometry"]["coarse_voxel_m"],
                                      config["roi"]["min_deck_span_m"])
        simplified_counts.append(len(simplified))
        turning_angles.extend(_turning_angles(simplified))
        for segment in perimeter_segments(seed, fov, config):
            segment_count += 1
            segment_lengths.append(segment.length_m)
            segment_orientations.append(
                float(math.degrees(math.atan2(segment.tangent[1], segment.tangent[0]))))
    return dict(
        node_id=node.node_id,
        level_index=node.level_index,
        level_m=node.level_m,
        seed_count=len(seeds),
        raw_contour_point_count=int(sum(raw_counts)),
        per_seed_raw_contour_point_counts=raw_counts,
        simplified_vertex_count=int(sum(simplified_counts)),
        per_seed_simplified_vertex_counts=simplified_counts,
        perimeter_segment_count=segment_count,
        segment_lengths_m=[float(value) for value in segment_lengths],
        segment_length_min_m=float(min(segment_lengths)) if segment_lengths else None,
        segment_length_median_m=float(np.median(segment_lengths)) if segment_lengths else None,
        segment_length_max_m=float(max(segment_lengths)) if segment_lengths else None,
        segment_orientation_deg=[float(value) for value in segment_orientations],
        turning_angles_deg=[float(value) for value in turning_angles],
        turning_angle_histogram=_turning_histogram(turning_angles),
        turning_angle_histogram_bins=[list(row) for row in TURNING_BINS],
    )


def _node_audit_p1(points, node, grid, fov, config, tree, baseline_selected):
    seeds = _node_seeds(node, grid, fov)
    ambiguity_records, lineage_records, ownership = [], [], {}
    contour_record = contour_audit(node, seeds, config, fov)
    contour_record["baseline_selected"] = baseline_selected
    for seed in seeds:
        segments = perimeter_segments(seed, fov, config)
        segment_decks = {
            segment.segment_id: estimate_segment_deck(points, segment, config, tree)
            for segment in segments
        }
        refined = refine_perimeter_boundaries(points, seed, segments, segment_decks, config, tree)
        profiles_by_segment = defaultdict(list)
        for row in refined["profiles"]:
            profiles_by_segment[row["profile_id"].rsplit("-p", 1)[0]].append(row)
        for segment in segments:
            deck, plane, forensic = segment_decks[segment.segment_id]
            lineage = segment_lineage(
                points, segment, deck, plane,
                profiles_by_segment.get(segment.segment_id, []), config, node.node_id)
            lineage["baseline_selected"] = baseline_selected
            lineage_records.append(lineage)
            if deck["status"] == "SEGMENT_DECK_AMBIGUOUS":
                detail, raw_ids = segment_ambiguity_detail(
                    points, segment, deck, forensic, config, node.node_id)
                detail["baseline_selected"] = baseline_selected
                ambiguity_records.append(detail)
                ownership[segment.segment_id] = dict(
                    strip=forensic["strip_raw_ids"],
                    candidates=raw_ids,
                )
    return ambiguity_records, lineage_records, contour_record, ownership


def audit_scene_p1(points, config, research, baseline_result):
    result, auxiliary, nodes = _hierarchy(points, config, research)
    if (_baseline_signature(result) != _baseline_signature(baseline_result)
            or result["region_count"] != baseline_result["region_count"]):
        raise ValueError("BASELINE004_BEHAVIOR_CHANGED")
    selected = {row["source_node"] for row in result["regions"]}
    grid = auxiliary["coarse"]
    fov, tree = FovMap(points, config), cKDTree(points[:, :2])
    ambiguity_records, lineage_records, contour_records, ownership = [], [], [], {}
    for node in nodes:
        node_ambiguity, node_lineage, node_contour, node_ownership = _node_audit_p1(
            points, node, grid, fov, config, tree, node.node_id in selected)
        ambiguity_records.extend(node_ambiguity)
        lineage_records.extend(node_lineage)
        contour_records.append(node_contour)
        ownership.update(node_ownership)
    return result, dict(
        ambiguous_segments=ambiguity_records,
        segments=lineage_records,
        nodes=contour_records,
        ownership=ownership,
    )


def _write_ambiguity_ply(directory, points, records):
    """One review PLY per scene: candidate points colored by candidate index."""
    clouds, colors = [], []
    for record in records:
        for candidate in record["plane_candidates"]:
            ids = record.get("_candidate_ids", [])[candidate["candidate_index"]]
            if not len(ids):
                continue
            clouds.append(points[ids])
            colors.append(np.tile(np.array(PALETTE[candidate["candidate_index"] % len(PALETTE)],
                                           dtype=np.uint8), (len(ids), 1)))
    if not clouds:
        return None
    write_colored_ply(directory / "ambiguity_plane_candidates.ply",
                      np.vstack(clouds),
                      np.vstack(colors).astype(np.uint8))
    return "ambiguity_plane_candidates.ply"


def run_batch_p1(run_id, baseline_root, manifest_path, data_root, output_root):
    if not run_id or run_id in (".", "..") or any(c in run_id for c in "/\\:"):
        raise ValueError("INVALID_RUN_ID")
    destination = Path(output_root) / run_id
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("P1_RUN_ALREADY_EXISTS:" + str(destination))
    config, _ = resolve_config(DEFAULT_CONFIG)
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    research_sha = hashlib.sha256(
        json.dumps(research, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scans = {scan["scan_id"]: scan for scan in manifest["scans"]}
    pair_rows, lineage_rows, contour_rows = [], [], []
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
            (Path(baseline_root) / scene / "hatch_regions.json").read_text(encoding="utf-8"))
        if input_sha != frozen["input_sha256"]:
            raise ValueError("BASELINE_INPUT_MISMATCH:" + scene)
        if (config_hash(config) != frozen["v15_config_hash"]
                or research_sha != frozen["research_config_sha256"]):
            raise ValueError("BASELINE_CONFIG_MISMATCH:" + scene)
        points, _ = decode(pcd)
        result, audit = audit_scene_p1(points, config, research, frozen)
        directory = destination / scene
        directory.mkdir(parents=True, exist_ok=False)
        # Keep raw IDs out of JSON; store them in a per-scene NPZ.
        ownership_path = directory / "ambiguity_ownership.npz"
        npz_payload = {}
        for segment_id, group in audit["ownership"].items():
            npz_payload[segment_id + "_strip"] = group["strip"]
            for index, ids in enumerate(group["candidates"]):
                npz_payload["%s_candidate_%02d" % (segment_id, index)] = ids
        np.savez_compressed(ownership_path, **npz_payload)
        for record in audit["ambiguous_segments"]:
            record["_candidate_ids"] = audit["ownership"][record["segment_id"]]["candidates"]
        _write_ambiguity_ply(directory, points, audit["ambiguous_segments"])
        for record in audit["ambiguous_segments"]:
            record.pop("_candidate_ids", None)

        _atomic_json(directory / "p1_ambiguity_forensics.json", dict(
            schema="ship_perception.v15r.p1_ambiguity_forensics.1",
            scene_id=scene, scan_id=scan_id, software_git_sha=_git_sha(),
            input_sha256=input_sha, baseline_004_software_git_sha=frozen["software_git_sha"],
            v15_config_hash=frozen["v15_config_hash"],
            research_config_sha256=research_sha,
            baseline_004_behavior_unchanged=True,
            selected_region_signature=_baseline_signature(result),
            ambiguous_segment_count=len(audit["ambiguous_segments"]),
            raw_ids_stored_in="ambiguity_ownership.npz",
            semantic="PER_CANDIDATE_PLANE_GEOMETRY_ONLY; NO_DECK_COAMING_LABEL_IN_DETECTOR",
            segments=audit["ambiguous_segments"],
        ))
        _atomic_json(directory / "p1_segment_lineage.json", dict(
            schema="ship_perception.v15r.p1_segment_lineage.1",
            scene_id=scene, scan_id=scan_id, software_git_sha=_git_sha(),
            segment_count=len(audit["segments"]),
            semantic="SEGMENT_CREATED->DECK->PROFILE->FACE->LINE; NOT_RUN!=FAILED",
            segments=audit["segments"],
        ))
        _atomic_json(directory / "p1_contour_audit.json", dict(
            schema="ship_perception.v15r.p1_contour_audit.1",
            scene_id=scene, scan_id=scan_id, software_git_sha=_git_sha(),
            node_count=len(audit["nodes"]),
            nodes=audit["nodes"],
        ))
        for record in audit["ambiguous_segments"]:
            for pair in record["plane_pairs"]:
                pair_rows.append(dict(
                    scene_id=scene, node_id=record["node_id"],
                    segment_id=record["segment_id"], **pair))
        for record in audit["segments"]:
            lineage_rows.append(dict(scene_id=scene, **record))
        for record in audit["nodes"]:
            contour_rows.append(dict(scene_id=scene, **record))
        summary.append(dict(
            scene_id=scene,
            node_count=len(audit["nodes"]),
            segment_count=len(audit["segments"]),
            ambiguous_segment_count=len(audit["ambiguous_segments"]),
        ))
        print("%s: %d segments, %d ambiguous, %d nodes"
              % (scene, len(audit["segments"]), len(audit["ambiguous_segments"]),
                 len(audit["nodes"])), flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "normal6_deck_plane_pair_table.csv", pair_rows)
    _write_csv(destination / "normal6_segment_failure_lineage.csv", lineage_rows)
    _write_csv(destination / "height_contour_audit.csv", contour_rows)
    _atomic_json(destination / "batch_summary.json", dict(
        schema="ship_perception.v15r.p1_batch.1", run_id=run_id,
        software_git_sha=_git_sha(), scenes=summary))
    return summary


def _write_csv(path, rows):
    if not rows:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            stream.write("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    ordered = ["scene_id", "node_id", "seed_id", "segment_id"] + [
        key for key in fieldnames if key not in ("scene_id", "node_id", "seed_id", "segment_id")]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ordered)
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
                        default=REPO / "Ship-Unloader-Work/r1_height_structure_forensics")
    args = parser.parse_args()
    run_batch_p1(args.run_id, args.baseline_root, args.dataset_manifest,
                 args.data_root, args.output_root)


if __name__ == "__main__":
    main()
