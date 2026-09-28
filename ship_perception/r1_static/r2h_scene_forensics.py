"""R2H-A offline stage accounting; never supplies evidence to the detector.

Only raw XYZ and the existing research stages are read. Output belongs under
Ship-Unloader-Work and is not a vessel or steel-boundary classification.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .batch import NORMAL6
from .family_reference import analyze_scene_s4r1
from .heightmap_batch import RESEARCH_CONFIG
from .heightmap_topology import detect_regions
from .rectangle_fusion import solve
from .rectangle_refinement import refine
from .rectangle_refinement_review import _read_validation_xyz
from .run import DEFAULT_CONFIG, REPO, _atomic_json, resolve_config


def axis_families(edges, bin_deg=5):
    """Descriptive angle bins, not semantic VesselHypotheses."""
    groups = defaultdict(list)
    for edge in edges:
        a = np.asarray(edge["a_raw"][:2], dtype=float)
        b = np.asarray(edge["b_raw"][:2], dtype=float)
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length <= 0:
            continue
        angle = math.degrees(math.atan2(delta[1], delta[0])) % 90
        bucket = int(angle // bin_deg)
        groups[bucket].append((a, b, min(float(edge["observed_support_length"]), 10.0),
                               edge.get("node_id")))
    result = []
    for bucket, rows in sorted(groups.items()):
        xy = np.asarray([point for a, b, _, _ in rows for point in (a, b)])
        result.append(dict(angle_bin_deg=[bucket * bin_deg, (bucket + 1) * bin_deg],
                           edge_count=len(rows), support_length_m=round(sum(r[2] for r in rows), 3),
                           node_ids=sorted({r[3] for r in rows if r[3] is not None}),
                           bbox_xy=[float(xy[:, 0].min()), float(xy[:, 1].min()),
                                    float(xy[:, 0].max()), float(xy[:, 1].max())]))
    return result


def global_axis_support(edges, axis_angle_deg, tolerance_deg):
    if axis_angle_deg is None:
        return dict(inlier_fraction=0.0, dispersion_deg=None, support_length_m=0.0)
    deltas, weights = [], []
    for row in edges:
        a = np.asarray(row["a_raw"][:2], dtype=float)
        b = np.asarray(row["b_raw"][:2], dtype=float)
        if np.linalg.norm(b - a) <= 0:
            continue
        angle = math.degrees(math.atan2((b - a)[1], (b - a)[0]))
        delta = (angle - axis_angle_deg + 45) % 90 - 45
        deltas.append(delta)
        weights.append(min(float(row["observed_support_length"]), 10.0))
    if not weights:
        return dict(inlier_fraction=0.0, dispersion_deg=None, support_length_m=0.0)
    inliers = np.abs(deltas) <= tolerance_deg
    weights = np.asarray(weights)
    return dict(inlier_fraction=float(weights[inliers].sum() / weights.sum()),
                dispersion_deg=float(np.sqrt(np.average(np.square(deltas), weights=weights))),
                support_length_m=float(weights.sum()))


def account_scene(path, config, research, expected_sha256=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("INPUT_SHA_MISMATCH:" + path.name)
    points = _read_validation_xyz(path)
    height_trace, fusion_trace = {}, {}
    height, _ = detect_regions(points, config, research, forensic_trace=height_trace)
    reference = analyze_scene_s4r1(points, config, research, height)
    initial = solve(points, reference["segment_resolutions"], reference["edges"],
                    height["level_hierarchy"], config, forensic_trace=fusion_trace)
    attempts = fusion_trace.pop("candidate_attempts", [])
    histogram = dict(sorted(Counter(row["reject_reason"] for row in attempts).items()))
    if "axes" in initial:
        refined = refine(points, reference["segment_resolutions"], reference["edges"],
                         initial, config)
        refined_count = len(refined["rectangles"])
    else:
        refined_count = 0
    families = axis_families(reference["edges"],
                            2 * config["boundary"]["merge_angle_deg"])
    axis = fusion_trace.get("global_structural_axes")
    axis_angle = math.degrees(math.atan2(axis[0][1], axis[0][0])) if axis else None
    axis_support = global_axis_support(reference["edges"], axis_angle,
                                       2 * config["boundary"]["merge_angle_deg"])
    secondary_stage = None
    if height["region_count"] == 0:
        first_bad = "HEIGHT_PROPOSAL"
        secondary_stage = "AXIS_RESOLUTION" if axis is None else None
    elif axis is None:
        first_bad = "AXIS_RESOLUTION"
    elif not attempts:
        first_bad = "RECTANGLE_PROPOSAL"
    elif not initial["rectangles"]:
        first_bad = "RECTANGLE_ACCEPTANCE" if histogram.get("FINAL_FOUR_SIDE_ACCEPTANCE_GATE") else "RECTANGLE_PROPOSAL"
    elif refined_count < len(initial["rectangles"]):
        first_bad = "R2G_REFINEMENT"
    else:
        first_bad = "NONE"
    return dict(schema_version="r2h_a_forensic.1", input_file=path.name,
                input_sha256=digest, raw_point_count=len(points),
                scene_support=height_trace, height_region_count=height["region_count"],
                global_structural_axis_deg=axis_angle,
                global_structural_axes=axis, raw_axis_families=families,
                global_axis_support=axis_support,
                structural_edge_count=len(reference["edges"]),
                profile_segment_count=len(reference["segment_resolutions"]),
                r2f_candidate_attempt_count=len(attempts),
                r2f_reject_reason_histogram=histogram,
                r2f_candidate_attempts=attempts,
                r2f_rectangle_count=len(initial["rectangles"]),
                r2g_rectangle_count=refined_count,
                first_bad_stage=first_bad,
                secondary_stage=secondary_stage,
                first_bad_stage_scope="MECHANICAL_PIPELINE_ONLY_NO_GT_OR_VESSEL_SEMANTICS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-set", choices=("normal6", "map-validation"), required=True)
    parser.add_argument("--validation-input-dir", type=Path)
    parser.add_argument("--only-scene")
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "Ship-Unloader-Work/review/r2h_a_forensics")
    args = parser.parse_args()
    config, _ = resolve_config(DEFAULT_CONFIG)
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    sources = []
    if args.scene_set == "normal6":
        manifest = json.loads((REPO / "ship_perception/datasets/v15_dataset_manifest.json")
                              .read_text(encoding="utf-8"))
        scans = {row["scan_id"]: row for row in manifest["scans"]}
        for scene, scan_id in NORMAL6.items():
            if args.only_scene and args.only_scene != scene:
                continue
            row = scans[scan_id]
            if row["split_role"] != "DEVELOPMENT":
                raise ValueError("NON_DEVELOPMENT_INPUT:" + scan_id)
            sources.append((scene, REPO / "Ship-Unloader-Data/legacy/hold_detector" /
                            row["pcd_path"], row["pcd_sha256"]))
    else:
        if args.validation_input_dir is None:
            parser.error("--validation-input-dir is required")
        for path in sorted(args.validation_input_dir.iterdir()):
            if path.suffix.lower() in (".pcd", ".ply") and (
                    args.only_scene is None or args.only_scene == path.stem):
                sources.append((path.stem, path, None))
    summary = {}
    for scene, path, digest in sources:
        result = account_scene(path, config, research, digest)
        _atomic_json(args.output_root / scene / "stage_accounting.json", result)
        summary[scene] = {key: result[key] for key in (
            "input_sha256", "raw_point_count", "height_region_count",
            "global_structural_axis_deg", "structural_edge_count",
            "global_axis_support",
            "r2f_candidate_attempt_count", "r2f_reject_reason_histogram",
            "r2f_rectangle_count", "r2g_rectangle_count", "first_bad_stage")}
        summary[scene]["secondary_stage"] = result["secondary_stage"]
        summary[scene]["scene_connected_component_count"] = result["scene_support"].get(
            "scene_connected_component_count", 0)
        summary[scene]["dropped_by_largest_component"] = result["scene_support"].get(
            "dropped_by_largest_component", [])
        summary[scene]["dropped_by_global_relative_score"] = result["scene_support"].get(
            "dropped_by_global_relative_score", [])
        print(scene, json.dumps(summary[scene], ensure_ascii=False), flush=True)
    _atomic_json(args.output_root / (args.scene_set + "_summary.json"), summary)


if __name__ == "__main__":
    main()
