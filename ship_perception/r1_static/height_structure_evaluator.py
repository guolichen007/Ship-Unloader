"""Post-freeze weak-label diagnostics for height-structure forensic outputs.

This module is never imported by the detector or structural audit path.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[2]
NORMAL6_SCENES = ("08-01", "08-08", "4-16", "6-8", "7-16", "8-22")


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _bbox(corners):
    if len(corners) < 4:
        raise ValueError("WEAK_ANNOTATION_HAS_NO_BOX")
    return [
        min(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[0] for point in corners),
        max(point[1] for point in corners),
    ]


def _metrics(candidate, label):
    x0, y0, x1, y1 = candidate
    u0, v0, u1, v1 = label
    intersection = max(0.0, min(x1, u1) - max(x0, u0)) * max(
        0.0, min(y1, v1) - max(y0, v0)
    )
    area_a, area_b = (x1 - x0) * (y1 - y0), (u1 - u0) * (v1 - v0)
    union = area_a + area_b - intersection
    center_distance = (
        ((x0 + x1 - u0 - u1) / 2) ** 2 + ((y0 + y1 - v0 - v1) / 2) ** 2
    ) ** 0.5
    return dict(
        bbox_iou=float(intersection / union) if union else 0.0,
        center_distance_m=float(center_distance),
        bbox_width_difference_m=float((x1 - x0) - (u1 - u0)),
        bbox_height_difference_m=float((y1 - y0) - (v1 - v0)),
    )


def evaluate_scene(frozen, annotation):
    """Compare only after the detector and structural audit files exist on disk."""
    labels = [_bbox(row["corners"]) for row in annotation["annotations"]]
    selected = frozen["selected_region_signature"]
    nodes = frozen["nodes"]
    rows = []
    for label_id, label in enumerate(labels):
        best_selected = max(
            ((name, _metrics(box, label)) for name, box, _ in selected),
            key=lambda row: row[1]["bbox_iou"],
            default=None,
        )
        best_node = max(
            ((row["node_id"], _metrics(row["bbox_xy"], label)) for row in nodes),
            key=lambda row: row[1]["bbox_iou"],
            default=None,
        )
        rows.append(
            dict(
                annotation_index=label_id,
                weak_bbox_xy=label,
                best_selected_node_id=best_selected[0] if best_selected else None,
                best_bbox_iou=best_selected[1]["bbox_iou"] if best_selected else None,
                center_distance_m=(
                    best_selected[1]["center_distance_m"] if best_selected else None
                ),
                bbox_width_difference_m=(
                    best_selected[1]["bbox_width_difference_m"]
                    if best_selected
                    else None
                ),
                bbox_height_difference_m=(
                    best_selected[1]["bbox_height_difference_m"]
                    if best_selected
                    else None
                ),
                best_all_node_id=best_node[0] if best_node else None,
                best_all_node_bbox_iou=(
                    best_node[1]["bbox_iou"] if best_node else None
                ),
            )
        )
    return dict(
        metric_semantics="WEAK_LABEL_DIAGNOSTIC_ONLY",
        caveat="PARTIAL_SINGLE_HATCH_LABEL_IS_NOT_GOLDEN_OR_FULL_RECALL",
        weak_annotation_count=len(labels),
        comparisons=rows,
    )


def _decision(scene, audit, fragments, weak):
    nodes = audit["nodes"]
    observed = [row for row in nodes if row["observed_edge_count"]]
    complete = [row for row in nodes if row["boundary_status"] == "COMPLETE_OBSERVED"]
    compatible = [
        row for row in fragments["pairs"] if row["association_status"] == "COMPATIBLE"
    ]
    recoverable = [
        row
        for row in compatible
        if row["boundary_recoverability"]
        == "RECOVERABLE_FROM_EXISTING_STRUCTURAL_EVIDENCE"
    ]
    if complete and any(not row["baseline_selected"] for row in complete):
        route, reason = ("ROUTE_A", "OBSERVED_COMPLETE_ALTERNATIVE_NODE_EXISTS")
    elif recoverable:
        route, reason = ("ROUTE_B", "OBSERVED_FRAGMENT_GROUP_BOUNDARY_RECOVERABLE")
    elif (
        weak is not None
        and weak["comparisons"]
        and all(
            row["best_all_node_bbox_iou"] is not None
            and row["best_all_node_bbox_iou"] < 0.5
            for row in weak["comparisons"]
        )
    ):
        route, reason = (
            "ROUTE_D",
            "NO_HEIGHT_NODE_OVERLAPS_WEAK_REGION_WELL; "
            "NO_RECOVERABLE_FRAGMENT_GROUP",
        )
    elif observed or nodes:
        route, reason = (
            "ROUTE_C",
            "HEIGHT_NODES_EXIST_BUT_CURRENT_PROFILE_FACE_IS_INCOMPLETE",
        )
    else:
        route, reason = ("ROUTE_D", "NO_USABLE_HEIGHT_NODE_OR_FRAGMENT")
    names = dict(
        ROUTE_A="HEIGHT_TREE_STRUCTURAL_FALLBACK_SUFFICIENT",
        ROUTE_B="HEIGHT_TREE_FRAGMENT_RECOMPOSITION_REQUIRED",
        ROUTE_C="GENERALIZED_PROFILE_REQUIRED",
        ROUTE_D="INDEPENDENT_STRUCTURE_PROPOSAL_REQUIRED",
    )
    selected_vs_node_gap = max(
        (
            row["best_all_node_bbox_iou"] - row["best_bbox_iou"]
            for row in (weak["comparisons"] if weak else [])
            if row["best_all_node_bbox_iou"] is not None
            and row["best_bbox_iou"] is not None
        ),
        default=0.0,
    )
    if route == "ROUTE_D":
        first_bad = "HEIGHTMAP_TOPOLOGY_DECOMPOSITION"
    elif selected_vs_node_gap > 0.2:
        first_bad = "HEIGHTMAP_CANDIDATE_SELECTION"
    elif route == "ROUTE_C":
        first_bad = "RAW_3D_STRUCTURAL_EVIDENCE_EXTRACTION"
    elif route == "ROUTE_A":
        first_bad = "CANDIDATE_SELECTION"
    else:
        first_bad = "FRAGMENT_BOUNDARY_COMPLETION"
    return dict(
        scene_id=scene,
        route=route,
        route_name=names[route],
        first_bad_stage_hypothesis=first_bad,
        reason=reason,
        route_is_forensic_hypothesis=True,
        node_count=len(nodes),
        observed_node_count=len(observed),
        complete_observed_node_count=len(complete),
        compatible_fragment_pair_count=len(compatible),
        recoverable_fragment_pair_count=len(recoverable),
        weak_label_diagnostic=(weak["comparisons"] if weak else None),
        limits="WEAK_LABEL_AND_CURRENT_PROFILE_ARE_DIAGNOSTIC; ROUTE_IS_NOT_ACCEPTANCE",
    )


def evaluate_run(run_root, manifest_path, data_root):
    run_root = Path(run_root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scans = {scan["scan_id"]: scan for scan in manifest["scans"]}
    frozen = {}
    for scene in NORMAL6_SCENES:
        directory = run_root / scene
        paths = [
            directory / name
            for name in (
                "node_structure_audit.json",
                "fragment_compatibility.json",
                "all_node_contours.ply",
                "node_structure_overlay.ply",
                "observed_profile_edges.ply",
                "observed_3d_faces.ply",
            )
        ]
        if not all(path.is_file() for path in paths):
            raise ValueError("AUDIT_OUTPUT_NOT_COMPLETE:" + scene)
        frozen[scene] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        }
    # This record is written before opening any weak annotation. Later evaluator
    # outputs cannot alter the frozen detector/audit signature.
    _atomic_json(
        run_root / "frozen_audit_outputs.json",
        dict(schema="ship_perception.v15r.frozen_audit.1", files=frozen),
    )
    decisions = []
    for scene in NORMAL6_SCENES:
        directory = run_root / scene
        audit = json.loads(
            (directory / "node_structure_audit.json").read_text(encoding="utf-8")
        )
        scan_id = audit["scan_id"]
        fragments = json.loads(
            (directory / "fragment_compatibility.json").read_text(encoding="utf-8")
        )
        scan = scans[scan_id]
        weak = None
        if scan["annotation_scope"] != "UNLABELED":
            canonical = []
            for name, digest in zip(
                scan["annotation_files"], scan["annotation_sha256"]
            ):
                path = Path(data_root) / name
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("WEAK_ANNOTATION_SHA_MISMATCH:" + scene)
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("cloud_file") == scan["pcd_filename"]:
                    canonical.append((name, data))
            if len(canonical) != 1:
                raise ValueError("CANONICAL_WEAK_ANNOTATION_NOT_UNIQUE:" + scene)
            weak = evaluate_scene(audit, canonical[0][1])
            weak.update(
                scene_id=scene,
                annotation_scope=scan["annotation_scope"],
                annotation_file=canonical[0][0],
                duplicate_alias_annotations_not_scored=True,
            )
            _atomic_json(directory / "weak_label_metrics.json", weak)
        decisions.append(_decision(scene, audit, fragments, weak))
    _atomic_json(
        run_root / "decision_matrix.json",
        dict(
            schema="ship_perception.v15r.height_structure_decision.1",
            evaluator_git_sha=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
            ).strip(),
            evidence_only=True,
            decisions=decisions,
            next_stage="REVIEW_PROFILE_AND_SEGMENT_DECK_FAILURES_BEFORE_ALGORITHM_CHANGE",
        ),
    )
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
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
    args = parser.parse_args()
    for row in evaluate_run(args.run_root, args.dataset_manifest, args.data_root):
        print("%s: %s (%s)" % (row["scene_id"], row["route"], row["reason"]))


if __name__ == "__main__":
    main()
