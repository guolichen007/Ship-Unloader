"""Post-freeze P1.5 evaluator: weak-label proximity annotation and the P15
decision table. Annotation is opened only after the detector-side P1.5 outputs
(plane families, roles, cross-hatch consensus, synthetic results) are SHA-frozen.
This module is never imported by the detector or the structural audit path."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .batch import NORMAL6
from .run import REPO, _atomic_json

STEEL_ROLES = ("BROAD_PERIMETER_SUPPORT", "NARROW_BOUNDARY_STRIP")


def _bbox(points_xy):
    return [float(points_xy[:, 0].min()), float(points_xy[:, 1].min()),
            float(points_xy[:, 0].max()), float(points_xy[:, 1].max())]


def _bbox_iou(first, second):
    x0, y0, x1, y1 = first
    u0, v0, u1, v1 = second
    intersection = max(0.0, min(x1, u1) - max(x0, u0)) * max(0.0, min(y1, v1) - max(y0, v0))
    area_a = (x1 - x0) * (y1 - y0)
    area_b = (u1 - u0) * (v1 - v0)
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def compute_p15_decision(summary_by_scene, cross_hatch_by_scene, synthetic, node_votes_by_scene,
                         selected_nodes_by_scene):
    # Criterion 1: 4-16 and 8-22 each have a stable interpretable family.
    def has_stable_structural(scene):
        summary = summary_by_scene.get(scene, {})
        return (summary.get("cross_segment_families", 0) >= 1 and
                summary.get("structural_families", 0) >= 1)
    criterion_1 = has_stable_structural("4-16") and has_stable_structural("8-22")

    # Criterion 2: 6-8 giant root not artificially favored over children. The
    # child/fragment structural evidence must be STRICTLY stronger than the
    # selected root; "child has any votes" is not enough.
    root_votes = 0
    child_votes = 0
    selected = set(selected_nodes_by_scene.get("6-8", []))
    for node_id, count in node_votes_by_scene.get("6-8", {}).items():
        if node_id in selected:
            root_votes = max(root_votes, count)
        else:
            child_votes = max(child_votes, count)
    criterion_2 = child_votes > root_votes

    criterion_3 = bool(synthetic.get("smooth_cargo_slope", {}).get("pass_"))
    criterion_4 = bool(synthetic.get("overflow_cargo", {}).get("pass_"))
    criterion_5 = bool(synthetic.get("roll_pitch_yaw_invariance", {}).get("pass_"))

    any_stable = any(has_stable_structural(scene) for scene in NORMAL6)
    if not any_stable:
        decision = "P15-A"
    elif not criterion_3:
        decision = "P15-B"
    elif not criterion_5:
        decision = "P15-A"
    elif criterion_1 and criterion_2 and criterion_3 and criterion_4 and criterion_5:
        decision = "P15-D"
    else:
        decision = "P15-C"
    next_algorithm = {
        "P15-A": "PLANE_CONSENSUS_NOT_SEPARABLE; REVISIT_CANDIDATE_GENERATION",
        "P15-B": "PLANE_CONSENSUS_USEFUL_ROLE_AMBIGUOUS; ADD_MORPHOLOGY_TOPOLOGY",
        "P15-C": "PLANE_CONSENSUS_AND_ROLE_USEFUL; RESOLVE_REMAINING_CRITERIA",
        "P15-D": "READY_FOR_S4_R1; UNLOCK_PROFILE_VIA_FAMILY_REFERENCE_PLANE",
    }[decision]
    return dict(
        p15_decision=decision,
        next_algorithm_decision=next_algorithm,
        criteria=dict(
            c1_stable_interpretable_4_16_8_22=criterion_1,
            c2_6_8_root_not_favored=criterion_2,
            c3_smooth_cargo_slope_not_steel=criterion_3,
            c4_overflow_cargo_fail_closed=criterion_4,
            c5_roll_pitch_invariance=criterion_5,
        ),
        evidence=dict(
            cross_hatch_consensus=cross_hatch_by_scene,
            six_eight_root_child_structural_votes=node_votes_by_scene.get("6-8", {}),
            synthetic=synthetic,
        ),
    )


def evaluate_run_p15(run_root, manifest_path, data_root):
    run_root = Path(run_root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scans = {scan["scan_id"]: scan for scan in manifest["scans"]}
    frozen = {}
    for scene in NORMAL6:
        names = ("plane_families.json", "plane_family_roles.json", "cross_hatch_consensus.json",
                 "plane_family_overlay.ply", "plane_family_ownership.npz")
        if not all((run_root / scene / name).is_file() for name in names):
            raise ValueError("P15_OUTPUT_NOT_COMPLETE:" + scene)
        frozen[scene] = {
            name: hashlib.sha256((run_root / scene / name).read_bytes()).hexdigest()
            for name in names
        }
    synthetic_path = run_root / "synthetic_adversarial_results.json"
    if synthetic_path.is_file():
        frozen["synthetic"] = {
            "synthetic_adversarial_results.json":
                hashlib.sha256(synthetic_path.read_bytes()).hexdigest()}
    _atomic_json(run_root / "frozen_audit_outputs.json",
                 dict(schema="ship_perception.v15r.frozen_audit.1", files=frozen))

    summary_by_scene = {}
    cross_hatch_by_scene = {}
    node_votes_by_scene = {}
    selected_nodes_by_scene = {}
    for scene in NORMAL6:
        families = json.loads((run_root / scene / "plane_families.json").read_text(encoding="utf-8"))
        cross = json.loads((run_root / scene / "cross_hatch_consensus.json").read_text(encoding="utf-8"))
        summary_by_scene[scene] = dict(
            candidate_count=families["candidate_count"],
            family_count=families["family_count"],
            cross_segment_families=sum(1 for family in families["families"]
                                       if family["unique_segment_votes"] >= 2),
            structural_families=sum(1 for family in families["families"]
                                    if family["role_hypothesis"] in STEEL_ROLES),
        )
        cross_hatch_by_scene[scene] = cross["current_ship_internal_consensus"]
        selected_nodes_by_scene[scene] = families.get("selected_nodes", [])
        scan = scans[NORMAL6[scene]]
        if scan["annotation_scope"] != "UNLABELED":
            from ship_perception.tools.v15_pcd import decode
            points, _ = decode(Path(data_root) / scan["pcd_path"])
            canonical = []
            for name, digest in zip(scan["annotation_files"], scan["annotation_sha256"]):
                path = Path(data_root) / name
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("WEAK_ANNOTATION_SHA_MISMATCH:" + scene)
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("cloud_file") == scan["pcd_filename"]:
                    canonical.append(data)
            if len(canonical) != 1:
                raise ValueError("CANONICAL_WEAK_ANNOTATION_NOT_UNIQUE:" + scene)
            ownership = np.load(run_root / scene / "plane_family_ownership.npz")
            annotation = _weak_label_proximity(scene, families["families"], points,
                                               ownership, canonical[0])
            _atomic_json(run_root / scene / "p15_weak_label_proximity.json", annotation)
        node_votes_by_scene[scene] = _node_votes_from_families(families["families"])

    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    decision = compute_p15_decision(summary_by_scene, cross_hatch_by_scene, synthetic,
                                    node_votes_by_scene, selected_nodes_by_scene)
    _atomic_json(run_root / "decision_matrix_p15.json", dict(
        schema="ship_perception.v15r.p15_decision.1",
        evaluator_git_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip(),
        evidence_only=True,
        summary_by_scene=summary_by_scene,
        **decision,
    ))
    return decision


def _weak_label_proximity(scene, families, points, ownership, annotation):
    labels = [_bbox(np.asarray(box["corners"])) for box in annotation["annotations"]]
    rows = []
    for label_id, label in enumerate(labels):
        best = None
        for family in families:
            if family["role_hypothesis"] not in STEEL_ROLES:
                continue
            ids = ownership[family["family_id"]]
            if not len(ids):
                continue
            box = _bbox(points[ids, :2])
            iou = _bbox_iou(box, label)
            if best is None or iou > best[1]:
                best = (family["family_id"], iou, family["role_hypothesis"])
        rows.append(dict(annotation_index=label_id, weak_bbox_xy=label,
                         best_structural_family_id=best[0] if best else None,
                         best_structural_family_iou=best[1] if best else None,
                         best_structural_family_role=best[2] if best else None))
    return dict(semantics="WEAK_LABEL_DIAGNOSTIC_ONLY",
                caveat="WEAK_LABEL_IS_NOT_GOLDEN; USED_ONLY_FOR_PROXIMITY_ANNOTATION",
                scene_id=scene, comparisons=rows)


def _node_votes_from_families(families):
    votes = {}
    for family in families:
        if family["role_hypothesis"] not in STEEL_ROLES:
            continue
        for node_id in family["node_ids"]:
            votes.setdefault(node_id, set()).add(family["family_id"])
    return {node_id: len(families) for node_id, families in votes.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path,
                        default=REPO / "ship_perception/datasets/v15_dataset_manifest.json")
    parser.add_argument("--data-root", type=Path,
                        default=REPO / "Ship-Unloader-Data/legacy/hold_detector")
    args = parser.parse_args()
    decision = evaluate_run_p15(args.run_root, args.dataset_manifest, args.data_root)
    print("%s -> %s" % (decision["p15_decision"], decision["next_algorithm_decision"]))


if __name__ == "__main__":
    main()
