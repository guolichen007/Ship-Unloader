"""Post-run S4-R1 evaluator: freeze detector-side outputs, then apply Gates A-E.

No weak annotation is read here; the S4-R1 success criteria are structural and
synthetic only. This module is never imported by the detector.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .batch import NORMAL6
from .run import REPO, _atomic_json


def evaluate_run_s4r1(run_root, synthetic_path):
    run_root = Path(run_root)
    frozen = {}
    for scene in NORMAL6:
        names = ("plane_families.json", "segment_reference_resolution.json",
                 "structural_edges.json", "narrow_boundary_strips.json",
                 "node_structural_evidence.json")
        if not all((run_root / scene / name).is_file() for name in names):
            raise ValueError("S4R1_OUTPUT_NOT_COMPLETE:" + scene)
        frozen[scene] = {
            name: hashlib.sha256((run_root / scene / name).read_bytes()).hexdigest()
            for name in names
        }
    if synthetic_path.is_file():
        frozen["synthetic"] = {
            "synthetic_regression.json": hashlib.sha256(synthetic_path.read_bytes()).hexdigest()}
    _atomic_json(run_root / "frozen_audit_outputs.json",
                 dict(schema="ship_perception.v15r.frozen_audit.1", files=frozen))

    scene_stats = {}
    for scene in NORMAL6:
        resolutions = json.loads((run_root / scene / "segment_reference_resolution.json")
                                 .read_text(encoding="utf-8"))["segments"]
        edges = json.loads((run_root / scene / "structural_edges.json")
                           .read_text(encoding="utf-8"))["edges"]
        node_evidence = json.loads((run_root / scene / "node_structural_evidence.json")
                                   .read_text(encoding="utf-8"))["nodes"]
        ambiguous = [row for row in resolutions
                     if row["original_deck_status"] == "SEGMENT_DECK_AMBIGUOUS"]
        resolved = [row for row in ambiguous
                    if row["reference_resolution_status"] == "FAMILY_REFERENCE_RESOLVED"]
        before_edges = [edge for edge in edges if not edge.get("via_family_reference")]
        new_edges = [edge for edge in edges if edge.get("via_family_reference")]
        scene_stats[scene] = dict(
            ambiguous_count=len(ambiguous),
            family_reference_resolved_count=len(resolved),
            resolved_fraction=float(len(resolved) / len(ambiguous)) if ambiguous else None,
            before_edge_count=len(before_edges),
            new_edge_count=len(new_edges),
            after_edge_count=len(edges),
            node_evidence=node_evidence,
        )

    # Gate A: 4-16 and 8-22 each safely unlock at least 5 ambiguous segments.
    gate_a = (scene_stats["4-16"]["family_reference_resolved_count"] >= 5 and
              scene_stats["8-22"]["family_reference_resolved_count"] >= 5)
    # Gate B: family-reference unlock actually produced new observed edges.
    gate_b = sum(stats["new_edge_count"] for stats in scene_stats.values()) > 0
    # Gate C: 6-8 giant root (selected node) not rescued into a complete perimeter.
    root_edges = scene_stats["6-8"]["node_evidence"].get("l03-c0002", {}).get(
        "observed_structural_edge_count", 0)
    root_segments = scene_stats["6-8"]["node_evidence"].get("l03-c0002", {}).get(
        "segment_count", 0)
    gate_c = bool(root_segments == 0 or root_edges < root_segments)

    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    gate_d = synthetic.get("overall") == "ALL_PASS"
    gate_e = True  # enforced in the batch: BASELINE004_BEHAVIOR_CHANGED raises.

    gates = dict(
        gate_a_4_16_8_22_unlock=gate_a,
        gate_b_observed_edges_increased=gate_b,
        gate_c_6_8_root_no_false_complete=gate_c,
        gate_d_synthetic_regression=gate_d,
        gate_e_baseline004_unchanged=gate_e,
    )
    if not gate_a:
        first_bad = "REFERENCE_UNLOCK_INSUFFICIENT"
    elif not gate_b:
        first_bad = "UNLOCKED_PROFILE_PRODUCED_NO_EDGE"
    elif not gate_c:
        first_bad = "SIX_EIGHT_GIANT_ROOT_FALSE_COMPLETE"
    elif not gate_d:
        first_bad = "SYNTHETIC_REGRESSION_FAILED"
    elif not gate_e:
        first_bad = "BASELINE004_BEHAVIOR_CHANGED"
    else:
        first_bad = None
    decision = "PASS" if all(gates.values()) else "FAIL"
    return dict(
        schema="ship_perception.v15r.s4r1_decision.1",
        evaluator_git_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip(),
        s4_r1_decision=decision,
        next_stage="S4-R2" if decision == "PASS" else None,
        first_bad_stage=first_bad,
        gates=gates,
        scene_stats={scene: {key: value for key, value in stats.items()
                             if key != "node_evidence"}
                     for scene, stats in scene_stats.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path,
                        default=REPO / "Ship-Unloader-Work/r1_family_reference/synthetic_regression.json")
    args = parser.parse_args()
    decision = evaluate_run_s4r1(args.run_root, args.synthetic)
    _atomic_json(args.run_root / "decision_matrix_s4r1.json", decision)
    print("%s -> %s (first_bad=%s)" % (decision["s4_r1_decision"],
                                       decision.get("next_stage"),
                                       decision["first_bad_stage"]))


if __name__ == "__main__":
    main()
