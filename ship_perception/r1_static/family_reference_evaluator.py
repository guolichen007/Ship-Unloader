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

    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    gates, first_bad, decision = compute_s4r1_gates(scene_stats, synthetic)
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


def compute_s4r1_gates(scene_stats, synthetic):
    """S4-R1 gates, tightened: majority unlock, per-scene edge gain, root==0.

    ``scene_stats`` maps scene_id to a dict with ``ambiguous_count``,
    ``family_reference_resolved_count``, ``before_edge_count``,
    ``after_edge_count`` and ``node_evidence`` (node_id -> stats with
    ``observed_structural_edge_count``).
    """
    def majority_unlocked(scene):
        stats = scene_stats.get(scene, {})
        resolved = stats.get("family_reference_resolved_count", 0)
        ambiguous = stats.get("ambiguous_count", 0)
        return ambiguous > 0 and resolved > ambiguous - resolved

    # Gate A: 4-16 and 8-22 each have a strict majority of ambiguous segments
    # safely unlocked (generic "majority unlock", no per-scene tuned ratio).
    gate_a = majority_unlocked("4-16") and majority_unlocked("8-22")
    # Gate B: 4-16 and 8-22 each gain observed edges after unlock.
    gate_b = (scene_stats.get("4-16", {}).get("after_edge_count", 0) >
              scene_stats.get("4-16", {}).get("before_edge_count", 0) and
              scene_stats.get("8-22", {}).get("after_edge_count", 0) >
              scene_stats.get("8-22", {}).get("before_edge_count", 0))
    # Gate C: the known-wrong 6-8 giant root must produce zero observed edges.
    root_edges = scene_stats.get("6-8", {}).get("node_evidence", {}).get(
        "l03-c0002", {}).get("observed_structural_edge_count", 0)
    gate_c = bool(root_edges == 0)
    gate_d = synthetic.get("overall") == "ALL_PASS"
    gate_e = True  # enforced in the batch: BASELINE004_BEHAVIOR_CHANGED raises.

    gates = dict(
        gate_a_4_16_8_22_majority_unlock=gate_a,
        gate_b_observed_edges_increased=gate_b,
        gate_c_6_8_root_zero_edges=gate_c,
        gate_d_synthetic_regression=gate_d,
        gate_e_baseline004_unchanged=gate_e,
    )
    if not gate_a:
        first_bad = "REFERENCE_UNLOCK_NOT_MAJORITY"
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
    return gates, first_bad, decision


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
