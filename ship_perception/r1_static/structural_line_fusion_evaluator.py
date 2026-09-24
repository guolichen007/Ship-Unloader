"""Post-run S4-R2 evaluator: synthetic regression + Gates A-G.

No weak annotation is read here; the S4-R2 criteria are structural and synthetic
only. This module is never imported by the detector.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .batch import NORMAL6
from .family_reference_synthetic import run_synthetic_regression
from .run import DEFAULT_CONFIG, REPO, _atomic_json
from .structural_line_fusion import fuse_lines


def _observation(segment_id, points, primitive_type="OBSERVED_PROFILE_BREAK", node_id="n0"):
    return dict(node_id=node_id, segment_id=segment_id, reference_family_id=None,
                primitive_type=primitive_type, positions=np.asarray(points, dtype=float))


def run_fusion_synthetic(config):
    # Case E: occlusion gap keeps two observed intervals.
    frag_a = [[i * 0.1, 0.0] for i in range(30)]
    frag_b = [[i * 0.1, 0.0] for i in range(60, 90)]
    occluded, _ = fuse_lines([_observation("s0", frag_a), _observation("s1", frag_b)], config)
    occlusion_pass = bool(len(occluded) == 1 and occluded[0]["observed_interval_count"] == 2)
    # Case F: parallel separate edges must not merge.
    edge_a = [[i * 0.1, 0.0] for i in range(100)]
    edge_b = [[i * 0.1, 1.0] for i in range(100)]
    parallel, _ = fuse_lines([_observation("s0", edge_a), _observation("s1", edge_b)], config)
    parallel_pass = bool(len(parallel) == 2)
    return {
        "occluded_interval": dict(pass_=occlusion_pass, line_count=len(occluded)),
        "parallel_separation": dict(pass_=parallel_pass, line_count=len(parallel)),
    }


def run_synthetic_s4r2(config):
    results = dict(run_synthetic_regression(config))
    fusion = run_fusion_synthetic(config)
    results["occluded_interval"] = fusion["occluded_interval"]
    results["parallel_separation"] = fusion["parallel_separation"]
    cases = ("smooth_cargo_slope", "overflow_cargo", "roll_pitch_yaw_invariance",
             "irregular_cargo", "occluded_interval", "parallel_separation")
    results["overall"] = "ALL_PASS" if all(results[case]["pass_"] for case in cases) else "SOME_FAIL"
    return results


def evaluate_run_s4r2(run_root, synthetic_path):
    run_root = Path(run_root)
    frozen = {}
    for scene in NORMAL6:
        names = ("structural_primitives.json", "structural_line_families.json",
                 "structure_bands.json", "node_line_evidence.json", "rejected_evidence.json")
        if not all((run_root / scene / name).is_file() for name in names):
            raise ValueError("S4R2_OUTPUT_NOT_COMPLETE:" + scene)
        frozen[scene] = {
            name: hashlib.sha256((run_root / scene / name).read_bytes()).hexdigest()
            for name in names
        }
    _atomic_json(run_root / "frozen_audit_outputs_s4r2.json",
                 dict(schema="ship_perception.v15r.frozen_audit_s4r2.1", files=frozen))

    scene_stats = {}
    for scene in NORMAL6:
        lines = json.loads((run_root / scene / "structural_line_families.json")
                           .read_text(encoding="utf-8"))["lines"]
        nodes = json.loads((run_root / scene / "node_line_evidence.json")
                           .read_text(encoding="utf-8"))["nodes"]
        primitives = json.loads((run_root / scene / "structural_primitives.json")
                                .read_text(encoding="utf-8"))["primitives"]
        rejected = json.loads((run_root / scene / "rejected_evidence.json")
                              .read_text(encoding="utf-8"))
        multi_segment = sum(1 for line in lines if len(line["source_segment_ids"]) >= 2)
        multi_modal = sum(1 for line in lines
                          if len([t for t in line["evidence_types"]
                                  if t in ("OBSERVED_PROFILE_BREAK", "OBSERVED_3D_FACE")]) >= 2)
        # Gate B: every primitive is either fused into a line or has a reject reason.
        accounted = sum(len(line["source_segment_ids"]) for line in lines)
        rejected_count = len(rejected["rejected_primitives"]) + len(rejected["rejected_observations"])
        gate_b_ok = accounted + rejected_count >= len(primitives)
        scene_stats[scene] = dict(
            primitive_count=len(primitives), line_count=len(lines),
            multi_segment_line_count=multi_segment, multi_modal_line_count=multi_modal,
            accounted_evidence=accounted, rejected_evidence=rejected_count,
            gate_b_ok=gate_b_ok, nodes=nodes)

    gate_a = (scene_stats["4-16"]["multi_segment_line_count"] > 0 and
              scene_stats["8-22"]["multi_segment_line_count"] > 0)
    gate_b = all(stats["gate_b_ok"] for stats in scene_stats.values())
    root_lines = scene_stats["6-8"]["nodes"].get("l03-c0002", {}).get("observed_line_count", 0)
    gate_c = bool(root_lines == 0)
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    gate_d = synthetic["occluded_interval"]["pass_"]
    gate_e = synthetic["parallel_separation"]["pass_"]
    gate_f = synthetic.get("overall") == "ALL_PASS"
    gate_g = True  # enforced in the batch: BASELINE004_BEHAVIOR_CHANGED raises.

    gates = dict(
        gate_a_multi_segment_line=gate_a,
        gate_b_no_silent_evidence_drop=gate_b,
        gate_c_6_8_root_no_lines=gate_c,
        gate_d_occluded_interval=gate_d,
        gate_e_parallel_separation=gate_e,
        gate_f_synthetic_regression=gate_f,
        gate_g_baseline004_unchanged=gate_g,
    )
    if not gate_a:
        first_bad = "NO_MULTI_SEGMENT_LINE"
    elif not gate_b:
        first_bad = "SILENT_EVIDENCE_DROP"
    elif not gate_c:
        first_bad = "SIX_EIGHT_ROOT_FALSE_LINE"
    elif not gate_d:
        first_bad = "OCCLUSION_INTERVAL_FAIL"
    elif not gate_e:
        first_bad = "PARALLEL_EDGE_MERGED"
    elif not gate_f:
        first_bad = "SYNTHETIC_REGRESSION_FAILED"
    else:
        first_bad = None
    decision = "PASS" if all(gates.values()) else "FAIL"
    return dict(
        schema="ship_perception.v15r.s4r2_decision.1",
        evaluator_git_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip(),
        s4_r2_decision=decision,
        next_stage="S4-R3" if decision == "PASS" else None,
        first_bad_stage=first_bad,
        gates=gates,
        scene_stats={scene: {key: value for key, value in stats.items() if key != "nodes"}
                     for scene, stats in scene_stats.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path,
                        default=REPO / "Ship-Unloader-Work/r1_structural_lines/synthetic_regression.json")
    args = parser.parse_args()
    config = json.loads((REPO / "ship_perception/config/v15.json").read_text(encoding="utf-8"))
    synthetic = run_synthetic_s4r2(config)
    _atomic_json(args.synthetic, synthetic)
    decision = evaluate_run_s4r2(args.run_root, args.synthetic)
    _atomic_json(args.run_root / "decision_matrix_s4r2.json", decision)
    print("%s -> %s (first_bad=%s)" % (decision["s4_r2_decision"],
                                       decision.get("next_stage"),
                                       decision["first_bad_stage"]))


if __name__ == "__main__":
    main()
