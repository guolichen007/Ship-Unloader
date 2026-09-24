"""P1.5 plane-consensus forensics stay diagnostic and rotation-covariant."""

import json
from pathlib import Path
import unittest

import numpy as np

from ship_perception.r1_static.plane_consensus import (
    centered_offset,
    classify_support,
    cluster_candidates,
    family_role,
    normalize_normal,
    run_consensus,
)
from ship_perception.r1_static.plane_consensus_evaluator import compute_p15_decision
from ship_perception.r1_static.plane_consensus_synthetic import (
    _rotation_matrix,
    apply_se3,
    build_case_a,
    build_case_b,
    build_case_c,
    run_synthetic_adversarial,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/v15.json").read_text(encoding="utf-8"))


def _candidate(node_id, segment_id, normal, offset, d_centered, raw_ids, width, span):
    return dict(scene_id="s", node_id=node_id, segment_id=segment_id, candidate_index=0,
                normal=normalize_normal(normal), offset=float(offset), d_centered=float(d_centered),
                support_count=int(len(raw_ids)), outward_width=width, along_span=span,
                along_coverage=1.0, raw_ids=np.asarray(raw_ids, dtype=np.int64))


class PlaneConsensus(unittest.TestCase):
    def test_normalize_normal_sign_is_gravity_up(self):
        normal = normalize_normal([0, 0, -1.0])
        self.assertAlmostEqual(normal[2], 1.0)

    def test_centered_offset_is_translation_covariant(self):
        normal = normalize_normal([0, 0, 1])
        d_centered = centered_offset(0.0, normal, np.array([5.0, 5.0, 5.0]))
        self.assertAlmostEqual(d_centered, 5.0)

    def test_classify_support_zones(self):
        bbox = (0.0, 0.0, 10.0, 8.0)
        xy = np.array([[5, 4], [-1, 4], [20, 4]])
        inside, boundary, outside = classify_support(xy, bbox, 4.0)
        self.assertTrue(inside[0])
        self.assertTrue(boundary[1])
        self.assertTrue(outside[2])

    def test_cluster_votes_at_most_once_per_segment(self):
        centroid = np.zeros(3)
        normal = [0, 0, 1]
        offset = 0.0
        ids = np.arange(10, dtype=np.int64)
        candidates = [
            _candidate("n", "seg_a", normal, offset, 0.0, ids, 3.0, 10.0),
            _candidate("n", "seg_a", normal, offset, 0.0, ids + 10, 3.0, 10.0),
            _candidate("n", "seg_a", normal, offset, 0.0, ids + 20, 3.0, 10.0),
            _candidate("n", "seg_b", normal, offset, 0.0, ids + 30, 3.0, 10.0),
        ]
        families = cluster_candidates(candidates, 3.0, 0.05)
        self.assertEqual(len(families), 1)
        members = families[0]["members"]
        self.assertEqual(len(members), 4)
        segments = {candidates[index]["segment_id"] for index in members}
        self.assertEqual(segments, {"seg_a", "seg_b"})

    def test_family_role_interior_surface(self):
        points = np.zeros((12, 3), dtype=np.float32)
        points[:, 0] = np.linspace(1, 9, 12)
        points[:, 1] = 4.0
        candidate = _candidate("n0", "seg", [0, 0, 1], 0.0, 0.0, np.arange(12), 3.0, 8.0)
        families = cluster_candidates([candidate], 3.0, 0.05)
        role = family_role(families[0], [candidate], points, {"n0": (0.0, 0.0, 10.0, 8.0)},
                           4.0, narrow_threshold=10.0)
        self.assertEqual(role["role_hypothesis"], "INTERIOR_SURFACE")

    def test_synthetic_adversarial_all_pass(self):
        results = run_synthetic_adversarial(CONFIG)
        for case in ("smooth_cargo_slope", "overflow_cargo", "roll_pitch_yaw_invariance"):
            self.assertTrue(results[case]["pass_"], case)
        self.assertEqual(results["overall"], "ALL_PASS")

    def test_overflow_cargo_stays_fail_closed(self):
        points, candidates, boxes = build_case_b()
        families, _ = run_consensus(candidates, points, boxes, CONFIG, 3.0, 0.05)
        roles = {family["role_hypothesis"] for family in families}
        self.assertIn("ROLE_AMBIGUOUS_OVERFLOW", roles)
        self.assertFalse(roles & {"BROAD_PERIMETER_SUPPORT", "NARROW_BOUNDARY_STRIP"})

    def test_c2_requires_child_strictly_stronger_than_root(self):
        synthetic = dict(smooth_cargo_slope=dict(pass_=True),
                         overflow_cargo=dict(pass_=True),
                         roll_pitch_yaw_invariance=dict(pass_=True))
        summary = {"4-16": dict(cross_segment_families=1, structural_families=1),
                   "8-22": dict(cross_segment_families=1, structural_families=1)}
        cross = {}
        selected = {"6-8": ["root"]}
        # root=100, child=1 must FAIL: child is not strictly stronger.
        d = compute_p15_decision(summary, cross, synthetic,
                                 {"6-8": {"root": 100, "child": 1}}, selected)
        self.assertFalse(d["criteria"]["c2_6_8_root_not_favored"])
        # root=1, child=37 passes (real data shape).
        d2 = compute_p15_decision(summary, cross, synthetic,
                                  {"6-8": {"root": 1, "child": 37}}, selected)
        self.assertTrue(d2["criteria"]["c2_6_8_root_not_favored"])

    def test_roll_pitch_invariance_preserves_roles(self):
        points, candidates, boxes = build_case_c()
        families_before, _ = run_consensus(candidates, points, boxes, CONFIG, 3.0, 0.05)
        rotation = _rotation_matrix(4.0, 2.0, 7.0)
        translation = np.array((3.0, -2.0, 5.0))
        points_rot, candidates_rot, boxes_rot = apply_se3(points, candidates, boxes,
                                                          rotation, translation)
        families_after, _ = run_consensus(candidates_rot, points_rot, boxes_rot,
                                          CONFIG, 3.0, 0.05)
        before = sorted((family["role_hypothesis"], tuple(sorted(family["segment_ids"])))
                        for family in families_before)
        after = sorted((family["role_hypothesis"], tuple(sorted(family["segment_ids"])))
                       for family in families_after)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
