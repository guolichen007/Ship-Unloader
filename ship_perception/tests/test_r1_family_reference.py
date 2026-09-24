"""S4-R1 deterministic family consensus and reference resolution invariants."""

import json
import math
from pathlib import Path
import unittest

import numpy as np

from ship_perception.r1_static.family_consensus import (
    deterministic_families,
    normalize_normal,
    role_from_zones,
)
from ship_perception.r1_static.family_reference import resolve_segment_reference
from ship_perception.r1_static.family_reference_evaluator import compute_s4r1_gates
from ship_perception.r1_static.family_reference_synthetic import run_synthetic_regression


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/v15.json").read_text(encoding="utf-8"))


def _candidate(segment_id, normal_deg, d_centered, support_count=100, node_id="n0"):
    rad = math.radians(normal_deg)
    normal = normalize_normal([math.sin(rad), 0.0, math.cos(rad)])
    return dict(node_id=node_id, segment_id=segment_id, candidate_index=0,
                normal=normal, offset=0.0, d_centered=d_centered,
                support_count=support_count, outward_width=2.0, along_span=10.0,
                along_coverage=1.0, raw_ids=np.arange(5, dtype=np.int64))


def _partition(families, candidates):
    parts = []
    for family in families:
        if family["members"]:
            parts.append(frozenset(candidates[index]["segment_id"]
                                   for index in family["members"]))
    return frozenset(parts)


class FamilyReference(unittest.TestCase):
    def test_deterministic_under_input_shuffle(self):
        candidates = [
            _candidate("seg_a", 0.0, 0.0, support_count=200),
            _candidate("seg_b", 0.5, 0.01, support_count=100),
            _candidate("seg_c", 1.0, 0.02, support_count=50),
            _candidate("seg_d", 5.0, 0.20, support_count=10),
            _candidate("seg_e", 5.5, 0.21, support_count=9),
        ]
        families_a, _ = deterministic_families(candidates, 3.0, 0.05, 6.0, 0.10)
        shuffled = candidates[::-1]
        families_b, _ = deterministic_families(shuffled, 3.0, 0.05, 6.0, 0.10)
        self.assertEqual(_partition(families_a, candidates),
                         _partition(families_b, shuffled))

    def test_relaxed_attachment_cannot_seed_its_own_family(self):
        # Drifting patches (4 deg apart) never form a strict core; they must not
        # be attached into a cross-segment family by the relaxed gate alone.
        candidates = [
            _candidate("seg_a", 0.0, 0.00),
            _candidate("seg_b", 4.0, 0.05),
            _candidate("seg_c", 8.0, 0.10),
        ]
        families, _ = deterministic_families(candidates, 3.0, 0.05, 6.0, 0.10)
        max_votes = max((len({candidates[i]["segment_id"] for i in family["members"]})
                         for family in families), default=0)
        self.assertEqual(max_votes, 1)

    def test_chain_bridging_guard_bounds_dispersion(self):
        candidates = [
            _candidate("seg_%d" % index, 2.5 * index, 0.05 * index)
            for index in range(6)
        ]
        families, _ = deterministic_families(candidates, 3.0, 0.05, 6.0, 0.10)
        for family in families:
            rep = family["rep_normal"]
            for index in family["members"]:
                angle = math.degrees(math.acos(float(np.clip(
                    float(candidates[index]["normal"] @ rep), -1.0, 1.0))))
                self.assertLessEqual(angle, 6.0 + 1e-9)
                self.assertLessEqual(abs(candidates[index]["d_centered"] -
                                         family["rep_d_centered"]), 0.10 + 1e-9)

    def test_role_from_zones_interior_surface(self):
        role = role_from_zones(interior_deep=80, boundary_inner=10, boundary_outer=10,
                               exterior_remote=0, median_width=3.0, narrow_threshold=1.0)
        self.assertEqual(role, "INTERIOR_SURFACE")

    def test_role_from_zones_overflow(self):
        role = role_from_zones(interior_deep=10, boundary_inner=10, boundary_outer=40,
                               exterior_remote=40, median_width=3.0, narrow_threshold=1.0)
        self.assertEqual(role, "ROLE_AMBIGUOUS_OVERFLOW")

    def test_synthetic_regression_all_pass(self):
        results = run_synthetic_regression(CONFIG)
        for case in ("smooth_cargo_slope", "overflow_cargo",
                     "roll_pitch_yaw_invariance", "irregular_cargo"):
            self.assertTrue(results[case]["pass_"], case)
        self.assertEqual(results["overall"], "ALL_PASS")

    def test_resolve_no_local_reference_observation(self):
        candidates = [
            dict(node_id="n0", segment_id="seg_a", candidate_index=0, normal=np.array([0, 0, 1.]),
                 offset=0.0, d_centered=0.0, support_count=10, outward_width=2.0,
                 along_span=10.0, along_coverage=1.0, raw_ids=np.arange(3, dtype=np.int64)),
        ]
        family_of = [None]
        families_by_id = {}
        resolution = resolve_segment_reference([0], family_of, families_by_id, candidates, CONFIG)
        self.assertEqual(resolution["status"], "NO_LOCAL_REFERENCE_OBSERVATION")


def _scene_stats(ambiguous, resolved, before, after, root_edges=0):
    return dict(ambiguous_count=ambiguous, family_reference_resolved_count=resolved,
                before_edge_count=before, after_edge_count=after,
                node_evidence={"l03-c0002": {"observed_structural_edge_count": root_edges}})


class S4R1GateHardening(unittest.TestCase):
    def _stats(self, ambiguous, resolved, before, after, root_edges=0):
        scenes = {"4-16": _scene_stats(ambiguous, resolved, before, after),
                  "8-22": _scene_stats(ambiguous, resolved, before, after),
                  "6-8": _scene_stats(ambiguous, resolved, before, after, root_edges=root_edges)}
        return scenes

    def test_gate_a_requires_majority_unlock(self):
        synthetic = dict(overall="ALL_PASS")
        # 5 of 10 is NOT a majority (5 > 5 is false) -> FAIL.
        gates, first_bad, _ = compute_s4r1_gates(
            self._stats(10, 5, 0, 10), synthetic)
        self.assertFalse(gates["gate_a_4_16_8_22_majority_unlock"])
        # 6 of 10 is a majority -> PASS.
        gates, _, decision = compute_s4r1_gates(self._stats(10, 6, 0, 10), synthetic)
        self.assertTrue(gates["gate_a_4_16_8_22_majority_unlock"])
        self.assertEqual(decision, "PASS")

    def test_gate_b_requires_per_scene_edge_gain(self):
        synthetic = dict(overall="ALL_PASS")
        gates, first_bad, _ = compute_s4r1_gates(self._stats(10, 6, 10, 10), synthetic)
        self.assertFalse(gates["gate_b_observed_edges_increased"])
        self.assertEqual(first_bad, "UNLOCKED_PROFILE_PRODUCED_NO_EDGE")
        gates, _, decision = compute_s4r1_gates(self._stats(10, 6, 9, 48), synthetic)
        self.assertTrue(gates["gate_b_observed_edges_increased"])

    def test_gate_c_requires_root_zero_edges(self):
        synthetic = dict(overall="ALL_PASS")
        # root with 51 of 52 fake edges must FAIL.
        gates, first_bad, _ = compute_s4r1_gates(self._stats(10, 6, 0, 10, root_edges=51), synthetic)
        self.assertFalse(gates["gate_c_6_8_root_zero_edges"])
        self.assertEqual(first_bad, "SIX_EIGHT_GIANT_ROOT_FALSE_COMPLETE")
        gates, _, decision = compute_s4r1_gates(self._stats(10, 6, 0, 10, root_edges=0), synthetic)
        self.assertTrue(gates["gate_c_6_8_root_zero_edges"])
        self.assertEqual(decision, "PASS")


if __name__ == "__main__":
    unittest.main()
