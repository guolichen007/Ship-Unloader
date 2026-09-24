"""P1 forensics remain pure diagnostics and never alter selected regions."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from ship_perception.r1_static.height_structure_forensics_p1 import (
    _classify_pair,
    _turning_angles,
    _turning_histogram,
    audit_scene_p1,
    segment_ambiguity_detail,
)
from ship_perception.r1_static.height_structure_evaluator_p1 import (
    _orientation_difference_deg,
    segment_first_bad_stage,
)
from ship_perception.r1_static.heightmap_topology import detect_regions
from ship_perception.tests.test_r1_heightmap_topology import rectangular_basins


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/v15.json").read_text(encoding="utf-8"))
RESEARCH = json.loads(
    (ROOT / "config/r1_heightmap_research.json").read_text(encoding="utf-8")
)


class P1Forensics(unittest.TestCase):
    def test_audit_scene_p1_preserves_baseline_signature(self):
        points = rectangular_basins(2).astype(np.float32)
        baseline, _ = detect_regions(points, CONFIG, RESEARCH)
        before = copy.deepcopy(baseline)
        rebuilt, audit = audit_scene_p1(points, CONFIG, RESEARCH, baseline)
        self.assertEqual(baseline, before)
        self.assertEqual(
            [row["source_node"] for row in rebuilt["regions"]],
            [row["source_node"] for row in baseline["regions"]],
        )
        self.assertTrue(all("baseline_selected" in row for row in audit["segments"]))
        self.assertTrue(all("node_id" in row for row in audit["segments"]))
        self.assertTrue(all("node_id" in row for row in audit["nodes"]))

    def test_lineage_first_bad_stage_ordering(self):
        ambiguous = dict(segment_id="s", deck_status="SEGMENT_DECK_AMBIGUOUS",
                         deck_reason="MULTIPLE_DECK_LIKE_PLANES", profile_executed=False,
                         line_fit_result=None, break_candidate_count=0, face_option_count=0)
        self.assertEqual(segment_first_bad_stage(ambiguous, {}),
                         "DECK_REFERENCE_AMBIGUOUS")
        unresolved = dict(segment_id="s", deck_status="SEGMENT_DECK_UNRESOLVED",
                          deck_reason="INSUFFICIENT_SEGMENT_SPAN", profile_executed=False,
                          line_fit_result=None, break_candidate_count=0, face_option_count=0)
        self.assertEqual(segment_first_bad_stage(unresolved, {}), "PROFILE_NOT_EXECUTED")
        no_break = dict(segment_id="s", deck_status="RESOLVED", deck_reason="BROAD_OUTWARD_DECK_SUPPORT",
                        profile_executed=True, line_fit_result=None,
                        break_candidate_count=0, face_option_count=0)
        self.assertEqual(segment_first_bad_stage(no_break, {}), "PROFILE_MODEL_INSUFFICIENT")
        bad_line = dict(segment_id="s", deck_status="RESOLVED", deck_reason="BROAD_OUTWARD_DECK_SUPPORT",
                        profile_executed=True, line_fit_result=None,
                        break_candidate_count=4, face_option_count=0)
        self.assertEqual(segment_first_bad_stage(bad_line, {}), "LINE_AGGREGATION_FAILURE")
        bad_face = dict(segment_id="s", deck_status="RESOLVED", deck_reason="BROAD_OUTWARD_DECK_SUPPORT",
                        profile_executed=True, line_fit_result=None,
                        break_candidate_count=0, face_option_count=3)
        self.assertEqual(segment_first_bad_stage(bad_face, {}), "FACE_MODEL_INSUFFICIENT")
        resolved = dict(segment_id="s", deck_status="RESOLVED", deck_reason="BROAD_OUTWARD_DECK_SUPPORT",
                        profile_executed=True, line_fit_result="OBSERVED_PROFILE_BREAK",
                        break_candidate_count=4, face_option_count=0)
        self.assertIsNone(segment_first_bad_stage(resolved, {}))

    def test_coaming_pair_requires_vertical_connector(self):
        config = CONFIG
        base = dict(
            height_difference_m=0.5,
            narrow_is_elevated=True,
            relative_width=0.3,
            relative_boundary_distance=0.5,
        )
        with_connector = dict(base, vertical_connector_support_count=20,
                              vertical_connector_span_m=0.5)
        self.assertEqual(_classify_pair(with_connector, config)[0], "DECK_COAMING_LIKE_PAIR")
        no_connector = dict(base, vertical_connector_support_count=2,
                            vertical_connector_span_m=0.0)
        self.assertEqual(_classify_pair(no_connector, config)[0], "UNKNOWN_PAIR")
        not_elevated = dict(base, narrow_is_elevated=False,
                            vertical_connector_support_count=20,
                            vertical_connector_span_m=0.5)
        self.assertEqual(_classify_pair(not_elevated, config)[0], "UNKNOWN_PAIR")
        too_high = dict(base, height_difference_m=4.0, vertical_connector_support_count=20,
                        vertical_connector_span_m=0.5)
        self.assertEqual(_classify_pair(too_high, config)[0], "UNKNOWN_PAIR")

    def test_orientation_difference_is_acute(self):
        self.assertAlmostEqual(_orientation_difference_deg(0.0, 30.0), 30.0)
        self.assertAlmostEqual(_orientation_difference_deg(0.0, 170.0), 10.0)
        self.assertAlmostEqual(_orientation_difference_deg(10.0, 190.0), 0.0)

    def test_turning_angle_histogram_covers_rectangle(self):
        rectangle = [(0, 0), (10, 0), (10, 5), (0, 5)]
        angles = _turning_angles(rectangle)
        self.assertEqual(len(angles), 4)
        histogram = _turning_histogram(angles)
        self.assertEqual(sum(histogram), 4)
        # A rectangle's four corners are all near +-90 degree turns.
        np.testing.assert_allclose(np.abs(angles), 90.0, atol=1e-9)

    def test_ambiguity_detail_builds_candidates_without_labels(self):
        # A small deck patch with two stacked horizontal planes at the same
        # anchor; only the geometry fields are expected, never Deck/Coaming.
        x = np.linspace(0, 5, 40)
        y = np.linspace(0, 1, 8)
        xx, yy = np.meshgrid(x, y)
        lower = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 0.0)))
        upper = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 0.8)))
        points = np.vstack((lower, upper)).astype(np.float32)
        from ship_perception.r1_static.model import PerimeterSegment
        segment = PerimeterSegment("seg", "seed", (0.0, 0.5), (5.0, 0.5),
                                   (1.0, 0.0), (0.0, 1.0), 5.0, "FULLY_OBSERVED")
        from ship_perception.r1_static.perimeter_deck import estimate_segment_deck
        from scipy.spatial import cKDTree
        deck, plane, forensic = estimate_segment_deck(
            points, segment, CONFIG, cKDTree(points[:, :2]))
        detail, raw_ids = segment_ambiguity_detail(
            points, segment, deck, forensic, CONFIG, node_id="n")
        self.assertEqual(len(raw_ids), len(detail["plane_candidates"]))
        for candidate in detail["plane_candidates"]:
            self.assertNotIn("label", candidate)
            self.assertIn("normal", candidate)
            self.assertIn("anchor_height_m", candidate)


if __name__ == "__main__":
    unittest.main()
