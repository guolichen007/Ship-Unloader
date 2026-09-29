"""R2H-A red tests: current stage failures, not relaxed acceptance gates.

Expected failures remain visible until R2H-B provides an evidence-backed scene
and proposal layer. Passing tests assert that ambiguity stays unconfirmed.
"""

import json
import math
import unittest

import numpy as np

from ship_perception.r1_static.heightmap_topology import detect_regions
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG
from ship_perception.r1_static.rectangle_fusion import solve
from ship_perception.r1_static.rectangle_refinement import _ship_width_consensus
from ship_perception.r1_static.scene_vessel import infer_scene_vessels
from ship_perception.tests.test_r1_boundary_topology import CONFIG, rectangle_case
from ship_perception.tests.test_r1_rectangle_fusion import raw_scene


def strong_hatch(width, vessel_id, x):
    polygon = [[x, 0], [x + 40, 0], [x + 40, width], [x, width]]
    return dict(vessel_hypothesis_id=vessel_id, polygon_xy=polygon,
                sides=[dict(side_id=key, structural_coverage=1.0, raster_coverage=1.0)
                       for key in ("U0", "U1", "V0", "V1")])


def vessel_cloud(cx, cy, angle_deg=0, step=.2, gap=False):
    longitudinal, transverse = np.meshgrid(np.arange(-20, 20, step),
                                            np.arange(-6, 6, step))
    longitudinal, transverse = longitudinal.ravel(), transverse.ravel()
    keep = np.abs(longitudinal) > 1 if gap else np.ones(len(longitudinal), bool)
    longitudinal, transverse = longitudinal[keep], transverse[keep]
    rim = (np.abs(longitudinal) > 19.2) | (np.abs(transverse) > 5.2)
    angle = math.radians(angle_deg)
    return np.column_stack((cx + longitudinal * math.cos(angle) - transverse * math.sin(angle),
                            cy + longitudinal * math.sin(angle) + transverse * math.cos(angle),
                            rim.astype(float))).astype(np.float32)


def scene_hypotheses(points):
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    result, _ = infer_scene_vessels(points, CONFIG, research)
    return result, [row for row in result["vessel_hypotheses"]
                    if row["classification_status"] == "VESSEL_HYPOTHESIS"]


class SceneCounterexamples(unittest.TestCase):
    def test_forensic_trace_does_not_change_height_or_rectangle_output(self):
        points = raw_scene()
        research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
        ordinary_height, _ = detect_regions(points, CONFIG, research)
        trace = {}
        traced_height, _ = detect_regions(points, CONFIG, research,
                                          forensic_trace=trace)
        self.assertEqual(ordinary_height, traced_height)
        self.assertEqual(trace["raw_point_count"], len(points))
        segments, edges, regions = rectangle_case()
        ordinary = solve(points, segments, edges, regions, CONFIG)
        candidate_trace = {}
        audited = solve(points, segments, edges, regions, CONFIG,
                        forensic_trace=candidate_trace)
        self.assertEqual(ordinary["rectangles"], audited["rectangles"])
        self.assertTrue(candidate_trace["candidate_attempts"])

    def test_two_parallel_vessels_are_not_largest_only(self):
        result, vessels = scene_hypotheses(np.vstack((vessel_cloud(0, 0),
                                                     vessel_cloud(0, 20))))
        self.assertEqual(len(vessels), 2)
        self.assertEqual(result["target_vessel_status"], "NOT_SELECTED")

    def test_two_different_yaw_vessels_have_two_local_axes(self):
        _, vessels = scene_hypotheses(np.vstack((vessel_cloud(0, 0),
                                                 vessel_cloud(0, 23, angle_deg=20))))
        self.assertEqual(len(vessels), 2)
        angles = [math.degrees(math.atan2(row["local_axes"][0][1],
                                             row["local_axes"][0][0])) for row in vessels]
        self.assertAlmostEqual(min(angles), 0, delta=2)
        self.assertAlmostEqual(max(angles), 20, delta=2)

    def test_three_vessels_unequal_density_retain_all_support(self):
        _, vessels = scene_hypotheses(np.vstack((vessel_cloud(0, 0),
                                                 vessel_cloud(0, 20, step=.3),
                                                 vessel_cloud(0, 40, step=.25))))
        self.assertEqual(len(vessels), 3)

    def test_wharf_larger_than_vessel_cannot_select_target(self):
        x, y = np.meshgrid(np.arange(30, 95, .2), np.arange(-20, 20, .2))
        wharf = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size))).astype(np.float32)
        result, vessels = scene_hypotheses(np.vstack((wharf, vessel_cloud(0, 0))))
        self.assertGreater(len(wharf), len(vessel_cloud(0, 0)))
        self.assertEqual(len(vessels), 1)
        self.assertLess(vessels[0]["bbox_xy"][2], 30)
        self.assertEqual(result["target_vessel_status"], "NOT_SELECTED")
        self.assertTrue(any(row["classification_status"] == "UNRESOLVED"
                            for row in result["vessel_hypotheses"]))

    def test_one_vessel_split_components_keeps_both_halves(self):
        result, vessels = scene_hypotheses(vessel_cloud(0, 0, gap=True))
        self.assertEqual(len(result["scene_support_candidates"]), 2)
        self.assertEqual(len(vessels), 1)
        self.assertEqual(len(vessels[0]["source_support_ids"]), 2)

    def test_two_vessels_noise_bridge_is_not_semantic_merge(self):
        bridge = np.column_stack((np.zeros(160), np.arange(6, 14, .05),
                                  np.zeros(160))).astype(np.float32)
        result, vessels = scene_hypotheses(np.vstack((vessel_cloud(0, 0),
                                                     vessel_cloud(0, 20), bridge)))
        self.assertEqual(len(vessels), 2)
        self.assertEqual(result["target_vessel_status"], "NOT_SELECTED")

    @unittest.expectedFailure
    def test_height_provider_miss_keeps_raw_proposal(self):
        segments, edges, _ = rectangle_case()
        result = solve(raw_scene(), segments, edges, [], CONFIG)
        self.assertGreater(len(result["rectangles"]), 0)

    @unittest.expectedFailure
    def test_three_strong_one_occluded_is_partial_not_silent(self):
        segments, edges, regions = rectangle_case()
        points = raw_scene()
        # One side has only a short measured transition and no 3D edge.
        points[(points[:, 0] <= .3) &
               ((points[:, 1] < 8) | (points[:, 1] > 12)), 2] = 0
        trace = {}
        result = solve(points, segments[:3], edges[:3], regions, CONFIG,
                       forensic_trace=trace)
        self.assertTrue(any(
            row["reject_reason"] == "FINAL_FOUR_SIDE_ACCEPTANCE_GATE"
            and row["strong_side_count"] == 3
            for row in trace["candidate_attempts"]
        ))
        self.assertTrue(any(row["status"] == "PARTIAL_HATCH" for row in result["rectangles"]))

    def test_two_edge_ambiguous_is_not_confirmed(self):
        segments, edges, regions = rectangle_case()
        result = solve(np.column_stack((raw_scene()[:, :2],
                                        np.zeros(len(raw_scene())))),
                       segments[:2], edges[:2], regions, CONFIG)
        self.assertFalse(any(row["status"] == "PROVISIONAL_RECTANGLE"
                             for row in result["rectangles"]))

    def test_crossbeam_trap_is_unresolved_without_opening_evidence(self):
        # A thin structural line is not a closed opening even if rectilinear.
        segments, edges, regions = rectangle_case()
        result = solve(np.column_stack((raw_scene()[:, :2],
                                        np.ones(len(raw_scene())))),
                       segments[:2], edges[:2], regions, CONFIG)
        self.assertFalse(any(row["status"] == "PROVISIONAL_RECTANGLE"
                             for row in result["rectangles"]))

    @unittest.expectedFailure
    def test_measured_crossbeam_strip_is_not_numbered_as_hatch(self):
        # A 2 m raised strip *inside* a larger opening has four measured
        # rectilinear sides, but its interior is steel, not an opening.
        segments, edges, regions = rectangle_case()
        for row in segments:
            row["profile_break_positions"] = [
                [x, 9 + y * 2 / 20] for x, y in row["profile_break_positions"]]
        for row in edges:
            row["a_raw"][1] = 9 + row["a_raw"][1] * 2 / 20
            row["b_raw"][1] = 9 + row["b_raw"][1] * 2 / 20
        regions[0]["bbox_xy"] = [0, 9, 40, 11]
        points = raw_scene()
        points[(points[:, 1] > 9) & (points[:, 1] < 11) &
               (points[:, 0] > 0) & (points[:, 0] < 40), 2] = 1
        result = solve(points, segments, edges, regions, CONFIG)
        self.assertFalse(result["rectangles"])

    def test_no_cross_vessel_axis_leak(self):
        _, vessels = scene_hypotheses(np.vstack((vessel_cloud(0, 0),
                                                 vessel_cloud(0, 23, angle_deg=20))))
        self.assertEqual(len(vessels), 2)
        self.assertGreater(abs(vessels[0]["local_axes"][0][1] -
                               vessels[1]["local_axes"][0][1]), .25)

    @unittest.expectedFailure
    def test_no_cross_vessel_width_leak(self):
        rectangles = [strong_hatch(20, "A", 0), strong_hatch(20, "B", 80)]
        target, _, _ = _ship_width_consensus(rectangles, np.eye(2), CONFIG)
        self.assertIsNone(target)


if __name__ == "__main__":
    unittest.main()
