"""R2H-A red tests: current stage failures, not relaxed acceptance gates.

Expected failures remain visible until R2H-B provides an evidence-backed scene
and proposal layer. Passing tests assert that ambiguity stays unconfirmed.
"""

import unittest

import numpy as np
from scipy import ndimage

from ship_perception.r1_static.boundary_topology import structural_axes
from ship_perception.r1_static.heightmap_topology import _vessel_support
from ship_perception.r1_static.heightmap_topology import detect_regions
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG
from ship_perception.r1_static.r2h_scene_forensics import axis_families
from ship_perception.r1_static.rectangle_fusion import solve
from ship_perception.r1_static.rectangle_refinement import _ship_width_consensus
from ship_perception.tests.test_r1_boundary_topology import CONFIG, rectangle_case
from ship_perception.tests.test_r1_rectangle_fusion import raw_scene
import json


def islands(*, count=2, bridge=False, wharf=False):
    valid = np.zeros((24, 75), dtype=bool)
    for index in range(count):
        valid[5:16, 4 + 22 * index:17 + 22 * index] = True
    if bridge:
        valid[10, 17:26] = True
    if wharf:
        valid[17:23, 0:75] = True
    return valid


def edge(angle_deg, node, x):
    angle = np.radians(angle_deg)
    return dict(node_id=node, a_raw=[x, 0, 0],
                b_raw=[x + 15 * np.cos(angle), 15 * np.sin(angle), 0],
                observed_support_length=15.0)


def strong_hatch(width, vessel_id, x):
    polygon = [[x, 0], [x + 40, 0], [x + 40, width], [x, width]]
    return dict(vessel_hypothesis_id=vessel_id, polygon_xy=polygon,
                sides=[dict(side_id=key, structural_coverage=1.0, raster_coverage=1.0)
                       for key in ("U0", "U1", "V0", "V1")])


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

    @unittest.expectedFailure
    def test_two_parallel_vessels_are_not_largest_only(self):
        valid = islands()
        self.assertEqual(int((_vessel_support(valid) & valid).sum()), int(valid.sum()))

    @unittest.expectedFailure
    def test_two_different_yaw_vessels_have_two_local_axes(self):
        axes = structural_axes([edge(0, "a", 0), edge(20, "b", 35)], 4)
        self.assertEqual(len(axes), 4)  # two orthogonal pairs required downstream

    @unittest.expectedFailure
    def test_three_vessels_unequal_density_retain_all_support(self):
        valid = islands(count=3)
        valid[5:13, 48:61] = False
        self.assertEqual(int((_vessel_support(valid) & valid).sum()), int(valid.sum()))

    @unittest.expectedFailure
    def test_wharf_larger_than_vessel_cannot_select_target(self):
        valid = islands(wharf=True)
        selected = _vessel_support(valid)
        self.assertTrue(selected[10, 10])
        self.assertFalse(selected[20, 10])

    @unittest.expectedFailure
    def test_one_vessel_split_components_keeps_both_halves(self):
        valid = islands()
        self.assertEqual(int((_vessel_support(valid) & valid).sum()), int(valid.sum()))

    @unittest.expectedFailure
    def test_two_vessels_noise_bridge_is_not_semantic_merge(self):
        valid = islands(bridge=True)
        labels, count = ndimage.label(valid, structure=np.ones((3, 3), bool))
        self.assertEqual(count, 2)

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

    @unittest.expectedFailure
    def test_no_cross_vessel_axis_leak(self):
        edges = [edge(0, "a", 0), edge(18, "b", 30)]
        self.assertEqual(len(axis_families(edges)), 2)
        self.assertEqual(len(structural_axes(edges, 4)), 4)

    @unittest.expectedFailure
    def test_no_cross_vessel_width_leak(self):
        rectangles = [strong_hatch(20, "A", 0), strong_hatch(20, "B", 80)]
        target, _, _ = _ship_width_consensus(rectangles, np.eye(2), CONFIG)
        self.assertIsNone(target)


if __name__ == "__main__":
    unittest.main()
