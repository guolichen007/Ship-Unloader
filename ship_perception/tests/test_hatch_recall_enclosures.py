"""Deep, cargo-divided basins must retain complete measured enclosure seeds."""

import json
import unittest

import numpy as np

from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG
from ship_perception.r1_static.heightmap_topology import detect_regions
from ship_perception.r1_static.corner_topology import (
    _strip_relation,
    _section_consensus,
)
from ship_perception.r1_static.raw_topview_evidence import build_topview
from ship_perception.r1_static.scene_vessel import infer_scene_vessels, _enclosed_trough
from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.static_showcase import _review_limits
from ship_perception.r1_static.run import resolve_config


CONFIG, _ = resolve_config()
RESEARCH = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))


def deep_basins(separator=False, crop=False):
    x, y = np.meshgrid(np.arange(0, 100, 0.15), np.arange(0, 30, 0.15))
    z = np.full(x.shape, -5.0)
    for left, right in ((10, 45), (50, 85)):
        inside = (x > left) & (x < right) & (y > 5) & (y < 23)
        z[inside] = -14 + 2 * np.exp(-(((y[inside] - 14) / 2) ** 2))
        if separator:
            z[inside & (y >= 13) & (y <= 15)] = -5
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
    return points[points[:, 0] > 20] if crop else points


class HatchRecallEnclosures(unittest.TestCase):
    def test_cargo_ridge_does_not_split_one_vessel_into_longitudinal_strips(self):
        points = deep_basins()
        scene, private = infer_scene_vessels(points, CONFIG, RESEARCH)
        vessels = [
            v
            for v in scene["vessel_hypotheses"]
            if v["classification_status"] == "VESSEL_HYPOTHESIS"
        ]
        self.assertEqual(len(vessels), 1)
        self.assertFalse(vessels[0]["split_seams"])
        self.assertEqual(
            len(private["vessel_point_indexes"][vessels[0]["vessel_hypothesis_id"]]),
            len(points),
        )

    def test_longitudinal_caps_do_not_mistake_open_water_for_an_enclosed_hold(self):
        points = deep_basins()
        # End caps in U remain, but the low return extends out of the V crop.
        # Such a trough must remain eligible as a scene ownership boundary.
        points[(points[:, 0] > 10) & (points[:, 0] < 85) & (points[:, 1] > 5), 2] = -14
        grid = R1HeightGrid.from_points(points, CONFIG["geometry"]["coarse_voxel_m"])
        central = np.ones(grid.count.shape[1], dtype=bool)
        valley = int((14 - grid.y0) / grid.cell_m)
        self.assertFalse(_enclosed_trough(grid, valley, -5, central, CONFIG))

    def test_local_ring_recovers_full_basins_above_global_low_modes(self):
        result, _ = detect_regions(deep_basins(), CONFIG, RESEARCH)
        # Global histogram truncates the remote higher family; all selected
        # low components are smaller than the actual enclosing depressions.
        self.assertTrue(all(level < -10 for level in result["levels_m"]))
        self.assertEqual(result["region_count"], 4)
        proposals = result["local_enclosure_proposals"]
        self.assertEqual(len(proposals), 2)
        np.testing.assert_allclose(
            [p["bbox_xy"] for p in proposals], [[10, 5, 45, 23], [50, 5, 85, 23]]
        )
        self.assertTrue(all(p["height_rise_side_count"] == 4 for p in proposals))
        self.assertTrue(
            all(
                p["proposal_source"] == "LOCAL_RING_HEIGHT_ENCLOSURE" for p in proposals
            )
        )
        self.assertFalse(any("polygon_xy" in p for p in proposals))

    def test_measured_deck_separator_is_not_bridged_by_a_local_level(self):
        result, _ = detect_regions(deep_basins(separator=True), CONFIG, RESEARCH)
        self.assertFalse(result["local_enclosure_proposals"])

    def test_missing_end_support_does_not_create_a_complete_enclosure(self):
        result, _ = detect_regions(deep_basins(crop=True), CONFIG, RESEARCH)
        self.assertFalse(
            any(p["bbox_xy"][0] < 45 for p in result["local_enclosure_proposals"])
        )

    def test_enclosure_proposals_are_translation_and_order_invariant(self):
        points = deep_basins()
        first, _ = detect_regions(points, CONFIG, RESEARCH)
        moved, _ = detect_regions((points + [100, 30, 9])[::-1], CONFIG, RESEARCH)
        np.testing.assert_allclose(
            np.asarray([p["bbox_xy"] for p in first["local_enclosure_proposals"]])
            + [100, 30, 100, 30],
            [p["bbox_xy"] for p in moved["local_enclosure_proposals"]],
            atol=0.001,
        )

    def test_review_viewport_includes_owned_support_without_a_detected_box(self):
        axial = np.array([[-50, 0], [50, 25], [200, 100]])
        private = dict(vessel_point_indexes={"A": [0, 1]})
        polygon = np.array([[0, 5], [40, 5], [40, 20], [0, 20]])
        limits = _review_limits(
            axial, private, [dict(vessel_hypothesis_id="A")], [(None, polygon)], 8
        )
        self.assertEqual(limits, (-58, 58, -8, 33))

    def test_dense_deck_and_deep_hold_are_a_transition_not_a_cargo_ridge(self):
        grid = build_topview(deep_basins(), np.eye(2), CONFIG)["fine"]
        for rho, sign in ((5, 1), (23, -1)):
            relation = _strip_relation(grid, rho, (12, 43), 1, sign, CONFIG)
            self.assertFalse(relation["opening_one_sided"])
            self.assertTrue(relation["opening_height_transition"])
            self.assertFalse(relation["interior_ridge"])
            sections = _section_consensus(grid, rho, (12, 43), 1, sign, CONFIG)
            self.assertEqual(sections["transition_count"], 5)
            self.assertEqual(sections["internal_ridge_section_count"], 0)

    def test_reversed_height_step_and_cargo_crest_are_not_opening_transitions(self):
        grid = build_topview(deep_basins(), np.eye(2), CONFIG)["fine"]
        reversed_side = _strip_relation(grid, 5, (12, 43), 1, -1, CONFIG)
        self.assertFalse(reversed_side["opening_height_transition"])
        self.assertTrue(reversed_side["interior_ridge"])
        crest = _strip_relation(grid, 14, (12, 43), 1, 1, CONFIG)
        self.assertFalse(crest["opening_height_transition"])
        self.assertTrue(crest["interior_ridge"])


if __name__ == "__main__":
    unittest.main()
