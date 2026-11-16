"""Physical IR recall and source ownership; truth stays in tests."""

import json
import unittest
import numpy as np

from ship_perception.io.frame import PointCloudFrame
from ship_perception.perception.context import SceneEvidenceContext
from ship_perception.r1_static.run import resolve_config
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG


def scene_points(include_boat=True, background=True):
    step = 0.19
    parts = []
    if include_boat:
        u, v = np.meshgrid(np.arange(0, 96, step), np.arange(0, 20, step))
        z = np.zeros(u.shape)
        walls = []
        for a, b, c, d in [(6, 4, 33, 16), (36, 4, 68, 16)]:
            z[(u > a) & (u < c) & (v > b) & (v < d)] = -3.8
            for x in (a, c):
                y, h = np.meshgrid(np.arange(b, d, step), np.arange(-3.8, 0, step))
                walls.append(np.c_[np.full(y.size, x), y.ravel(), h.ravel()])
            for y in (b, d):
                x, h = np.meshgrid(np.arange(a, c, step), np.arange(-3.8, 0, step))
                walls.append(np.c_[x.ravel(), np.full(x.size, y), h.ravel()])
        parts.extend([np.c_[u.ravel(), v.ravel(), z.ravel()], *walls])
    if background:
        u, v = np.meshgrid(np.arange(-6, 102, step), np.arange(23, 35, step))
        parts.append(np.c_[u.ravel(), v.ravel(), np.full(u.size, -6.0)])
        u, v = np.meshgrid(np.arange(-6, 102, step), np.arange(35, 42, step))
        parts.append(np.c_[u.ravel(), v.ravel(), np.full(u.size, 2.0)])
        u, h = np.meshgrid(np.arange(-6, 102, step), np.arange(-6, 2, step))
        parts.append(np.c_[u.ravel(), np.full(u.size, 35.0), h.ravel()])
    return np.vstack(parts) + [13.271, -27.349, 1.613]


class UnselectedRawIRContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ship_perception.perception.unselected_review_ir import (
            unselected_opening_ir,
        )

        cls.run_ir = staticmethod(unselected_opening_ir)
        cls.config, cls.config_sha = resolve_config()
        cls.research = json.loads(RESEARCH_CONFIG.read_bytes())

    def execute(self, points):
        frame = PointCloudFrame(
            points, None, "synthetic", 0, 0, "", ("x", "y", "z"), len(points)
        )
        scene = SceneEvidenceContext.build(
            frame, self.config, self.research, self.config_sha, "SYNTHETIC_TEST"
        )
        return scene, self.run_ir(frame, scene, self.config, self.research)

    def assert_source_and_safety(self, scene, records):
        owners = {v.vessel_id: set(v.point_indexes.tolist()) for v in scene.vessels}
        for key, record in records.items():
            self.assertFalse(record["main_review_visible"])
            self.assertFalse(record["control_eligible"])
            self.assertFalse(record["steel_certified"])
            for candidate in record["candidates"]:
                self.assertEqual(
                    candidate["identity_state"], "UNRESOLVED_VESSEL_SELECTION"
                )
                self.assertFalse(candidate["main_review_visible"])
                self.assertFalse(candidate["control_eligible"])
                self.assertFalse(candidate["steel_certified"])
                self.assertNotIn("corners_xyz", candidate)
                ids = candidate["source_point_indexes"]
                self.assertTrue(ids)
                self.assertTrue(set(ids).issubset(owners[key]))
                self.assertEqual(candidate["raw_point_count"], len(ids))

    def test_true_openings_with_quay_keep_own_raw_candidates_without_selecting_berth(
        self,
    ):
        scene, records = self.execute(scene_points())
        self.assertEqual(sum(len(v["candidates"]) for v in records.values()), 2)
        self.assert_source_and_safety(scene, records)

    def test_water_quay_and_single_wall_without_opening_do_not_create_ir_openings(self):
        scene, records = self.execute(scene_points(include_boat=False))
        self.assertEqual(sum(len(v["candidates"]) for v in records.values()), 0)
        self.assert_source_and_safety(scene, records)

    def test_two_boats_retain_disjoint_raw_ownership_and_no_active_target(self):
        first = scene_points(background=False)
        scene, records = self.execute(np.vstack([first, first + [2, 40, -1]]))
        self.assertEqual(sum(len(v["candidates"]) for v in records.values()), 4)
        self.assert_source_and_safety(scene, records)
        groups = [
            set(i for c in r["candidates"] for i in c["source_point_indexes"])
            for r in records.values()
            if r["candidates"]
        ]
        self.assertEqual(len(groups), 2)
        self.assertFalse(groups[0] & groups[1])

    def test_offline_pipeline_retains_ir_without_creating_main_frames(self):
        from ship_perception.perception.review_pipeline import run_perception

        points = scene_points()
        frame = PointCloudFrame(
            points, None, "synthetic", 0, 0, "", ("x", "y", "z"), len(points)
        )
        result = run_perception(
            frame, mode="review-first", opening_topology_v2=True, local_rim_review=True
        )
        self.assertIsNone(result["berth_selection"]["active_vessel_id"])
        self.assertEqual(result["geometry_only"]["selected_opening_cells"], [])
        self.assertEqual(result["geometry_only"]["review_opening_count"], 0)
        self.assertEqual(
            sum(
                v["unresolved_candidates"]["total_count"]
                for v in result["per_vessel_status"]
            ),
            2,
        )
        self.assertIn("UNSELECTED_VESSEL_RAW_IR", result["stage_elapsed"])
        self.assertEqual(result["pipeline_health"]["b1_execution_count"], 1)

    def test_bounded_mode_does_not_enter_the_offline_ir_path(self):
        from unittest.mock import patch
        from ship_perception.perception.review_pipeline import run_perception

        points = scene_points()
        frame = PointCloudFrame(
            points, None, "synthetic", 0, 0, "", ("x", "y", "z"), len(points)
        )
        with patch(
            "ship_perception.perception.review_pipeline.unselected_opening_ir"
        ) as ir:
            result = run_perception(frame, mode="bounded", opening_topology_v2=True)
            ir.assert_not_called()
        self.assertEqual(result["review_mode"], "bounded")


if __name__ == "__main__":
    unittest.main()
