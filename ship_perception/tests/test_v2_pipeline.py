"""Production V2 executes a lossless union and one shared raw refiner."""

import unittest
from unittest.mock import patch
import numpy as np
from ship_perception.io.frame import PointCloudFrame
from ship_perception.tests.test_hatch_metamorphic import vessel_points
from ship_perception.perception.pipeline_v2 import run_perception
from ship_perception.perception.output import auxiliary_review


class V2Pipeline(unittest.TestCase):
    def run_points(self, points):
        return run_perception(
            PointCloudFrame(
                points,
                np.ones(len(points)),
                "map",
                0,
                0,
                "",
                ("x", "y", "z", "intensity"),
                len(points),
            )
        )

    def test_three_unequal_holds_one_union_one_precision_xyz_only(self):
        xyz, boxes = vessel_points()
        result = self.run_points(xyz)
        self.assertEqual(result["pipeline_health"]["isolated_failures"], [])
        self.assertEqual(result["geometry_only"]["hatch_count"], 3)
        self.assertFalse(result["intensity_status"]["used"])
        profile = result["precision_profiler"]
        self.assertGreaterEqual(
            profile["proposal_count"], profile["physical_query_count"]
        )
        self.assertGreaterEqual(profile["raw_profile_cache_hit"], 1)
        self.assertEqual(profile["normal_points_evaluated"], 0)
        actual = sorted(
            result["geometry_only"]["hatches"],
            key=lambda h: np.asarray(h["boundary_xy"])[:, 0].mean(),
        )
        for hatch, box in zip(actual, boxes):
            corners = np.asarray(hatch["boundary_xy"])
            np.testing.assert_allclose(corners.min(0), box[:2], atol=0.15)
            np.testing.assert_allclose(corners.max(0), box[2:], atol=0.15)
            self.assertFalse(hatch["layout"]["boundary_modified"])
        with patch(
            "ship_perception.perception.output.IntensityEvidence",
            side_effect=AssertionError("Intensity scorer must not be constructed"),
        ):
            fusion = auxiliary_review(
                result,
                PointCloudFrame(
                    xyz,
                    np.arange(len(xyz)),
                    "map",
                    0,
                    0,
                    "",
                    ("x", "y", "z", "intensity"),
                    len(xyz),
                ),
                {},
            )
        self.assertFalse(fusion["intensity_used"])
        self.assertEqual(
            fusion["intensity_fusion_status"], "DISABLED_PRODUCTION_XYZ_ONLY"
        )

    def test_two_unreferenced_vessels_never_enter_experts(self):
        xyz, _ = vessel_points(1)
        points = np.vstack((xyz, xyz + np.array([0.0, 32.0, 0.0])))
        with patch(
            "ship_perception.perception.pipeline_v2.LegacyProposalExpert.propose",
            side_effect=AssertionError("Unselected ship entered expert"),
        ):
            result = self.run_points(points)
        self.assertIsNone(result["berth_selection"]["active_vessel_id"])
        self.assertEqual(result["geometry_only"]["hatch_count"], 0)
        self.assertEqual(
            result["berth_selection"]["status"], "MULTI_VESSEL_BERTH_AMBIGUOUS"
        )
