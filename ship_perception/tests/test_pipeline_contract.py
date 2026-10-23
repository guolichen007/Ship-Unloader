"""Verify reuse at the actual pipeline boundary, plus per-hatch axis isolation."""

import unittest
from unittest import mock

import numpy as np

from ship_perception.io.frame import PointCloudFrame
from ship_perception.perception.pipeline import run_perception
from ship_perception.perception.output import auxiliary_review
from ship_perception.perception.context import infer_scene_vessels
from ship_perception.tests.test_hatch_metamorphic import vessel_points


class PipelineContracts(unittest.TestCase):
    def test_real_b1_and_structural_provider_reuse_with_legacy_enabled(self):
        xyz, _ = vessel_points(1, spacing=0.12)
        frame = PointCloudFrame(xyz, None, "map", 0, 0, "", ("x", "y", "z"), len(xyz))
        with mock.patch(
            "ship_perception.perception.context.infer_scene_vessels",
            wraps=infer_scene_vessels,
        ) as b1, mock.patch(
            "ship_perception.r1_static.corner_topology_review.infer_scene_vessels",
            side_effect=AssertionError("B1 repeated by legacy consumer"),
        ):
            result = run_perception(frame, legacy_evidence=True)
        self.assertEqual(b1.call_count, 1)
        self.assertGreater(result["geometry_only"]["hatch_count"], 0)
        for vessel in result["per_vessel_status"]:
            cache = vessel.get("provider_cache")
            if cache:
                self.assertEqual(cache["raw3d_build_count"], 1)
                self.assertEqual(cache["bev_build_count"], 1)
                self.assertTrue(
                    all(n == 1 for n in cache["grid_build_counts"].values())
                )

    def test_intensity_scorer_is_keyed_by_each_immutable_hatch_axis(self):
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
        frame = PointCloudFrame(
            xyz,
            np.array([1.0, 2.0, 3.0]),
            "map",
            0,
            0,
            "",
            ("x", "y", "z", "intensity"),
            3,
        )
        yaw = 0.3
        rotated = [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]
        identity = np.eye(2).tolist()
        hatches = [
            dict(
                vessel_hypothesis_id="V0",
                vessel_local_axes=a,
                boundary_xy=[[0, 0], [1, 0], [1, 1], [0, 1]],
            )
            for a in (rotated, identity, rotated)
        ]
        payload = dict(
            geometry_only=dict(hatches=hatches),
            intensity_status=dict(status="AVAILABLE"),
        )
        with mock.patch(
            "ship_perception.perception.output.IntensityEvidence"
        ) as scorer:
            scorer.return_value.edge.return_value = dict(
                intensity_edge_score=0.5, intensity_used=True
            )
            output = auxiliary_review(payload, frame, {"V0": np.arange(3)})
        self.assertEqual(scorer.call_count, 2)
        np.testing.assert_array_equal(scorer.call_args_list[0].args[2], rotated)
        np.testing.assert_array_equal(scorer.call_args_list[1].args[2], identity)
        self.assertEqual(
            [h["boundary_xy"] for h in output["hatches"]],
            [h["boundary_xy"] for h in hatches],
        )


if __name__ == "__main__":
    unittest.main()
