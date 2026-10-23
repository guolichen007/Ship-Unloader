"""Independent annotation identity, signed coverage and corner evaluation."""

import copy
import unittest

from ship_perception.evaluation.hatch_precision import evaluate
from ship_perception.r1_static.run import resolve_config


class PrecisionEvaluation(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()
        self.manifest = dict(
            schema="ship_perception.hatch_golden.1",
            usage="EVALUATION_ONLY_NEVER_RUNTIME",
            pcd_sha256="a" * 64,
            annotation_version="independent-test.1",
            coordinate_frame="map",
            reviewer="independent reviewer",
            hatches=[dict(prediction_hatch_id="H000", edges={})],
        )
        for name, start, end, normal in (
            ("U0", [0, 0], [0, 8], [1, 0]),
            ("U1", [10, 0], [10, 8], [-1, 0]),
            ("V0", [0, 0], [10, 0], [0, 1]),
            ("V1", [0, 8], [10, 8], [0, -1]),
        ):
            self.manifest["hatches"][0]["edges"][name] = dict(
                start_xy=start,
                end_xy=end,
                inward_normal_xy=normal,
                visibility="VISIBLE",
                ignore_segments=[],
            )
        self.result = dict(
            input=dict(frame_id="map", provenance=dict(wire_sha256="a" * 64)),
            geometry_only=dict(
                hatches=[
                    dict(
                        hatch_id="H000",
                        best_estimate_boundary_xy=[[0, 0], [10, 0], [10, 8], [0, 8]],
                        per_side_uncertainty={
                            name: 0.05 for name in ("U0", "U1", "V0", "V1")
                        },
                    )
                ]
            ),
        )

    def status(self):
        return evaluate(self.manifest, self.result, self.config)["hatches"][0]

    def test_independent_exact_edges_pass(self):
        self.assertEqual(self.status()["status"], "PASS")

    def test_visible_inner_edge_crop_is_top_failure(self):
        self.result["geometry_only"]["hatches"][0]["best_estimate_boundary_xy"][0][
            0
        ] = 0.5
        self.result["geometry_only"]["hatches"][0]["best_estimate_boundary_xy"][3][
            0
        ] = 0.5
        self.assertEqual(self.status()["status"], "UNDER_COVERAGE")
        self.assertEqual(self.status()["edges"]["U0"]["undercoverage_m"], 0.5)

    def test_occluded_not_claimed_precision_pass(self):
        self.manifest["hatches"][0]["edges"]["U0"]["visibility"] = "OCCLUDED"
        self.assertEqual(self.status()["status"], "REVIEW")

    def test_frame_and_raw_sha_are_hard_requirements(self):
        for key, value in (
            ("frame_id", "other"),
            ("provenance", dict(wire_sha256="b" * 64)),
        ):
            result = copy.deepcopy(self.result)
            result["input"][key] = value
            with self.assertRaises(ValueError):
                evaluate(self.manifest, result, self.config)


if __name__ == "__main__":
    unittest.main()
