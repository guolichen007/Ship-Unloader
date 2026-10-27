"""Corner exports must preserve the full-scene geometry and unresolved meaning."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from ship_perception.perception.coordinates import (
    boundary_dimensions,
    corner_coordinates,
)


def payload():
    return dict(
        schema="ship_perception.hatch_once.2",
        software_git_sha="a" * 40,
        software_source_identity=dict(runtime_source_sha256="b" * 64),
        config_sha256="c" * 64,
        input=dict(frame_id="map", semantic_sha256="d" * 64),
        geometry_only=dict(
            hatch_count_status="AMBIGUOUS",
            hatches=[
                dict(
                    hatch_id="H000",
                    vessel_hypothesis_id="V0",
                    status="PARTIAL_HATCH",
                    geometry_role="REVIEW_NOT_CERTIFIED_STEEL",
                    boundary_xy=[[-12.2, 3.1], [2.3, 3.5], [2.1, 15.4], [-12.4, 15.0]],
                    undercoverage_risk=True,
                    corners=[dict(uncertainty_m=x) for x in (0.1, 0.2, 0.3, 0.4)],
                ),
                dict(
                    hatch_id="H001",
                    vessel_hypothesis_id="V1",
                    status="UNRESOLVED",
                    geometry_role="UNRESOLVED_ENCLOSURE_REVIEW",
                    candidate_type="UNRESOLVED_OPENING_GROUP",
                    individual_hatch_count=None,
                    candidate_blockers=["HATCH_COUNT_AMBIGUOUS"],
                    boundary_xy=[[30, 4], [70, 5], [69, 20], [29, 19]],
                    undercoverage_risk=False,
                ),
            ],
        ),
    )


class CornerExportTests(unittest.TestCase):
    def test_all_hatches_keep_original_global_coordinates_and_semantics(self):
        source = payload()
        before = copy.deepcopy(source)
        result = corner_coordinates(source, "e" * 64)
        self.assertEqual(result["review_candidate_count"], 2)
        self.assertEqual(result["coordinate_frame"], "map")
        self.assertEqual(result["source_result_sha256"], "e" * 64)
        self.assertEqual(result["hatch_count_status"], "AMBIGUOUS")
        self.assertFalse(result["steel_edge_certified"])
        for old, new in zip(source["geometry_only"]["hatches"], result["hatches"]):
            self.assertEqual(new["boundary_xy_m"], old["boundary_xy"])
            self.assertEqual(
                [[p["x_m"], p["y_m"]] for p in new["corners"]], old["boundary_xy"]
            )
            self.assertEqual(new["status"], old["status"])
            self.assertEqual(len(new["corners"]), 4)
            self.assertEqual(
                [point["display_label"] for point in new["corners"]],
                ["P1", "P2", "P3", "P4"],
            )
            self.assertTrue(all(p["z_m"] is None for p in new["corners"]))
            self.assertTrue(all(p["z_status"] == "UNOBSERVED" for p in new["corners"]))
        self.assertIsNone(result["hatches"][1]["individual_hatch_count"])
        self.assertEqual(
            result["hatches"][1]["candidate_type"], "UNRESOLVED_OPENING_GROUP"
        )
        self.assertEqual(source, before)
        json.dumps(result, allow_nan=False)

    def test_dimensions_follow_rotated_final_edges_instead_of_xy_extents(self):
        angle = np.deg2rad(37)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        polygon = np.array([[0, 0], [40, 0], [40, 16], [0, 16]])
        transformed = polygon @ rotation.T + [-103.7, 57.4]
        dimensions = boundary_dimensions(transformed)
        self.assertAlmostEqual(dimensions["length_m"], 40)
        self.assertAlmostEqual(dimensions["width_m"], 16)
        for side, expected in (("V0", 40), ("V1", 40), ("U0", 16), ("U1", 16)):
            self.assertAlmostEqual(dimensions["side_lengths_m"][side], expected)

    def test_nonparallel_measured_edges_keep_both_lengths_and_mean(self):
        # Real line intersections need not form a perfect rectangle.
        dimensions = boundary_dimensions([[0, 0], [40, 0], [39, 16], [1, 16]])
        self.assertAlmostEqual(dimensions["length_m"], 39)
        self.assertAlmostEqual(dimensions["width_m"], np.hypot(1, 16))
        self.assertEqual(dimensions["side_lengths_m"]["V0"], 40)
        self.assertEqual(dimensions["side_lengths_m"]["V1"], 38)

    def test_no_candidates_exports_empty_list(self):
        source = payload()
        source["geometry_only"]["hatches"] = []
        result = corner_coordinates(source, "e" * 64)
        self.assertEqual(result["review_candidate_count"], 0)
        self.assertEqual(result["hatches"], [])

    def test_invalid_or_inconsistent_geometry_is_not_exported_as_four_corners(self):
        for polygon in ([[0, 0]], [[float("nan"), 0]] * 4, [[0, 0, 0]] * 4):
            source = payload()
            source["geometry_only"]["hatches"][0]["boundary_xy"] = polygon
            with self.assertRaisesRegex(ValueError, "INVALID_FOUR_CORNER_COORDINATES"):
                corner_coordinates(source, "e" * 64)
        source = payload()
        source["geometry_only"]["hatches"][0]["best_estimate_boundary_xy"] = [
            [0, 0]
        ] * 4
        with self.assertRaisesRegex(ValueError, "FINAL_BOUNDARY_COORDINATE_MISMATCH"):
            corner_coordinates(source, "e" * 64)

    def test_atomic_output_includes_hash_linked_corner_file(self):
        from ship_perception.io.frame import PointCloudFrame
        from ship_perception.perception.output import save_output
        from ship_perception.perception.pipeline import run_perception, Timing

        points = np.array([[0.0, 0.0, 0.0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        frame = PointCloudFrame(points, None, "map", 0, 0, "", ("x", "y", "z"), 4)
        source = run_perception(frame, legacy_evidence=False)
        with tempfile.TemporaryDirectory() as directory:
            destination = save_output(
                copy.deepcopy(source), frame, directory, False, Timing(None)
            )
            self.assertEqual(
                {p.name for p in destination.iterdir()},
                {
                    "01_geometry_only.png",
                    "02_intensity_fusion.png",
                    "03_results.json",
                    "04_input_summary.txt",
                    "05_hatch_corners.json",
                },
            )
            full = (destination / "03_results.json").read_bytes()
            corners = json.loads((destination / "05_hatch_corners.json").read_text())
            self.assertEqual(
                corners["source_result_sha256"], hashlib.sha256(full).hexdigest()
            )
            self.assertEqual(corners["hatches"], [])
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "ship_perception.perception.coordinates.corner_coordinates",
            side_effect=ValueError("INVALID_FOUR_CORNER_COORDINATES"),
        ):
            with self.assertRaisesRegex(ValueError, "INVALID_FOUR_CORNER_COORDINATES"):
                save_output(
                    copy.deepcopy(source), frame, directory, False, Timing(None)
                )
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
