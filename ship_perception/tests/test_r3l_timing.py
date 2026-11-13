import copy
import unittest
from unittest.mock import patch

from ship_perception.perception.e2e_timing import ReviewTiming


class GeometryReadiness(unittest.TestCase):
    def test_readiness_precedes_publication_and_preserves_unknown_z_and_payload(self):
        polygon = [[0.0, 0.0], [8.0, 0.0], [8.0, 3.0], [0.0, 3.0]]
        review = [[1.0, 0.0], [8.0, 0.0], [8.0, 3.0], [1.0, 3.0]]
        geometry = dict(
            boundary_xy=polygon, corners_xyz=[[x, y, 4.0] for x, y in polygon]
        )
        proposed = dict(
            state="REVIEW_ESTIMATED",
            adopted_for_review=True,
            boundary_xy=review,
            corners_xyz=[[x, y, None] for x, y in review],
        )
        payload = dict(
            geometry_only=dict(
                selected_opening_cells=[
                    dict(
                        cell_id="a",
                        vessel_id="v",
                        strict_rectangle_state="INFEASIBLE",
                        geometry=geometry,
                        review_geometry=proposed,
                    )
                ]
            )
        )
        frozen = copy.deepcopy(payload)
        observer = ReviewTiming(0.0, 1.0, 3.0, "full", "sha", 1000)
        with patch(
            "ship_perception.perception.e2e_timing.time.perf_counter",
            side_effect=[9.0, 10.0, 17.0],
        ):
            observer.geometry_ready(payload)
            ready = observer.report()
            self.assertFalse(ready["publication_complete"])
            observer.published()
        report = observer.report()
        self.assertEqual(report["read_start_to_geometry_ready_s"], 9.0)
        self.assertEqual(report["read_start_to_publication_s"], 16.0)
        self.assertEqual(report["geometry_ready_to_publication_s"], 7.0)
        self.assertEqual(report["objects"][0]["length_m"], 7.0)
        self.assertTrue(all(p[2] is None for p in report["objects"][0]["corners_xyz"]))
        self.assertEqual(payload, frozen)

    def test_missing_geometry_remains_unknown_and_is_not_a_zero_size_success(self):
        observer = ReviewTiming(0.0, 1.0, 2.0, "preview", "sha", 3)
        observer.geometry_ready(
            dict(
                geometry_only=dict(
                    selected_opening_cells=[dict(cell_id="a", geometry=None)]
                )
            )
        )
        report = observer.report()
        self.assertFalse(report["is_full_phase"])
        self.assertIsNone(report["objects"][0]["length_m"])
        self.assertEqual(report["objects"][0]["state"], "UNRESOLVED")
