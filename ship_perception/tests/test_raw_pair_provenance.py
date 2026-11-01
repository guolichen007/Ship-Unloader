"""Raw pairing witnesses retain identity and never certify steel or corner Z."""

import unittest
from types import SimpleNamespace
import numpy as np
from ship_perception.perception.raw_pair_provenance import trace_surface_pairs
from ship_perception.perception.raw_pair_provenance import trace_section_pairs
from ship_perception.perception.measurement import RawSpatialIndex
from ship_perception.r1_static.rectangle_refinement import _raw_edge_rho
from ship_perception.r1_static.run import resolve_config


class RawPairProvenance(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()
        rows = []
        for along in (0.05, 0.35, 0.65, 0.95):
            rows.extend((5.0 + d, along, -3.0) for d in np.linspace(-0.14, -0.02, 8))
            rows.extend((5.0 + d, along, -5.0) for d in np.linspace(0.04, 0.14, 8))
        self.points = np.asarray(rows)
        self.indexes = np.arange(len(rows)) * 7 + 1000
        self.owned = np.arange(len(rows)) % 3 == 0

    def trace(
        self,
        points=None,
        indexes=None,
        owned=None,
        axis=0,
        sign=1,
        rho=5.0,
        span=(0.0, 1.2),
    ):
        return trace_surface_pairs(
            self.points if points is None else points,
            self.indexes if indexes is None else indexes,
            self.owned if owned is None else owned,
            axis,
            rho,
            span,
            sign,
            self.config,
        )

    def test_witnesses_reproduce_existing_primitive_and_original_returns(self):
        row = self.trace()
        expected, _ = _raw_edge_rho(self.points, 0, 5.0, (0.0, 1.2), 1, self.config)
        self.assertEqual(row["raw_midpoint_rho_m"], expected)
        source = {int(i): p for i, p in zip(self.indexes, self.points)}
        for pair in row["pairs"]:
            outer = [source[i][0] for i in pair["outside_point_indexes"]]
            inner = [source[i][0] for i in pair["inside_point_indexes"]]
            self.assertEqual(pair["midpoint_rho_m"], (max(outer) + min(inner)) / 2)
            self.assertAlmostEqual(pair["bracket_width_m"], min(inner) - max(outer))
        self.assertEqual(row["paired_interval_count"], 4)
        self.assertEqual(row["independent_section_vote_count"], 1)

    def test_axis_polarity_and_large_origin_follow_same_raw_primitive(self):
        for axis, sign in ((0, -1), (1, 1), (1, -1)):
            points = self.points.copy()
            if sign < 0:
                points[:, 0] = 10 - points[:, 0]
            if axis:
                points[:, [0, 1]] = points[:, [1, 0]]
            shift = np.array([1024.0, -2048.0, 8192.0])
            points += shift
            rho, span = 5 + shift[axis], (shift[1 - axis], shift[1 - axis] + 1.2)
            row = self.trace(points=points, axis=axis, sign=sign, rho=rho, span=span)
            expected, _ = _raw_edge_rho(points, axis, rho, span, sign, self.config)
            self.assertEqual(row["raw_midpoint_rho_m"], expected)

    def test_tied_duplicate_returns_are_all_retained_after_reordering(self):
        points = np.r_[self.points, self.points[7:8]]
        indexes = np.r_[self.indexes, 90001]
        owned = np.r_[self.owned, False]
        row = self.trace(points=points, indexes=indexes, owned=owned)
        self.assertIn(90001, row["pairs"][0]["outside_point_indexes"])
        order = np.arange(len(points))[::-1]
        self.assertEqual(row, self.trace(points[order], indexes[order], owned[order]))

    def test_context_returns_do_not_become_owned_steel_or_corner_z(self):
        row = self.trace(owned=np.zeros(len(self.points), bool))
        self.assertEqual(row["query_owned_point_count"], 0)
        self.assertEqual(row["raw_midpoint_rho_m"], self.trace()["raw_midpoint_rho_m"])
        self.assertFalse(row["ownership_promoted"])
        self.assertFalse(row["corner_z_source"])
        self.assertFalse(row["decision_input"])
        self.assertFalse(row["steel_certified"])
        self.assertTrue(all(not p["inside_owned_point_indexes"] for p in row["pairs"]))

    def test_missing_height_step_and_missing_pairs_are_not_filled(self):
        points = self.points.copy()
        points[:, 2] = -3.0
        self.assertIsNone(self.trace(points=points)["raw_midpoint_rho_m"])
        points = self.points[:48]
        row = self.trace(
            points=points, indexes=self.indexes[:48], owned=self.owned[:48]
        )
        self.assertEqual(row["status"], "INSUFFICIENT_PAIRED_INTERVALS")
        self.assertIsNone(row["raw_midpoint_rho_m"])

    def test_misaligned_source_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "INDEX_ALIGNMENT"):
            self.trace(indexes=self.indexes[:-1])

    def test_absent_trace_does_not_invent_an_anchor_or_original_indexes(self):
        row = trace_section_pairs(
            SimpleNamespace(), {"rho": 5.0}, 0, (0.0, 1.2), 1, self.config
        )
        self.assertEqual(row["status"], "UNAVAILABLE_ORIGINAL_RAW_PAIR_TRACE_INPUT")
        self.assertFalse(row["decision_input"])
        self.assertEqual(row["source_pair_point_indexes"], [])

    def test_section_trace_uses_original_spatial_indexes_without_moving_measurement(
        self,
    ):
        expected, _ = _raw_edge_rho(self.points, 0, 5.0, (0.0, 1.2), 1, self.config)
        section = {"rho": expected, "section_record": {"rho": 5.0}}
        measurement = SimpleNamespace(
            local=SimpleNamespace(
                vessel=SimpleNamespace(point_indexes=self.indexes[self.owned])
            ),
            raw_uvz=self.points,
            raw_point_indexes=self.indexes,
            spatial=RawSpatialIndex(self.points),
        )
        trace = trace_section_pairs(measurement, section, 0, (0.0, 1.2), 1, self.config)
        self.assertEqual(trace["continuous_rho_difference_m"], 0.0)
        self.assertEqual(section, {"rho": expected, "section_record": {"rho": 5.0}})
        self.assertTrue(set(trace["source_pair_point_indexes"]).issubset(self.indexes))


if __name__ == "__main__":
    unittest.main()
