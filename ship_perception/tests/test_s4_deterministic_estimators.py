"""Spatial hypotheses preserve modes and never sample random array slots."""

import unittest
from unittest.mock import patch
import numpy as np
from ship_perception.r1_static.run import resolve_config
from ship_perception.r1_static.model import PerimeterSegment
from ship_perception.r1_static.perimeter_deck import estimate_segment_deck
from ship_perception.r1_static.boundary_refinement import _fit_line
from ship_perception.r1_static.boundary_topology import (
    structural_axis_estimate,
    orientation_search_budget,
)


class S4DeterministicEstimators(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()

    def test_spatial_planes_keep_two_broad_observed_modes(self):
        x, y = np.meshgrid(np.linspace(0, 4, 35), np.linspace(0.1, 2, 25))
        xy = np.c_[x.ravel(), y.ravel()]
        raw = np.vstack((np.c_[xy, np.zeros(len(xy))], np.c_[xy, np.ones(len(xy))]))
        segment = PerimeterSegment(
            "S", "H", (0.0, 0.0), (4.0, 0.0), (1.0, 0.0), (0.0, 1.0), 4.0, "UNKNOWN"
        )
        with patch(
            "numpy.random.default_rng",
            side_effect=AssertionError("Random plane hypotheses forbidden"),
        ):
            deck, _, private = estimate_segment_deck(raw, segment, self.config)
        self.assertEqual(deck["status"], "SEGMENT_DECK_AMBIGUOUS")
        self.assertEqual(deck["plane_candidate_count"], 2)
        self.assertEqual(
            sorted(round(z, 8) for z in deck["candidate_plane_heights"]), [0.0, 1.0]
        )
        self.assertTrue(
            all(
                len(ids) >= self.config["geometry"]["min_plane_points"]
                for ids in private["candidate_raw_ids"]
            )
        )

    def test_deterministic_pairs_keep_huber_line_under_permutation(self):
        xy = np.c_[np.linspace(-4, 4, 60), 0.02 * np.linspace(-4, 4, 60)]
        xy = np.vstack((xy, [[0, 2], [3, -3], [-2, 4]]))
        with patch(
            "numpy.random.default_rng",
            side_effect=AssertionError("Random pairs forbidden"),
        ):
            a = _fit_line(xy, self.config)
            b = _fit_line(xy[::-1], self.config)
        for key in ("a_xy", "b_xy"):
            np.testing.assert_allclose(a[key], b[key], atol=1e-12)
        self.assertEqual(a["support_count"], 60)

    def test_axis_is_orthogonal_and_low_concentration_is_ambiguous(self):
        def edge(angle, node):
            t = np.radians(angle)
            return dict(
                a_raw=[0, 0, 0],
                b_raw=[np.cos(t), np.sin(t), 0],
                observed_support_length=1.0,
                node_id=node,
            )

        estimate = structural_axis_estimate(
            [edge(2, "A"), edge(92, "B"), edge(182, "C")], 10
        )
        self.assertEqual(estimate["status"], "AXIS_ESTIMATED")
        self.assertAlmostEqual(np.dot(estimate["axis_a"], estimate["axis_b"]), 0.0)
        self.assertAlmostEqual(np.degrees(np.arctan2(*estimate["axis_a"][::-1])), 2.0)
        self.assertEqual(
            structural_axis_estimate([edge(0, "A"), edge(45, "B")], 10)["status"],
            "AXIS_AMBIGUOUS",
        )

    def test_last_node_edge_fades_continuously_instead_of_casting_one_full_vote(self):
        def edge(angle, node, support):
            t = np.radians(angle)
            return dict(
                a_raw=[0, 0, 0],
                b_raw=[np.cos(t), np.sin(t), 0],
                observed_support_length=support,
                node_id=node,
            )

        fixed = [edge(0, "A", 4.0), edge(90, "B", 4.0)]
        present = structural_axis_estimate(fixed + [edge(5, "C", 1e-8)], 10)
        absent = structural_axis_estimate(fixed, 10)
        np.testing.assert_allclose(present["axis_a"], absent["axis_a"], atol=1e-9)

    def test_orientation_budget_uses_region_lever_arm(self):
        estimate = dict(axis_uncertainty_deg=1.0)
        short = orientation_search_budget(estimate, (0, 0, 2, 1), 4.0, 0.15)
        long = orientation_search_budget(estimate, (0, 0, 100, 20), 4.0, 0.15)
        self.assertGreater(
            long["predicted_lateral_error_m"], short["predicted_lateral_error_m"]
        )
        self.assertEqual(
            long["precision_orientation_status"], "UNRESOLVED_ORIENTATION_BUDGET"
        )


if __name__ == "__main__":
    unittest.main()
