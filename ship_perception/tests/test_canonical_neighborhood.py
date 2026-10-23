"""KD-tree order and raw point order cannot change seeded physical sampling."""

import unittest

import numpy as np

from ship_perception.r1_static.boundary_refinement import _fit_line
from ship_perception.r1_static.model import PerimeterSegment
from ship_perception.r1_static.perimeter_deck import estimate_segment_deck
from ship_perception.r1_static.run import resolve_config


class CanonicalNeighborhood(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()

    def test_tree_traversal_and_message_order_do_not_change_plane(self):
        x, y = np.meshgrid(np.arange(0, 10, 0.1), np.arange(-3, 0, 0.1))
        points = np.column_stack(
            (x.ravel(), y.ravel(), (0.003 * x + 0.002 * y).ravel())
        )
        segment = PerimeterSegment(
            "s",
            "seed",
            (0.0, 0.0),
            (10.0, 0.0),
            (1.0, 0.0),
            (0.0, -1.0),
            10.0,
            "VISIBLE",
        )

        class Tree:
            def __init__(self, order):
                self.order = order

            def query_ball_point(self, *args, **kwargs):
                return self.order

        first = estimate_segment_deck(
            points, segment, self.config, Tree(list(range(len(points))))
        )
        second = estimate_segment_deck(
            points, segment, self.config, Tree(list(reversed(range(len(points)))))
        )
        shuffled = points[np.random.default_rng(27).permutation(len(points))]
        third = estimate_segment_deck(
            shuffled, segment, self.config, Tree(list(range(len(points))))
        )
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[0], third[0])
        np.testing.assert_array_equal(
            first[2]["strip_raw_ids"], second[2]["strip_raw_ids"]
        )
        self.assertEqual(first[0]["status"], "RESOLVED")

    def test_line_fit_uses_physical_order(self):
        x = np.linspace(0.0, 10.0, 100)
        xy = np.column_stack((x, 0.02 * x))
        first = _fit_line(xy, self.config)
        second = _fit_line(xy[::-1], self.config)
        np.testing.assert_allclose(first["a_xy"], second["a_xy"], atol=1e-12)
        np.testing.assert_allclose(first["b_xy"], second["b_xy"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
