"""Layout cannot edit geometry, and unknown ownership cannot seed water."""

import copy
import unittest
from types import SimpleNamespace
import numpy as np
from ship_perception.perception.layout import VesselLayoutModel
from ship_perception.perception.validation import ExteriorTopology
from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.run import resolve_config


class LayoutAndExterior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()

    def cache(self):
        u, v = np.meshgrid(np.arange(0, 100, 0.5), np.arange(0, 20, 0.5))
        grid = R1HeightGrid.from_points(
            np.column_stack((u.ravel(), v.ravel(), np.zeros(u.size))), 0.5
        )
        return SimpleNamespace(config=self.config, context_grid=lambda _: grid)

    def test_equal_width_family_is_only_a_ranking_measurement(self):
        rows = [
            dict(bounds=[5, 2, 25, 18], independent_support_sides=4),
            dict(bounds=[30, 2, 55, 18.1], independent_support_sides=4),
        ]
        box = [60, 5, 95, 16]
        original = copy.deepcopy(box)
        model = VesselLayoutModel(self.cache()).fit(rows)
        detail = model.review(box)
        self.assertEqual(box, original)
        self.assertFalse(detail["boundary_modified"])
        self.assertTrue(detail["layout_prior_used"])
        self.assertEqual(detail["anomaly"], "LAYOUT_WIDTH_OUTLIER")
        self.assertEqual(detail["measured_width_m"], 11)

    def test_single_hold_and_weak_rows_cannot_define_width_family(self):
        model = VesselLayoutModel(self.cache()).fit(
            [
                dict(bounds=[5, 2, 95, 18], independent_support_sides=4),
                dict(bounds=[1, 1, 5, 5], independent_support_sides=2),
            ]
        )
        self.assertIsNone(model.width_family)
        self.assertFalse(model.review([5, 2, 95, 18])["layout_prior_used"])

    def test_unknown_low_component_does_not_become_confirmed_water(self):
        u, v = np.meshgrid(np.arange(0, 10, 0.5), np.arange(0, 10, 0.5))
        grid = R1HeightGrid.from_points(
            np.column_stack((u.ravel(), v.ravel(), -np.ones(u.size) * 4)), 0.5
        )
        owned = np.zeros(grid.count.shape, bool)
        owned[4:16, 4:16] = True
        unknown = grid.occupancy & ~owned
        exterior = ExteriorTopology(
            grid,
            owned,
            self.config,
            unresolved_context=unknown,
            boundary_seeds_confirmed=False,
        )
        result = exterior.review([2, 2, 8, 8], 0.0)
        self.assertEqual(result["status"], "UNRESOLVED_EXTERIOR_CONNECTIVITY")
        self.assertFalse(result["known_exterior_low_connected"])
        confirmed = ExteriorTopology(
            grid,
            owned,
            self.config,
            unresolved_context=np.zeros_like(owned),
            boundary_seeds_confirmed=False,
        )
        self.assertEqual(
            confirmed.review([2, 2, 8, 8], 0.0)["status"], "REJECT_EXTERIOR_LOW_REGION"
        )
