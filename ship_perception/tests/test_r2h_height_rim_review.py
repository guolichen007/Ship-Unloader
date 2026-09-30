"""Raw XYZ height-review counterexamples; no steel ground truth is inferred."""

import unittest
from unittest.mock import patch

import numpy as np

from ship_perception.r1_static.height_rim_review import (
    _guard_review_displacement, _section_transition, refine_height_review,
)
from ship_perception.r1_static.run import DEFAULT_CONFIG, resolve_config


CONFIG, _ = resolve_config(DEFAULT_CONFIG)


class SyntheticGrid:
    cell_m = 0.15

    def __init__(self, slope_deg=0, occupied_outside=True):
        self.u, self.v = np.meshgrid(np.arange(0, 42, self.cell_m),
                                     np.arange(0, 22, self.cell_m))
        shift = np.tan(np.radians(slope_deg)) * self.u
        interior = ((self.v > 4 + shift) & (self.v < 16 + shift))
        self.median = np.where(interior, -4.0, 0.0)
        # An interior cargo patch must not win a boundary search.
        self.median[(self.v > 9) & (self.v < 12)] += 2.0
        # A bright exterior strip represents quay/outer structure.
        self.median[self.v < 1] = 8.0
        self.count = np.full(self.median.shape, 6, dtype=int)
        if not occupied_outside:
            self.count[self.v > 16 + shift] = 0

    def centers(self):
        return self.u, self.v


class HeightRimReview(unittest.TestCase):
    def test_underfilled_rectangle_expands_to_persistent_transitions(self):
        grid = SyntheticGrid()
        lower = _section_transition(grid, (0, 40), 5.2, "V0", CONFIG)
        upper = _section_transition(grid, (0, 40), 14.8, "V1", CONFIG)
        self.assertTrue(lower["accepted"])
        self.assertTrue(upper["accepted"])
        self.assertAlmostEqual(np.median(np.asarray(lower["samples"])[:, 1]), 4.0,
                               delta=.4)
        self.assertAlmostEqual(np.median(np.asarray(upper["samples"])[:, 1]), 16.0,
                               delta=.4)

    def test_missing_outside_returns_diagnostic_only(self):
        grid = SyntheticGrid(occupied_outside=False)
        upper = _section_transition(grid, (0, 40), 15.0, "V1", CONFIG)
        self.assertFalse(upper["accepted"])

    def test_one_ship_axis_shared_after_drift(self):
        grid = SyntheticGrid(slope_deg=1.0)
        polygon = [[0, 5], [40, 5], [40, 16.2], [0, 16.2]]
        rectangles = [dict(hatch_id=ident, polygon_after=polygon,
                           side_stage_arbitration={}) for ident in ("H000", "H001")]
        vessel = dict(vessel_hypothesis_id="V0000", rectangles=rectangles)
        b4 = dict(rectangles=[dict(vessel_hypothesis_id="V0000",
                                   axes=np.eye(2).tolist())])
        with patch("ship_perception.r1_static.height_rim_review.build_topview",
                   return_value={"fine": grid}):
            result = refine_height_review(np.empty((0, 3)), vessel, {}, b4, CONFIG)
        self.assertAlmostEqual(result["axis_delta_deg"], 1.0, delta=.3)
        self.assertEqual(result["rectangles"]["H000"]["axes"],
                         result["rectangles"]["H001"]["axes"])
        self.assertEqual(result["status"], "HEIGHT_TRANSITION_REVIEW_NOT_STEEL")

    def test_b4_side_cannot_jump_to_remote_outer_transition(self):
        detail = dict(accepted=True, samples=[[2, .7, 2], [20, .7, 2],
                                              [38, .7, 2]])
        guarded = _guard_review_displacement(detail, 1.5, "V0", False, CONFIG)
        self.assertFalse(guarded["accepted"])
        self.assertEqual(guarded["reason"],
                         "LARGE_MOVE_REQUIRES_STRUCTURAL_CORROBORATION")


if __name__ == "__main__":
    unittest.main()
