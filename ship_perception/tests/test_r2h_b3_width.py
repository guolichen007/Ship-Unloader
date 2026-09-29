"""B3 vessel-width isolation and sustained parallel-mode counterexamples."""

import unittest

import numpy as np

from ship_perception.r1_static.raw_topview_evidence import build_topview
from ship_perception.r1_static.rectangle_refinement import (
    _local_rho_consensus, _mode_profile, _rho_mode_clusters,
    _select_joint, _ship_width_consensus,
)
from ship_perception.tests.test_r1_boundary_topology import CONFIG


def strong_hatch(width, vessel_id, offset):
    return dict(vessel_hypothesis_id=vessel_id,
                polygon_xy=[[offset, 0], [offset + 40, 0],
                            [offset + 40, width], [offset, width]],
                sides=[dict(side_id=side, structural_coverage=1.0,
                            raster_coverage=1.0)
                       for side in ("U0", "U1", "V0", "V1")])


class VesselWidthIsolation(unittest.TestCase):
    def test_two_vessels_same_width_do_not_form_a_shared_prior(self):
        rows = [strong_hatch(20, "A", 0), strong_hatch(20, "B", 80)]
        self.assertIsNone(_ship_width_consensus(rows, np.eye(2), CONFIG, "A")[0])
        self.assertIsNone(_ship_width_consensus(rows, np.eye(2), CONFIG, "B")[0])

    def test_two_vessels_different_width_do_not_form_a_shared_prior(self):
        rows = [strong_hatch(20, "A", 0), strong_hatch(14, "B", 80)]
        self.assertIsNone(_ship_width_consensus(rows, np.eye(2), CONFIG, "A")[0])
        self.assertIsNone(_ship_width_consensus(rows, np.eye(2), CONFIG, "B")[0])

    def test_parallel_vessels_can_consense_only_within_one_identity(self):
        rows = [strong_hatch(20, "A", 0), strong_hatch(20, "A", 45),
                strong_hatch(20, "B", 90)]
        self.assertEqual(_ship_width_consensus(rows, np.eye(2), CONFIG, "A")[0], 20)
        self.assertIsNone(_ship_width_consensus(rows, np.eye(2), CONFIG, "B")[0])

    def test_close_vessels_cannot_share_width_or_missing_identity(self):
        rows = [strong_hatch(20, "A", 0), strong_hatch(20, "B", 1)]
        self.assertIsNone(_ship_width_consensus(rows, np.eye(2), CONFIG, "A")[0])
        self.assertIsNone(_ship_width_consensus(rows, np.eye(2), CONFIG)[0])
        unknown = [strong_hatch(20, "", 0), strong_hatch(20, "", 45)]
        self.assertIsNone(_ship_width_consensus(unknown, np.eye(2), CONFIG, "")[0])

    def test_width_prior_cannot_invent_an_unmeasured_mode(self):
        def side(rho, coverage):
            return dict(rho_refined=rho, role="INNER_EDGE",
                        structural_coverage=coverage, raster_coverage=coverage,
                        opening_side_relation=dict(canonical_inner=True),
                        uncertainty_m=0.05)

        rows = [[side(0, .8)], [side(40, .8)], [side(0, .8)],
                [side(20, .8), side(21, 0)]]
        selected, used = _select_joint(rows, (0, 40, 0, 20), CONFIG, 21)
        self.assertFalse(used)
        self.assertEqual(selected[3]["rho_refined"], 20)


class WeakSideModeConflict(unittest.TestCase):
    def test_neighbor_hatch_line_outside_local_span_cannot_support_side(self):
        mode = dict(rho=4.0, along_min=40.0, along_max=60.0,
                    along_values=np.arange(40, 60, .15).tolist(),
                    evidence_types=["HEIGHT_TRANSITION"])
        channels, structural, raster = _mode_profile(
            [mode], [], 1, 4.0, (0, 30), CONFIG)
        self.assertEqual((structural, raster), (0, 0))
        self.assertEqual(channels["HEIGHT_TRANSITION"], 0)

    def test_small_intermediate_rho_gaps_do_not_merge_distant_rims(self):
        values = [2.00, 2.01, 2.12, 2.42, 2.56, 2.58, 2.72]
        clusters = _rho_mode_clusters(values, CONFIG)
        self.assertGreaterEqual(len(clusters), 2)
        self.assertLess(clusters[0]["rho_median"], 2.2)
        self.assertGreater(clusters[-1]["rho_median"], 2.4)

    def test_h001_v0_historical_mode_jump_is_rejected_when_two_modes_persist(self):
        x, y = np.meshgrid(np.arange(0, 31, .15), np.arange(0, 8, .15))
        # The two sustained measured ridges mirror the 08-01 V0 forensic
        # modes. Width is intentionally absent from all evidence generation.
        boundary = np.where(x < 18, 4.073, 4.823)
        points = np.column_stack((x.ravel(), y.ravel(),
                                  (y >= boundary).ravel().astype(float)))
        grid = build_topview(points, np.eye(2), CONFIG)["fine"]

        def mode(rho, start, end):
            return dict(rho=rho, along_min=start, along_max=end,
                        along_values=np.arange(start, end, .15).tolist(),
                        evidence_types=["HEIGHT_TRANSITION"])

        candidates = [dict(rho_grid=rho, role="INNER_EDGE")
                      for rho in (4.073, 4.823)]
        result = _local_rho_consensus(
            candidates, [mode(4.073, 0, 18), mode(4.823, 18, 30)], [],
            grid, points, 1, 1, 3.975, (0, 30), CONFIG,
        )
        self.assertTrue(result["mode_conflict"], result)
        self.assertFalse(result["confident"], result)
        self.assertGreaterEqual(len(result["rho_mode_clusters"]), 2)


if __name__ == "__main__":
    unittest.main()
