"""Integer evidence boundary, frozen-provider parity and support provenance."""

import math
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from ship_perception.perception.exact_density_evidence import (
    ExactReviewEvidence,
    exact_bev_provider,
    exact_density_transition,
    exact_raster_modes,
)
from ship_perception.perception.local_evidence import VesselLocalEvidence
from ship_perception.r1_static.hatch_proposal import _bev_provider
from ship_perception.r1_static.raw_topview_evidence import raster_modes
from ship_perception.r1_static.run import resolve_config


class ExactDensityContracts(unittest.TestCase):
    def test_inclusive_measured_ratio_ties_and_one_count_neighbors(self):
        low = np.arange(1, 10000, dtype=np.int32)
        equal = 2 * low + 1
        for a, b in ((low, equal), (equal, low)):
            self.assertTrue(exact_density_transition(a, b, 1).all())
        self.assertFalse(exact_density_transition(low, equal - 1, 1).any())
        self.assertTrue(exact_density_transition(low, equal + 1, 1).all())

    def test_no_new_evidence_from_empty_or_insufficiently_sampled_cells(self):
        np.testing.assert_array_equal(
            exact_density_transition([0, 1, 2, 2], [100, 3, 5, 6], 5),
            [False, False, True, True],
        )

    def test_off_boundary_agrees_with_logarithmic_physical_definition(self):
        a, b = np.meshgrid(np.arange(1, 256), np.arange(1, 256))
        non_ties = np.maximum(a, b) + 1 != 2 * (np.minimum(a, b) + 1)
        expected = np.abs(np.log1p(a) - np.log1p(b)) >= math.log(2.0)
        np.testing.assert_array_equal(
            exact_density_transition(a, b, 1)[non_ties], expected[non_ties]
        )

    def test_int32_grid_counts_do_not_overflow_at_ratio_boundary(self):
        low = np.array([2**30 - 1, 2**30], dtype=np.int32)
        high = np.array([2**31 - 1, 2**31 - 1], dtype=np.int32)
        np.testing.assert_array_equal(
            exact_density_transition(low, high, 1), [True, False]
        )

    def test_fragment_and_provider_semantics_equal_frozen_away_from_ties(self):
        cfg, _ = resolve_config()
        rng = np.random.RandomState(72)
        # Multiples of four cannot make (high+1)/(low+1) exactly two.
        count = rng.randint(0, 7, size=(90, 110)).astype(np.int32) * 4
        median = rng.normal(size=count.shape)
        minimum = median - rng.uniform(0, 2, size=count.shape)
        maximum = median + rng.uniform(0, 2, size=count.shape)
        grid = SimpleNamespace(
            count=count,
            occupancy=count > 0,
            median=median,
            minimum=minimum,
            maximum=maximum,
            x0=-13.73,
            y0=2.19,
            cell_m=0.15,
        )
        self.assertEqual(exact_raster_modes(grid, cfg), raster_modes(grid, cfg))
        vessel = dict(
            vessel_hypothesis_id="SYNTHETIC",
            source_support_ids=["MEASURED"],
            local_axes=[[1, 0], [0, 1]],
        )
        old, _ = _bev_provider(None, vessel, cfg, fine_grid=grid)
        new, used = exact_bev_provider(None, vessel, cfg, fine_grid=grid)
        self.assertIs(used, grid)
        self.assertTrue(old)
        self.assertEqual(new, old)

        # A V2 view must not overwrite the historical B2 cache, even when
        # the legacy expert has already populated it before review starts.
        frozen = VesselLocalEvidence(
            SimpleNamespace(
                public=vessel,
                axes=np.eye(2),
                point_index_sha256="measured-point-index-digest",
            ),
            np.empty((0, 3)),
            cfg,
            grids={"fine": grid},
            providers=old,
        )
        original = copy.deepcopy(old)
        review = ExactReviewEvidence.from_frozen(frozen)
        self.assertIs(review.grid("fine"), grid)
        self.assertIsNot(review.structural_providers(), old)
        self.assertIs(frozen.structural_providers(), old)
        self.assertEqual(old, original)
        self.assertEqual(review.provider_build_count, 1)


if __name__ == "__main__":
    unittest.main()
