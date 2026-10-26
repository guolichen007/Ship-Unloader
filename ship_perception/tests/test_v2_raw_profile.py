"""Raw indexing is exact; profiles retain signed physical changes and modes."""

import unittest
import numpy as np
from ship_perception.perception.measurement import RawSpatialIndex
from ship_perception.perception.final_boundary import FinalBoundaryRefiner
from ship_perception.r1_static.run import resolve_config


class RawProfileContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()

    def test_spatial_queries_match_inclusive_raw_masks_with_duplicate_points(self):
        points = np.random.default_rng(7).normal(size=(10003, 3))
        points[1:4] = points[0]
        index = RawSpatialIndex(points)
        for box in ((-0.3, -0.5, 0.7, 0.9), (0, 0, 0, 0), (-10, -10, 10, 10)):
            mask = (
                (points[:, 0] >= box[0])
                & (points[:, 0] <= box[2])
                & (points[:, 1] >= box[1])
                & (points[:, 1] <= box[3])
            )
            np.testing.assert_array_equal(index.query(box), points[mask])

    def test_raw_pair_midpoint_uses_original_returns_without_repeated_scans(self):
        u, v = np.meshgrid(np.arange(-2, 2, 0.025), np.arange(0, 4, 0.025))
        xyz = np.column_stack(
            (u.ravel(), v.ravel(), np.where(u > 0, -4.0, 0.0).ravel())
        )
        for points in (
            xyz,
            xyz[::-1],
            xyz[np.random.default_rng(9).permutation(len(xyz))],
        ):
            refiner = FinalBoundaryRefiner(points, [], self.config)
            new = refiner._search_raw_section(points, 0, 0, (0, 4), 1)
            old = refiner._search_raw_section_v1(points, 0, 0, (0, 4), 1)
            self.assertEqual(new["status"], old["status"])
            expected = (
                points[points[:, 2] == 0, 0].max() + points[points[:, 2] == -4, 0].min()
            ) / 2
            self.assertAlmostEqual(new["rho"], expected, places=10)
            self.assertLessEqual(
                abs(new["rho"] - old["rho"]), self.config["geometry"]["refine_voxel_m"]
            )
            self.assertLessEqual(refiner.profiler["continuous_refinement_count"], 3)
            self.assertEqual(refiner.profiler["raw_profile_build_count"], 1)

    def test_wrong_polarity_does_not_become_an_opening(self):
        u, v = np.meshgrid(np.arange(-2, 2, 0.03), np.arange(0, 4, 0.03))
        xyz = np.column_stack((u.ravel(), v.ravel(), np.where(u > 0, 4.0, 0.0).ravel()))
        result = FinalBoundaryRefiner(xyz, [], self.config)._search_raw_section(
            xyz, 0, 0, (0, 4), 1
        )
        self.assertEqual(result["status"], "INSUFFICIENT_RAW_SUPPORT")
