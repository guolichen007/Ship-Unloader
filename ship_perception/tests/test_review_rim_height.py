"""Z must consume current owned role evidence rather than XY model flags."""

import copy
from types import SimpleNamespace
import unittest
import numpy as np

from ship_perception.perception.measurement import VesselMeasurementContext
from ship_perception.perception.review_height import (
    ReviewCornerHeight,
    cross_segment_consensus,
    SEMANTIC,
)
from ship_perception.perception.rim_top_height import absolute_surface_evidence
from ship_perception.r1_static.run import resolve_config


class ReviewRimHeightContracts(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()

    def measurement(self, points, owned_count=None):
        indexes = np.arange(len(points), dtype=np.int64)
        owned = indexes if owned_count is None else indexes[:owned_count]
        vessel = SimpleNamespace(vessel_id="V", point_indexes=owned)
        context = SimpleNamespace(point_indexes=indexes)
        local = SimpleNamespace(
            vessel=vessel,
            xyz=points,
            context_xyz=points,
            config=self.config,
            measurement_context=context,
        )
        return VesselMeasurementContext(local)

    def fixture(self, cargo=False, halo=False):
        along = np.linspace(0, 8, 161)

        def patch(xs, offset):
            x, y = np.meshgrid(xs, along)
            z = 0.01 * x + 0.02 * y + 3 + offset
            return np.column_stack((x.ravel(), y.ravel(), z.ravel()))

        rim = patch(np.linspace(-0.35, 0, 8), 0)
        floor = patch(np.linspace(0.03, 0.5, 12), 2 if cargo else -2)
        deck = patch(np.linspace(-0.9, -0.45, 12), -0.6)
        points = np.vstack((rim, floor, deck))
        owned_count = len(points)
        if halo:
            points = np.vstack((points, patch(np.linspace(-0.35, 0, 8), 10)))
        measurement = self.measurement(points, owned_count)
        line = dict(
            mode_id="RAW_SIDE",
            source="MEASURED_RAW",
            normal_uv=[1.0, 0.0],
            rho=0.0,
            support_span=[0.0, 8.0],
            independent_opening_role_gain=True,
            source_point_indexes=[],
        )
        result = dict(
            opening_identity="OPEN",
            boundary_uv=[[0.0, 0.0], [20.0, 0.0], [20.0, 8.0], [0.0, 8.0]],
            sides={
                name: dict(independent_geometry_support=False)
                for name in ["U0", "U1", "V0", "V1"]
            },
            corners=[dict(uncertainty_m=100.0) for _ in range(4)],
            constrained_review_rectangle=dict(sides={"U0": dict(measured_line=line)}),
            raw_evidence_geometry=dict(sides={}),
        )
        return measurement, result

    def test_post_rectangle_flags_do_not_block_current_rim_top(self):
        measurement, result = self.fixture()
        before = copy.deepcopy(result)
        estimator = ReviewCornerHeight(measurement, self.config)
        heights = estimator.measure(result)
        self.assertEqual(result, before)
        self.assertTrue(estimator.audit["z_logic_dead_path_removed"])
        self.assertGreater(estimator.audit["rim_candidate_count"], 0)
        self.assertAlmostEqual(heights[0]["z"], 3.0, places=7)
        self.assertAlmostEqual(heights[3]["z"], 3.16, places=7)
        self.assertEqual(heights[0]["z_state"], "MEASURED")
        self.assertTrue(all(heights[i]["z"] is None for i in [1, 2]))
        self.assertEqual(heights[0]["z_definition"], SEMANTIC)
        self.assertFalse(heights[0]["steel_certified"])

    def test_cargo_plane_is_not_promoted_to_rim(self):
        measurement, result = self.fixture(cargo=True)
        estimator = ReviewCornerHeight(measurement, self.config)
        heights = estimator.measure(result)
        self.assertTrue(all(h["z"] is None for h in heights))
        self.assertEqual(estimator.audit["rim_candidate_count"], 0)

    def test_context_halo_height_cannot_supply_rim_z(self):
        measurement, result = self.fixture(halo=True)
        heights = ReviewCornerHeight(measurement, self.config).measure(result)
        self.assertAlmostEqual(heights[0]["z"], 3.0, places=7)
        owned = set(measurement.local.vessel.point_indexes)
        self.assertTrue(set(heights[0]["source_point_indexes"]).issubset(owned))

    def test_deck_reference_does_not_fill_rim_top(self):
        x, y = np.meshgrid(np.linspace(0, 3, 16), np.linspace(0, 3, 16))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 2.0)))
        measurement = self.measurement(points)
        result = dict(
            opening_identity="R",
            boundary_uv=[[0, 0], [3, 0], [3, 3], [0, 3]],
            corners=[{}] * 4,
        )
        sources = []
        for i, ids in enumerate(np.array_split(np.arange(len(points)), 4)):
            sources.append(
                dict(
                    segment_id="S" + str(i),
                    vessel_id="V",
                    plane_role="TRUE_DECK",
                    role_confirmed=True,
                    source_roles=["TRUE_DECK"],
                    normal_uvz=[0, 0, 1],
                    offset_m=-2.0,
                    fit_residual=0.0,
                    fit_extent=[[0, 0], [3, 3]],
                    source_point_indexes=ids.tolist(),
                    tier="DECK_PLANE",
                )
            )
        heights = ReviewCornerHeight(measurement, self.config).measure(result, sources)
        self.assertTrue(
            all(h["z"] is None and h["deck_reference_z"] == 2.0 for h in heights)
        )
        self.assertEqual(heights[0]["deck_reference_semantic"], "DECK_REFERENCE_Z")

    def test_two_current_rim_modes_are_ambiguous_not_averaged(self):
        x, y = np.meshgrid(np.linspace(0, 0.2, 8), np.linspace(0, 0.2, 8))
        xy = np.column_stack((x.ravel(), y.ravel()))
        points = np.vstack(
            (
                np.column_stack((xy, np.full(len(xy), 3.0))),
                np.column_stack((xy, np.full(len(xy), 4.0))),
            )
        )
        measurement = self.measurement(points)
        result = dict(
            opening_identity="R",
            boundary_uv=[[0, 0], [3, 0], [3, 3], [0, 3]],
            corners=[{}] * 4,
        )
        sources = []
        for i, z in enumerate([3.0, 4.0]):
            sources.append(
                dict(
                    segment_id="S" + str(i),
                    vessel_id="V",
                    plane_role="RIM_TOP",
                    role_confirmed=True,
                    source_roles=["RIM_TOP"],
                    normal_uvz=[0, 0, 1],
                    offset_m=-z,
                    fit_residual=0.0,
                    fit_extent=[[0, 0], [0.2, 0.2]],
                    source_point_indexes=list(range(i * len(xy), (i + 1) * len(xy))),
                    tier="DIRECT_RIM_TOP",
                )
            )
        heights = ReviewCornerHeight(measurement, self.config).measure(result, sources)
        self.assertEqual(heights[0]["z_state"], "AMBIGUOUS")
        self.assertIsNone(heights[0]["z"])
        self.assertEqual(len(heights[0]["z_candidates"]), 2)

    def test_profile_absolute_z_retains_original_point_indexes(self):
        measurement, _ = self.fixture(halo=True)
        evidence = absolute_surface_evidence(
            measurement, 0, 0.0, [0.0, 8.0], 1, self.config
        )
        self.assertLess(evidence["inside_surface_z"], evidence["outside_surface_z"])
        self.assertTrue(evidence["upper_surface_point_indexes"])
        self.assertTrue(
            set(evidence["source_point_indexes"]).issubset(
                set(measurement.local.vessel.point_indexes)
            )
        )

    def test_duplicate_physical_support_does_not_create_segment_votes(self):
        rows = [
            dict(
                segment_id="S" + str(i),
                vessel_id="V",
                plane_role="TRUE_DECK",
                role_confirmed=True,
                source_roles=["TRUE_DECK"],
                normal_uvz=[0, 0, 1],
                offset_m=-2.0,
                fit_residual=0.0,
                source_point_indexes=[1, 2, 3],
                fit_extent=[[0, 0], [1, 1]],
            )
            for i in range(4)
        ]
        self.assertIsNone(cross_segment_consensus(rows, np.zeros(2), self.config, "V"))

    def test_normal_scaling_preserves_deck_consensus_height(self):
        rows = [
            dict(
                segment_id="S" + str(i),
                vessel_id="V",
                plane_role="TRUE_DECK",
                role_confirmed=True,
                source_roles=["TRUE_DECK"],
                normal_uvz=[0, 0, 2],
                offset_m=-6.0,
                fit_residual=0.0,
                source_point_indexes=[i * 10 + j for j in range(10)],
                fit_extent=[[0, 0], [1, 1]],
            )
            for i in range(4)
        ]
        chosen = cross_segment_consensus(rows, np.zeros(2), self.config, "V")
        self.assertIsNotNone(chosen)
        self.assertAlmostEqual(-chosen["offset_m"] / chosen["normal_uvz"][2], 3.0)


if __name__ == "__main__":
    unittest.main()
