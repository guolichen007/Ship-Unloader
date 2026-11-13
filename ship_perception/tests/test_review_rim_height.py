"""Z must consume current owned role evidence rather than XY model flags."""

import copy
from types import SimpleNamespace
import unittest
import numpy as np

from ship_perception.perception.measurement import (
    VesselMeasurementContext,
    BudgetedLocalSurfaceNormals,
)
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

    def test_direct_corner_rim_survives_missing_remote_side_sections(self):
        measurement, result = self.fixture()
        points = measurement.raw_uvz
        # Only the first end of this rim is visible; floor and deck remain.
        rim = (points[:, 0] >= -0.35) & (points[:, 0] <= 0)
        visible = points[~rim | (points[:, 1] <= 1.8)]
        before = copy.deepcopy(result)
        estimator = ReviewCornerHeight(self.measurement(visible), self.config)
        heights = estimator.measure(result)
        self.assertEqual(result, before)
        self.assertIsNotNone(heights[0]["z"])
        self.assertAlmostEqual(heights[0]["z"], 3.0, places=7)
        self.assertEqual(heights[0]["segment_vote_count"], 1)
        self.assertEqual(len(heights[0]["source_segment_ids"]), 1)
        self.assertIsNone(heights[3]["z"])

    def test_deck_reference_still_requires_independent_remote_segments(self):
        x, y = np.meshgrid(np.linspace(0, 3, 16), np.linspace(0, 3, 16))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 2.0)))
        source = dict(
            segment_id="SINGLE",
            vessel_id="V",
            plane_role="TRUE_DECK",
            role_confirmed=True,
            source_roles=["TRUE_DECK"],
            normal_uvz=[0, 0, 1],
            offset_m=-2.0,
            fit_residual=0.0,
            fit_extent=[[0, 0], [3, 3]],
            source_point_indexes=list(range(len(points))),
            tier="DECK_PLANE",
        )
        result = dict(
            opening_identity="R",
            boundary_uv=[[0, 0], [3, 0], [3, 3], [0, 3]],
            corners=[{}] * 4,
        )
        heights = ReviewCornerHeight(self.measurement(points), self.config).measure(
            result, [source]
        )
        self.assertTrue(
            all(h["z"] is None and h["deck_reference_z"] is None for h in heights)
        )

    def test_plane_bbox_cannot_hide_missing_actual_corner_support(self):
        for size, tier in [(3.0, "LOCAL_RIM_PLANE"), (11.0, "SAME_SIDE_RIM_PLANE")]:
            x, y = np.meshgrid(np.linspace(0, 0.2, 8), np.linspace(0, 0.2, 8))
            patch = np.c_[x.ravel(), y.ravel(), np.full(x.size, 3.0)]
            points = np.vstack((patch, patch + [size - 0.2, size - 0.2, 0]))
            source = dict(
                segment_id="OBSERVED_PATCHES",
                vessel_id="V",
                plane_role="RIM_TOP",
                role_confirmed=True,
                source_roles=["RIM_TOP"],
                normal_uvz=[0, 0, 1],
                offset_m=-3.0,
                fit_residual=0.0,
                fit_extent=[[0, 0], [size, size]],
                source_point_indexes=list(range(len(points))),
                tier=tier,
            )
            result = dict(
                opening_identity="R",
                boundary_uv=[[0, 0], [size, 0], [size, size], [0, size]],
                corners=[{}] * 4,
            )
            heights = ReviewCornerHeight(self.measurement(points), self.config).measure(
                result, [source]
            )
            self.assertEqual(heights[0]["z"], 3.0)
            self.assertEqual(heights[2]["z"], 3.0)
            self.assertIsNone(heights[1]["z"])
            self.assertIsNone(heights[3]["z"])
            self.assertEqual(
                heights[0]["extrapolation_distance_semantic"],
                "NEAREST_ORIGINAL_RAW_SUPPORT_XY",
            )

    def test_context_plane_fit_is_unclassified_and_cannot_claim_ownership(self):
        x, y = np.meshgrid(np.linspace(0, 3, 31), np.linspace(0, 3, 31))
        points = np.c_[x.ravel(), y.ravel(), np.full(x.size, 3.0)]
        measurement = self.measurement(points, owned_count=0)
        measurement.local.primary_cache = SimpleNamespace(
            xyz=points[:0], vessel=measurement.local.vessel
        )
        estimator = ReviewCornerHeight(measurement, self.config)
        callback = lambda ids, normal: estimator._normal_support(
            ids, normal, context=True
        )
        planes = estimator._plane_candidates(
            points, np.arange(len(points)), "CONTEXT", "UNCLASSIFIED_SURFACE", callback
        )
        self.assertTrue(planes)
        self.assertTrue(
            all(p["vessel_id"] is None and not p["role_confirmed"] for p in planes)
        )
        for plane in planes:
            forged = dict(
                plane,
                vessel_id="V",
                plane_role="RIM_TOP",
                role_confirmed=True,
                source_roles=["RIM_TOP"],
            )
            self.assertFalse(
                estimator._eligible(forged, np.array([], dtype=int), {"RIM_TOP"})
            )
        with self.assertRaisesRegex(ValueError, "UNCLASSIFIED_ROLE"):
            estimator._plane_candidates(
                points, np.arange(len(points)), "CONTEXT", "RIM_TOP", callback
            )

    def test_same_side_plane_does_not_extrapolate_across_the_opening(self):
        x, y = np.meshgrid(np.linspace(0, 10, 101), np.linspace(0, 0.3, 8))
        points = np.c_[x.ravel(), y.ravel(), np.full(x.size, 3.0)]
        source = dict(
            segment_id="SIDE",
            vessel_id="V",
            plane_role="RIM_TOP",
            role_confirmed=True,
            source_roles=["RIM_TOP"],
            normal_uvz=[0, 0, 1],
            offset_m=-3.0,
            fit_residual=0.0,
            fit_extent=[[0, 0], [10, 0.3]],
            source_point_indexes=list(range(len(points))),
            tier="SAME_SIDE_RIM_PLANE",
            rim_side_normal_uv=[0, 1],
        )
        result = dict(
            opening_identity="R",
            boundary_uv=[[0, 2], [10, 2], [10, 3], [0, 3]],
            corners=[{}] * 4,
        )
        heights = ReviewCornerHeight(self.measurement(points), self.config).measure(
            result, [source]
        )
        self.assertTrue(all(z["z"] is None for z in heights))

    def test_thin_rim_surface_is_not_suppressed_by_more_numerous_floor_returns(self):
        _, result = self.fixture()
        along = np.linspace(0, 8, 161)

        def patch(xs, offset):
            x, y = np.meshgrid(xs, along)
            return np.column_stack(
                (x.ravel(), y.ravel(), (3 + 0.01 * x + 0.02 * y + offset).ravel())
            )

        points = np.vstack(
            (
                patch(np.linspace(-0.2, 0, 8), 0),
                patch(np.linspace(-0.4, -0.21, 32), -2),
                patch(np.linspace(0.03, 0.5, 12), -2),
                patch(np.linspace(-0.9, -0.45, 12), -0.6),
            )
        )
        estimator = ReviewCornerHeight(self.measurement(points), self.config)
        heights = estimator.measure(result)
        self.assertAlmostEqual(heights[0]["z"], 3.0, places=7)
        self.assertAlmostEqual(heights[3]["z"], 3.16, places=7)
        self.assertEqual(heights[0]["z_state"], "MEASURED")

    def test_horizontal_slices_of_vertical_walls_are_not_top_surfaces(self):
        along, height = np.meshgrid(np.linspace(0, 3, 121), np.linspace(-2, 2, 161))
        points = np.vstack(
            [
                np.column_stack((np.full(along.size, x), along.ravel(), height.ravel()))
                for x in (-0.3, -0.05)
            ]
        )
        measurement = self.measurement(points)
        sources = ReviewCornerHeight(measurement, self.config)._plane_candidates(
            points, np.arange(len(points)), "WALLS", "UNCLASSIFIED_SURFACE"
        )
        self.assertEqual(sources, [])

    def test_plane_raw_support_and_observed_normal_support_are_separate(self):
        x, y = np.meshgrid(np.linspace(0, 2, 16), np.linspace(0, 2, 16))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 3.0)))
        measurement = self.measurement(points)
        normals = BudgetedLocalSurfaceNormals(
            points, self.config, None, measurement.counters
        )
        normals._estimate(np.arange(len(points)))
        observed = np.flatnonzero(normals.valid)[::16]
        normals.valid[:] = False
        normals.valid[observed] = True
        measurement.review_primary_normals = normals
        self.assertGreaterEqual(
            len(observed), self.config["geometry"]["normal_min_points"]
        )
        self.assertLess(len(observed), self.config["geometry"]["min_plane_points"])
        sources = ReviewCornerHeight(measurement, self.config)._plane_candidates(
            points, np.arange(len(points)), "PARTIAL_NORMALS", "UNCLASSIFIED_SURFACE"
        )
        self.assertTrue(sources)
        self.assertEqual(len(sources[0]["source_point_indexes"]), len(points))

    def test_rim_adjacency_uses_side_gap_not_vertical_plane_residual(self):
        _, result = self.fixture()
        along = np.linspace(0, 8, 161)

        def patch(xs, offset):
            x, y = np.meshgrid(xs, along)
            return np.column_stack(
                (x.ravel(), y.ravel(), (3 + 0.01 * x + 0.02 * y + offset).ravel())
            )

        for gap, expected in [(0.10, True), (0.25, False)]:
            points = np.vstack(
                (
                    patch(np.linspace(-0.35, -gap, 8), 0),
                    patch(np.linspace(0.03, 0.5, 12), -2),
                    patch(np.linspace(-0.9, -0.45, 12), 0),
                )
            )
            heights = ReviewCornerHeight(self.measurement(points), self.config).measure(
                result
            )
            self.assertEqual(heights[0]["z"] is not None, expected)
            if expected:
                self.assertAlmostEqual(heights[0]["z"], 3.0, places=7)

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
