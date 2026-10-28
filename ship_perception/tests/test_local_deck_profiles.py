"""Exact profile equivalence and Raw face ownership counterexamples."""

import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from ship_perception.r1_static.run import resolve_config
from ship_perception.r1_static.boundary_refinement import _profile
from ship_perception.perception.plane_profile import observed_profile, profile_record
from ship_perception.perception.measurement import BudgetedLocalSurfaceNormals
from ship_perception.perception.measurement import RawSpatialIndex
from ship_perception.perception.local_deck_profiles import LocalDeckProfileCache
from ship_perception.perception.sections import robust_height_support
from ship_perception.perception.raw_face_refinement import (
    measured_faces,
    raw_face_section,
    primary_core_attachment,
)


class LocalDeckProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()

    def wall_fixture(self, bottom=-3.0, top=0.0, plane_side=-1):
        x, y = np.meshgrid(
            np.arange(0.1, 4.01, 0.1) * plane_side, np.arange(0.0, 12.01, 0.1)
        )
        deck = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        y, z = np.meshgrid(
            np.arange(0.0, 12.01, 0.1), np.arange(bottom, top + 0.01, 0.1)
        )
        wall = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
        x, y = np.meshgrid(np.arange(0.1, 4.01, 0.1), np.arange(0.0, 12.01, 0.1))
        cargo = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.6)))
        points = np.vstack((deck, wall, cargo))
        measurement = SimpleNamespace(
            raw_uvz=points,
            raw_point_indexes=np.arange(len(points)),
            counters={},
            budget=None,
        )
        measurement.spatial = RawSpatialIndex(points, measurement.counters)
        cache = LocalDeckProfileCache(measurement, self.config)
        normals = BudgetedLocalSurfaceNormals(points, self.config, None, {})
        faces = measured_faces(
            normals, 0, 0.0, (0.0, 12.0), [(0, 3), (3, 6), (6, 9), (9, 12)], self.config
        )
        self.assertTrue(faces)
        face = min(faces, key=lambda f: abs(f["a_xy"][0]))
        face.update(
            local_deck_wall_height_support=True, _primary_normal_points=face["points"]
        )
        return points, face, cache

    def test_measured_wall_height_does_not_require_a_horizontal_floor_profile(self):
        points, face, cache = self.wall_fixture()
        original = points.copy()
        for span in ((0, 3), (3, 6), (6, 9), (9, 12)):
            result = raw_face_section(points, face, 0, span, 1, self.config, cache)
            self.assertIsNotNone(result)
            self.assertFalse(result["surface_pair_present"])
            self.assertFalse(result["continuous"][3])
            self.assertIn("MEASURED_WALL_HEIGHT", result["measurement_source"])
            self.assertGreaterEqual(
                min(result["height"]["counts"]),
                self.config["boundary"]["profile_min_points"],
            )
            self.assertEqual(
                result["raw_normal_face_count"],
                int(
                    np.count_nonzero(
                        (face["points"][:, 1] >= span[0])
                        & (face["points"][:, 1] <= span[1])
                    )
                ),
            )
            self.assertTrue(result["height"]["local_deck_profiles"]["alternatives"])
        self.assertEqual(
            cache.measurement.counters.get("local_deck_raw_profile_query_count", 0), 0
        )
        np.testing.assert_array_equal(points, original)

    def test_plane_only_reference_keeps_the_same_original_plane_families(self):
        points, face, cache = self.wall_fixture()
        plane_only = cache.section(0, 0.0, (0.0, 3.0), 1, include_profiles=False)
        profiles = cache.section(0, 0.0, (0.0, 3.0), 1)
        self.assertEqual(plane_only["planes"], profiles["planes"])
        self.assertEqual(plane_only["profiles"], [])
        np.testing.assert_array_equal(plane_only["points"], profiles["points"])

    def test_wall_height_rejects_missing_contact_missing_lower_and_wrong_plane_side(
        self,
    ):
        for args in ((-3.0, -1.0, -1), (0.0, 3.0, -1), (-3.0, 0.0, 1)):
            with self.subTest(args=args):
                points, face, cache = self.wall_fixture(*args)
                self.assertIsNone(
                    raw_face_section(points, face, 0, (3, 6), 1, self.config, cache)
                )

    def test_wall_height_cannot_borrow_an_unowned_quay_face(self):
        points, face, cache = self.wall_fixture()
        face["_primary_normal_points"] = points[:0]
        self.assertIsNone(
            raw_face_section(points, face, 0, (3, 6), 1, self.config, cache)
        )

    def test_two_horizontal_return_layers_do_not_become_an_observed_wall(self):
        x, y = np.meshgrid(np.arange(-2.0, 2.01, 0.1), np.arange(0.0, 12.01, 0.1))
        xy = np.column_stack((x.ravel(), y.ravel()))
        points = np.vstack(
            (
                np.column_stack((xy, np.zeros(len(xy)))),
                np.column_stack((xy, np.full(len(xy), -2.0))),
            )
        )
        normals = BudgetedLocalSurfaceNormals(points, self.config, None, {})
        self.assertEqual(
            measured_faces(
                normals,
                0,
                0.0,
                (0.0, 12.0),
                [(0, 3), (3, 6), (6, 9), (9, 12)],
                self.config,
            ),
            [],
        )

    def test_batched_quantiles_preserve_observed_breaks_and_rejections(self):
        rng = np.random.default_rng(818)
        for dtype in (np.float32, np.float64):
            for scenario in range(40):
                xy = rng.uniform([-2.0, -1.0], [2.0, 1.0], (1400, 2))
                center = np.array([100.0, -60.0])
                z = np.where(xy[:, 0] < 0, 0.0, -1.0) + rng.normal(0, 0.015, len(xy))
                if scenario % 4 == 1:  # Competing cargo steps.
                    z[(xy[:, 0] > 0.4) & (xy[:, 0] < 0.8)] = 0.0
                if scenario % 4 == 2:  # No opening.
                    z[:] = 0.0
                if scenario % 4 == 3:  # Sparse or occluded returns.
                    xy, z = xy[::20], z[::20]
                points = np.column_stack((xy + center, z)).astype(dtype)
                tree = cKDTree(points[:, :2])
                plane = (np.array([0.0, 0.0, 1.0]), 0.0)
                args = (
                    points,
                    tree,
                    center,
                    np.array([0.0, 1.0]),
                    np.array([1.0, 0.0]),
                    plane,
                    self.config,
                )
                old = _profile(*args, "ORIGINAL")
                new = observed_profile(*args)
                record = profile_record(*args, "ORIGINAL")
                self.assertEqual(record["face_position"], old["face_position"])
                for key in (
                    "valid",
                    "reason",
                    "sample_count",
                    "deck_support_count",
                    "lower_support_count",
                ):
                    self.assertEqual(record[key], old[key])
                self.assertEqual(old["valid"], new is not None)
                if new is not None:
                    np.testing.assert_allclose(
                        new["break_position"], old["break_position"], atol=1e-12
                    )
                    self.assertAlmostEqual(
                        new["break_strength"], old["break_strength"], places=10
                    )
                    self.assertEqual(new["residual"], old["residual"])

    def test_canonical_normal_indexes_map_back_to_the_actual_input_wall(self):
        along, height = np.meshgrid(np.arange(0, 13, 0.1), np.arange(-3, 0, 0.1))
        wall = np.column_stack(
            (np.full(along.size, 4.0), along.ravel(), height.ravel())
        )
        floor = np.column_stack(
            (
                np.arange(len(wall)) * 0.001,
                np.full(len(wall), 16.0),
                np.zeros(len(wall)),
            )
        )
        points = np.vstack((wall, floor))
        points = points[np.random.default_rng(19).permutation(len(points))]
        normals = BudgetedLocalSurfaceNormals(points, self.config, None, {})
        faces = measured_faces(
            normals, 0, 4.0, (0.0, 12.0), [(0, 3), (3, 6), (6, 9), (9, 12)], self.config
        )
        self.assertTrue(faces)
        for face in faces:
            reconstructed = points[face["point_indexes"]]
            self.assertTrue(np.all(reconstructed[:, 0] == 4.0))
            self.assertEqual(face["point_index_frame"], "INPUT_RAW")
            np.testing.assert_allclose(
                reconstructed[np.lexsort(reconstructed.T[::-1])],
                face["points"][np.lexsort(face["points"].T[::-1])],
            )

    def test_local_surface_profiles_cannot_fill_a_missing_core_wall_section(self):
        points = np.column_stack(
            (np.full(30, 4.0), np.linspace(0, 3, 30), np.linspace(-3, 0, 30))
        )
        face = dict(
            points=points,
            normal_axis=0,
            a_xy=[4.0, 0.0],
            b_xy=[4.0, 12.0],
            residual_p95_m=0.01,
            point_index_sha256="raw",
            local_deck_profile_support=True,
            _primary_normal_points=points[:0],
            local_deck_profile_reference=dict(rho=4.0, span=[0, 12]),
        )

        def forbidden(*args):
            self.fail(
                "Missing primary wall must fail before profile or deck measurements"
            )

        profiles = SimpleNamespace(section=forbidden, section_evidence=forbidden)
        self.assertIsNone(
            raw_face_section(points, face, 0, (0, 3), 1, self.config, profiles)
        )

    def test_multiple_observed_deck_levels_retain_raw_support_and_exact_cache_results(
        self,
    ):
        x, y = np.meshgrid(np.arange(0, 8, 0.075), np.arange(0, 12.05, 0.1))
        level = np.where(y < 6, 0.0, 1.0)
        z = level - np.where(x > 4, 2.0, 0.0)
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        inside = points[points[:, 0] > 4].copy()
        inside[:, 2] += 2.0
        # High mixed returns destroy strip-wide quartile polarity. A local
        # measured deck-to-lower family remains observable in every section.
        points = np.vstack((points, inside))
        self.assertFalse(
            robust_height_support(points, 0, 4.0, (0, 3), 1, self.config)["transition"]
        )
        measurement = SimpleNamespace(
            raw_uvz=points,
            raw_point_indexes=np.arange(len(points)),
            counters={},
            budget=None,
        )
        measurement.spatial = RawSpatialIndex(points, measurement.counters)
        cache = LocalDeckProfileCache(measurement, self.config)
        face = dict(normal_axis=0, a_xy=[4.0, 0.0], b_xy=[4.0, 12.0])
        first = cache.evidence(0, 4.0, (0.0, 12.0), 1, face)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(row["supported"] for row in first))
        second = cache.evidence(0, 4.0, (0.0, 12.0), 1, face)
        self.assertEqual(first, second)
        self.assertGreater(measurement.counters["local_deck_profile_cache_hit"], 0)
        cache.enabled = False
        uncached = cache.evidence(0, 4.0, (0.0, 12.0), 1, face)
        self.assertEqual(first, uncached)
        for row in first:
            self.assertGreaterEqual(
                min(row["counts"]), self.config["boundary"]["profile_min_points"]
            )
            self.assertFalse(row["geometry_inferred"])

    def test_core_patch_links_context_without_filling_an_absent_wall_section(self):
        minimum = self.config["boundary"]["line_min_points"]
        along = np.linspace(0, self.config["roi"]["min_deck_span_m"], minimum)
        points = np.column_stack((np.full(minimum, 4.0), along, np.zeros(minimum)))
        ids = np.arange(minimum)
        attachment = primary_core_attachment(points, ids, 0, self.config)
        self.assertTrue(attachment["supported"])
        self.assertFalse(attachment["ownership_confirmed"])
        self.assertFalse(
            primary_core_attachment(points[:0], ids[:0], 0, self.config)["supported"]
        )
        self.assertFalse(
            primary_core_attachment(points[:-1], ids[:-1], 0, self.config)["supported"]
        )
        face = dict(points=points, primary_core_attachment=attachment)
        profiles = SimpleNamespace(
            section=lambda *args: self.fail("Missing actual wall")
        )
        self.assertIsNone(
            raw_face_section(points, face, 0, (3, 6), 1, self.config, profiles)
        )


if __name__ == "__main__":
    unittest.main()
