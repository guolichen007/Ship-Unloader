"""Corner Z must be locally observed and must not change detection geometry."""

from types import SimpleNamespace
import copy
import unittest
import numpy as np

from ship_perception.r1_static.run import resolve_config
from ship_perception.perception.measurement import VesselMeasurementContext
from ship_perception.perception.corner_height import CornerHeightMeasurement
from ship_perception.perception.coordinates import corner_coordinates
from ship_perception.tests.test_corner_export import payload


class CornerHeightContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()

    def scene(self):
        u, v = np.meshgrid(np.arange(0.0, 10.01, 0.1), np.arange(0.0, 10.01, 0.1))
        inside = (u > 2) & (u < 8) & (v > 2) & (v < 8)
        xyz = np.column_stack(
            (
                u.ravel(),
                v.ravel(),
                np.where(inside, -4.0, 3.0 + 0.02 * u + 0.01 * v).ravel(),
            )
        )
        result = dict(
            boundary_uv=[[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]],
            sides={
                name: dict(independent_geometry_support=True)
                for name in ("U0", "U1", "V0", "V1")
            },
            corners=[dict(uncertainty_m=0.1) for _ in range(4)],
        )
        return xyz, result

    def measure(self, xyz, result):
        cache = SimpleNamespace(
            config=self.config,
            xyz=xyz,
            context_xyz=xyz,
            measurement_context=None,
            vessel=SimpleNamespace(
                point_indexes=np.arange(len(xyz)), point_index_sha256="synthetic-index"
            ),
        )
        return CornerHeightMeasurement(
            VesselMeasurementContext(cache), self.config
        ).measure(result)

    def test_local_raw_upper_surface_recovers_sloped_corner_z_without_geometry_change(
        self,
    ):
        xyz, result = self.scene()
        before = copy.deepcopy(result)
        heights = self.measure(xyz, result)
        for corner, height in zip(result["boundary_uv"], heights):
            self.assertEqual(height["z_status"], "OBSERVED_LOCAL_SURFACE_ESTIMATE")
            self.assertAlmostEqual(
                height["z_m"], 3.0 + 0.02 * corner[0] + 0.01 * corner[1], places=10
            )
            self.assertGreaterEqual(
                height["near_corner_raw_point_count"],
                self.config["boundary"]["line_min_points"],
            )
            self.assertFalse(height["used_for_detection"])
        self.assertEqual(result, before)

    def test_missing_corner_is_null_even_when_a_distant_deck_plane_exists(self):
        xyz, result = self.scene()
        xyz = xyz[
            np.linalg.norm(xyz[:, :2] - [2, 2], axis=1)
            > self.config["boundary"]["corner_join_m"]
        ]
        height = self.measure(xyz, result)[0]
        self.assertIsNone(height["z_m"])
        self.assertEqual(height["z_status"], "UNOBSERVED")

    def test_unmeasured_adjacent_side_cannot_create_an_observed_corner_height(self):
        xyz, result = self.scene()
        result["sides"]["U0"]["independent_geometry_support"] = False
        heights = self.measure(xyz, result)
        self.assertIsNone(heights[0]["z_m"])
        self.assertIsNone(heights[3]["z_m"])
        self.assertIsNotNone(heights[1]["z_m"])

    def test_input_translation_and_z_offset_preserve_measurement(self):
        xyz, result = self.scene()
        before = self.measure(xyz, result)
        xyz = xyz + [-103.2, 57.4, 7.25]
        result["boundary_uv"] = (
            np.asarray(result["boundary_uv"]) + [-103.2, 57.4]
        ).tolist()
        after = self.measure(xyz, result)
        np.testing.assert_allclose(
            [h["z_m"] for h in after],
            np.array([h["z_m"] for h in before]) + 7.25,
            atol=1e-10,
        )

    def test_export_copies_z_and_retains_nullable_missing_heights(self):
        source = payload()
        source["geometry_only"]["hatches"][0]["corner_heights"] = [
            dict(
                z_m=z,
                z_status=(
                    "OBSERVED_LOCAL_SURFACE_ESTIMATE" if z is not None else "UNOBSERVED"
                ),
                z_uncertainty_m=0.05 if z is not None else None,
            )
            for z in (3.1, 3.2, None, 3.4)
        ]
        exported = corner_coordinates(source, "sha")
        self.assertEqual(exported["coordinate_dimensions"], ["x", "y", "z"])
        self.assertEqual(
            [c["z_m"] for c in exported["hatches"][0]["corners"]], [3.1, 3.2, None, 3.4]
        )
        self.assertEqual(
            exported["hatches"][0]["boundary_xy_m"],
            source["geometry_only"]["hatches"][0]["boundary_xy"],
        )
        source["geometry_only"]["hatches"][0]["corner_heights"][0]["z_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "INVALID_CORNER_HEIGHT"):
            corner_coordinates(source, "sha")
