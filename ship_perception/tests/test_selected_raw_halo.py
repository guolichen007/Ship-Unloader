"""Exterior measurement context preserves Raw values and never borrows a ship."""

from types import SimpleNamespace
import unittest
import numpy as np

from ship_perception.tests.test_v2_berth import fixture
from ship_perception.io.frame import PointCloudFrame
from ship_perception.perception.context import BoundaryMeasurementContext
from ship_perception.perception.selected_context import SelectedVesselRawContext
from ship_perception.r1_static.run import resolve_config


class SelectedRawHalo(unittest.TestCase):
    def build(self, yaw=0.0, translation=(0.0, 0.0)):
        frame, scene, reference = fixture(offset=-5.0, yaw=yaw, translation=translation)
        rotation = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        x, y = np.meshgrid(np.arange(0.0, 30.0, 0.5), np.arange(-1.0, 4.0, 0.5))
        xy = np.column_stack((x.ravel(), y.ravel())) @ rotation + translation
        quay = np.column_stack((xy, np.full(len(xy), 0.25)))
        points = np.vstack((frame.xyz, quay))
        source = PointCloudFrame(
            points,
            np.arange(len(points), dtype=float),
            "map",
            0,
            0,
            "",
            ("x", "y", "z", "intensity"),
            len(points),
        )
        near = scene.vessels[0]
        measurement = BoundaryMeasurementContext.build(near, scene.vessels)
        context = SimpleNamespace(
            vessels=scene.vessels, boundary_context=lambda _: measurement
        )
        config, _ = resolve_config()
        selected = SelectedVesselRawContext.build(
            source, context, dict(active_vessel_id="NEAR", reference=reference), config
        )
        return source, scene, selected, config

    def test_halo_is_available_to_measurement_and_excludes_other_known_vessel(self):
        frame, scene, selected, config = self.build()
        self.assertGreater(
            len(selected.raw_point_indexes), len(selected.measurement.point_indexes)
        )
        self.assertGreater(len(selected.exterior_point_indexes), 0)
        self.assertFalse(
            np.isin(selected.raw_point_indexes, scene.vessels[1].point_indexes).any()
        )
        self.assertTrue(
            np.isin(scene.vessels[0].point_indexes, selected.raw_point_indexes).all()
        )
        self.assertGreaterEqual(
            selected.halo_margin_m,
            config["roi"]["support_search_m"]
            + config["boundary"]["profile_half_length_m"],
        )
        self.assertEqual(selected.audit()["exterior_halo_ownership"], "UNKNOWN_CONTEXT")
        np.testing.assert_array_equal(
            frame.intensity[selected.raw_point_indexes], selected.raw_point_indexes
        )
        self.assertFalse(frame.xyz.flags.writeable)
        self.assertFalse(selected.raw_point_indexes.flags.writeable)

    def test_halo_rotates_and_translates_with_vessel_instead_of_using_fixed_y(self):
        _, _, original, _ = self.build()
        _, _, rotated, _ = self.build(0.73, (120.0, -80.0))
        np.testing.assert_array_equal(
            original.raw_point_indexes, rotated.raw_point_indexes
        )
        np.testing.assert_array_equal(
            original.exterior_point_indexes, rotated.exterior_point_indexes
        )
