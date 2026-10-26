"""Berth distance is site-referenced and cannot be replaced by area or count."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ship_perception.io.frame import PointCloudFrame
from ship_perception.perception.context import VesselEvidence, freeze
from ship_perception.perception.berth import (
    select_berth_vessel,
    load_berth_reference,
    AutomaticQuayDetector,
)
from ship_perception.r1_static.run import resolve_config


def fixture(offset=0.0, yaw=0.0, translation=(0.0, 0.0)):
    rotation = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    blocks = []
    vessels = []
    cursor = 0
    for ident, shift, spacing in (("NEAR", 4.0, 0.5), ("FAR", 19.0 + offset, 0.2)):
        u, v = np.meshgrid(
            np.arange(0, 30, spacing), np.arange(shift, shift + 8, spacing)
        )
        xy = np.column_stack((u.ravel(), v.ravel())) @ rotation + translation
        xyz = np.column_stack((xy, np.zeros(len(xy))))
        indexes = np.arange(cursor, cursor + len(xyz))
        cursor += len(xyz)
        blocks.append(xyz)
        vessels.append(
            VesselEvidence(
                freeze(
                    dict(
                        vessel_hypothesis_id=ident,
                        local_axes=rotation.tolist(),
                        classification_status="VESSEL_HYPOTHESIS",
                        source_support_ids=[ident],
                    )
                ),
                indexes,
                ident,
            )
        )
    xyz = np.vstack(blocks)
    frame = PointCloudFrame(xyz, None, "map", 0, 0, "", ("x", "y", "z"), len(xyz))
    reference = dict(
        enabled=True,
        origin=list(translation),
        tangent=rotation[0].tolist(),
        normal=rotation[1].tolist(),
        working_longitudinal_range=[-1, 31],
        source="SURVEYED_SITE_REFERENCE",
    )
    return frame, SimpleNamespace(vessels=tuple(vessels)), reference


class BerthContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()

    def test_nearest_berth_ship_wins_even_with_far_fewer_points(self):
        f, c, r = fixture()
        s = select_berth_vessel(f, c, self.config, r)
        self.assertEqual(s["active_vessel_id"], "NEAR")
        self.assertEqual(s["ignored_vessel_ids"], ["FAR"])

    def test_quay_and_vessels_transform_together(self):
        for yaw, translation in ((0.71, (120, -80)), (-1.3, (-50, 20))):
            f, c, r = fixture(yaw=yaw, translation=translation)
            self.assertEqual(
                select_berth_vessel(f, c, self.config, r)["active_vessel_id"], "NEAR"
            )

    def test_overlapping_distance_uncertainty_is_ambiguous(self):
        f, c, r = fixture(offset=-14.9)
        self.assertIsNone(select_berth_vessel(f, c, self.config, r)["active_vessel_id"])

    def test_two_ships_without_quay_reference_are_not_guessed(self):
        f, c, _ = fixture()
        s = select_berth_vessel(f, c, self.config)
        self.assertEqual(s["status"], "MULTI_VESSEL_BERTH_AMBIGUOUS")
        self.assertIsNone(s["active_vessel_id"])

    def test_map_origin_is_not_assumed_to_be_lidar_position(self):
        f, _, _ = fixture()
        self.assertEqual(
            AutomaticQuayDetector(self.config).detect(f)["reason"],
            "NO_TRUSTED_QUAY_OBSERVER_POSITION",
        )

    def test_invalid_frame_and_ship_prior_fields_are_rejected(self):
        reference = dict(
            enabled=True,
            frame_id="wrong",
            quay_line=[[0, 0], [30, 0]],
            quay_normal_direction=[0, 1],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "site.yaml"
            path.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, "FRAME_MISMATCH"):
                load_berth_reference(path, "map")
            reference["frame_id"] = "map"
            reference["hatch_count"] = 2
            path.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, "REFERENCE_FIELDS"):
                load_berth_reference(path, "map")
