"""ROS byte-layout, missing-intensity and exact dual-output contracts."""

import copy
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace as Obj
import unittest
from unittest.mock import patch

import numpy as np

from ship_perception.ros.pointcloud2_reader import decode_pointcloud2, read_once
from ship_perception.ros.ros_input_adapter import PointCloudFrame
from ship_perception.ros.ship_scan_once import run_frame
from ship_perception.r1_static.corner_topology import _side_rank
from ship_perception.r1_static.intensity_evidence import IntensityEvidence
from ship_perception.r1_static.run import DEFAULT_CONFIG, resolve_config


def message(big_endian=False, with_intensity=True):
    """2x2 organized cloud with reordered fields, point and row padding."""
    names = ("intensity", "z", "x", "y") if with_intensity else ("z", "x", "y")
    offsets = {"intensity": 0, "z": 4, "x": 12, "y": 20}
    fields = [
        Obj(name=name, offset=offsets[name], datatype=7, count=1) for name in names
    ]
    data = bytearray(2 * 64)
    for index in range(4):
        base = (index // 2) * 64 + (index % 2) * 28
        for name, value in dict(
            x=index, y=index + 10, z=-index, intensity=100 + index
        ).items():
            struct.pack_into(
                (">" if big_endian else "<") + "f", data, base + offsets[name], value
            )
    return Obj(
        width=2,
        height=2,
        point_step=28,
        row_step=64,
        data=bytes(data),
        is_bigendian=big_endian,
        fields=fields,
        header=Obj(frame_id="map", stamp=Obj(secs=123, nsecs=456)),
    )


class PointCloudInput(unittest.TestCase):
    def test_offsets_endian_padding_and_alignment(self):
        for endian in (False, True):
            with self.subTest(endian=endian):
                frame = decode_pointcloud2(message(endian))
                self.assertEqual(frame.xyz.dtype, np.dtype("<f4"))
                np.testing.assert_array_equal(frame.xyz[:, 0], np.arange(4))
                np.testing.assert_array_equal(frame.xyz[:, 1], np.arange(4) + 10)
                np.testing.assert_array_equal(frame.intensity, np.arange(4) + 100)
                self.assertEqual(frame.stamp, "123.000000456")
                self.assertEqual(frame.frame_id, "map")

    def test_nonfinite_xyz_keeps_intensity_aligned(self):
        msg = message()
        data = bytearray(msg.data)
        struct.pack_into("<f", data, 12, float("nan"))
        struct.pack_into("<f", data, 28, float("inf"))
        msg.data = bytes(data)
        frame = decode_pointcloud2(msg)
        np.testing.assert_array_equal(frame.xyz[:, 0], [1, 2, 3])
        self.assertTrue(np.isnan(frame.intensity[0]))
        np.testing.assert_array_equal(frame.intensity[1:], [102, 103])
        self.assertEqual(frame.source_point_count, 4)
        self.assertFalse(frame.xyz.flags.writeable)

    def test_missing_intensity(self):
        self.assertIsNone(decode_pointcloud2(message(with_intensity=False)).intensity)

    def test_invalid_layout_and_fields_fail_closed(self):
        for change in (
            dict(row_step=1),
            dict(data=b""),
            dict(width=0),
            dict(fields=[Obj(name="x", offset=12, datatype=7, count=1)]),
        ):
            with self.subTest(change=change):
                msg = message()
                for key, value in change.items():
                    setattr(msg, key, value)
                with self.assertRaises(ValueError):
                    decode_pointcloud2(msg)
        for change in (dict(count=2), dict(datatype=99), dict(offset=28)):
            msg = message()
            for key, value in change.items():
                setattr(msg.fields[0], key, value)
            with self.assertRaises(ValueError):
                decode_pointcloud2(msg)

    def test_integer_intensity(self):
        msg = message()
        msg.fields[0].datatype = 4
        data = bytearray(msg.data)
        for index in range(4):
            struct.pack_into("<H", data, (index // 2) * 64 + (index % 2) * 28, index)
        msg.data = bytes(data)
        np.testing.assert_array_equal(decode_pointcloud2(msg).intensity, np.arange(4))

    def test_invalid_timeout(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                read_once("/ship_scan/map", value)

    def test_ros_wait_type_timeout_and_single_message(self):
        class Timeout(Exception):
            pass

        msg = message()
        ros = Obj(
            core=Obj(is_initialized=lambda: True),
            get_published_topics=lambda: [
                ("/ship_scan/map", "sensor_msgs/PointCloud2")
            ],
            resolve_name=lambda topic: topic,
            ROSException=Timeout,
        )
        ros.wait_for_message = lambda *args, **kwargs: msg
        with patch.dict(
            "sys.modules",
            {
                "rospy": ros,
                "sensor_msgs": Obj(),
                "sensor_msgs.msg": Obj(PointCloud2=Obj),
            },
        ):
            self.assertEqual(len(read_once("/ship_scan/map", 1).xyz), 4)
            ros.get_published_topics = lambda: [("/ship_scan/map", "std_msgs/String")]
            with self.assertRaisesRegex(ValueError, "TOPIC_TYPE"):
                read_once("/ship_scan/map", 1)
            ros.get_published_topics = lambda: []

            def timeout(*args, **kwargs):
                raise Timeout("timeout")

            ros.wait_for_message = timeout
            with self.assertRaisesRegex(RuntimeError, "No PointCloud2"):
                read_once("/ship_scan/map", 1)


class IntensityAuxiliary(unittest.TestCase):
    def setUp(self):
        u, v = np.meshgrid(np.arange(0.05, 4, 0.1), np.arange(0.05, 4, 0.1))
        self.xyz = np.column_stack((u.ravel(), v.ravel(), np.zeros(u.size)))
        self.values = np.where(v.ravel() > 2, 30.0, 10.0)

    def test_scale_offset_invariance_and_missing_support(self):
        first = IntensityEvidence(self.xyz, self.values, np.eye(2), 0.1)
        second = IntensityEvidence(self.xyz, self.values * 17 + 1000, np.eye(2), 0.1)
        self.assertGreater(first.edge(1, 2.0, (0, 4))["intensity_edge_score"], 0.8)
        self.assertEqual(first.edge(1, 2.0, (0, 4)), second.edge(1, 2.0, (0, 4)))
        self.assertEqual(first.edge(1, -2.0, (0, 4))["intensity_edge_score"], 0)
        self.assertEqual(first.edge(1, 1.0, (0, 4))["intensity_edge_score"], 0)

    def test_no_contrast_no_finite_and_sparse(self):
        for values in (np.ones(len(self.xyz)), np.full(len(self.xyz), np.nan)):
            scorer = IntensityEvidence(self.xyz, values, np.eye(2), 0.1)
            self.assertIsNone(scorer.grid)
            self.assertEqual(scorer.edge(1, 2, (0, 4))["intensity_edge_score"], 0)
        keep = self.xyz[:, 1] > 2
        scorer = IntensityEvidence(self.xyz[keep], self.values[keep], np.eye(2), 0.1)
        self.assertEqual(scorer.edge(1, 2, (0, 4))["intensity_edge_score"], 0)

    def test_corner_requires_local_dual_direction_contrast(self):
        one_direction = IntensityEvidence(self.xyz, self.values, np.eye(2), 0.1)
        self.assertEqual(one_direction.corner(np.array([2.0, 2.0]), 0.5), 0.0)
        dual_values = 10.0 + 20.0 * (self.xyz[:, 0] > 2) + 30.0 * (self.xyz[:, 1] > 2)
        dual = IntensityEvidence(self.xyz, dual_values, np.eye(2), 0.1)
        self.assertGreater(dual.corner(np.array([2.0, 2.0]), 0.5), 0.0)

    def test_auxiliary_cannot_make_geometry_viable(self):
        config, _ = resolve_config(DEFAULT_CONFIG)
        row = dict(
            primary_raw3d=0.0,
            primary_bev=0.0,
            height_level_count=0,
            rho=2.0,
            height_rho_m=None,
            height_distance_m=None,
            height_coverage=0.0,
        )
        relation = dict(
            opening_one_sided=False, interior_ridge=False, boundary_asymmetry=0.2
        )
        baseline = _side_rank(row, relation, 2.0, config, "V1")
        row["intensity_bonus"] = 0.05
        fusion = _side_rank(row, relation, 2.0, config, "V1")
        self.assertEqual(baseline[:10], fusion[:10])
        self.assertEqual(fusion[0], 0)
        self.assertGreater(fusion[10], baseline[10])


class DualOutput(unittest.TestCase):
    def test_real_renderer_exact_files_fallback_and_snapshot(self):
        # Providers are replaced here to isolate file publication and actual
        # rendering. Full real-cloud provider parity is checked separately.
        u, v = np.meshgrid(np.arange(0, 5, 0.25), np.arange(0, 3, 0.25))
        xyz = np.column_stack((u.ravel(), v.ravel(), -np.ones(u.size)))
        base_report = dict(
            scene_id="synthetic",
            input_sha256="test",
            static_calibration_status="UNVERIFIED",
            target_vessel_status="NOT_SELECTED",
            vessels=[],
        )
        b2 = dict(scene=dict(vessel_hypotheses=[]))
        with tempfile.TemporaryDirectory() as directory:
            for values, disabled, expected in (
                (None, False, "UNAVAILABLE"),
                (np.ones(len(xyz)), True, "DISABLED"),
                (np.ones(len(xyz)), False, "UNAVAILABLE_NO_VESSEL_CONTRAST"),
            ):
                frame = PointCloudFrame(
                    xyz,
                    values,
                    "map",
                    1,
                    2,
                    "/ship_scan/map",
                    ("x", "y", "z"),
                    len(xyz),
                )
                with patch(
                    "ship_perception.ros.ship_scan_once.prepare_geometry",
                    return_value=(b2, dict(rectangles=[]), {}),
                ), patch(
                    "ship_perception.ros.ship_scan_once.review_points",
                    side_effect=lambda *args, **kwargs: (
                        copy.deepcopy(base_report),
                        xyz,
                        {},
                    ),
                ):
                    output = run_frame(
                        frame,
                        Path(directory),
                        save_snapshot=True,
                        disable_intensity_fusion=disabled,
                    )
                self.assertEqual(
                    {p.name for p in output.iterdir()},
                    {
                        "01_geometry_only.png",
                        "02_intensity_fusion.png",
                        "03_results.json",
                        "04_input_summary.txt",
                        "input_snapshot.npz",
                    },
                )
                result = json.loads(
                    (output / "03_results.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    result["intensity_fusion"]["intensity_fusion_status"], expected
                )
                self.assertEqual(
                    result["geometry_only"]["hatches"],
                    result["intensity_fusion"]["hatches"],
                )
                with np.load(
                    output / "input_snapshot.npz", allow_pickle=False
                ) as snapshot:
                    np.testing.assert_array_equal(snapshot["xyz"], xyz)
                self.assertEqual(len(list(output.glob("*.png"))), 2)

    def test_failure_does_not_publish_partial_run(self):
        frame = decode_pointcloud2(message())
        report = dict(vessels=[])
        with tempfile.TemporaryDirectory() as directory, patch(
            "ship_perception.ros.ship_scan_once.prepare_geometry",
            return_value=({}, {}, {}),
        ), patch(
            "ship_perception.ros.ship_scan_once.review_points",
            return_value=(report, frame.xyz, {}),
        ), patch(
            "ship_perception.ros.ship_scan_once.render_showcase",
            side_effect=RuntimeError("render failed"),
        ):
            with self.assertRaises(RuntimeError):
                run_frame(frame, Path(directory), disable_intensity_fusion=True)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
