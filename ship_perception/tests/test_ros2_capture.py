"""Real DDS tests, registered in colcon; no mocked ROS APIs."""

import struct
import subprocess
import sys
import signal
import os
import threading
import time
import unittest
import uuid

try:
    import rclpy
except ImportError:
    rclpy = None

from ship_perception.ros2.pointcloud2_source import CaptureError, read_once


@unittest.skipIf(rclpy is None, "ROS2 runtime required; run with colcon test in Humble")
class CaptureIntegration(unittest.TestCase):
    def setUp(self):
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor

        self.context = Context()
        rclpy.init(args=[], context=self.context)
        self.node = Node("hatch_test_writer", context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        self.topic = "/hatch_test_" + uuid.uuid4().hex

    def tearDown(self):
        self.executor.shutdown()
        self.thread.join(timeout=3)
        self.node.destroy_node()
        self.context.try_shutdown()

    def writer(
        self,
        *,
        reliable=True,
        transient=False,
        intensity=False,
        repeat=True,
        invalid_intensity=False,
    ):
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import PointCloud2, PointField

        qos = QoSProfile(
            depth=2,
            reliability=(
                ReliabilityPolicy.RELIABLE
                if reliable
                else ReliabilityPolicy.BEST_EFFORT
            ),
            durability=(
                DurabilityPolicy.TRANSIENT_LOCAL
                if transient
                else DurabilityPolicy.VOLATILE
            ),
        )
        publisher = self.node.create_publisher(PointCloud2, self.topic, qos)
        msg = PointCloud2()
        msg.header.frame_id = "lidar_map"
        msg.header.stamp.sec = 123
        msg.header.stamp.nanosec = 456
        msg.width, msg.height, msg.point_step, msg.row_step = 2, 2, 16, 40
        msg.fields = [
            PointField(name=name, offset=offset, datatype=7, count=1)
            for name, offset in (("z", 8), ("x", 0), ("y", 4))
        ]
        if intensity:
            msg.fields.append(
                PointField(name="intensity", offset=12, datatype=7, count=1)
            )
        rows = []
        for row in range(2):
            payload = bytearray(40)
            for col in range(2):
                x = float("nan") if row == col == 1 else float(row * 2 + col)
                signal = float("inf") if invalid_intensity and row == 0 else 42.0
                struct.pack_into("<ffff", payload, col * 16, x, 5.0, -7.0, signal)
            rows.append(payload)
        msg.data = bytes(rows[0] + rows[1])
        if repeat:
            self.node.create_timer(0.04, lambda: publisher.publish(msg))
        else:
            publisher.publish(msg)
        time.sleep(0.4)
        return publisher

    def check_capture(self, **options):
        publisher = self.writer(**options)
        frame = read_once(self.topic, 4.0)
        runtime = frame.provenance["runtime"]
        self.assertEqual(runtime["messages_consumed"], 1)
        self.assertEqual((frame.stamp_sec, frame.stamp_nanosec), (123, 456))
        self.assertEqual(frame.source_point_count, 4)
        self.assertEqual(len(frame.xyz), 3)
        self.assertEqual(frame.frame_id, "lidar_map")
        self.assertEqual(frame.intensity is not None, options.get("intensity", False))
        return runtime

    def test_reliable_volatile_layout(self):
        self.assertIn("MAP_IS_VOLATILE", self.check_capture()["warnings"][0])

    def test_best_effort_intensity(self):
        self.assertEqual(
            self.check_capture(reliable=False, intensity=True)["selected_qos"][
                "reliability"
            ],
            "BEST_EFFORT",
        )

    def test_late_join_transient_before_subscriber(self):
        self.assertEqual(
            self.check_capture(transient=True, repeat=False)["selected_qos"][
                "durability"
            ],
            "TRANSIENT_LOCAL",
        )

    def test_invalid_intensity_keeps_xyz_alignment(self):
        self.check_capture(intensity=True, invalid_intensity=True)

    def test_no_publisher(self):
        with self.assertRaises(CaptureError) as raised:
            read_once(self.topic, 0.3)
        self.assertEqual(raised.exception.code, "NO_PUBLISHER")

    def test_volatile_no_future_publish_timeout(self):
        publisher = self.writer(repeat=False)
        with self.assertRaises(CaptureError) as raised:
            read_once(self.topic, 0.7)
        self.assertEqual(raised.exception.code, "CAPTURE_TIMEOUT")

    @unittest.skipIf(os.name == "nt", "POSIX SIGINT integration")
    def test_sigint_releases_private_context_and_exits(self):
        publisher = self.writer(repeat=False)
        code = (
            "from ship_perception.ros2.pointcloud2_source import read_once; read_once("
            + repr(self.topic)
            + ", 20.0)"
        )
        process = subprocess.Popen(
            [sys.executable, "-s", "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            time.sleep(1.2)
            self.assertIsNone(process.poll())
            process.send_signal(signal.SIGINT)
            _, error = process.communicate(timeout=5)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn(b"KeyboardInterrupt", error)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)

    def test_wrong_type(self):
        from std_msgs.msg import String

        publisher = self.node.create_publisher(String, self.topic, 1)
        time.sleep(0.4)
        with self.assertRaises(CaptureError) as raised:
            read_once(self.topic, 2.0)
        self.assertEqual(raised.exception.code, "WRONG_TOPIC_TYPE")

    def test_multiple_publishers_conflicting_qos(self):
        first = self.writer(transient=True)
        second = self.writer(reliable=False)
        with self.assertRaises(CaptureError) as raised:
            read_once(self.topic, 2.0)
        self.assertEqual(raised.exception.code, "MULTIPLE_PUBLISHERS_AMBIGUOUS")
        self.assertEqual(raised.exception.diagnostics["publisher_count"], 2)

    def test_explicit_incompatible_map(self):
        publisher = self.writer(reliable=False)
        with self.assertRaises(CaptureError) as raised:
            read_once(self.topic, 2.0, "map")
        self.assertEqual(raised.exception.code, "INCOMPATIBLE_QOS")


if __name__ == "__main__":
    unittest.main()
