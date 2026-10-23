"""Wire identity, bounded allocation and transport-neutral numeric contracts."""

import copy
import struct
from types import SimpleNamespace as Obj
import unittest

import numpy as np

from ship_perception.io.frame import PointCloudFrame
from ship_perception.io.pointcloud2_codec import decode_pointcloud2
from ship_perception.tests.test_ros_hatch_once import message


def ros2_message(**options):
    msg = message(**options)
    msg.header.stamp = Obj(sec=123, nanosec=456)
    return msg


class PointCloud2Codec(unittest.TestCase):
    def test_layout_endian_padding_and_nonfinite_alignment(self):
        for endian in (False, True):
            msg = ros2_message(big_endian=endian)
            data = bytearray(msg.data)
            struct.pack_into((">" if endian else "<") + "f", data, 12, float("nan"))
            msg.data = data
            frame = decode_pointcloud2(msg)
            np.testing.assert_array_equal(frame.xyz[:, 0], [1, 2, 3])
            np.testing.assert_array_equal(frame.intensity, [101, 102, 103])
            self.assertEqual(frame.stamp, "123.000000456")
            self.assertFalse(frame.xyz.flags.writeable)
            self.assertFalse(frame.intensity.flags.writeable)

    def test_wire_padding_and_header_identity_distinguish_equal_semantic_points(self):
        first = ros2_message()
        second = copy.deepcopy(first)
        second.data = bytearray(second.data)
        second.data[27] = 42
        a, b = decode_pointcloud2(first), decode_pointcloud2(second)
        self.assertEqual(a.sha256, b.sha256)
        self.assertNotEqual(a.provenance["wire_sha256"], b.provenance["wire_sha256"])
        second.header.stamp.nanosec += 1
        c = decode_pointcloud2(second)
        self.assertEqual(b.provenance["wire_sha256"], c.provenance["wire_sha256"])
        self.assertNotEqual(
            b.provenance["raw_message_identity_sha256"],
            c.provenance["raw_message_identity_sha256"],
        )

    def test_missing_intensity_and_scalar_float64_integer_intensity(self):
        self.assertIsNone(
            decode_pointcloud2(ros2_message(with_intensity=False)).intensity
        )
        fields = [
            Obj(name=n, offset=i * 8, datatype=8, count=1)
            for i, n in enumerate(("z", "x", "y"))
        ]
        fields.append(Obj(name="intensity", offset=24, datatype=6, count=1))
        msg = Obj(
            width=1,
            height=1,
            point_step=28,
            row_step=28,
            fields=fields,
            data=struct.pack("<dddI", 3.0, 1.000000001, 2.0, 123456),
            is_bigendian=False,
            header=Obj(frame_id="map", stamp=Obj(sec=0, nanosec=0)),
        )
        frame = decode_pointcloud2(msg)
        self.assertEqual(frame.xyz.dtype, np.dtype("<f8"))
        self.assertEqual(frame.xyz[0, 0], 1.000000001)
        self.assertEqual(frame.intensity[0], 123456)

    def test_damage_and_large_input_fail_before_arrays(self):
        for attr, value in (("row_step", 1), ("data", b""), ("width", 20000000)):
            msg = ros2_message()
            setattr(msg, attr, value)
            with self.assertRaises(ValueError):
                decode_pointcloud2(msg)
        with self.assertRaisesRegex(ValueError, "POINT_LIMIT"):
            decode_pointcloud2(ros2_message(), max_points=3)
        msg = ros2_message()
        msg.fields[-1].offset = 12
        with self.assertRaisesRegex(ValueError, "OVERLAPPING"):
            decode_pointcloud2(msg)

    def test_frame_default_copy_and_immutable_provenance(self):
        xyz = np.zeros((3, 3), dtype=np.float32)
        frame = PointCloudFrame(xyz, None, "map", 0, 0, "test", ("x", "y", "z"), 3)
        xyz[0] = 9
        self.assertEqual(frame.xyz.max(), 0)
        identity = frame.provenance
        identity["transport"] = "CHANGED"
        self.assertEqual(frame.provenance, {})


if __name__ == "__main__":
    unittest.main()
