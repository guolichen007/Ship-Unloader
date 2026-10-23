"""PCD precision, padding and immutable acquisition metadata."""

import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
import numpy as np
from ship_perception.io.frame import PointCloudFrame
from ship_perception.io.pcd_source import read_pcd
from ship_perception.perception.intensity import admission_stats


class OfflineInput(unittest.TestCase):
    def test_pcd_float64_integer_intensity_nonfinite_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.pcd"
            header = "VERSION .7\nFIELDS z intensity y x\nSIZE 8 2 8 8\nTYPE F U F F\nCOUNT 1 1 1 1\nWIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA binary\n"
            rows = np.array(
                [(3.0, 1, 2.0, 1.0), (3.0, 2, 2.0, np.nan), (6.0, 3, 5.0, 4.0)],
                dtype=[("z", "<f8"), ("intensity", "<u2"), ("y", "<f8"), ("x", "<f8")],
            )
            path.write_bytes(header.encode() + rows.tobytes())
            result = read_pcd(path)
            np.testing.assert_equal(result.intensity, [1.0, 3.0])
            np.testing.assert_equal(result.xyz, [[1, 2, 3], [4, 5, 6]])
            self.assertEqual(result.source_point_count, 3)

    def test_float32_one_ulp_signal_is_unavailable(self):
        values = np.array(
            [1.0, np.nextafter(np.float32(1.0), np.float32(2.0))] * 50, dtype=float
        )
        result = admission_stats(values, "<f4")
        self.assertEqual(result["status"], "UNAVAILABLE_CONSTANT_DEVICE_SIGNAL")
        self.assertEqual(result["effective_unique_count"], 1)

    def test_replacing_metadata_with_invalid_intensity_keeps_geometry(self):
        frame = PointCloudFrame(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([np.inf, 42.0]),
            "map",
            0,
            0,
            "topic",
            ("x", "y", "z", "intensity"),
            2,
        )
        other = replace(
            frame, provenance_json=json.dumps(dict(runtime="test")), copy_arrays=False
        )
        self.assertEqual(frame.sha256, other.sha256)
        self.assertTrue(np.isnan(other.intensity[0]))
        self.assertFalse(other.xyz.flags.writeable)

    def test_ascii_scalar_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scalar.pcd"
            path.write_text(
                "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n1 2 3 4\n5 6 7 8\n"
            )
            frame = read_pcd(path)
            np.testing.assert_allclose(frame.xyz, [[1, 2, 3], [5, 6, 7]])
            np.testing.assert_allclose(frame.intensity, [4, 8])


if __name__ == "__main__":
    unittest.main()
