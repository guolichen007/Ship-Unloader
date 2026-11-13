"""Field-major LZF PCD parity and bounded corrupt-payload rejection."""

import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
from ship_perception.io.lzf import decompress
from ship_perception.io.pcd_source import read_pcd


class CompressedPCD(unittest.TestCase):
    def test_overlapping_lzf_and_invalid_references(self):
        self.assertEqual(decompress(bytes([0, 65, 224, 4, 0]), 14), b"A" * 14)
        for value, size in (
            (bytes([0]), 1),
            (bytes([32, 0]), 3),
            (bytes([0, 65]), 2),
            (bytes([2, 65, 66]), 3),
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decompress(value, size)

    def test_reordered_float64_and_uint16_field_major_payload(self):
        fields = [
            ("z", "<f8", [3, 6]),
            ("intensity", "<u2", [11, 22]),
            ("y", "<f8", [2, 5]),
            ("x", "<f8", [1, 4]),
        ]
        raw = b"".join(
            np.array(values, dtype=kind).tobytes() for _, kind, values in fields
        )
        encoded = b"".join(
            bytes([len(raw[i : i + 32]) - 1]) + raw[i : i + 32]
            for i in range(0, len(raw), 32)
        )
        header = b"FIELDS z intensity y x\nSIZE 8 2 8 8\nTYPE F U F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary_compressed\n"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "map.pcd"
            path.write_bytes(
                header + struct.pack("<II", len(encoded), len(raw)) + encoded
            )
            frame = read_pcd(path)
            np.testing.assert_array_equal(frame.xyz, [[1, 2, 3], [4, 5, 6]])
            np.testing.assert_array_equal(frame.intensity, [11, 22])
            path.write_bytes(header + struct.pack("<II", len(encoded), 2**31) + encoded)
            with self.assertRaisesRegex(ValueError, "COMPRESSED_SIZE_INVALID"):
                read_pcd(path)
