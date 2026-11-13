"""Lossless audit serialization and strict atomic failure semantics."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from ship_perception.perception.output import _atomic_results_json
from ship_perception.r1_static.run import _atomic_json


class CompactOutput(unittest.TestCase):
    def test_family_index_preserves_float_boundary_ties_duplicates_and_raw_order(self):
        from ship_perception.perception.measurement import RawSpatialIndex
        from ship_perception.perception.structural_family import _family_raw_indexes

        rng = np.random.default_rng(7321)
        for dtype in [np.float32, np.float64]:
            for origin in [0.0, 10000.0]:
                for axis in [0, 1]:
                    xyz = rng.uniform(-3, 3, (2000, 3)).astype(dtype)
                    xyz[:, :2] += origin
                    rho, band, low, high = origin + 0.1, 0.05, origin - 1, origin + 1
                    edges = np.asarray([rho - band, rho + band], dtype=dtype)
                    values = np.r_[
                        edges,
                        np.nextafter(edges, -np.inf, dtype=dtype),
                        np.nextafter(edges, np.inf, dtype=dtype),
                    ]
                    xyz[: len(values), axis] = values
                    xyz[: len(values), 1 - axis] = origin
                    xyz = np.r_[xyz, xyz[: len(values)]]
                    expected = np.flatnonzero(
                        (abs(xyz[:, axis] - rho) <= band)
                        & (xyz[:, 1 - axis] >= low)
                        & (xyz[:, 1 - axis] <= high)
                    )
                    actual = _family_raw_indexes(
                        xyz, RawSpatialIndex(xyz), axis, rho, low, high, band
                    )
                    np.testing.assert_array_equal(actual, expected)

    def test_all_values_and_ieee_coordinates_match_pretty_output(self):
        coordinates = [
            0.0,
            -0.0,
            1.0e-300,
            float(np.nextafter(1.0, 2.0)),
            123.45678901234567,
        ]
        payload = dict(
            text="船舱/甲板",
            corners=coordinates,
            evidence=[dict(id=str(i), raw=[i, True, None]) for i in range(100)],
        )
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / "old.json", Path(directory) / "new.json"
            _atomic_json(old, payload)
            _atomic_results_json(new, payload)
            a, b = json.loads(old.read_bytes()), json.loads(new.read_bytes())
            self.assertEqual(a, b)
            np.testing.assert_array_equal(
                np.asarray(a["corners"]).view("u8"), np.asarray(b["corners"]).view("u8")
            )
            self.assertLess(new.stat().st_size, old.stat().st_size)
            self.assertFalse(new.with_name("new.json.tmp").exists())

    def test_nonfinite_value_does_not_replace_existing_document(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "audit.json"
            target.write_bytes(b'{"existing":true}\n')
            for invalid in [float("nan"), float("inf"), -float("inf")]:
                with self.assertRaises(ValueError):
                    _atomic_results_json(target, dict(nested=[invalid]))
                self.assertEqual(target.read_bytes(), b'{"existing":true}\n')
                self.assertFalse(target.with_name("audit.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
