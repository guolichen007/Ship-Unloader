from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from planar_replay_fixture import transform


class PlanarReplayFixture(unittest.TestCase):
    def test_inverse_preserves_original_rows_and_gravity(self):
        xyz = np.array([[4.0, -2.5, 6.0], [-13.0, 50.0, -4.5]], dtype=np.float32)
        origin = np.array([60.0, 12.0])
        translation = np.array([17.27, -11.34, 5.0])
        moved, rotation = transform(xyz, -23.9, origin, translation)
        recovered_xy = (moved[:, :2] - origin - translation[:2]) @ rotation + origin
        np.testing.assert_allclose(recovered_xy, xyz[:, :2], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(moved[:, 2] - translation[2], xyz[:, 2])
        self.assertEqual(moved.dtype, np.float64)
