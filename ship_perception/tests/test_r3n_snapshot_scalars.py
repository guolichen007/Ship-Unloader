"""Verify complete snapshot payload equivalence, not a scalar helper mirror."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

from ship_perception.perception.context import freeze, VesselEvidence
from ship_perception.perception.recall_snapshot import RecallSnapshot
from ship_perception.r1_static.height_grid import R1HeightGrid


class SnapshotScalarContract(unittest.TestCase):
    def test_complete_whitelisted_snapshot_members_equal_with_fast_path_on_and_off(
        self,
    ):
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.2, 2.0], [1.0, 1.0, 3.0]])
        state = dict(
            scalars=[
                None,
                True,
                False,
                0,
                2**63,
                -0.0,
                0.0,
                1.25,
                float("nan"),
                float("inf"),
                float("-inf"),
                "中文",
            ],
            numpy=[np.int32(7), np.float32(1.5), np.bool_(True)],
            keyed=freeze({0: (1, "x"), 1: np.array([[np.nan, 3.0], [4.0, 5.0]])}),
            vessel=VesselEvidence(
                freeze(
                    dict(
                        vessel_hypothesis_id="SYNTHETIC",
                        local_axes=[[1.0, 0.0], [0.0, 1.0]],
                    )
                ),
                np.array([0, 1, 2], dtype="<i8"),
                "RAW_HASH",
            ),
            grid=R1HeightGrid.from_points(points, 0.5),
        )
        identity = dict(input_sha256="INPUT", config_sha256="CONFIG", frame_id="FRAME")
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, cls, flag in [
                ("slow", RecallSnapshot, "0"),
                ("fast", RecallSnapshot, "1"),
            ]:
                path = Path(directory) / (name + ".snapshot")
                with patch.dict(os.environ, SHIP_REVIEW_SNAPSHOT_FAST_SCALARS=flag):
                    cls(path, identity).save(state)
                restored = cls(path, identity)
                self.assertEqual(restored.audit["status"], "HIT")
                self.assertFalse(restored.value["keyed"][1].flags.writeable)
                np.testing.assert_array_equal(
                    restored.value["vessel"].point_indexes, [0, 1, 2]
                )
                paths.append(path)
            with zipfile.ZipFile(paths[0]) as original:
                for path in paths[1:]:
                    with zipfile.ZipFile(path) as actual:
                        self.assertEqual(actual.namelist(), original.namelist())
                        for name in original.namelist():
                            self.assertEqual(actual.read(name), original.read(name))

    def test_object_arrays_remain_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for flag in ("0", "1"):
                with patch.dict(os.environ, SHIP_REVIEW_SNAPSHOT_FAST_SCALARS=flag):
                    with self.assertRaisesRegex(ValueError, "SNAPSHOT_OBJECT_ARRAY"):
                        RecallSnapshot(Path(directory) / ("bad" + flag), {}).save(
                            np.array([object()], dtype=object)
                        )


if __name__ == "__main__":
    unittest.main()
