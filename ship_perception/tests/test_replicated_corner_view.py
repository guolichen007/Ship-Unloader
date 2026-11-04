from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np

from ship_perception.perception.measurement import replicated_corner_view


@dataclass
class Vessel:
    point_indexes: np.ndarray
    point_index_sha256: str = "original"


class ReplicatedCornerView(unittest.TestCase):
    def measurement(self, xyz, indexes):
        xyz = np.asarray(xyz, dtype=float)
        original = np.arange(len(xyz))
        primary = SimpleNamespace(vessel=Vessel(np.asarray(indexes)), xyz=xyz[indexes])
        return SimpleNamespace(
            local=SimpleNamespace(primary_cache=primary),
            raw_uvz=xyz, raw_point_indexes=original, counters={},
        )

    def test_uniform_replication_retains_original_groups_and_native_multiplicity(self):
        base = [[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]]
        raw = self.measurement(np.repeat(base, 4, axis=0), list(range(12)))
        view, groups, audit = replicated_corner_view(raw)
        np.testing.assert_array_equal(view.raw_uvz, base)
        np.testing.assert_array_equal(groups, np.arange(16).reshape(4, 4))
        np.testing.assert_array_equal(view.local.primary_cache.vessel.point_indexes, [0, 4, 8])
        np.testing.assert_array_equal(raw.raw_point_indexes, np.arange(16))
        self.assertEqual(audit["replica_factor"], 4)

    def test_partial_repeats_and_cross_owner_groups_do_not_normalize(self):
        for xyz, owned in (
            ([[0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 0]], list(range(5))),
            ([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]], [0, 2]),
        ):
            self.assertIsNone(replicated_corner_view(self.measurement(xyz, owned)))
