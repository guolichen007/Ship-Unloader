import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ship_perception.perception.exact_normal_pool import exact_normal_instance


class ExactNormalPoolContracts(unittest.TestCase):
    def test_all_raw_ids_coordinates_frame_and_full_config_must_match(self):
        points = np.arange(36, dtype=float).reshape(12, 3)
        ids = np.arange(12, dtype=np.int64)
        points.setflags(write=False)
        ids.setflags(write=False)
        cfg = dict(normal_k=24, normal_radius_m=0.5)
        m = SimpleNamespace(counters={})

        def get(p=points, i=ids, c=cfg, a=None):
            return exact_normal_instance(
                m, p, i, c, np.eye(2) if a is None else a, object
            )

        with patch.dict(os.environ, SHIP_REVIEW_EXACT_NORMAL_POOL="1"):
            first = get()
            p2, i2 = points.copy(), ids.copy()
            p2.setflags(write=False)
            i2.setflags(write=False)
            self.assertIs(get(p=p2, i=i2), first)
            other_ids = ids + 100
            other_ids.setflags(write=False)
            self.assertIsNot(get(i=other_ids), first)
            shifted = points + 0.1
            shifted.setflags(write=False)
            self.assertIsNot(get(p=shifted), first)
            self.assertIsNot(get(c=dict(normal_k=25, normal_radius_m=0.5)), first)
            self.assertIsNot(get(a=-np.eye(2)), first)
            self.assertIsNot(get(p=points.copy()), first)
            self.assertIsNot(
                exact_normal_instance(
                    SimpleNamespace(counters={}), points, ids, cfg, np.eye(2), object
                ),
                first,
            )
        with patch.dict(os.environ, SHIP_REVIEW_EXACT_NORMAL_POOL="0"):
            self.assertIsNot(get(), first)

    def test_duplicate_xyz_does_not_replace_the_original_index_contract(self):
        p = np.zeros((20, 3))
        a = np.arange(20)
        b = a[::-1].copy()
        for q in (p, a, b):
            q.setflags(write=False)
        m = SimpleNamespace(counters={})
        with patch.dict(os.environ, SHIP_REVIEW_EXACT_NORMAL_POOL="1"):
            one = exact_normal_instance(m, p, a, {}, np.eye(2), object)
            two = exact_normal_instance(m, p, b, {}, np.eye(2), object)
        self.assertIsNot(one, two)
