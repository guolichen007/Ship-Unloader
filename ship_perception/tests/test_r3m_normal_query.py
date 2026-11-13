import os
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from ship_perception.perception.measurement import BudgetedLocalSurfaceNormals
from ship_perception.perception.normal_query import ExactQueryExecution
from ship_perception.r1_static.run import resolve_config


class ExactNormalQueryContracts(unittest.TestCase):
    def test_parallel_query_preserves_exact_neighbors_and_full_normal_results(self):
        rng = np.random.RandomState(315)
        raw = rng.uniform(-2, 2, (9000, 3))
        raw[:, 0] *= 0.01
        raw[100:120] = raw[0]  # Ties must retain canonical Raw index order.
        tree = cKDTree(raw)
        a = tree.query(raw, k=24, distance_upper_bound=0.5, workers=1)
        b = ExactQueryExecution(tree, 4).query(
            raw, k=24, distance_upper_bound=0.5, workers=1
        )
        for first, second in zip(a, b):
            np.testing.assert_array_equal(first, second)
        cfg = resolve_config()[0]
        with patch.dict(os.environ, SHIP_REVIEW_NORMAL_WORKERS="1"):
            one = BudgetedLocalSurfaceNormals(raw, cfg, None, {})
        with patch.dict(os.environ, SHIP_REVIEW_NORMAL_WORKERS="4"):
            four = BudgetedLocalSurfaceNormals(raw, cfg, None, {})
        ids = np.arange(len(raw))
        one._estimate(ids)
        four._estimate(ids)
        for attribute in ("xyz", "computed", "valid", "normals", "input_point_indexes"):
            np.testing.assert_array_equal(
                getattr(one, attribute), getattr(four, attribute)
            )
