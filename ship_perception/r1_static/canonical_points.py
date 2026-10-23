"""Physical ordering before seeded sampling; independent of KD-tree traversal."""

import numpy as np


def point_order(points):
    points = np.asarray(points)
    return np.lexsort(
        tuple(points[:, column] for column in reversed(range(points.shape[1])))
    )


def canonical_indexes(points, indexes):
    indexes = np.asarray(indexes, dtype=np.int64)
    return indexes[point_order(np.asarray(points)[indexes])]
