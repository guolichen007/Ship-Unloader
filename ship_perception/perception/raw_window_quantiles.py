"""Exact range quantiles on immutable Raw sequences, without value binning.

Rank queries use the wavelet-matrix construction of Claude, Navarro and
Ordonez (Information Systems 47, 2015). Integer codes identify exact value
order, not rounded heights. Interpolation follows NumPy's linear recipe.
"""

import numpy as np


class RawWindowQuantileIndex:
    def __init__(self, values):
        order = np.argsort(values, kind="stable")
        self.values = values[order]
        codes = np.empty(len(values), dtype=np.uint32)
        codes[order] = np.arange(len(values), dtype=np.uint32)
        self.levels = []
        for bit in range(max(1, (len(values) - 1).bit_length()) - 1, -1, -1):
            zero = ((codes >> bit) & 1) == 0
            prefix = np.empty(len(values) + 1, dtype=np.uint32)
            prefix[0] = 0
            np.cumsum(zero, dtype=np.uint32, out=prefix[1:])
            prefix.setflags(write=False)
            self.levels.append((bit, prefix, int(prefix[-1])))
            codes = np.r_[codes[zero], codes[~zero]]
        self.values.setflags(write=False)

    def select(self, left, right, rank):
        codes = np.zeros(len(rank), dtype=np.uint32)
        left, right, rank = left.copy(), right.copy(), rank.copy()
        for bit, prefix, total in self.levels:
            a = prefix[left].astype(np.int64)
            b = prefix[right].astype(np.int64)
            count = b - a
            one = rank >= count
            codes |= one.astype(np.uint32) << bit
            rank -= np.where(one, count, 0)
            left = np.where(one, total + left - a, a)
            right = np.where(one, total + right - b, b)
        return self.values[codes]

    def quantiles(self, left, right, quantiles):
        if not len(left):
            return np.empty(0, dtype=float)
        rank = (right - left - 1) * quantiles
        low, high = np.floor(rank).astype(np.int64), np.ceil(rank).astype(np.int64)
        a, b = self.select(
            np.r_[left, left], np.r_[right, right], np.r_[low, high]
        ).reshape(2, -1)
        fraction, delta = rank - low, b - a
        return np.where(
            fraction >= 0.5, b - delta * (1 - fraction), a + delta * fraction
        )
