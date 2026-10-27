"""Nominal line adjacency; signed Raw roles are resolved after compilation."""

import numpy as np


class BoundaryLineGraph:
    def __init__(self, families, config, height=None, budget=None):
        self.families = {
            axis: [f for f in rows if f["raw3d"] and f["bev"]]
            for axis, rows in families.items()
        }
        self.config, self.height, self.budget = config, height, budget
        self.join = config["boundary"]["corner_join_m"]
        self.minimum = config["roi"]["min_deck_span_m"]
        self.support = config["roi"]["min_enclosure_ratio"]
        self._signs = {}
        self.nodes = sum(map(len, self.families.values()))
        self.edges = 0

    def signs(self, axis, family):
        key = (axis, family["provider_ids"])
        if key not in self._signs:
            measured = [
                sign
                for sign in (1, -1)
                if self.height is not None
                and (self.height(axis, family["rho"], family["interval"], sign) or 0)
                >= self.config["roi"]["opening_drop_m"]
            ]
            # A measured vertical face with unknown polarity may nominate a
            # search. It is never labelled as a signed physical edge here.
            self._signs[key] = measured or ([1, -1] if family["normal_face"] else [])
        return self._signs[key]

    def overlap(self, family, span):
        return max(
            0.0,
            min(family["interval"][1], span[1]) - max(family["interval"][0], span[0]),
        ) / max(self.minimum, span[1] - span[0])

    def enclosures(self):
        output, seen = [], set()
        longitudinal = self.families[1]
        transverse = self.families[0]
        for left in transverse:
            if self.budget is not None:
                self.budget.check()
            if 1 not in self.signs(0, left):
                continue
            neighbors = [
                v
                for v in longitudinal
                if left["interval"][0] - self.join
                <= v["rho"]
                <= left["interval"][1] + self.join
                and v["interval"][0] - self.join
                <= left["rho"]
                <= v["interval"][1] + self.join
            ]
            self.edges += len(neighbors)
            for low in neighbors:
                if 1 not in self.signs(1, low):
                    continue
                highs = [
                    v
                    for v in neighbors
                    if v["rho"] - low["rho"] >= self.minimum and -1 in self.signs(1, v)
                ]
                for high in highs:
                    span = (low["rho"], high["rho"])
                    rights = [
                        u
                        for u in transverse
                        if u["rho"] - left["rho"] >= self.minimum
                        and self.overlap(u, span) >= self.support
                        and self.overlap(left, span) >= self.support
                        and self.overlap(low, (left["rho"], u["rho"])) >= self.support
                        and self.overlap(high, (left["rho"], u["rho"])) >= self.support
                        and -1 in self.signs(0, u)
                    ]
                    if not rights:
                        continue
                    right = rights[0]
                    box = (left["rho"], low["rho"], right["rho"], high["rho"])
                    key = tuple(float(v) for v in box)
                    if key not in seen:
                        seen.add(key)
                        faces = (left, right, low, high)
                        output.append(
                            dict(
                                bounds_axial=list(box),
                                source_provider_ids=[
                                    pid for f in faces for pid in f["provider_ids"]
                                ],
                                physical_modes=[f["provider_ids"] for f in faces],
                                normal_face_sides=sum(f["normal_face"] for f in faces),
                                geometry_role="NOMINAL_ADJACENT_CLOSURE_NOT_STEEL",
                                signed_roles_resolved=False,
                            )
                        )
                    # Only the nearest compatible opposite side closes this
                    # adjacency. A farther parallel line is a different local
                    # search from its own neighbor, not a Cartesian product.
                    break
        return output
