"""Nominal line adjacency; signed Raw roles are resolved after compilation."""


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
            # Cargo can reverse a height comparison across a real coaming.
            # A verified face nominates either side at this cheap stage;
            # candidate-local Raw measurement must establish opening polarity.
            self._signs[key] = [1, -1] if family["normal_face"] else measured
        return self._signs[key]

    def overlap(self, family, span):
        return max(
            0.0,
            min(family["interval"][1], span[1]) - max(family["interval"][0], span[0]),
        ) / max(self.minimum, span[1] - span[0])

    def _extra_envelope_support(self, faces, box):
        # An outer search is retained only with independent measured structure
        # beyond the nominal adjacency. Cargo height is never a hard veto on
        # the adjacent hypothesis; this gate only adds a wider alternative.
        normal_count = sum(face["normal_face"] for face in faces)
        if normal_count < 3:
            return False
        for face, axis, sign in zip(faces, (0, 0, 1, 1), (1, -1, 1, -1)):
            if face["normal_face"]:
                continue
            span = (box[1 - axis], box[3 - axis])
            if (
                self.height is None
                or (self.height(axis, face["rho"], span, sign) or 0)
                < self.config["roi"]["opening_drop_m"]
            ):
                return False
        return True

    def enclosures(self):
        output, seen = [], set()
        longitudinal, transverse = self.families[1], self.families[0]

        def emit(left, right, low, high, kind):
            box = (left["rho"], low["rho"], right["rho"], high["rho"])
            if box in seen:
                return
            faces = (left, right, low, high)
            if kind != "ADJACENT" and not self._extra_envelope_support(faces, box):
                return
            seen.add(box)
            skipped = sum(
                f["normal_face"]
                and box[axis] + self.join < f["rho"] < box[axis + 2] - self.join
                and self.overlap(f, (box[1 - axis], box[3 - axis])) >= self.support
                for axis in (0, 1)
                for f in self.families[axis]
            )
            transverse = max(
                (
                    self.overlap(f, (box[1], box[3]))
                    for f in self.families[0]
                    if f["normal_face"]
                    and box[0] + self.join < f["rho"] < box[2] - self.join
                ),
                default=0.0,
            )
            output.append(
                dict(
                    bounds_axial=list(box),
                    source_provider_ids=[
                        pid for face in faces for pid in face["provider_ids"]
                    ],
                    physical_modes=[face["provider_ids"] for face in faces],
                    normal_face_sides=sum(face["normal_face"] for face in faces),
                    geometry_role="NOMINAL_STRUCTURAL_CLOSURE_NOT_STEEL",
                    closure_kind=kind,
                    signed_roles_resolved=False,
                    skipped_verified_face_count=int(skipped),
                    internal_transverse_face_coverage=float(transverse),
                )
            )

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

                def closure(ordered):
                    for high in ordered:
                        span = (low["rho"], high["rho"])
                        rights = [
                            u
                            for u in transverse
                            if u["rho"] - left["rho"] >= self.minimum
                            and self.overlap(u, span) >= self.support
                            and self.overlap(left, span) >= self.support
                            and self.overlap(low, (left["rho"], u["rho"]))
                            >= self.support
                            and self.overlap(high, (left["rho"], u["rho"]))
                            >= self.support
                            and -1 in self.signs(0, u)
                        ]
                        if rights:
                            return high, rights
                    return None

                first = closure(highs)
                if first is None:
                    continue
                high, rights = first
                emit(left, rights[0], low, high, "ADJACENT")
                emit(left, rights[-1], low, high, "SUPPORTED_OUTER_ENVELOPE")
                last = closure(reversed(highs))
                if last is not None:
                    high, rights = last
                    emit(left, rights[0], low, high, "SUPPORTED_OUTER_ENVELOPE")
                    emit(left, rights[-1], low, high, "SUPPORTED_OUTER_ENVELOPE")
        return output
