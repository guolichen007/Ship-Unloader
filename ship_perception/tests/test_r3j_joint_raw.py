"""Joint negatives use original point clouds and the real recovery entry point."""

import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from test_r3j_interior_search import fixture
from ship_perception.r1_static.run import resolve_config
from ship_perception.perception.measurement import RawSpatialIndex
from ship_perception.perception.local_rim_evidence import LocalRimEvidence
from ship_perception.perception.review_geometry import ReviewGeometryAssembler, SIDES
from ship_perception.perception.constrained_rectangle import constrain_review_geometry
from ship_perception.perception.vessel_rim_recovery import recover_vessel_sides


def pair(kind):
    m, r, c, p = fixture("water" if kind == "unobservable" else "wall")
    q = m.raw_uvz.copy()
    q[:, 0] += 23.0
    if kind == "unequal":
        q[:, 1] *= 1.2
    combined = np.vstack((m.raw_uvz, q))
    ids = np.arange(len(combined))
    m.raw_uvz = combined
    m.raw_point_indexes = ids
    m.local.vessel.point_indexes = ids
    m.spatial = RawSpatialIndex(combined, {})
    bounds = (23.0, 0.0, 41.0, 14.4 if kind == "unequal" else 12.0)
    own = (q[:, 1] >= 0) & (q[:, 1] <= bounds[3])
    rc = np.unique(
        np.floor((q[own, :2][:, [1, 0]] + 1.0) / 0.5).astype(int), axis=0
    ).tolist()
    r2 = replace(
        r,
        region_id="R2",
        conservative_bounds=bounds,
        search_envelope=bounds,
        boundary_measurements={},
        identity_evidence=dict(
            member_support_grid=r.identity_evidence["member_support_grid"],
            member_support_cells_rc=rc,
        ),
    )
    c.update(bounds_uv=r.bounds, separator_constraints=[])
    c2 = dict(
        c, cell_id="C2", primary_region_id="R2", candidate_ids=["R2"], bounds_uv=bounds
    )
    return m, [r, r2], [c, c2]


class JointRawNegatives(unittest.TestCase):
    def verify(self, kind):
        m, regions, cells = pair(kind)
        cfg = resolve_config()[0]
        asm = ReviewGeometryAssembler(cfg)
        seeds = {
            r.region_id: asm._result(
                r, np.eye(2), {n: asm._prior(r, n, a, p, s) for n, a, p, s in SIDES}
            )
            for r in regions
        }
        old = [
            constrain_review_geometry(r, seeds[r.region_id], cfg, {}) for r in regions
        ]
        before = copy.deepcopy(old)
        final, audit = recover_vessel_sides(
            cells,
            old,
            seeds,
            {r.region_id: r for r in regions},
            LocalRimEvidence(m, cfg),
            asm,
            cfg,
            {},
        )
        self.assertEqual(
            {x["opening_identity"]: x["boundary_xy"] for x in old},
            {x["opening_identity"]: x["boundary_xy"] for x in final},
        )
        self.assertEqual(old, before)
        self.assertIsNone(audit["V"]["common_width_m"])
        self.assertFalse(audit["V"]["two_width_vote_allowed"])
        return cells, final

    def test_real_unequal_widths_are_retained_without_forced_equalization(self):
        cells, final = self.verify("unequal")
        widths = [
            np.linalg.norm(
                np.asarray(h["boundary_xy"])[3] - np.asarray(h["boundary_xy"])[0]
            )
            for h in final
        ]
        self.assertNotAlmostEqual(*widths)

    def test_both_openings_on_same_wrong_hull_have_no_self_validating_width_vote(self):
        cells, _ = self.verify("both_hull")
        self.assertTrue(
            all(
                c["rim_error_recovery"]["status"] == "FAST_PATH_RETAINED" for c in cells
            )
        )

    def test_unobservable_inner_rim_never_becomes_a_measured_equal_width(self):
        cells, _ = self.verify("unobservable")
        self.assertTrue(
            all(
                c["rim_error_recovery"]["status"] == "FAST_PATH_RETAINED" for c in cells
            )
        )


if __name__ == "__main__":
    unittest.main()
