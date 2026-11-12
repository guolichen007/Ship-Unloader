import copy
import unittest
import numpy as np
from ship_perception.r1_static.run import resolve_config
from ship_perception.perception.opening_interior_support import spatial_interior_core
from ship_perception.perception.review_side_evidence import heldout_side


class ReviewEvidence(unittest.TestCase):
    def fixture(self, kind="step"):
        u, v = np.meshgrid(np.arange(0, 24.001, 0.05), np.arange(-1, 12.001, 0.05))
        cut = (
            np.where((np.floor(u / 3).astype(int) % 2) == 0, 2.0, 4.0)
            if kind == "train_only"
            else np.full(u.shape, 2.0)
        )
        z = np.where(v < cut, 0.0, -2.0)
        if kind == "ridge":
            z = np.where((v > 1.0) & (v < 2.0), 3.0, 0.0)
        if kind == "water":
            z = np.zeros(u.shape)
        p = np.c_[u.ravel(), v.ravel(), z.ravel()]
        ids = np.arange(len(p))
        member = (p[:, 1] >= 0) & (p[:, 1] <= 12)
        polygon = np.array([[0.0, 0.0], [24.0, 0.0], [24.0, 12.0], [0.0, 12.0]])
        return p, ids, member, polygon, resolve_config()[0]

    def evidence(self, kind):
        p, ids, member, polygon, cfg = self.fixture(kind)
        before = p.copy()
        frozen = member.copy()
        core, a = spatial_interior_core(p, member, ids, polygon, cfg)
        e = heldout_side(p, ids, core, member, "V0", polygon, [-1, -1, 25, 13], cfg)
        np.testing.assert_array_equal(p, before)
        np.testing.assert_array_equal(member, frozen)
        return e, a

    def test_multiple_material_heights_do_not_destroy_spatial_ownership(self):
        p, ids, m, poly, cfg = self.fixture()
        p[:, 2] = 3 * np.sin(p[:, 0]) + 4 * np.cos(p[:, 1])
        core, a = spatial_interior_core(p, m, ids, poly, cfg)
        self.assertGreater(core.sum(), 0)
        self.assertFalse(a["material_continuity_asserted"])
        self.assertFalse(a["original_mask_modified"])

    def test_empty_core_is_unobservable_and_not_a_vacuous_pass(self):
        p, ids, m, poly, cfg = self.fixture()
        core, a = spatial_interior_core(p, np.zeros(len(m), bool), ids, poly, cfg)
        self.assertFalse(core.any())
        self.assertEqual(a["state"], "SPATIAL_CORE_UNOBSERVABLE")

    def test_real_step_has_heldout_support_without_a_visible_vertical_plane(self):
        e, a = self.evidence("step")
        good = [m for m in e["candidates"] if m["qualified_for_review"]]
        self.assertTrue(good)
        for m in good:
            self.assertAlmostEqual(m["position_m"], 2.0, places=1)
            self.assertFalse(m["steel_face_contact"])
            self.assertFalse(m["steel_certified"])
            self.assertEqual(
                set(m["train_section_ids"]) & set(m["holdout_section_ids"]), set()
            )
            train = set(m["nomination_raw_indexes"])
            val = {
                i
                for r in m["holdout_evidence"]
                if r.get("raw_support_ref")
                for i in e["source_supports"][r["raw_support_ref"]]
            }
            self.assertFalse(train & val)

    def test_training_only_line_is_rejected_by_disjoint_holdout(self):
        e, _ = self.evidence("train_only")
        self.assertFalse(any(m["qualified_for_review"] for m in e["candidates"]))

    def test_cargo_ridge_is_a_two_sided_peak_not_an_opening_transition(self):
        e, _ = self.evidence("ridge")
        self.assertFalse(any(m["qualified_for_review"] for m in e["candidates"]))

    def test_flat_water_does_not_produce_an_edge(self):
        e, _ = self.evidence("water")
        self.assertFalse(any(m["qualified_for_review"] for m in e["candidates"]))


if __name__ == "__main__":
    unittest.main()
