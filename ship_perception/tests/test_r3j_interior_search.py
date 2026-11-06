import copy
from types import SimpleNamespace
import unittest
import numpy as np

from ship_perception.r1_static.run import resolve_config
from ship_perception.perception.measurement import RawSpatialIndex
from ship_perception.perception.local_rim_evidence import LocalRimEvidence
from ship_perception.perception.interior_rim_search import InteriorRimSearch
from ship_perception.perception.region_model import OpeningRegion, TopologySupport
from ship_perception.perception.review_geometry import ReviewGeometryAssembler, SIDES
from ship_perception.perception.constrained_rectangle import constrain_review_geometry
from ship_perception.perception.vessel_rim_recovery import recovery_ab


def fixture(kind="wall"):
    along = np.arange(0.0, 18.001, 0.05)
    ss, tt = np.meshgrid(along, np.arange(-0.8, 2.0, 0.05))
    deck = np.column_stack((ss.ravel(), tt.ravel(), np.zeros(ss.size)))
    ss, tt = np.meshgrid(along, np.arange(2.05, 12.001, 0.05))
    floor = np.column_stack((ss.ravel(), tt.ravel(), np.full(ss.size, -2.0)))
    ss, zz = np.meshgrid(along, np.arange(-2.0, 0.001, 0.05))
    wall = np.column_stack((ss.ravel(), np.full(ss.size, 2.0), zz.ravel()))
    points = np.vstack((deck, floor, wall))
    ids = np.arange(len(points))
    owned = ids
    if kind == "ridge":
        points = points[: len(deck) + len(floor)]
        points[:, 2] = -2.5 * np.maximum(points[:, 1] - 2.0, 0.0)
        ids = np.arange(len(points))
        owned = ids
    elif kind == "water":
        points = points[: len(deck) + len(floor)]
        points[:, 2] = -2.0
        ids = np.arange(len(points))
        owned = ids
    elif kind == "foreign_wall":
        owned = ids[: len(deck) + len(floor)]
    vessel = SimpleNamespace(vessel_id="V", point_indexes=owned, axes=np.eye(2))
    measurement = SimpleNamespace(
        raw_uvz=points,
        raw_point_indexes=ids,
        spatial=RawSpatialIndex(points, {}),
        local=SimpleNamespace(vessel=vessel),
    )
    grid = dict(x0=-1.0, y0=-1.0, cell_m=0.5)
    in_box = (
        (points[:, 0] >= 0)
        & (points[:, 0] <= 18)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= 12)
    )
    rc = np.floor((points[in_box, :2][:, [1, 0]] + 1) / 0.5).astype(int)
    region = OpeningRegion(
        "R",
        "V",
        (0.0, 0.0, 18.0, 12.0),
        (0.0, 0.0, 18.0, 12.0),
        (TopologySupport("STRUCTURAL_LINE_FAMILY", "SUPPORTING"),),
        identity_role="CONFIRMED_INDEPENDENT_OPENING",
        identity_evidence=dict(
            member_support_grid=grid,
            member_support_cells_rc=np.unique(rc, axis=0).tolist(),
        ),
    )
    cell = dict(
        cell_id="C",
        vessel_id="V",
        candidate_ids=["R"],
        primary_region_id="R",
        width_recovery_requested_sides=["V0"],
    )
    polygon = np.array([[0.0, 0.0], [18.0, 0.0], [18.0, 12.0], [0.0, 12.0]])
    return measurement, region, cell, polygon


class InteriorSearchTests(unittest.TestCase):
    def run_search(self, kind="wall"):
        m, region, cell, p = fixture(kind)
        cfg = resolve_config()[0]
        before = copy.deepcopy(region.identity_evidence)
        original = p.copy()
        search = InteriorRimSearch(LocalRimEvidence(m, cfg))
        result = search.search(cell, p, {"R": region}, [cell])
        self.assertEqual(region.identity_evidence, before)
        np.testing.assert_array_equal(p, original)
        return result, m

    def test_real_raw_wall_meters_inside_old_side_is_a_measured_candidate(self):
        result, m = self.run_search()
        modes = result["candidates"]["V0"]["candidates"]
        proven = [mode for mode in modes if mode["qualified_for_recovery"]]
        self.assertTrue(
            proven,
            [
                (
                    x["rho"],
                    x["independent_raw_sections"],
                    x["observed_section_coverage"],
                )
                for x in modes
            ],
        )
        mode = min(proven, key=lambda x: abs(x["rho"] - 2.0))
        self.assertAlmostEqual(mode["rho"], 2.0, places=5)
        self.assertGreaterEqual(mode["observed_section_coverage"], 0.8)
        self.assertFalse(mode["steel_certified"])
        self.assertTrue(mode["source_point_indexes"])
        self.assertLess(
            result["member_partition"]["trusted_member_count"],
            result["member_partition"]["original_member_count"],
        )

    def test_steep_material_ridge_cannot_become_steel(self):
        result, _ = self.run_search("ridge")
        self.assertFalse(
            any(
                x["qualified_for_recovery"]
                for x in result["candidates"].get("V0", {}).get("candidates", [])
            )
        )

    def test_water_without_inner_edge_remains_unresolved(self):
        result, _ = self.run_search("water")
        self.assertFalse(
            any(
                x["qualified_for_recovery"]
                for x in result["candidates"].get("V0", {}).get("candidates", [])
            )
        )

    def test_foreign_vertical_structure_cannot_supply_a_raw_face(self):
        result, m = self.run_search("foreign_wall")
        self.assertGreater(result["contamination_raw_count"], 0)
        self.assertFalse(
            any(
                x["qualified_for_recovery"]
                for x in result["candidates"].get("V0", {}).get("candidates", [])
            )
        )

    def test_missing_member_provenance_does_not_guess_an_anchor(self):
        m, r, c, p = fixture()
        r.identity_evidence = {}
        result = InteriorRimSearch(LocalRimEvidence(m, resolve_config()[0])).search(
            c, p, {"R": r}, [c]
        )
        self.assertEqual(result["status"], "CELL_MEMBER_PROVENANCE_INCOMPLETE")

    def test_unexplained_boundary_member_is_not_silently_deleted(self):
        m, r, c, p = fixture()
        cfg = resolve_config()[0]
        # This original occupied member is neither the observed deck nor face.
        m.raw_uvz = np.vstack((m.raw_uvz, [[5.13, 1.17, -0.41]]))
        m.raw_point_indexes = np.arange(len(m.raw_uvz))
        m.local.vessel.point_indexes = m.raw_point_indexes
        m.spatial = RawSpatialIndex(m.raw_uvz, {})
        assembler = ReviewGeometryAssembler(cfg)
        seed = assembler._result(
            r,
            np.eye(2),
            {
                name: assembler._prior(r, name, axis, pos, sign)
                for name, axis, pos, sign in SIDES
            },
        )
        legacy = constrain_review_geometry(r, seed, cfg, {})
        search = InteriorRimSearch(LocalRimEvidence(m, cfg))
        report = search.search(c, p, {"R": r}, [c])
        self.assertIsNone(
            recovery_ab(r, c, legacy, seed, report, search, assembler, cfg, {})
        )
        self.assertGreater(
            report["shadow_ab"]["members"]["unexplained_clipped_count"], 0
        )
        self.assertIn(
            len(m.raw_uvz) - 1,
            report["shadow_ab"]["members"]["unexplained_clipped_raw_indexes"],
        )

    def test_raw_search_preserves_physical_line_under_permutation_and_vertical_origin(
        self,
    ):
        baseline, _ = self.run_search()
        expected = next(
            x
            for x in baseline["candidates"]["V0"]["candidates"]
            if x["qualified_for_recovery"]
        )
        m, r, c, p = fixture()
        order = np.random.default_rng(31).permutation(len(m.raw_uvz))
        m.raw_uvz = m.raw_uvz[order]
        m.raw_uvz[:, 2] += 136.0
        m.raw_point_indexes = m.raw_point_indexes[order]
        m.spatial = RawSpatialIndex(m.raw_uvz, {})
        result = InteriorRimSearch(LocalRimEvidence(m, resolve_config()[0])).search(
            c, p, {"R": r}, [c]
        )
        mode = next(
            x
            for x in result["candidates"]["V0"]["candidates"]
            if x["qualified_for_recovery"]
        )
        self.assertAlmostEqual(mode["rho"], expected["rho"], places=8)
        # Floating point neighbourhood ties can select different returns after
        # a Z origin change; the fitted physical line and observed span cannot.
        self.assertGreater(len(mode["source_point_indexes"]), 0)
        self.assertGreaterEqual(
            mode["observed_section_coverage"],
            resolve_config()[0]["boundary"]["min_edge_coverage"],
        )
        self.assertLessEqual(
            abs(
                mode["observed_section_coverage"]
                - expected["observed_section_coverage"]
            )
            * np.linalg.norm(p[1] - p[0]),
            mode["uncertainty_m"] + expected["uncertainty_m"],
        )

    def test_own_raw_recovery_moves_real_solver_corners_without_relaxing_local_guard(
        self,
    ):
        m, r, c, p = fixture()
        cfg = resolve_config()[0]
        before_region = copy.deepcopy(r.audit())
        assembler = ReviewGeometryAssembler(cfg)
        seed = assembler._result(
            r,
            np.eye(2),
            {
                name: assembler._prior(r, name, axis, pos, sign)
                for name, axis, pos, sign in SIDES
            },
        )
        legacy = constrain_review_geometry(r, seed, cfg, {})
        before = copy.deepcopy(legacy)
        search = InteriorRimSearch(LocalRimEvidence(m, cfg))
        report = search.search(c, p, {"R": r}, [c])
        alternate = recovery_ab(r, c, legacy, seed, report, search, assembler, cfg, {})
        self.assertIsNotNone(alternate, report["shadow_ab"])
        self.assertAlmostEqual(alternate["boundary_uv"][0][1], 2.0, places=5)
        self.assertAlmostEqual(alternate["boundary_uv"][1][1], 2.0, places=5)
        self.assertEqual(cfg["boundary"]["corner_join_m"], 0.3)
        self.assertEqual(legacy, before)
        self.assertEqual(r.audit(), before_region)
        audit = report["shadow_ab"]
        self.assertEqual(audit["status"], "COMMITTED_RAW_ERROR_RECOVERY")
        self.assertEqual(audit["members"]["lost_trusted_count"], 0)
        self.assertEqual(audit["members"]["unexplained_clipped_count"], 0)
        self.assertGreater(
            audit["members"]["reclassified_actual_deck_or_face_count"], 0
        )
        self.assertFalse(alternate["steel_certified"])


if __name__ == "__main__":
    unittest.main()
