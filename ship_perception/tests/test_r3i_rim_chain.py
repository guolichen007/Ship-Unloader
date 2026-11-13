"""Physical role counterexamples and a real solver A/B using synthetic Raw."""

import copy
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from scipy.spatial import cKDTree
from ship_perception.perception.local_rim_evidence import (
    LocalRimEvidence,
    gravity_surfaces,
    isotropic_face,
    shadow_rectangle_ab,
    shadow_review_only_cell,
)
from ship_perception.perception.measurement import RawSpatialIndex
from ship_perception.perception.review_geometry import ReviewGeometryAssembler, SIDES
from ship_perception.r1_static.run import resolve_config


class LocalRimChainContracts(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()

    def fixture(self, lower=-1.0, offset=0.18, length=6.0):
        s = np.arange(0.0, length + 0.001, 0.05)
        ss, tt = np.meshgrid(s, np.arange(-0.8, offset, 0.05))
        deck = np.column_stack((tt.ravel(), ss.ravel(), np.zeros(ss.size)))
        ss, tt = np.meshgrid(s, np.arange(offset + 0.05, 2.001, 0.05))
        inside = np.column_stack((tt.ravel(), ss.ravel(), np.full(ss.size, lower)))
        ss, zz = np.meshgrid(s, np.linspace(min(0.0, lower), max(0.0, lower), 31))
        wall = np.column_stack((np.full(ss.size, offset), ss.ravel(), zz.ravel()))
        xyz = np.vstack((deck, inside, wall))
        ids = np.arange(len(xyz), dtype=np.int64)
        vessel = SimpleNamespace(point_indexes=ids, axes=np.eye(2), vessel_id="V")
        measurement = SimpleNamespace(
            raw_uvz=xyz,
            raw_point_indexes=ids,
            spatial=RawSpatialIndex(xyz, {}),
            local=SimpleNamespace(vessel=vessel),
        )
        members = np.r_[
            np.zeros(len(deck), bool),
            np.ones(len(inside), bool),
            np.zeros(len(wall), bool),
        ]
        return (
            measurement,
            members,
            np.array([[0.0, 0.0], [10.0, 0.0], [10.0, length], [0.0, length]]),
        )

    def test_mixed_surfaces_preserve_deck_despite_dense_vertical_returns(self):
        measurement, _, _ = self.fixture()
        xyz = measurement.raw_uvz[:, [1, 0, 2]]
        wall = xyz[abs(xyz[:, 1] - 0.18) < 1e-8]
        points = np.vstack((xyz, np.repeat(wall, 10, axis=0)))
        surfaces = gravity_surfaces(points, np.arange(len(points)), self.config)
        self.assertTrue(
            any(
                abs(s["coefficients"][2]) < 0.05 and s["cross_span_m"] >= 0.4
                for s in surfaces
            )
        )

    def test_full_normal_neighbourhood_rejects_thin_slice_of_cargo(self):
        s, t = np.meshgrid(np.arange(0, 4, 0.05), np.arange(-1, 1, 0.05))
        points = np.column_stack((s.ravel(), t.ravel(), 0.7 * t.ravel()))
        ids = np.arange(len(points))
        face = isotropic_face(
            points, ids, points, ids, cKDTree(points), 0.0, self.config
        )
        self.assertFalse(face["qualified"])

    def test_deck_face_and_current_members_nominate_actual_raw_line(self):
        for lower in (-1.0, 1.0):
            measurement, members, polygon = self.fixture(lower)
            entry = LocalRimEvidence(measurement, self.config).side(
                polygon, "U0", members, np.zeros(len(members), bool)
            )
            self.assertEqual(len(entry["side_candidates"]), 1)
            mode = entry["side_candidates"][0]
            self.assertTrue(mode["qualified_for_shadow"])
            self.assertAlmostEqual(mode["rho"], 0.18, places=5)
            self.assertFalse(mode["steel_certified"])
            self.assertGreaterEqual(mode["uncertainty_m"], 0.05)

    def test_no_frozen_member_attachment_cannot_promote_geometry(self):
        measurement, _, polygon = self.fixture()
        entry = LocalRimEvidence(measurement, self.config).side(polygon, "U0")
        self.assertEqual(entry["side_candidates"], [])
        self.assertIsNone(entry["new_line"])

    def test_partial_disjoint_measurements_do_not_certify_the_full_side(self):
        measurement, members, polygon = self.fixture(length=24.0)
        members &= measurement.raw_uvz[:, 1] < 12.0
        entry = LocalRimEvidence(measurement, self.config).side(
            polygon, "U0", members, np.zeros(len(members), bool)
        )
        self.assertEqual(entry["state"], "PROVISIONAL_INNER_RIM_SEGMENTS_OBSERVED")
        self.assertEqual(entry["measurement_state"], "estimated")
        self.assertTrue(entry["side_candidates"])
        self.assertFalse(entry["side_candidates"][0]["qualified_for_shadow"])
        self.assertIsNone(entry["new_line"])

    def test_material_on_both_sides_is_not_inner_rim(self):
        measurement, members, polygon = self.fixture()
        entry = LocalRimEvidence(measurement, self.config).side(
            polygon, "U0", np.ones(len(members), bool), np.zeros(len(members), bool)
        )
        self.assertEqual(entry["side_candidates"], [])

    def test_parallel_foreign_returns_cannot_change_owned_measurement(self):
        measurement, members, polygon = self.fixture()
        size = len(members)
        foreign = measurement.raw_uvz.copy()
        foreign[:, 0] += 0.25
        measurement.raw_uvz = np.vstack((measurement.raw_uvz, foreign))
        measurement.raw_point_indexes = np.arange(2 * size)
        measurement.spatial = RawSpatialIndex(measurement.raw_uvz, {})
        members = np.r_[members, np.ones(size, bool)]
        entry = LocalRimEvidence(measurement, self.config).side(
            polygon, "U0", members, np.zeros(2 * size, bool)
        )
        self.assertGreater(entry["contamination_raw_count"], 0)
        self.assertAlmostEqual(entry["side_candidates"][0]["rho"], 0.18, places=5)
        self.assertTrue(
            all(i < size for i in entry["side_candidates"][0]["source_point_indexes"])
        )

    def test_local_chain_commutes_with_rigid_transform_and_point_permutation(self):
        measurement, members, polygon = self.fixture()
        theta = 0.31
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        shift = np.array([1150.123, -730.456])
        order = np.random.default_rng(4).permutation(len(members))
        xyz = measurement.raw_uvz.copy()
        xyz[:, :2] = xyz[:, :2] @ rotation.T + shift
        xyz[:, 2] += 17.317
        measurement.raw_uvz = xyz[order]
        measurement.raw_point_indexes = order
        measurement.spatial = RawSpatialIndex(measurement.raw_uvz, {})
        entry = LocalRimEvidence(measurement, self.config).side(
            polygon @ rotation.T + shift,
            "U0",
            members[order],
            np.zeros(len(members), bool),
        )
        self.assertEqual(len(entry["side_candidates"]), 1)
        mode = entry["side_candidates"][0]
        expected = rotation[:, 0]
        np.testing.assert_allclose(mode["normal_uv"], expected, atol=1e-8, rtol=0)
        self.assertAlmostEqual(mode["rho"] - expected @ shift, 0.18, places=7)

    def test_verified_new_line_reaches_actual_four_corner_solver(self):
        measurement, members, polygon = self.fixture()
        rim = LocalRimEvidence(measurement, self.config)
        entry = rim.side(polygon, "U0", members, np.zeros(len(members), bool))
        region = SimpleNamespace(
            region_id="R",
            vessel_id="V",
            conservative_bounds=(0.0, 0.0, 10.0, 6.0),
            identity_evidence={},
            identity_role="CONFIRMED_CONNECTED_OPENING",
            source_proposal_ids=[],
        )
        assembler = ReviewGeometryAssembler(self.config)
        sides = {
            name: assembler._prior(region, name, axis, pos, sign)
            for name, axis, pos, sign in SIDES
        }
        candidate = assembler._result(region, np.eye(2), sides)
        before = copy.deepcopy(candidate)
        with patch(
            "ship_perception.perception.opening_rim_source.member_mask",
            return_value=members,
        ):
            result = shadow_rectangle_ab(
                region,
                candidate,
                {"U0": entry},
                measurement,
                assembler,
                self.config,
                {},
            )
        self.assertEqual(result["local_rim_shadow_ab"]["status"], "COMMITTED_GEOMETRY")
        self.assertAlmostEqual(result["boundary_xy"][0][0], 0.18, places=5)
        self.assertAlmostEqual(result["boundary_xy"][3][0], 0.18, places=5)
        self.assertEqual(candidate, before)
        self.assertFalse(result["steel_edge_certified"])

    def review_fixture(self, separator=None):
        from ship_perception.perception.region_model import (
            OpeningRegion,
            TopologySupport,
        )

        measurement, members, polygon = self.fixture()
        evidence = {
            "U0": LocalRimEvidence(measurement, self.config).side(
                polygon, "U0", members, np.zeros(len(members), bool)
            )
        }
        identity = (
            {}
            if separator is None
            else {"side_ownership_constraints": {"U0": {"lower_u": separator}}}
        )
        region = OpeningRegion(
            "R",
            "V",
            (0.0, 0.0, 10.0, 6.0),
            (0.0, 0.0, 10.0, 6.0),
            (TopologySupport("STRUCTURAL_LINE_FAMILY", "SUPPORTING"),),
            identity_role="CONFIRMED_CONNECTED_OPENING",
            identity_evidence=identity,
        )
        cell = dict(
            cell_id="OC:R",
            vessel_id="V",
            primary_region_id="R",
            candidate_ids=["R"],
            bounds_uv=[0.0, 0.0, 10.0, 6.0],
            separator_constraints=(
                [] if separator is None else [dict(bound="lower_u", value=separator)]
            ),
        )
        assembler = ReviewGeometryAssembler(self.config)
        sides = {
            name: assembler._prior(region, name, axis, pos, sign)
            for name, axis, pos, sign in SIDES
        }
        candidate = assembler._result(region, np.eye(2), sides)
        return measurement, members, region, cell, assembler, candidate, evidence

    def test_verified_side_can_resolve_infeasible_legacy_without_removing_separator(
        self,
    ):
        measurement, members, region, cell, assembler, candidate, evidence = (
            self.review_fixture(0.1)
        )
        with patch(
            "ship_perception.perception.opening_rim_source.member_mask",
            return_value=members,
        ):
            result = shadow_rectangle_ab(
                region,
                candidate,
                evidence,
                measurement,
                assembler,
                self.config,
                {},
                opening_cell=cell,
            )
        self.assertEqual(
            result["local_rim_shadow_ab"]["legacy_solver_state"], "INFEASIBLE"
        )
        self.assertEqual(result["local_rim_shadow_ab"]["status"], "COMMITTED_GEOMETRY")
        self.assertAlmostEqual(result["boundary_xy"][0][0], 0.18, places=5)
        self.assertTrue(result["constrained_review_rectangle"]["separator_constraints"])

    def test_solver_cannot_trade_verified_raw_residual_for_a_feasible_rectangle(self):
        from ship_perception.perception.constrained_rectangle import RectangleInfeasible

        measurement, members, region, cell, assembler, candidate, evidence = (
            self.review_fixture(0.4)
        )
        with patch(
            "ship_perception.perception.opening_rim_source.member_mask",
            return_value=members,
        ):
            with self.assertRaises(RectangleInfeasible):
                shadow_rectangle_ab(
                    region,
                    candidate,
                    evidence,
                    measurement,
                    assembler,
                    self.config,
                    {},
                    opening_cell=cell,
                )
        self.assertIsNone(evidence["U0"]["new_line"])

    def test_verified_side_can_promote_existing_review_only_identity(self):
        measurement, members, region, cell, assembler, _, evidence = (
            self.review_fixture()
        )
        before = copy.deepcopy(region.audit())
        with patch(
            "ship_perception.perception.opening_rim_source.member_mask",
            return_value=members,
        ):
            result = shadow_review_only_cell(
                region,
                cell,
                evidence,
                {"R": region},
                measurement,
                assembler,
                self.config,
                {},
            )
        self.assertIsNotNone(result)
        self.assertEqual(result["opening_identity"], "R")
        self.assertEqual(result["local_rim_shadow_ab"]["status"], "COMMITTED_GEOMETRY")
        self.assertEqual(region.audit(), before)

    def test_review_only_cell_without_complete_new_chain_stays_review_only(self):
        measurement, _, region, cell, assembler, _, evidence = self.review_fixture()
        evidence["U0"]["side_candidates"][0]["qualified_for_shadow"] = False
        result = shadow_review_only_cell(
            region,
            cell,
            evidence,
            {"R": region},
            measurement,
            assembler,
            self.config,
            {},
        )
        self.assertIsNone(result)
        self.assertIsNone(evidence["U0"]["new_line"])

    def test_review_only_cell_requires_every_alias_member_provenance(self):
        measurement, members, region, cell, assembler, _, evidence = (
            self.review_fixture()
        )
        cell["candidate_ids"].append("MISSING")
        with patch(
            "ship_perception.perception.opening_rim_source.member_mask",
            return_value=members,
        ):
            result = shadow_review_only_cell(
                region,
                cell,
                evidence,
                {"R": region},
                measurement,
                assembler,
                self.config,
                {},
            )
        self.assertIsNone(result)
        self.assertIn("PROVENANCE_INCOMPLETE", evidence["U0"]["selected_reason"])

    def test_new_side_cannot_drop_another_alias_of_the_same_cell(self):
        from dataclasses import replace

        measurement, members, region, cell, assembler, _, evidence = (
            self.review_fixture()
        )
        alias = replace(region, region_id="A")
        cell["candidate_ids"].append("A")
        alias_members = (measurement.raw_uvz[:, 0] > 0.01) & (
            measurement.raw_uvz[:, 0] < 0.15
        )

        def frozen_mask(points, source):
            return members if source.region_id == "R" else alias_members

        with patch(
            "ship_perception.perception.opening_rim_source.member_mask",
            side_effect=frozen_mask,
        ):
            result = shadow_review_only_cell(
                region,
                cell,
                evidence,
                {"R": region, "A": alias},
                measurement,
                assembler,
                self.config,
                {},
            )
        self.assertIsNone(result)
        self.assertIsNone(evidence["U0"]["new_line"])

    def test_foreign_vessel_alias_cannot_enter_the_review_only_solver(self):
        from dataclasses import replace

        measurement, members, region, cell, assembler, _, evidence = (
            self.review_fixture()
        )
        alias = replace(region, region_id="A", vessel_id="OTHER")
        cell["candidate_ids"].append("A")
        with patch(
            "ship_perception.perception.opening_rim_source.member_mask",
            return_value=members,
        ):
            result = shadow_review_only_cell(
                region,
                cell,
                evidence,
                {"R": region, "A": alias},
                measurement,
                assembler,
                self.config,
                {},
            )
        self.assertIsNone(result)
        self.assertIn("FOREIGN_VESSEL", evidence["U0"]["selected_reason"])


if __name__ == "__main__":
    unittest.main()
