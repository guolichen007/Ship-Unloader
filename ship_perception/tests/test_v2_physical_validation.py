"""Weak load height remains a raw investigation, never a confirmed opening."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from ship_perception.perception.candidate_union import (
    CandidateProposal,
    PhysicalBoundaryQuery,
)
from ship_perception.perception.physical_validation import UnifiedPhysicalValidation
from ship_perception.perception.measurement import BudgetedLocalSurfaceNormals
from ship_perception.r1_static.raw_surface_normals import LocalSurfaceNormals
from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.run import resolve_config


class PhysicalValidationContracts(unittest.TestCase):
    def test_empty_short_side_and_guarded_search_do_not_count_as_raw_support(self):
        from ship_perception.perception.pipeline_v2 import final_support_sides

        config, _ = resolve_config()
        valid = dict(
            independent_geometry_support=True,
            raw_subvoxel_section_count=3,
            positive_height_section_count=3,
            support_section_count=3,
            phase_runs=[dict(physical_section_count=3)],
        )
        empty = dict(
            valid,
            raw_subvoxel_section_count=0,
            positive_height_section_count=0,
            support_section_count=0,
            phase_runs=[dict(physical_section_count=-1)],
        )
        guarded = dict(valid, independent_geometry_support=False)
        result = dict(sides=dict(U0=empty, U1=guarded, V0=valid, V1=valid))
        self.assertEqual(final_support_sides(result, config), ["V0", "V1"])

    def test_loaded_enclosure_can_be_investigated_without_becoming_validated(self):
        config, _ = resolve_config()
        u, v = np.meshgrid(np.arange(0, 25, 0.5), np.arange(0, 12, 0.5))
        grid = R1HeightGrid.from_points(
            np.column_stack((u.ravel(), v.ravel(), np.zeros(u.size))), 0.5
        )
        cache = SimpleNamespace(
            config=config,
            context_grid=lambda _: grid,
            vessel=SimpleNamespace(vessel_id="V0"),
        )
        exterior = SimpleNamespace(
            review=lambda *args: dict(status="NO_LOWER_INTERIOR_COMPONENT")
        )
        proposal = CandidateProposal.from_evidence(
            "LOADED_HOLD", "V0", dict(bounds_axial=[2, 2, 20, 10])
        )
        query = PhysicalBoundaryQuery("Q", "V0", proposal.bounds, (proposal,), ())
        ring = dict(
            measured_strong_side_count=3,
            positive_height_side_count=0,
            sides={
                name: dict(vertical_face_support=False)
                for name in ("U0", "U1", "V0", "V1")
            },
        )
        opening = dict(
            status="RAISED_INTERIOR_OR_CARGO", outside_side_median_z=[0.0] * 4
        )
        with patch(
            "ship_perception.perception.physical_validation.deck_ring",
            return_value=ring,
        ), patch(
            "ship_perception.perception.physical_validation._opening_evidence",
            return_value=opening,
        ):
            row = UnifiedPhysicalValidation(cache, exterior).evaluate(query)
        self.assertEqual(row["state"], "PARTIAL")
        self.assertTrue(row["precision_allowed"])
        self.assertNotEqual(row["state"], "VALIDATED")

    def test_budgeted_normals_keep_the_exact_frozen_estimator_values(self):
        config, _ = resolve_config()
        xyz = np.random.default_rng(31).normal(size=(600, 3))
        old = LocalSurfaceNormals(xyz, config)
        new = BudgetedLocalSurfaceNormals(xyz, config, None, {})
        ids = np.arange(len(xyz))
        old._estimate(ids)
        new._estimate(ids)
        np.testing.assert_array_equal(new.valid, old.valid)
        np.testing.assert_array_equal(new.normals, old.normals)
        self.assertEqual(new.counters["normal_points_evaluated"], len(xyz))
        new._estimate(ids)
        self.assertEqual(new.counters["normal_points_evaluated"], len(xyz))

    def test_two_coarse_sides_do_not_block_local_raw_probe(self):
        config, _ = resolve_config()
        grid = R1HeightGrid.from_points(
            np.array([[x, y, 0.0] for x in range(25) for y in range(12)]), 0.5
        )
        cache = SimpleNamespace(
            config=config,
            context_grid=lambda _: grid,
            vessel=SimpleNamespace(vessel_id="V0"),
        )
        proposal = CandidateProposal.from_evidence(
            "LEGACY", "V0", dict(bounds_axial=[2, 2, 20, 10])
        )
        query = PhysicalBoundaryQuery("Q", "V0", proposal.bounds, (proposal,), ())
        ring = dict(
            measured_strong_side_count=2,
            positive_height_side_count=0,
            sides={
                name: dict(vertical_face_support=False)
                for name in ("U0", "U1", "V0", "V1")
            },
        )
        exterior = SimpleNamespace(
            review=lambda *args: dict(status="NO_LOWER_INTERIOR_COMPONENT")
        )
        with patch(
            "ship_perception.perception.physical_validation.deck_ring",
            return_value=ring,
        ):
            row = UnifiedPhysicalValidation(cache, exterior).evaluate(query)
        self.assertEqual(row["state"], "UNRESOLVED")
        self.assertTrue(row["raw_edge_probe_allowed"])
        self.assertFalse(row["precision_allowed"])
