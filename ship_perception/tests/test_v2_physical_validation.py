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
