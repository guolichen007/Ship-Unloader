"""Outliers, actual competing lines and execution limits remain distinct."""

import unittest
from unittest import mock
import numpy as np
from ship_perception.perception.consensus import SpatialEvidenceConsensus
from ship_perception.perception.final_boundary import guard_move
from ship_perception.perception.line_geometry import guard_line_move
from ship_perception.perception.budget import (
    PerformanceBudget,
    PerformanceBudgetExceeded,
)
from ship_perception.r1_static.run import resolve_config


class ConsensusBudgetContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()

    def test_one_outlier_does_not_invent_a_second_line(self):
        rows = [
            dict(
                along=float(i * 3),
                rho=10.0 + i * 0.03,
                measurement_source=["RAW3D", "HEIGHT", "BEV", "RAW_FACE"][i % 4],
            )
            for i in range(9)
        ]
        rows[4]["rho"] += 2
        rho, detail = SpatialEvidenceConsensus(self.config).fit(rows, 10.0)
        self.assertEqual(detail["status"], "CONSENSUS")
        self.assertEqual(detail["inlier_section_count"], 8)
        self.assertEqual(detail["outlier_section_indexes"], [4])
        self.assertEqual(len(detail["evidence_sources"]), 4)
        self.assertAlmostEqual(detail["slope"], 0.01, places=8)

    def test_two_stable_observed_lines_remain_multimode(self):
        rows = [
            dict(along=float(i * 3), rho=10.0 + shift, measurement_source="RAW3D")
            for shift in (0.0, 2.0)
            for i in range(6)
        ]
        _, detail = SpatialEvidenceConsensus(self.config).fit(rows, 10.0)
        self.assertEqual(detail["status"], "MULTIMODE_BOUNDARY_CONFLICT")
        self.assertTrue(detail["second_stable_line_mode"])

    def test_different_physical_families_cannot_vote_together(self):
        rows = [
            dict(along=i, rho=10.0, physical_line_family=str(i % 2)) for i in range(8)
        ]
        _, detail = SpatialEvidenceConsensus(self.config).fit(rows, 10.0)
        self.assertEqual(detail["reason"], "DISTINCT_PHYSICAL_LINE_FAMILIES")

    def test_outward_move_also_requires_independent_raw_support(self):
        self.assertEqual(guard_move(0.0, -2.0, 1, 1, False, self.config)[0], 0.0)
        self.assertEqual(
            guard_line_move(0.0, -2.0, 0.0, 0.0, (0, 10), 1, 1, False, self.config)[0],
            0.0,
        )
        self.assertEqual(
            guard_line_move(0.0, -2.0, 0.0, 0.0, (0, 10), 1, 6, True, self.config)[0],
            -2.0,
        )

    def test_precision_budget_interrupts_without_changing_geometry_thresholds(self):
        with mock.patch(
            "ship_perception.perception.budget.time.perf_counter", return_value=0.0
        ) as clock:
            budget = PerformanceBudget()
            with self.assertRaisesRegex(
                PerformanceBudgetExceeded, "PERFORMANCE_BUDGET_EXCEEDED"
            ):
                with budget.stage("FINAL_PRECISION", hard=True):
                    clock.return_value = 20.1
                    budget.check()
            self.assertEqual(budget.events[0]["status"], "PERFORMANCE_BUDGET_EXCEEDED")
