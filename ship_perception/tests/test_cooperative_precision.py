"""Scheduling yields preserve evidence and leave geometry decisions unchanged."""

from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from ship_perception.perception.budget import (
    PerformanceBudget,
    CandidateDeferredPerformance,
    PerformanceBudgetExceeded,
)
from ship_perception.perception.final_boundary import FinalBoundaryRefiner
from ship_perception.perception.precision_scheduler import PrecisionScheduler
from ship_perception.r1_static.run import resolve_config
from ship_perception.tests.test_precision_diagnostics import Clock


class CooperativePrecision(unittest.TestCase):
    def test_candidate_yield_does_not_consume_other_regions_global_budget(self):
        clock = Clock()
        with patch("ship_perception.perception.budget.time.perf_counter", clock):
            budget = PerformanceBudget()
            with budget.stage("FINAL_PRECISION", hard=True):
                with self.assertRaises(CandidateDeferredPerformance):
                    with budget.candidate_slice(3):
                        clock.advance(5.01)
                        budget.check()
                # A different region can execute after the local yield.
                with budget.candidate_slice(2):
                    clock.advance(1)
                    budget.check()
                with self.assertRaises(PerformanceBudgetExceeded):
                    clock.advance(15)
                    budget.check()

    def test_next_physical_region_runs_before_deferred_candidate_resumes(self):
        config, _ = resolve_config()
        ready = []
        for i in range(3):
            q = SimpleNamespace(
                query_id="Q" + str(i),
                vessel_id="V",
                bounds=(50 * i, 0, 50 * i + 30, 12),
            )
            ready.append((q, {}))
        scheduler = PrecisionScheduler(ready, config)
        iterator = iter(scheduler)
        first, row = next(iterator)
        scheduler.defer_performance(first, row)
        second, _ = next(iterator)
        third, _ = next(iterator)
        resumed, again = next(iterator)
        self.assertEqual(
            [first.query_id, second.query_id, third.query_id, resumed.query_id],
            ["Q0", "Q1", "Q2", "Q0"],
        )
        self.assertEqual(again["precision_disposition"], "DEFERRED_PERFORMANCE")
        self.assertNotEqual(again.get("state"), "REJECTED")

    def test_only_completed_side_progress_is_retained_and_is_immutable(self):
        config, _ = resolve_config()
        refiner = FinalBoundaryRefiner(
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), [], config
        )
        output = {
            "U0": dict(
                status="NO_CORROBORATED_INNER_REPLACEMENT", rho=1.0, provenance=[4, 8]
            )
        }
        with refiner._role_side_progress("key", "U0", output):
            pass
        output["U0"]["provenance"].append(99)
        self.assertEqual(refiner.role_progress_cache["key"]["U0"]["provenance"], [4, 8])
        with self.assertRaises(CandidateDeferredPerformance):
            with refiner._role_side_progress("key", "U1", {}):
                raise CandidateDeferredPerformance("DEFERRED_PERFORMANCE")
        self.assertNotIn("U1", refiner.role_progress_cache["key"])
