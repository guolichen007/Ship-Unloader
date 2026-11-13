"""A slow preview cannot reset the complete CLI's existing 300 s watchdog."""

import unittest
from unittest.mock import patch

from ship_perception.perception.budget import PerformanceBudgetExceeded
from ship_perception.perception.review_pipeline import (
    ProgressOnly,
    review_precision_deadline,
)


class SharedProcessBudget(unittest.TestCase):
    def test_full_phase_keeps_publication_reserve_after_a_slow_preview(self):
        # Worker starts at zero, preview consumes 120 s; watchdog and reserve
        # remain the existing 300/60 s. A per-call reset would allow 360 s.
        deadline = review_precision_deadline(120.0, 300.0, 60.0, process_deadline=300.0)
        callbacks = []
        budget = ProgressOnly(lambda: callbacks.append(True), deadline)
        with patch(
            "ship_perception.perception.review_pipeline.time.perf_counter",
            return_value=239.0,
        ):
            budget.check()
        with patch(
            "ship_perception.perception.review_pipeline.time.perf_counter",
            return_value=241.0,
        ):
            with self.assertRaises(PerformanceBudgetExceeded):
                budget.check()
        self.assertEqual(len(callbacks), 1)

    def test_single_execute_contract_and_nonfinite_deadline(self):
        self.assertEqual(review_precision_deadline(120.0, 300.0, 60.0), 360.0)
        with self.assertRaisesRegex(ValueError, "NONFINITE"):
            review_precision_deadline(0.0, 300.0, 60.0, float("nan"))
