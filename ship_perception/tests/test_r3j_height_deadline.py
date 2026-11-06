"""A soft precision deadline must publish honest partial height, not lose XY."""

import copy
import unittest
from unittest.mock import patch
import test_review_rim_height as raw_fixture
from ship_perception.perception.review_height import ReviewCornerHeight
from ship_perception.perception.budget import PerformanceBudgetExceeded


class HeightDeadline(unittest.TestCase):
    def fixture(self):
        fixture = raw_fixture.ReviewRimHeightContracts()
        fixture.setUp()
        m, result = fixture.fixture()
        return ReviewCornerHeight(m, fixture.config), result

    def test_real_raw_probe_can_defer_without_losing_original_geometry(self):
        estimator, result = self.fixture()
        before = copy.deepcopy(result)

        class Expired:
            def check(self):
                raise PerformanceBudgetExceeded("PRECISION_DEFERRED_FOR_REVIEW_OUTPUT")

        estimator.measurement.budget = Expired()
        rows, deferred = estimator.measure_for_review(result, allow_defer=True)
        self.assertTrue(deferred)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["z_m"] is None for r in rows))
        self.assertEqual(result, before)
        self.assertEqual(estimator.audit["status"], "DEFERRED_PERFORMANCE")
        self.assertTrue(estimator.audit["partial_raw_evidence_preserved"])

    def test_completed_current_raw_height_and_source_are_retained(self):
        estimator, result = self.fixture()
        original = estimator.measure(result)
        good = copy.deepcopy(original[0])
        self.assertIsNotNone(good["z_m"])

        def expire(*args, **kwargs):
            estimator._completed_corners = [good]
            raise PerformanceBudgetExceeded("PRECISION_DEFERRED_FOR_REVIEW_OUTPUT")

        with patch.object(estimator, "measure", side_effect=expire):
            rows, deferred = estimator.measure_for_review(result, allow_defer=True)
        self.assertTrue(deferred)
        self.assertEqual(rows[0], good)
        self.assertTrue(all(r["z_m"] is None for r in rows[1:]))
        self.assertEqual(estimator.audit["deferred_corner_indexes"], [1, 2, 3])

    def test_non_review_and_unrelated_budget_errors_keep_raising(self):
        estimator, result = self.fixture()
        for allow, code in [
            (False, "PRECISION_DEFERRED_FOR_REVIEW_OUTPUT"),
            (True, "OTHER_BUDGET_FAILURE"),
        ]:
            with patch.object(
                estimator, "measure", side_effect=PerformanceBudgetExceeded(code)
            ):
                with self.assertRaises(PerformanceBudgetExceeded):
                    estimator.measure_for_review(result, allow_defer=allow)

    def test_solver_or_programming_errors_are_not_masked_as_height_deferred(self):
        estimator, result = self.fixture()
        with patch.object(estimator, "measure", side_effect=ValueError("bad geometry")):
            with self.assertRaisesRegex(ValueError, "bad geometry"):
                estimator.measure_for_review(result, allow_defer=True)


if __name__ == "__main__":
    unittest.main()
