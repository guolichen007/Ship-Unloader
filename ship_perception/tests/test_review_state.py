"""An incomplete or risky boundary must never get a cosmetic promotion."""

import copy
import unittest
from ship_perception.perception.review_state import apply_review_state


class ReviewStates(unittest.TestCase):
    def complete(self):
        return dict(
            status="PARTIAL_HATCH",
            review_support_side_count=4,
            undercoverage_risk=False,
            boundary_xy=[[0, 0], [20, 0], [20, 12], [0, 12]],
            corner_geometry=dict(intersection_stable=True),
            sides={
                n: dict(
                    independent_geometry_support=True,
                    undercoverage_risk=False,
                    line=dict(source="RAW_SECTION_ROBUST_LINE"),
                )
                for n in ("U0", "U1", "V0", "V1")
            },
        )

    def test_geometry_does_not_certify_steel_or_operator_confirmation(self):
        result = apply_review_state(self.complete())
        self.assertEqual(result["geometry_status"], "VALIDATED")
        self.assertEqual(result["steel_edge_benchmark_status"], "NOT_CERTIFIED")
        self.assertEqual(result["operator_confirmation"], "PENDING")

    def test_three_sides_undercoverage_or_conservative_line_stay_partial(self):
        incomplete = self.complete()
        incomplete["review_support_side_count"] = 3
        risky = self.complete()
        risky["undercoverage_risk"] = True
        inferred = self.complete()
        inferred["sides"]["V1"]["line"]["source"] = "CONSERVATIVE_AXIS_LINE"
        unsupported = self.complete()
        unsupported["sides"]["V1"]["independent_geometry_support"] = False
        for hatch in (incomplete, risky, inferred, unsupported):
            before = copy.deepcopy(hatch["boundary_xy"])
            result = apply_review_state(hatch)
            self.assertEqual(result["geometry_status"], "PARTIAL")
            self.assertEqual(result["status"], "PARTIAL_HATCH")
            self.assertEqual(result["boundary_xy"], before)
