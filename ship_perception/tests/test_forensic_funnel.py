"""Lifecycle diagnostics distinguish absent measurements from measured failure."""

import unittest

from ship_perception.tools.forensic_funnel import funnel


class ForensicFunnelContracts(unittest.TestCase):
    def result(self):
        proposals = [
            dict(
                proposal_id="LEGACY:B2",
                expert="LEGACY",
                bounds_axial=[0, 0, 20, 10],
                evidence={},
            ),
            dict(
                proposal_id="STRUCTURAL:S",
                expert="STRUCTURAL",
                bounds_axial=[0, 0, 20, 10],
                evidence={},
            ),
            dict(
                proposal_id="HEIGHT:H",
                expert="HEIGHT",
                bounds_axial=[21, 0, 40, 10],
                evidence={},
            ),
        ]
        return dict(
            per_vessel_status=[
                dict(vessel_hypothesis_id="V", proposal_registry=proposals)
            ],
            physical_candidate_registry=[
                dict(
                    query_id="Q0",
                    source_proposal_ids=["LEGACY:B2", "STRUCTURAL:S"],
                    bounds_axial=[0, 0, 20, 10],
                    state="PARTIAL",
                    precision_allowed=True,
                    reason="RAW_INVESTIGATION",
                ),
                dict(
                    query_id="Q1",
                    source_proposal_ids=["HEIGHT:H"],
                    bounds_axial=[21, 0, 40, 10],
                    state="REJECTED",
                    precision_allowed=False,
                    reason="STRONG_EXTERIOR_WATER_CONNECTION",
                ),
            ],
            precision_attempts=[],
            geometry_only=dict(hatches=[]),
            input=dict(semantic_sha256="input"),
            software_git_sha="source",
            software_source_identity={},
        )

    def test_every_proposal_keeps_first_blocker_and_shared_query_identity(self):
        report = funnel(self.result(), 4)
        self.assertEqual(len(report["proposal_trace"]), 3)
        first, second, exterior = report["proposal_trace"]
        self.assertEqual(first["physical_query_id"], second["physical_query_id"])
        self.assertEqual(first["first_blocking_stage"], "FULL_PRECISION")
        self.assertEqual(report["full_precision"]["budget_deferred"], 1)
        self.assertEqual(exterior["first_blocking_stage"], "PHYSICAL_VALIDATION")
        self.assertEqual(
            exterior["first_blocking_reason"], "STRONG_EXTERIOR_WATER_CONNECTION"
        )
        self.assertEqual(report["exact_duplicate_count"], 1)

    def test_baseline_without_probe_never_claims_zero_observed_raw_edges(self):
        report = funnel(self.result(), 4)
        self.assertIsNone(report["raw_probe"]["attempted"])
        self.assertIsNone(report["raw_probe"]["3-side"])
        self.assertFalse(report["raw_probe"]["instrumented"])

    def test_zero_or_negative_physical_sections_cannot_count_as_supported_side(self):
        result = self.result()
        side = dict(
            independent_geometry_support=True,
            raw_subvoxel_section_count=0,
            positive_height_section_count=0,
            support_section_count=0,
            phase_runs=[dict(physical_section_count=-1)],
        )
        result["precision_attempts"] = [
            dict(
                query_id="Q0",
                result=dict(sides={name: side for name in ("U0", "U1", "V0", "V1")}),
            )
        ]
        report = funnel(result, 4)
        self.assertEqual(report["full_precision"]["unresolved"], 1)
        self.assertEqual(report["full_precision"]["accepted"], 0)
        self.assertEqual(
            report["proposal_trace"][0]["first_blocking_reason"],
            "INSUFFICIENT_FINAL_PHYSICAL_SIDES",
        )
