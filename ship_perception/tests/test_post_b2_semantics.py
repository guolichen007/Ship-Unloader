"""Semantic Golden cannot be satisfied by retained IDs without physical evidence."""

import unittest
from ship_perception.tools.check_post_b2_semantics import check


class PostB2SemanticGate(unittest.TestCase):
    def data(self):
        expected = dict(
            candidate_id="legacy:H000",
            b2_proposal_id="V:H0",
            separator_topology="UNSPLIT",
            support_side_count=4,
            side_evidence_availability={
                name: dict(raw3d=True, bev=True) for name in ("U0", "U1", "V0", "V1")
            },
            profile_observations_available=True,
        )
        golden = dict(
            scenes=[
                dict(input_sha256="input", candidate_count=1, candidates=[expected])
            ]
        )
        evidence = dict(
            query_id="C",
            source_proposal_ids=["LEGACY:V:H0"],
            separator_topology="UNSPLIT",
            support_side_count=4,
            side_evidence_availability={
                name: True for name in ("U0", "U1", "V0", "V1")
            },
            profile_observations=[dict(evidence_type="OBSERVED_PROFILE_BREAK")],
            local_physical_candidate_available=True,
        )
        result = dict(
            input=dict(semantic_sha256="input"),
            software_git_sha="sha",
            compiled_candidate_registry=[dict(group_id="C")],
            per_vessel_status=[dict(legacy_post_b2_evidence=[evidence])],
        )
        return result, golden, evidence

    def test_retained_b2_identity_without_local_profile_does_not_pass(self):
        result, golden, evidence = self.data()
        evidence["profile_observations"] = []
        report = check(result, golden)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "LOCAL_PROFILE_EVIDENCE_NOT_RECOVERED",
            report["candidate_checks"][0]["failures"],
        )

    def test_measured_semantic_availability_can_pass_without_certifying_steel(self):
        result, golden, evidence = self.data()
        report = check(result, golden)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["steel_edge_certified"])

    def test_unassigned_historical_path_is_explicit_failure(self):
        result, golden, evidence = self.data()
        golden["scenes"][0]["candidates"][0]["b2_proposal_id"] = None
        report = check(result, golden)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("LINEAGE_UNASSIGNED", report["candidate_checks"][0]["reason"])
