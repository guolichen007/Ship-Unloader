"""Lossless expert union and physical identity, independently of ranking."""

import unittest
from ship_perception.perception.candidate_union import (
    CandidateProposal,
    candidate_union,
    physical_queries,
)
from ship_perception.perception.legacy_expert import (
    LegacyProposalBatch,
    verify_legacy_sources,
)
from ship_perception.r1_static.run import resolve_config


class CandidateUnionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()

    def proposal(self, expert, vessel="V0", box=(0, 0, 20, 10), separator=None):
        return CandidateProposal.from_evidence(
            expert, vessel, dict(bounds_axial=box, measured_separator_faces=separator)
        )

    def test_expert_absence_cannot_delete_another_experts_proposal(self):
        original = self.proposal("LEGACY")
        union = candidate_union([(original,), ()])
        self.assertEqual(union, (original,))
        copy = original.evidence()
        copy["bounds_axial"][0] = 999
        self.assertEqual(original.bounds[0], 0)

    def test_near_duplicates_preserve_every_source_and_original_proposal(self):
        proposals = tuple(self.proposal(e) for e in ("LEGACY", "HEIGHT", "ENCLOSURE"))
        queries = physical_queries(proposals, self.config)
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].proposals, proposals)
        self.assertEqual(len(queries[0].sources), 3)

    def test_large_envelope_or_other_ship_does_not_erase_smaller_opening(self):
        proposals = [
            self.proposal("HEIGHT"),
            self.proposal("LEGACY", box=(-2, -2, 30, 15)),
            self.proposal("HEIGHT", "V1"),
        ]
        self.assertEqual(len(physical_queries(proposals, self.config)), 3)

    def test_distinct_separator_modes_do_not_merge(self):
        proposals = [
            self.proposal("HEIGHT", separator=[9, 10]),
            self.proposal("LEGACY", separator=[8, 11]),
        ]
        self.assertEqual(
            len(
                physical_queries(proposals, self.config, lambda _: ("A", "B", "C", "D"))
            ),
            2,
        )

    def test_legacy_sources_and_output_are_frozen(self):
        self.assertEqual(
            verify_legacy_sources(), "51410e3f4f932fcbdcc7fc88b8f9979c60c2b954"
        )
        batch = LegacyProposalBatch('{"rows":[1]}', "identity", ())
        copy = batch.raw_output()
        copy["rows"].append(2)
        self.assertEqual(batch.raw_output(), {"rows": [1]})

    def test_frozen_legacy_matches_recorded_synthetic_proposals(self):
        import json
        from pathlib import Path
        import numpy as np
        from ship_perception.io.frame import PointCloudFrame
        from ship_perception.tests.test_hatch_metamorphic import vessel_points
        from ship_perception.perception.context import VesselEvidence, freeze
        from ship_perception.perception.local_evidence import VesselLocalEvidence
        from ship_perception.perception.legacy_expert import LegacyProposalExpert
        from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG

        xyz, _ = vessel_points(1, spacing=0.2)
        frame = PointCloudFrame(xyz, None, "map", 0, 0, "", ("x", "y", "z"), len(xyz))
        vessel = VesselEvidence(
            freeze(
                dict(
                    vessel_hypothesis_id="V_SYNTHETIC",
                    local_axes=[[1, 0], [0, 1]],
                    classification_status="VESSEL_HYPOTHESIS",
                    source_support_ids=["S_SYNTHETIC"],
                    point_count=len(xyz),
                    bbox_xy=[*xyz[:, :2].min(0), *xyz[:, :2].max(0)],
                )
            ),
            np.arange(len(xyz)),
            "synthetic",
        )
        result = LegacyProposalExpert().propose(
            VesselLocalEvidence.build(frame, vessel, self.config),
            json.loads(RESEARCH_CONFIG.read_bytes()),
        )
        summary = [
            dict(
                bounds_axial=[round(x, 6) for x in p.bounds],
                state=p.evidence()["state"],
                measured_strong_side_count=p.evidence()["measured_strong_side_count"],
                role_conflicts=p.evidence()["role_conflicts"],
            )
            for p in result.proposals
        ]
        golden = json.loads(
            (
                Path(__file__).parent / "fixtures/legacy_proposal_golden.json"
            ).read_bytes()
        )
        self.assertEqual(summary, golden["proposal_summary"])
