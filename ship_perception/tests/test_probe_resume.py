"""A scheduling yield must not restart completed physical measurements."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ship_perception.perception.budget import CandidateDeferredPerformance
from ship_perception.perception.final_boundary import FinalBoundaryRefiner
from ship_perception.perception.measurement import (
    VesselMeasurementContext,
    RawSectionProfile,
)
from ship_perception.perception.raw_boundary_probe import SignedRoleResolver
from ship_perception.r1_static.run import resolve_config
from ship_perception.tests.test_hatch_metamorphic import vessel_points


class VisitBudget:
    def __init__(self, allowance):
        self.allowance = allowance
        self.remaining = allowance

    def check(self):
        if self.remaining <= 0:
            raise CandidateDeferredPerformance("TEST_YIELD")
        self.remaining -= 1

    def next_visit(self):
        self.remaining = self.allowance


class ProbeResume(unittest.TestCase):
    def resolver(self):
        config, _ = resolve_config()
        xyz, _ = vessel_points(1, spacing=0.15)
        local = SimpleNamespace(
            config=config,
            xyz=xyz,
            context_xyz=xyz,
            measurement_context=None,
            vessel=SimpleNamespace(point_indexes=np.arange(len(xyz))),
            structural_providers=lambda: [],
        )
        measurement = VesselMeasurementContext(local)
        refiner = FinalBoundaryRefiner(xyz, [], config, measurement)
        return SignedRoleResolver(measurement, refiner)

    def complete(self, operation, budget):
        for visit in range(100):
            budget.next_visit()
            try:
                return operation(), visit
            except CandidateDeferredPerformance:
                pass
        self.fail("A finite physical search failed to make progress across yields")

    def test_completed_sections_are_not_remeasured_after_yield(self):
        from ship_perception.r1_static.rectangle_refinement import _raw_edge_rho

        expected = self.resolver().side(0, 8.0, (3.0, 14.0), 1, [])
        resolver = self.resolver()
        budget = VisitBudget(1)
        resolver.refiner.budget = budget
        calls = []

        def measured(points, axis, rho, span, sign, config):
            calls.append((axis, rho, tuple(span), sign))
            return _raw_edge_rho(points, axis, rho, span, sign, config)

        with patch(
            "ship_perception.r1_static.rectangle_refinement._raw_edge_rho", measured
        ):
            actual, visits = self.complete(
                lambda: resolver.side(0, 8.0, (3.0, 14.0), 1, []), budget
            )
        self.assertGreater(visits, 0)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["status"], "STRONG_UNIQUE_EDGE")
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(len(calls), 4)

    def test_nomination_centers_resume_without_rebuilding_or_rescanning(self):
        expected = self.resolver().raw_nominations(0, 9.0, (3.0, 14.0), 1)
        resolver = self.resolver()
        budget = VisitBudget(1)
        resolver.refiner.budget = budget
        calls = []
        edge_modes = RawSectionProfile.edge_modes

        def measured(profile, anchor, sign):
            calls.append(anchor)
            return edge_modes(profile, anchor, sign)

        with patch.object(RawSectionProfile, "edge_modes", measured):
            actual, visits = self.complete(
                lambda: resolver.raw_nominations(0, 9.0, (3.0, 14.0), 1), budget
            )
        self.assertGreater(visits, 0)
        self.assertEqual(actual, expected)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(resolver.measurement.counters["raw_profile_build_count"], 1)

    def test_signed_measurements_do_not_share_opposite_polarity(self):
        resolver = self.resolver()
        correct = resolver.side(0, 8.0, (3.0, 14.0), 1, [])
        opposite = resolver.side(0, 8.0, (3.0, 14.0), -1, [])
        self.assertEqual(correct["status"], "STRONG_UNIQUE_EDGE")
        self.assertNotEqual(opposite["status"], "STRONG_UNIQUE_EDGE")

    def test_cheap_missing_pair_does_not_hide_extended_raw_measurement(self):
        resolver = self.resolver()
        with patch(
            "ship_perception.r1_static.rectangle_refinement._raw_edge_rho",
            return_value=(None, 0),
        ), patch.object(resolver, "raw_nominations", return_value=()):
            initial = resolver.side(0, 8.0, (3.0, 14.0), 1, [])
            extended = resolver.side(0, 8.0, (3.0, 14.0), 1, [], extended=True)
        self.assertNotEqual(initial["status"], "STRONG_UNIQUE_EDGE")
        self.assertEqual(extended["status"], "STRONG_UNIQUE_EDGE")

    def test_explicit_compiler_identity_precedes_an_overlapping_fragment(self):
        from ship_perception.perception.candidate_union import (
            CandidateProposal,
            PhysicalBoundaryQuery,
        )
        from ship_perception.perception.opening_regions import (
            OpeningRegionSeed,
            associate_region,
        )
        from ship_perception.perception.precision_scheduler import PrecisionScheduler

        full = OpeningRegionSeed("FULL", "V", (0.0, 0.0, 40.0, 12.0), (), ())
        fragment = OpeningRegionSeed("SMALL", "V", (0.0, 0.0, 10.0, 12.0), (), ())
        proposal = CandidateProposal.from_evidence(
            "STRUCTURAL_LINE",
            "V",
            dict(bounds_axial=full.bounds, opening_region_id="FULL"),
            0,
        )
        query = PhysicalBoundaryQuery("Q", "V", full.bounds, (proposal,), ())
        row = dict()
        associate_region(query, row, [fragment, full])
        self.assertEqual(row["opening_region_id"], "FULL")
        self.assertEqual(row["opening_region_ids"], ["FULL"])
        self.assertEqual(set(row["opening_search_region_ids"]), {"FULL", "SMALL"})
        competing = PhysicalBoundaryQuery("A", "V", fragment.bounds, (), ())
        scheduler = PrecisionScheduler(
            [(competing, dict(opening_region_id="FULL")), (query, row)],
            self.resolver().config,
        )
        self.assertEqual(next(iter(scheduler))[0].query_id, "Q")

    def test_foreign_declared_identity_does_not_get_owned_nomination_priority(self):
        from ship_perception.perception.candidate_union import (
            CandidateProposal,
            PhysicalBoundaryQuery,
        )
        from ship_perception.perception.opening_regions import (
            OpeningRegionSeed,
            associate_region,
        )

        foreign = OpeningRegionSeed("OTHER", "W", (0.0, 0.0, 40.0, 12.0), (), ())
        proposal = CandidateProposal.from_evidence(
            "STRUCTURAL_LINE",
            "V",
            dict(bounds_axial=foreign.bounds, opening_region_id="OTHER"),
            0,
        )
        query = PhysicalBoundaryQuery("Q", "V", foreign.bounds, (proposal,), ())
        row = dict()
        associate_region(query, row, [foreign])
        self.assertFalse(row["region_identity_nomination"])
        self.assertNotEqual(row["opening_region_id"], "OTHER")

    def test_deferred_s4_keeps_finished_evidence_and_existing_identity(self):
        from ship_perception.perception.s4_topology_bridge import (
            S4TopologyBridge,
            S4ObservationDeferred,
        )

        bridge = S4TopologyBridge.__new__(S4TopologyBridge)
        bridge.observation_progress = dict(
            completed_deck_segments=[], completed_profile_segments=[]
        )
        bridge.budget = SimpleNamespace(
            started=0.0,
            total_s=55.0,
            LIMITS=dict(FINAL_PRECISION=20.0),
            check=lambda: None,
        )
        record = dict(segment_id="OBSERVED", deck=dict(status="RESOLVED"))
        with patch(
            "ship_perception.perception.s4_topology_bridge.time.perf_counter",
            return_value=36.0,
        ):
            with self.assertRaises(S4ObservationDeferred):
                bridge._checkpoint("SEGMENT_DECK", record)
        self.assertEqual(
            bridge.observation_progress["completed_deck_segments"], [record]
        )
        bridge.axis = dict(status="DEFERRED_PERFORMANCE")
        prior = SimpleNamespace(region_id="KEEP")
        self.assertEqual(bridge.compile_regions([prior]), [prior])


if __name__ == "__main__":
    unittest.main()
