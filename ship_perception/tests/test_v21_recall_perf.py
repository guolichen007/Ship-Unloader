"""Counterexamples for compilation, local probing and exact shared measurements."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from ship_perception.r1_static.run import resolve_config
from ship_perception.perception.candidate_union import (
    CandidateProposal,
    PhysicalBoundaryQuery,
)
from ship_perception.perception.proposal_compiler import ProposalCompiler
from ship_perception.perception.measurement import VesselMeasurementContext
from ship_perception.perception.final_boundary import FinalBoundaryRefiner
from ship_perception.perception.raw_boundary_probe import RawBoundaryProbe
from ship_perception.perception.precision_scheduler import PrecisionScheduler
from ship_perception.perception.boundary_line_graph import BoundaryLineGraph
from ship_perception.tests.test_hatch_metamorphic import vessel_points


class RecallPerfContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()

    def proposal(self, expert="LEGACY", box=(0, 0, 20, 10), ident="P", separator=None):
        return CandidateProposal.from_evidence(
            expert,
            "V",
            dict(
                proposal_id=ident,
                bounds_axial=box,
                source_provider_ids=["A", "B", "C", "D"],
                measured_separator_faces=separator,
            ),
        )

    def test_cargo_height_polarity_cannot_remove_a_verified_face_search_side(self):
        face = dict(
            provider_ids=("face",),
            rho=10.0,
            interval=(0.0, 20.0),
            raw3d=True,
            bev=True,
            normal_face=True,
        )
        graph = BoundaryLineGraph(
            {0: [face], 1: []},
            self.config,
            height=lambda axis, rho, span, sign: sign * 2.0,
        )
        self.assertEqual(set(graph.signs(0, face)), {1, -1})
        nominal = dict(face, provider_ids=("height",), normal_face=False)
        self.assertEqual(graph.signs(0, nominal), [1])

    def test_scheduler_measurable_enclosure_precedes_thin_many_face_strip(self):
        p = self.proposal(ident="broad")
        broad = PhysicalBoundaryQuery("broad", "V", p.bounds, (p,), ())
        p = self.proposal(box=(30, 0, 60, 1), ident="thin")
        thin = PhysicalBoundaryQuery("thin", "V", p.bounds, (p,), ())
        broad_row = dict(legacy_post_b2=dict(support_side_count=3))
        thin_row = dict(
            deck_ring=dict(
                sides={
                    name: dict(vertical_face_support=True)
                    for name in ("U0", "U1", "V0", "V1")
                }
            )
        )
        ordered = list(
            PrecisionScheduler([(thin, thin_row), (broad, broad_row)], self.config)
        )
        self.assertEqual(
            [q.query_id for q, _ in ordered], [broad.query_id, thin.query_id]
        )

    def test_narrow_regions_cannot_preempt_the_next_measurable_hypothesis(self):
        p = self.proposal()
        a = PhysicalBoundaryQuery("A", "V", p.bounds, (p,), ())
        b = PhysicalBoundaryQuery("B", "V", p.bounds, (p,), ())
        p = self.proposal(box=(40, 0, 60, 1))
        narrow = PhysicalBoundaryQuery("N", "V", p.bounds, (p,), ())
        ordered = list(
            PrecisionScheduler([(a, {}), (b, {}), (narrow, {})], self.config)
        )
        self.assertEqual([q.query_id for q, _ in ordered], ["A", "B", "N"])

    def test_parent_overlap_cannot_put_two_adjacent_openings_in_one_search_region(self):
        ready = []
        for ident, box in [
            ("parent", (0, 0, 60, 12)),
            ("left", (0, 0, 28, 12)),
            ("right", (32, 0, 60, 12)),
        ]:
            p = self.proposal(box=box)
            ready.append((PhysicalBoundaryQuery(ident, "V", p.bounds, (p,), ()), {}))
        scheduler = PrecisionScheduler(ready, self.config)
        self.assertGreaterEqual(len(scheduler.regions), 2)
        regions = {
            q.query_id: i
            for i, r in enumerate(scheduler.regions)
            for q, _ in r["queue"]
        }
        self.assertNotEqual(regions["left"], regions["right"])
        self.assertNotEqual(regions["parent"], regions["left"])
        self.assertNotEqual(regions["parent"], regions["right"])

    def test_outer_envelope_cannot_preempt_independent_interior_closures(self):
        ready = []
        for ident, box, skipped in [
            ("parent", (0, 0, 60, 12), 2),
            ("left", (0, 0, 28, 12), 0),
            ("right", (32, 0, 60, 12), 0),
        ]:
            p = self.proposal(box=box)
            ready.append(
                (
                    PhysicalBoundaryQuery(ident, "V", p.bounds, (p,), ()),
                    dict(
                        skipped_verified_face_count=skipped,
                        deck_ring=dict(
                            sides={
                                side: dict(vertical_face_support=True)
                                for side in ("U0", "U1", "V0", "V1")
                            }
                        ),
                    ),
                )
            )
        ordered = [q.query_id for q, _ in PrecisionScheduler(ready, self.config)]
        self.assertEqual(set(ordered[:2]), {"left", "right"})
        self.assertEqual(ordered[-1], "parent")

    def test_scheduler_alternates_independent_paths_within_one_region(self):
        p = self.proposal()
        ring = dict(
            sides={
                name: dict(vertical_face_support=True)
                for name in ("U0", "U1", "V0", "V1")
            }
        )
        structural = [
            (
                PhysicalBoundaryQuery("S%02d" % i, "V", p.bounds, (p,), ()),
                dict(deck_ring=ring),
            )
            for i in range(30)
        ]
        legacy = (
            PhysicalBoundaryQuery("L", "V", p.bounds, (p,), ()),
            dict(
                legacy_post_b2=dict(derived_search_evidence=True, support_side_count=3)
            ),
        )
        scheduled = iter(PrecisionScheduler(structural + [legacy], self.config))
        self.assertEqual(next(scheduled)[0].query_id, "S00")
        self.assertEqual(next(scheduled)[0].query_id, "L")
        self.assertEqual(next(scheduled)[0].query_id, "S01")

    def test_post_b2_tier_requires_actual_local_side_evidence_not_legacy_label(self):
        p = self.proposal()
        post = dict(
            local_physical_candidate_available=True,
            support_side_count=3,
            side_evidence_availability=dict(U0=True, U1=False, V0=True, V1=True),
        )
        historical = (
            PhysicalBoundaryQuery("L", "V", p.bounds, (p,), ()),
            dict(
                legacy_post_b2=post,
                deck_ring=dict(
                    sides={
                        name: dict(vertical_face_support=True)
                        for name in ("U0", "V0", "V1")
                    }
                ),
            ),
        )
        measured = (
            PhysicalBoundaryQuery("S", "V", p.bounds, (p,), ()),
            dict(
                deck_ring=dict(
                    sides={
                        name: dict(vertical_face_support=True)
                        for name in ("U0", "U1", "V0", "V1")
                    }
                )
            ),
        )
        self.assertEqual(
            next(iter(PrecisionScheduler([measured, historical], self.config)))[
                0
            ].query_id,
            "L",
        )
        post["side_evidence_availability"]["U0"] = False
        self.assertEqual(
            next(iter(PrecisionScheduler([measured, historical], self.config)))[
                0
            ].query_id,
            "S",
        )

    def measurement(self, xyz, enabled=True):
        cache = SimpleNamespace(
            config=self.config,
            context_xyz=xyz,
            xyz=xyz,
            measurement_context=None,
            vessel=SimpleNamespace(point_indexes=np.arange(len(xyz))),
            structural_providers=lambda: [],
        )
        measurement = VesselMeasurementContext(cache)
        measurement.budget = None
        measurement.side_measurements.enabled = enabled
        measurement.rasters.enabled = enabled
        return measurement

    def test_identical_multi_expert_geometry_compiles_once_with_all_provenance(self):
        proposals = [
            self.proposal(expert=e)
            for e in ("LEGACY", "STRUCTURAL_LINE", "LARGE_SAND_HOLD")
        ]
        compiler = ProposalCompiler(self.config)
        with patch(
            "ship_perception.perception.experts.role_families",
            side_effect=AssertionError("Eager Raw role work"),
        ):
            groups = compiler.compile(proposals)
        self.assertEqual(len(groups), 1)
        self.assertEqual(compiler.exact_duplicate_count, 2)
        self.assertEqual(
            groups[0].query().sources, ["LARGE_SAND_HOLD", "LEGACY", "STRUCTURAL_LINE"]
        )
        self.assertEqual(
            set(groups[0].audit()["source_proposal_ids"]),
            {p.proposal_id for p in proposals},
        )

    def test_adjacent_hatches_and_distinct_separator_topology_never_compile_together(
        self,
    ):
        proposals = [
            self.proposal(box=(0, 0, 20, 10), separator=[20, 21]),
            self.proposal(box=(21, 0, 41, 10), ident="Q", separator=[20, 21]),
            self.proposal(box=(0.1, 0, 20.1, 10), ident="R", separator=[19, 22]),
        ]
        self.assertEqual(len(ProposalCompiler(self.config).compile(proposals)), 3)

    def test_compiled_family_retains_member_bounds_and_union_search_envelope(self):
        a, b = self.proposal(), self.proposal("STRUCTURAL", box=(0.1, 0, 20.1, 10))
        (group,) = ProposalCompiler(self.config).compile([a, b])
        self.assertEqual(list(group.hypotheses()), [a, b])
        np.testing.assert_array_equal(group.search_envelope, [0, 0, 20.1, 10])

    def test_adjacency_graph_has_no_fixed_hatch_count_or_all_pairs_envelopes(self):
        families = {0: [], 1: []}
        roles = {}
        for index in range(12):
            for rho, sign in ((index * 21.0, 1), (index * 21.0 + 20, -1)):
                families[0].append(
                    dict(
                        rho=rho,
                        interval=(0, 10),
                        raw3d=True,
                        bev=True,
                        normal_face=True,
                        provider_ids=(str(rho),),
                    )
                )
                roles[0, rho] = sign
        for rho, sign in ((0.0, 1), (10.0, -1)):
            families[1].append(
                dict(
                    rho=rho,
                    interval=(0, 251),
                    raw3d=True,
                    bev=True,
                    normal_face=True,
                    provider_ids=("V" + str(rho),),
                )
            )
            roles[1, rho] = sign
        graph = BoundaryLineGraph(
            families,
            self.config,
            lambda axis, rho, span, sign: 4.0 if roles[axis, rho] == sign else -4.0,
        )
        rows = graph.enclosures()
        adjacent = [r for r in rows if r["closure_kind"] == "ADJACENT"]
        widths = [r["bounds_axial"][2] - r["bounds_axial"][0] for r in adjacent]
        self.assertEqual(widths.count(20), 12)
        # Reversed face polarity can nominate the next closure across a deck
        # gap. These eleven alternatives are searches, not accepted hatches.
        self.assertEqual(widths.count(21), 11)
        self.assertTrue(all(not r["signed_roles_resolved"] for r in rows))
        outer = [r for r in rows if r["closure_kind"] == "SUPPORTED_OUTER_ENVELOPE"]
        self.assertTrue(outer)
        self.assertTrue(all(r["normal_face_sides"] >= 3 for r in outer))
        self.assertLessEqual(len(rows), 2 * len(adjacent))

    def test_outer_envelope_needs_extra_structure_and_does_not_remove_adjacent_closure(
        self,
    ):
        families = {0: [], 1: []}
        roles = {(0, 0.0): 1, (0, 20.0): -1, (0, 40.0): -1, (1, 0.0): 1, (1, 10.0): -1}
        for axis, rhos, span in (
            (0, (0.0, 20.0, 40.0), (0.0, 10.0)),
            (1, (0.0, 10.0), (0.0, 40.0)),
        ):
            for rho in rhos:
                families[axis].append(
                    dict(
                        rho=rho,
                        interval=span,
                        raw3d=True,
                        bev=True,
                        normal_face=False,
                        provider_ids=(str((axis, rho)),),
                    )
                )
        height = lambda axis, rho, span, sign: 1.0 if roles[axis, rho] == sign else -1.0
        rows = BoundaryLineGraph(families, self.config, height).enclosures()
        self.assertTrue(any(r["bounds_axial"] == [0.0, 0.0, 20.0, 10.0] for r in rows))
        self.assertFalse(any(r["bounds_axial"] == [0.0, 0.0, 40.0, 10.0] for r in rows))
        for groups in families.values():
            for face in groups:
                face["normal_face"] = True
        rows = BoundaryLineGraph(families, self.config, height).enclosures()
        self.assertTrue(any(r["bounds_axial"] == [0.0, 0.0, 20.0, 10.0] for r in rows))
        self.assertTrue(
            any(
                r["bounds_axial"] == [0.0, 0.0, 40.0, 10.0]
                and r["closure_kind"] == "SUPPORTED_OUTER_ENVELOPE"
                for r in rows
            )
        )

    def test_side_line_index_preserves_exact_mode_values_and_provider_order(self):
        from ship_perception.perception.local_evidence import VesselLocalEvidence
        from ship_perception.r1_static.r2h_b5_corner_forensics import _side_modes

        providers = []
        for axis in (0, 1):
            for index, rho in enumerate((10.0, 0.0, 20.0, 5.0, 15.0)):
                for kind in ("RAW3D_STRUCTURAL_PROVIDER", "BEV_RECTILINEAR_PROVIDER"):
                    providers.append(
                        dict(
                            provider_id=str(len(providers)),
                            provider_type=kind,
                            rho_or_region=dict(normal_axis=axis, rho=rho),
                            support_interval=(-10.0, 30.0),
                            source_support_ids=["S"],
                            coverage=0.9,
                        )
                    )
        cache = VesselLocalEvidence(
            SimpleNamespace(vessel_id="V"),
            np.zeros((1, 3)),
            self.config,
            providers=providers,
        )
        box = (1.0, 2.0, 19.0, 16.0)
        indexed = cache.side_candidates(box)
        lines = cache._v21_side_lines[0]
        for name, axis, pos in (("U0", 0, 0), ("U1", 0, 2), ("V0", 1, 1), ("V1", 1, 3)):
            search = self.config["roi"]["support_search_m"]
            old = _side_modes(
                lines,
                axis,
                (box[1 - axis], box[3 - axis]),
                (box[pos] - search, box[pos] + search),
                self.config,
            )
            self.assertEqual(indexed[name], old)

    def test_coarse_two_sides_can_discover_additional_raw_sides_without_normals_or_rasters(
        self,
    ):
        xyz, boxes = vessel_points(1, spacing=0.15)
        m = self.measurement(xyz)
        refiner = FinalBoundaryRefiner(xyz, [], self.config, m)
        p = self.proposal(box=tuple(boxes[0]))
        query = PhysicalBoundaryQuery("Q", "V", p.bounds, (p,), ())
        with patch(
            "ship_perception.r1_static.height_grid.R1HeightGrid.from_points",
            side_effect=AssertionError("Probe raster"),
        ), patch(
            "ship_perception.perception.final_boundary.BudgetedLocalSurfaceNormals",
            side_effect=AssertionError("Probe normals"),
        ):
            result = RawBoundaryProbe(m, refiner).evaluate(
                query, dict(independent_support_sides=2), {}
            )
        self.assertGreaterEqual(result["raw_probe_support_side_count"], 3)
        self.assertTrue(result["full_precision_allowed"])
        self.assertEqual(m.counters["normal_points_evaluated"], 0)

    def test_two_raw_sides_without_third_remain_unresolved(self):
        u, v = np.meshgrid(np.arange(-2, 22, 0.1), np.arange(0, 10, 0.1))
        xyz = np.column_stack(
            (u.ravel(), v.ravel(), np.where((u > 0) & (u < 20), -4.0, 0.0).ravel())
        )
        m = self.measurement(xyz)
        p = self.proposal(box=(0, 1, 20, 9))
        query = PhysicalBoundaryQuery("Q", "V", p.bounds, (p,), ())
        result = RawBoundaryProbe(
            m, FinalBoundaryRefiner(xyz, [], self.config, m)
        ).evaluate(query, dict(independent_support_sides=2), {})
        self.assertEqual(result["raw_probe_support_side_count"], 2)
        self.assertFalse(result["full_precision_allowed"])

    def test_raw_probe_nominates_a_missing_edge_beyond_fine_refinement_band(self):
        xyz, boxes = vessel_points(1, spacing=0.15)
        measurement = self.measurement(xyz)
        box = list(boxes[0])
        box[0] += 1.2
        box[2] -= 1.2
        p = self.proposal(box=box)
        query = PhysicalBoundaryQuery("Q", "V", p.bounds, (p,), ())
        with patch(
            "ship_perception.r1_static.height_grid.R1HeightGrid.from_points",
            side_effect=AssertionError("probe raster"),
        ), patch(
            "ship_perception.perception.final_boundary.BudgetedLocalSurfaceNormals",
            side_effect=AssertionError("probe normals"),
        ):
            result = RawBoundaryProbe(
                measurement, FinalBoundaryRefiner(xyz, [], self.config, measurement)
            ).evaluate(query, dict(independent_support_sides=2), {})
        self.assertGreaterEqual(result["raw_probe_support_side_count"], 3)
        self.assertTrue(result["full_precision_allowed"])

    def test_nearby_anchors_reuse_profile_without_caching_or_rounding_final_rho(self):
        u, v = np.meshgrid(np.arange(-2, 2, 0.025), np.arange(0, 4, 0.025))
        xyz = np.column_stack(
            (u.ravel(), v.ravel(), np.where(u > 0, -4.0, 0.0).ravel())
        )
        cached, uncached = self.measurement(xyz), self.measurement(xyz, False)
        a, b = (
            FinalBoundaryRefiner(xyz, [], self.config, m) for m in (cached, uncached)
        )
        for anchor in (0.0, 0.05):
            one = a._search_raw_section(xyz, 0, anchor, (0, 4), 1)
            two = b._search_raw_section(xyz, 0, anchor, (0, 4), 1)
            self.assertEqual(one, two)
        self.assertEqual(cached.counters["raw_profile_build_count"], 1)
        self.assertEqual(uncached.counters["raw_profile_build_count"], 2)

    def test_profile_cache_toggle_keeps_final_coordinates_and_all_four_phases(self):
        xyz, boxes = vessel_points(1, spacing=0.15)
        proposal = dict(
            bounds_axial=list(boxes[0]),
            vessel_hypothesis_id="V",
            measured_side_candidates={},
            deck_ring=dict(
                sides={
                    name: dict(raw3d_support=True, bev_support=True)
                    for name in ("U0", "U1", "V0", "V1")
                }
            ),
        )
        results = []
        for enabled in (False, True):
            m = self.measurement(xyz, enabled)
            results.append(
                FinalBoundaryRefiner(xyz, [], self.config, m).refine(
                    proposal, np.eye(2)
                )
            )
        np.testing.assert_array_equal(
            results[0]["boundary_xy"], results[1]["boundary_xy"]
        )
        for name in ("U0", "U1", "V0", "V1"):
            self.assertEqual(len(results[1]["sides"][name]["phase_runs"]), 4)
            self.assertEqual(
                results[0]["sides"][name]["rho_after_m"],
                results[1]["sides"][name]["rho_after_m"],
            )

    def test_overlapping_section_spans_share_sort_without_changing_raw_measurement(
        self,
    ):
        u, v = np.meshgrid(np.arange(-2, 2, 0.025), np.arange(0, 6, 0.025))
        xyz = np.column_stack(
            (u.ravel(), v.ravel(), np.where(u > 0, -4.0, 0.0).ravel())
        )
        cached, uncached = self.measurement(xyz), self.measurement(xyz, False)
        a, b = (
            FinalBoundaryRefiner(xyz, [], self.config, m) for m in (cached, uncached)
        )
        for span in ((0.0, 4.0), (0.2, 4.2), (0.3, 4.3)):
            self.assertEqual(
                a._search_raw_section(xyz, 0, 0.0, span, 1),
                b._search_raw_section(xyz, 0, 0.0, span, 1),
            )
        self.assertEqual(cached.counters["raw_profile_build_count"], 1)
        self.assertEqual(uncached.counters["raw_profile_build_count"], 3)

    def test_scheduler_visits_second_region_before_thirty_competing_hypotheses(self):
        ready = []
        for index in range(30):
            p = self.proposal(box=(index * 0.02, 0, 20, 10), ident=str(index))
            ready.append(
                (PhysicalBoundaryQuery("A%02d" % index, "V", p.bounds, (p,), ()), {})
            )
        p = self.proposal(box=(40, 0, 60, 10))
        ready.append((PhysicalBoundaryQuery("Z", "V", p.bounds, (p,), ()), {}))
        scheduled = iter(PrecisionScheduler(ready, self.config))
        self.assertEqual(next(scheduled)[0].query_id, "A00")
        self.assertEqual(next(scheduled)[0].query_id, "Z")

    def test_measured_face_enclosure_precedes_legacy_id_without_physical_agreement(
        self,
    ):
        p = self.proposal()
        legacy = (
            PhysicalBoundaryQuery("A", "V", p.bounds, (p,), ()),
            dict(
                legacy_post_b2=dict(derived_search_evidence=True, support_side_count=4),
                independent_support_sides=4,
                deck_ring=dict(positive_height_side_count=4, sides={}),
            ),
        )
        enclosure = (
            PhysicalBoundaryQuery("Z", "V", p.bounds, (p,), ()),
            dict(
                independent_support_sides=3,
                deck_ring=dict(
                    positive_height_side_count=3,
                    sides={
                        name: dict(vertical_face_support=True)
                        for name in ("U0", "U1", "V0")
                    },
                ),
            ),
        )
        self.assertEqual(
            next(iter(PrecisionScheduler([legacy, enclosure], self.config)))[
                0
            ].query_id,
            "Z",
        )

    def test_measurement_admission_does_not_lower_final_three_raw_side_requirement(
        self,
    ):
        from ship_perception.perception.pipeline_v2 import final_support_sides

        u, v = np.meshgrid(np.arange(-2, 22, 0.1), np.arange(0, 10, 0.1))
        xyz = np.column_stack(
            (u.ravel(), v.ravel(), np.where((u > 0) & (u < 20), -4.0, 0.0).ravel())
        )
        m = self.measurement(xyz)
        p = self.proposal(box=(0, 1, 20, 9))
        q = PhysicalBoundaryQuery("Q", "V", p.bounds, (p,), ())
        ring = dict(
            sides={
                name: dict(
                    vertical_face_support=True, raw3d_support=True, bev_support=True
                )
                for name in ("U0", "U1", "V0", "V1")
            }
        )
        row = dict(independent_support_sides=3, deck_ring=ring)
        refiner = FinalBoundaryRefiner(xyz, [], self.config, m)
        probe = RawBoundaryProbe(m, refiner).evaluate(q, row, {})
        self.assertTrue(probe["full_precision_allowed"])
        result = refiner.refine(
            dict(bounds_axial=list(p.bounds), vessel_hypothesis_id="V", deck_ring=ring),
            np.eye(2),
        )
        self.assertEqual(len(final_support_sides(result, self.config)), 2)

    def test_section_capacity_is_conservative_about_existing_structural_search(self):
        xyz, _ = vessel_points(1, spacing=0.15)
        m = self.measurement(xyz)
        p = self.proposal(box=(0, 0, 20, 0.1))
        q = PhysicalBoundaryQuery("Q", "V", p.bounds, (p,), ())
        probe = RawBoundaryProbe(m, FinalBoundaryRefiner(xyz, [], self.config, m))
        narrow = probe.evaluate(q, {}, {})
        self.assertFalse(narrow["full_measurement_possible"])
        modes = dict(
            V0=[dict(rho=-3.9, primary_raw3d=1.0, primary_bev=1.0)],
            V1=[dict(rho=4.0, primary_raw3d=1.0, primary_bev=1.0)],
        )
        expanded = probe.evaluate(q, {}, modes)
        self.assertTrue(expanded["full_measurement_possible"])
        self.assertFalse(expanded["final_coordinates_created"])

    def test_parent_envelope_cannot_explain_away_a_separate_child_opening(self):
        parent = PhysicalBoundaryQuery("P", "V", (0, 0, 40, 10), (), ())
        child = PhysicalBoundaryQuery("C", "V", (0, 0, 20, 10), (), ())
        scheduler = PrecisionScheduler([(parent, {}), (child, {})], self.config)
        scheduler.record(
            parent,
            {},
            dict(
                boundary_uv=[[0, 0], [40, 0], [40, 10], [0, 10]], status="PARTIAL_HATCH"
            ),
        )
        self.assertFalse(scheduler.explained(child, {}))
        self.assertTrue(scheduler.explained(parent, {}))

    def test_large_sand_hold_shares_geometry_and_only_adds_interpretation(self):
        from ship_perception.perception.experts import LargeSandHoldExpert

        original = self.proposal("STRUCTURAL_LINE")
        cache = SimpleNamespace(_v21_structural_proposals=(original,))
        (adapted,) = LargeSandHoldExpert().propose(cache, {})
        self.assertIs(adapted.bounds, original.bounds)
        self.assertFalse(adapted.evidence()["interior_height_hard_gate"])
        self.assertEqual(adapted.lineage, original.lineage)

    def test_cached_raster_tiles_match_independent_exact_raw_quantiles(self):
        from ship_perception.r1_static.height_grid import R1HeightGrid

        xyz = np.random.default_rng(21).uniform([-3, -3, -5], [5, 5, 0], (4000, 3))
        m = self.measurement(xyz)
        for phase in ((0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)):
            grid = m.rasters.grid([0, 0, 2, 2], 1.0, phase)
            full = R1HeightGrid.from_points(
                xyz,
                self.config["geometry"]["refine_voxel_m"],
                origin_xy=m.rasters.origin + np.array(phase) * m.rasters.cell,
            )
            x, y = grid.centers()
            rr, cc = full.cell_indices(np.column_stack((x.ravel(), y.ravel())))
            for key in ("count", "q10", "median", "q90", "minimum", "maximum"):
                np.testing.assert_array_equal(
                    getattr(grid, key).ravel(), getattr(full, key)[rr, cc]
                )
            m.rasters.grid([0.1, 0.1, 2.1, 2.1], 1.0, phase)
        self.assertGreater(m.counters["raster_tile_cache_hit"], 0)
