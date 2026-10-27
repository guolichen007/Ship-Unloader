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
        self.assertEqual(len(rows), 12)
        self.assertTrue(
            all(r["bounds_axial"][2] - r["bounds_axial"][0] == 20 for r in rows)
        )
        self.assertTrue(all(not r["signed_roles_resolved"] for r in rows))

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
