"""Region grouping cannot create final edges or impose a hatch count."""

import unittest

from ship_perception.perception.candidate_union import CandidateProposal
from ship_perception.perception.opening_regions import (
    OpeningRegionCompiler,
    OpeningRegionSeed,
)
from ship_perception.perception.boundary_line_graph import BoundaryLineGraph
from ship_perception.r1_static.run import resolve_config


class OpeningRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()

    def proposal(self, box, index, expert="HEIGHT_BASIN", **extra):
        row = dict(
            bounds_axial=box,
            height_rise_side_count=4,
            surrounding_height_m=2.0,
            interior_height_m=-3.0,
            touches_grid_boundary=False,
        )
        row.update(extra)
        return CandidateProposal.from_evidence(expert, "VESSEL", row, index)

    def test_nested_height_levels_keep_two_distinct_regions_and_compound_diagnostic(
        self,
    ):
        proposals = [
            self.proposal((0, 0, 20, 12), 0),
            self.proposal((1, 1, 19, 11), 1),
            self.proposal((30, 0, 55, 12), 2),
            self.proposal((0, 0, 55, 12), 3),
        ]
        compiler = OpeningRegionCompiler(self.config)
        regions = compiler.compile("VESSEL", proposals)
        self.assertEqual(len(regions), 2)
        self.assertEqual(len(compiler.compound), 1)
        for region in regions:
            self.assertEqual(
                region.audit()["geometry_role"], "SEARCH_REGION_NOT_FINAL_BOUNDARY"
            )
            self.assertEqual(region.audit()["boundary_status"], "BOUNDARY_UNRESOLVED")

    def test_widthwise_floor_lobes_join_when_a_measured_height_component_connects_them(
        self,
    ):
        proposals = [
            self.proposal((0, 0, 25, 6), 0),
            self.proposal((0, 9, 25, 15), 1),
            self.proposal((0, 0, 25, 15), 2),
        ]
        regions = OpeningRegionCompiler(self.config).compile("VESSEL", proposals)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].bounds, (0.0, 0.0, 25.0, 15.0))
        self.assertEqual(len(regions[0].source_proposal_ids), 3)

    def test_adjacent_floor_components_share_existing_legacy_opening_search(self):
        p = [
            self.proposal((0, 0, 25, 12), 0),
            self.proposal((25.5, 0, 50, 12), 1),
            self.proposal(
                (-1, 0, 51, 12),
                2,
                "LEGACY",
                opening_evidence=dict(status="OPENING_LOWER"),
                measured_strong_side_count=3,
            ),
        ]
        regions = OpeningRegionCompiler(self.config).compile("VESSEL", p)
        self.assertEqual(len(regions), 1)
        self.assertEqual(len(regions[0].component_bounds), 2)

    def test_missing_reference_is_not_replaced_by_zero_height(self):
        compiler = OpeningRegionCompiler(self.config)
        regions = compiler.compile(
            "VESSEL", [self.proposal((0, 0, 25, 12), 0, surrounding_height_m=None,
                                     height_rise_side_count=0)]
        )
        self.assertEqual(regions, ())
        self.assertEqual(len(compiler.secondary), 1)
        self.assertIn("NOT_REJECTED", compiler.secondary[0]["reason"])

    def test_region_graph_retains_all_side_alternatives_without_rectangle_product(self):
        families = {
            axis: [
                dict(
                    rho=float(i),
                    interval=(-20.0, 200.0),
                    provider_ids=(str(axis) + ":" + str(i),),
                    raw3d=True,
                    bev=True,
                    normal_face=True,
                )
                for i in range(100)
            ]
            for axis in (0, 1)
        }
        regions = [
            OpeningRegionSeed(
                "R" + str(i), "V", (i * 20.0, 10.0, i * 20.0 + 15.0, 25.0), (), ()
            )
            for i in range(4)
        ]
        graph = BoundaryLineGraph(families, self.config)
        result = graph.region_searches(regions)
        self.assertEqual(len(result), len(regions))
        self.assertEqual(graph.region_metrics["rectangle_combination_count"], 0)
        self.assertGreater(graph.region_metrics["left_right_pair_count"], len(regions))
        self.assertGreater(len(result[0]["region_side_nominations"][0]), 1)
        self.assertTrue(
            all(r["geometry_role"] == "REGION_LOCAL_SEARCH_NOT_STEEL" for r in result)
        )
