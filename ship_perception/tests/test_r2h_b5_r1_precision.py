"""Physical counterexamples for stage arbitration and section consensus."""

import unittest

import numpy as np


from ship_perception.r1_static.corner_topology import (
    _has_neighboring_opening, _section_rho_modes,
    arbitrate_unresolved_side, solve_three_corners,
)
from ship_perception.r1_static.corner_topology_review import _initial_seed_groups
from ship_perception.r1_static.run import DEFAULT_CONFIG, resolve_config
from ship_perception.r1_static.static_showcase import (
    _showcase_axes, classify_review_points,
)


CONFIG, _ = resolve_config(DEFAULT_CONFIG)


def _stage_side(sources):
    ident = "V0001"
    return dict(
        rho_before=24.34, rho_after=22.42, previous_role="INNER_EDGE",
        selected_source_vessel_ids=sources,
        candidate_modes=[
            dict(rho=24.34, raw3d_coverage=.76, bev_coverage=.41,
                 provider_ids=[ident + ":RAW3D_STRUCTURAL_PROVIDER:1",
                               ident + ":BEV_RECTILINEAR_PROVIDER:2"],
                 opening_relation=dict(canonical_inner=True,
                                       outside_minus_inside_m=2.43)),
            dict(rho=22.42, raw3d_coverage=.76, bev_coverage=.37,
                 provider_ids=[ident + ":RAW3D_STRUCTURAL_PROVIDER:3",
                               "V0002:BEV_RECTILINEAR_PROVIDER:4"],
                 opening_relation=dict(canonical_inner=True,
                                       outside_minus_inside_m=.19)),
        ],
    )


class B5R1Precision(unittest.TestCase):
    def test_neighbor_context_survives_independent_group_recursion(self):
        context = [("middle", [-2.36, 4.9, 22.42, 19.7]),
                   ("next", [26.69, 4.9, 53.09, 20.6])]
        self.assertTrue(_has_neighboring_opening(
            "U1", 24.34, context[0][1], "middle", context, CONFIG))
        self.assertFalse(_has_neighboring_opening(
            "U1", 24.34, context[0][1], "middle", context[:1], CONFIG))

    def test_older_end_edge_survives_mixed_internal_transition(self):
        result = arbitrate_unresolved_side(
            _stage_side(["V0001", "V0002"]), "U1", "V0001", CONFIG,
            neighboring_opening=True)
        self.assertEqual(result["final_stage"], "R2G")
        self.assertAlmostEqual(result["final_rho"], 24.34)

    def test_single_vessel_outer_hull_does_not_trigger_rollback(self):
        result = arbitrate_unresolved_side(
            _stage_side(["V0001", "V0002"]), "U1", "V0001", CONFIG,
            neighboring_opening=False)
        self.assertEqual(result["final_stage"], "B4")

    def test_clean_b4_source_is_preserved(self):
        result = arbitrate_unresolved_side(
            _stage_side(["V0001"]), "U1", "V0001", CONFIG,
            neighboring_opening=True)
        self.assertEqual(result["final_stage"], "B4")

    def test_single_strong_false_section_cannot_move_long_side(self):
        def mode(rho, asymmetry):
            return dict(rho=rho, evidence_rank=[1], primary_raw3d=.7,
                        primary_bev=.6, height_level_count=0,
                        section_consensus=dict(sections=[
                            dict(opening_one_sided=True, interior_ridge=False,
                                 boundary_asymmetry=value) for value in asymmetry]))
        true = mode(4.53, [.9, .9, .9, .9, .6])
        local = mode(4.89, [.1, .1, 1.0, .1, .1])
        result = _section_rho_modes([true, local], "V0", CONFIG)
        self.assertEqual(result["section_count"], 5)
        self.assertAlmostEqual(result["section_rho_median"], 4.53)
        self.assertAlmostEqual(result["section_rho_mad"], 0)

    def test_bootstrap_prefers_opening_cell_to_edge_strip(self):
        def proposal(ident, box):
            return dict(hatch_hypothesis_id=ident, state="PARTIAL_HATCH",
                        measured_strong_side_count=3,
                        opening_evidence=dict(status="OPENING_LOWER"),
                        role_conflicts=[], bounds_axial=box)
        broad = proposal("cell", [0, 0, 20, 10])
        strip = proposal("strip", [1, 0, 19, 1])
        groups = _initial_seed_groups([strip, broad], CONFIG)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0]["hatch_hypothesis_id"], "cell")

    def test_persistent_height_context_closes_only_partial_corner(self):
        corners = [dict(corner_xy=xy, role=role,
                        local_u_persistence=dict(level_count=levels),
                        v_height_levels=height)
                   for xy, role, levels, height in (
                       ((0, 0), "MEASURED_INTERSECTION_CANDIDATE", 0, 0),
                       ((2, 0), "FOOTPRINT_CONTEXT_CORNER", 3, 3),
                       ((2, 1), "FOOTPRINT_CONTEXT_CORNER", 3, 3),
                       ((0, 1), "UNRESOLVED_CORNER", 0, 0))]
        result = solve_three_corners(corners)
        self.assertEqual(result["state"], "PARTIAL_HATCH_3C")
        self.assertEqual(result["inferred_corner"], [0, 1])
        corners[1]["local_u_persistence"]["level_count"] = 0
        corners[2]["local_u_persistence"]["level_count"] = 0
        self.assertEqual(solve_three_corners(corners)["state"], "UNRESOLVED")

    def test_showcase_categories_are_geometric_review_only(self):
        points = np.asarray([[1, 1, 0], [.02, 1, 0], [3, 1, 0], [-2, -2, 0]])
        polygon = [[0, 0], [2, 0], [2, 2], [0, 2]]
        report = dict(vessels=[dict(vessel_hypothesis_id="V0000",
                                    rectangles=[dict(polygon_after=polygon)])])
        private = dict(vessel_point_indexes={"V0000": np.asarray([0, 1, 2])})
        categories = classify_review_points(points, private, report, np.eye(2), CONFIG)
        self.assertEqual(categories.tolist(), [3, 4, 2, 1])

    def test_showcase_uses_rectangle_axes_when_vessel_axis_differs(self):
        b2 = dict(scene=dict(vessel_hypotheses=[dict(
            vessel_hypothesis_id="V0001", local_axes=np.eye(2).tolist())]))
        rotated = [[.99, -.14], [.14, .99]]
        b4 = dict(rectangles=[dict(vessel_hypothesis_id="V0001", axes=rotated)])
        axes, source = _showcase_axes("V0001", b2, b4)
        self.assertEqual(source, "B4_RECTANGLE_AXES")
        np.testing.assert_allclose(axes, rotated)



if __name__ == "__main__":
    unittest.main()
