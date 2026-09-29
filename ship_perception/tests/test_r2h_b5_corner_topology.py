"""Counterexamples for the B5 static corner review solver."""

import unittest

import numpy as np

from ship_perception.r1_static.corner_topology import (
    _corner, _merge_candidates, _pair_rank, _side_rank, _strip_relation,
    _upper_multimode_conflict,
    solve_three_corners,
)
from ship_perception.r1_static.raw_topview_evidence import build_topview
from ship_perception.r1_static.run import DEFAULT_CONFIG, resolve_config
from ship_perception.r1_static.vessel_height_topology import (
    build_vessel_scoped_topview, persistent_height_modes,
)


CONFIG, _ = resolve_config(DEFAULT_CONFIG)


def _cloud(raised=True):
    u, v = np.meshgrid(np.arange(-4, 15, .15), np.arange(-4, 15, .15))
    interior = (u >= 0) & (u <= 10) & (v >= 0) & (v <= 10)
    z = np.where(interior, 2.0 if raised else -2.0, 0.0)
    return np.column_stack((u.ravel(), v.ravel(), z.ravel()))


def _mode(rho, height=0, raw=0, bev=0, footprint=False):
    return dict(rho=float(rho), height_level_count=height,
                height_coverage=0.8 if height else 0.0,
                primary_raw3d=raw, primary_bev=bev,
                support_limit_validated=footprint)


def _corner_row(u, v, role):
    return dict(corner_xy=[u, v], role=role)


class B5CornerTopology(unittest.TestCase):
    def test_ship_next_to_high_quay_uses_point_provenance(self):
        ship = _cloud()
        quay = _cloud() + np.array([0, -30, 8])
        points = np.vstack((ship, quay))
        scoped = build_vessel_scoped_topview(
            points, np.arange(len(ship)), np.eye(2), CONFIG)
        self.assertEqual(scoped["rejected_external_points"], len(quay))
        self.assertAlmostEqual(np.median(scoped["points"][:, 2]),
                               np.median(ship[:, 2]))
        self.assertLess(scoped["grids"]["coarse"].y0, 0)
        self.assertGreater(scoped["grids"]["coarse"].y0, -10)

    def test_low_hold_points_are_retained(self):
        points = _cloud(raised=False)
        scoped = build_vessel_scoped_topview(
            points, np.arange(len(points)), np.eye(2), CONFIG)
        self.assertEqual(len(scoped["points"]), len(points))
        self.assertLess(scoped["points"][:, 2].min(), -1.0)
        self.assertEqual(scoped["point_selection"],
                         "B1_VESSEL_XY_FOOTPRINT_RESTORE_ALL_RAW_Z")

    def test_low_raw_returns_inside_shell_footprint_are_restored(self):
        shell = _cloud()
        deep = shell[::50].copy()
        deep[:, 2] = -12
        points = np.vstack((shell, deep))
        scoped = build_vessel_scoped_topview(
            points, np.arange(len(shell)), np.eye(2), CONFIG)
        self.assertEqual(scoped["restored_inside_footprint_points"], len(deep))
        self.assertEqual(scoped["points"][:, 2].min(), -12)

    def test_internal_cargo_ridge_does_not_win_by_line_strength(self):
        points = _cloud()
        grid = build_topview(points, np.eye(2), CONFIG)["fine"]
        relation = _strip_relation(grid, 5, (1, 9), 1, 1, CONFIG)
        self.assertTrue(relation["interior_ridge"])
        rank = _side_rank(_mode(5, raw=1, bev=1), relation, 5, CONFIG, "V1")
        self.assertEqual(rank[0], 0)

    def test_parallel_inner_mode_cannot_win_only_by_more_raw_points(self):
        opening_onset = dict(opening_one_sided=True, interior_ridge=False,
                             boundary_asymmetry=.97)
        deeper_mode = dict(opening_one_sided=True, interior_ridge=False,
                           boundary_asymmetry=.65)
        edge = _side_rank(_mode(4.5, raw=.55, bev=.52),
                          opening_onset, 2.0, CONFIG, "V0")
        deep = _side_rank(_mode(4.9, raw=1, bev=.68),
                          deeper_mode, 2.0, CONFIG, "V0")
        self.assertGreater(edge, deep)

    def test_upper_rim_prefers_measured_persistent_contour_position(self):
        relation = dict(opening_one_sided=False, interior_ridge=False,
                        boundary_asymmetry=.1)
        near = _mode(17.23, height=4, raw=1, bev=.6)
        near["height_distance_m"] = .05
        near["height_rho_m"] = 17.28
        far = _mode(17.50, height=4, raw=1, bev=.6)
        far["height_distance_m"] = .22
        far["height_rho_m"] = 17.28
        self.assertGreater(_side_rank(near, relation, 17.1, CONFIG, "V1"),
                           _side_rank(far, relation, 17.1, CONFIG, "V1"))

    def test_nearer_persistent_family_beats_outer_hull_family(self):
        relation = dict(opening_one_sided=False, interior_ridge=True,
                        boundary_asymmetry=0)
        inner = _mode(13.04, height=2, raw=1, bev=.8)
        inner.update(height_rho_m=13.07, height_distance_m=.03)
        outer = _mode(15.09, height=4, raw=1, bev=.8)
        outer.update(height_rho_m=15.10, height_distance_m=.01)
        self.assertGreater(_side_rank(inner, relation, 13.45, CONFIG, "V1"),
                           _side_rank(outer, relation, 13.45, CONFIG, "V1"))

    def test_two_loaded_upper_height_families_require_review(self):
        modes = [dict(rho=13.07, level_count=2),
                 dict(rho=15.10, level_count=4)]
        candidates = [dict(evidence_rank=[1], height_level_count=2,
                           opening_relation=dict(interior_ridge=True))]
        self.assertTrue(_upper_multimode_conflict(modes, candidates, CONFIG))
        self.assertFalse(_upper_multimode_conflict(modes[:1], candidates, CONFIG))

    def test_three_corners_are_partial_with_inferred_fourth(self):
        roles = ["MEASURED_INTERSECTION_CANDIDATE"] * 3
        corners = [_corner_row(0, 0, roles[0]),
                   _corner_row(10, 0, roles[1]),
                   _corner_row(10, 8, roles[2]),
                   _corner_row(0, 8, "UNRESOLVED_CORNER")]
        result = solve_three_corners(corners)
        self.assertEqual(result["state"], "PARTIAL_HATCH_3C")
        self.assertEqual(result["inferred_corner"], [0, 8])
        self.assertEqual(result["inferred_role"], "INFERRED_GEOMETRIC_CORNER")

    def test_three_physical_corners_beat_four_on_an_inner_parallel_mode(self):
        three = [_corner_row(u, v, "MEASURED_INTERSECTION_CANDIDATE")
                 for u, v in ((0, 0), (10, 0), (10, 8))]
        three.append(_corner_row(0, 8, "UNRESOLVED_CORNER"))
        four = [_corner_row(u, v, "MEASURED_INTERSECTION_CANDIDATE")
                for u, v in ((0, 0), (10, 0), (10, 8), (0, 8))]
        upper = dict(evidence_rank=[1, 2, 0])
        onset = dict(evidence_rank=[1, 1, .97])
        inner = dict(evidence_rank=[1, 1, .65])
        self.assertGreater(_pair_rank(onset, upper, [three]),
                           _pair_rank(inner, upper, [four]))

    def test_shared_separator_three_corners_preserves_unequal_cells(self):
        shared_u = 7
        first = [_corner_row(u, v, "MEASURED_INTERSECTION_CANDIDATE")
                 for u, v in ((0, 0), (shared_u, 0), (shared_u, 8), (0, 8))]
        second = [_corner_row(u, v, "MEASURED_INTERSECTION_CANDIDATE")
                  for u, v in ((shared_u, 0), (20, 0), (20, 8), (shared_u, 8))]
        second[2]["role"] = "UNRESOLVED_CORNER"
        self.assertEqual(solve_three_corners(first)["state"], "PARTIAL_HATCH_4C_REVIEW")
        self.assertEqual(solve_three_corners(second)["state"], "PARTIAL_HATCH_3C")
        self.assertEqual(first[1]["corner_xy"], second[0]["corner_xy"])
        self.assertNotEqual(shared_u, 20 - shared_u)

    def test_three_footprint_context_corners_do_not_create_a_hatch(self):
        corners = [_corner_row(u, v, "FOOTPRINT_CONTEXT_CORNER")
                   for u, v in ((0, 0), (10, 0), (10, 8))]
        self.assertEqual(solve_three_corners(corners)["state"], "UNRESOLVED")

    def test_parallel_height_modes_remain_distinct_from_old_mode(self):
        height = [dict(rho=0.0, level_count=3, position_mad_m=0,
                       coverage=1, support_levels_m=[.2, .4, .6]),
                  dict(rho=.24, level_count=3, position_mad_m=0,
                       coverage=1, support_levels_m=[.2, .4, .6])]
        rows = _merge_candidates([], height, [], CONFIG)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[1]["rho"] - rows[0]["rho"], .24)

    def test_height_corner_without_raw_line_is_not_measured_steel(self):
        points = _cloud()
        grid = build_topview(points, np.eye(2), CONFIG)["fine"]
        corner = _corner(_mode(0, height=3), _mode(0, height=3),
                         [], grid, 2, CONFIG)
        self.assertNotEqual(corner["role"], "MEASURED_INTERSECTION_CANDIDATE")
        self.assertEqual(corner["u_raw3d"], 0)
        self.assertEqual(corner["v_raw3d"], 0)

    def test_outer_hull_l_corner_is_not_a_hatch_corner(self):
        points = _cloud()
        points = points[(points[:, 0] >= 0) & (points[:, 1] >= 0)]
        grid = build_topview(points, np.eye(2), CONFIG)["fine"]
        corner = _corner(_mode(0, height=3), _mode(0, height=3),
                         [], grid, 2, CONFIG)
        self.assertTrue(corner["outer_hull_conflict"])
        self.assertEqual(corner["role"], "UNRESOLVED_CORNER")

    def test_persistent_height_topology_is_z_shift_invariant(self):
        points = _cloud()
        first = build_topview(points, np.eye(2), CONFIG)["fine"]
        shifted = build_topview(points + np.array([0, 0, 20]), np.eye(2), CONFIG)["fine"]
        box = [0, 0, 10, 10]
        a = persistent_height_modes(first, box, CONFIG)
        b = persistent_height_modes(shifted, box, CONFIG)
        self.assertEqual(a["topology"], b["topology"])
        self.assertEqual(len(a["persistent_corners"]), 4)
        self.assertEqual(len(b["persistent_corners"]), 4)
        self.assertTrue(all(row["level_count"] >= 2 for row in a["persistent_corners"]))
        self.assertTrue(all(row["evidence_type"] ==
                            "PERSISTENT_HEIGHT_CORNER_NOT_STEEL"
                            for row in a["persistent_corners"]))
        for axis in (0, 1):
            self.assertEqual([round(row["rho"], 3) for row in a["modes"][axis]],
                             [round(row["rho"], 3) for row in b["modes"][axis]])


if __name__ == "__main__":
    unittest.main()
