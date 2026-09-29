"""Precision contracts for shared-axis and four-side R2G refinement."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ship_perception.r1_static.rectangle_fusion import solve
from ship_perception.r1_static.rectangle_refinement import (
    _joint_score,
    _local_rho_consensus,
    _select_joint,
    _ship_width_consensus,
    diagnose_axes,
    refine,
)
from ship_perception.r1_static.raw_topview_evidence import build_topview
from ship_perception.r1_static.rectangle_refinement_review import _read_validation_xyz
from ship_perception.tests.test_r1_boundary_topology import CONFIG, rectangle_case
from ship_perception.tests.test_r1_rectangle_fusion import single_case


class RectangleRefinement(unittest.TestCase):
    def test_axis_drift_uses_one_shared_ship_axis_despite_bad_edges(self):
        angle = math.radians(4)
        rotation = np.asarray(
            ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
        )
        points, segments, edges, regions = single_case()
        points[:, :2] = points[:, :2] @ rotation.T
        for row in segments:
            row["profile_break_positions"] = (
                np.asarray(row["profile_break_positions"]) @ rotation.T
            ).tolist()
        for row in edges:
            row["a_raw"][:2] = (np.asarray(row["a_raw"][:2]) @ rotation.T).tolist()
            row["b_raw"][:2] = (np.asarray(row["b_raw"][:2]) @ rotation.T).tolist()
        bad = dict(edges[0])
        bad["node_id"] = "bad_node"
        bad["a_raw"] = [0, 0, 0]
        bad["b_raw"] = [40, 0, 0]
        bad["observed_support_length"] = 40
        edges.append(bad)
        initial = solve(points, segments, edges[:-1], regions, CONFIG)
        self.assertTrue(initial["rectangles"])
        wrong = math.radians(2)
        initial_axes = (
            (math.cos(wrong), math.sin(wrong)),
            (-math.sin(wrong), math.cos(wrong)),
        )
        axes, diagnostics = diagnose_axes(
            points, segments, edges, initial["rectangles"], initial_axes, CONFIG
        )
        self.assertLess(
            abs(math.degrees(math.atan2(axes[0, 1], axes[0, 0])) - 4), 1, diagnostics
        )
        self.assertEqual(diagnostics["status"], "AXIS_DRIFT_SUSPECTED")
        self.assertLessEqual(max(diagnostics["per_node_axis_contribution"].values()), 1)
        self.assertTrue(np.allclose(axes @ axes.T, np.eye(2)))

    def test_parallel_rims_below_proposal_dedup_are_both_reviewed(self):
        points, segments, edges, regions = single_case()
        outer = np.column_stack((np.full(201, -0.17), np.linspace(0, 20, 201)))
        segments.append(
            dict(
                node_id="n0",
                seed_id="seed",
                segment_id="outer_close",
                profile_executed_after=True,
                profile_break_positions=outer.tolist(),
                face_positions=[],
            )
        )
        edges.append(
            dict(
                node_id="n0",
                a_raw=[-0.17, 0, 0],
                b_raw=[-0.17, 20, 0],
                observed_support_length=20,
            )
        )
        initial = solve(points, segments, edges, regions, CONFIG)
        result = refine(points, segments, edges, initial, CONFIG)
        profile = result["rectangles"][0]["side_refinement_profiles"]["U0"]
        rhos = [row["rho_grid"] for row in profile]
        self.assertTrue(any(abs(rho + 0.17) < 0.05 for rho in rhos), rhos)
        self.assertTrue(any(abs(rho) < 0.05 for rho in rhos))
        self.assertLess(abs(result["rectangles"][0]["sides"]["U0"]["rho_after"]), 0.12)
        self.assertTrue(result["raster_dedup_precision_bypass"])

    def test_subvoxel_continuous_edge_improves_grid_interface(self):
        x, y = np.meshgrid(np.arange(-3, 43, 0.15), np.arange(-3, 23, 0.15))
        inside = (x > 0.075) & (x < 40) & (y > 0) & (y < 20)
        points = np.column_stack(
            (x.ravel(), y.ravel(), np.where(inside, 0.0, 1.0).ravel())
        )
        segments, edges, regions = rectangle_case()
        # The left side has no 3D line, forcing a raster-grid proposal.
        initial = solve(points, segments[:3], edges[:3], regions, CONFIG)
        result = refine(points, segments[:3], edges[:3], initial, CONFIG)
        side = result["rectangles"][0]["sides"]["U0"]
        self.assertLess(abs(side["rho_after"] - 0.075), abs(side["rho_grid"] - 0.075))
        self.assertLess(abs(side["rho_after"] - 0.075), 0.05)

    def test_area_never_ranks_a_larger_outer_rectangle_higher(self):
        def side(rho):
            return dict(
                rho_refined=rho,
                role="INNER_EDGE",
                structural_coverage=0.8,
                raster_coverage=0.8,
                opening_side_relation=dict(canonical_inner=True),
                uncertainty_m=0.05,
            )

        inner = [side(0), side(40), side(0), side(20)]
        outer = [side(-1), side(41), side(-1), side(21)]
        self.assertEqual(
            _joint_score(inner, (0, 40, 0, 20), CONFIG),
            _joint_score(outer, (-1, 41, -1, 21), CONFIG),
        )

    def test_shared_width_is_only_a_measured_local_evidence_tie_breaker(self):
        def box(width, coverage):
            return [
                dict(
                    rho_refined=rho,
                    role="INNER_EDGE",
                    structural_coverage=(0.1 if index == 2 else coverage),
                    raster_coverage=coverage,
                    opening_side_relation=dict(canonical_inner=True),
                    uncertainty_m=0.05,
                )
                for index, rho in enumerate((0, 40, 0, width))
            ]

        equal_width = _joint_score(box(20, 0.8), (0, 40, 0, 20), CONFIG)
        unequal_width = _joint_score(box(21, 0.8), (0, 40, 0, 21), CONFIG)
        strong_unequal = _joint_score(box(21, 0.9), (0, 40, 0, 21), CONFIG)
        self.assertEqual(equal_width[:8], unequal_width[:8])
        self.assertGreater(strong_unequal, equal_width)
        sides = box(20, 0.8)
        alternative = box(21, 0.8)[3]
        chosen, used = _select_joint([[side] for side in sides[:3]] +
                                     [[sides[3], alternative]],
                                     (0, 40, 0, 20), CONFIG, 21)
        self.assertTrue(used)
        self.assertEqual(chosen[3]["rho_refined"], 21)

    def test_local_noise_trap_uses_repeated_boundary_not_one_strong_peak(self):
        x, y = np.meshgrid(np.arange(8, 13, 0.15), np.arange(0, 21, 0.15))
        true_rho = np.where((y >= 9) & (y <= 11), 10.8, 10.0)
        points = np.column_stack(
            (x.ravel(), y.ravel(), (x <= true_rho).ravel().astype(float))
        )
        grid = build_topview(points, np.eye(2), CONFIG)["fine"]

        def mode(rho, along):
            return dict(
                rho=rho,
                along_min=float(min(along)),
                along_max=float(max(along)),
                along_values=list(along),
                evidence_types=["HEIGHT_TRANSITION"],
            )

        modes = [
            mode(10.0, np.r_[np.arange(0, 9, 0.15), np.arange(11, 20, 0.15)]),
            mode(10.8, np.arange(9, 11, 0.03)),
        ]
        candidates = [dict(rho_grid=rho, role="INNER_EDGE") for rho in (10.0, 10.8)]
        result = _local_rho_consensus(
            candidates, modes, [], grid, points, 0, 1, 10.0, (0, 20), CONFIG
        )
        self.assertTrue(result["confident"], result)
        self.assertLess(abs(result["rho_median"] - 10.0), 0.2, result)
        self.assertGreaterEqual(
            result["independent_along_support_bins"],
            CONFIG["boundary"]["profile_min_sections"],
        )

    def test_weak_side_width_tie_break_needs_two_strong_same_ship_hatches(self):
        def hatch(x0, width, weak=False):
            return dict(
                vessel_hypothesis_id="V_A",
                polygon_xy=[[x0, 0], [x0 + 40, 0], [x0 + 40, width], [x0, width]],
                sides=[
                    dict(
                        side_id=side,
                        structural_coverage=(0.1 if weak and side == "V0" else 0.8),
                        raster_coverage=0.8,
                    )
                    for side in ("U0", "U1", "V0", "V1")
                ],
            )

        strong = [hatch(0, 20), hatch(50, 20)]
        target, widths, mad = _ship_width_consensus(strong, np.eye(2), CONFIG, "V_A")
        self.assertEqual(target, 20)
        self.assertEqual(widths, [20, 20])
        self.assertEqual(mad, 0)
        target_with_weak, _, _ = _ship_width_consensus(
            [strong[0], hatch(50, 22, weak=True)], np.eye(2), CONFIG, "V_A"
        )
        self.assertIsNone(target_with_weak)

    def test_no_cross_scene_width_leakage(self):
        def hatch(width):
            return dict(
                vessel_hypothesis_id="V_A",
                polygon_xy=[[0, 0], [40, 0], [40, width], [0, width]],
                sides=[
                    dict(side_id=side, structural_coverage=0.8, raster_coverage=0.8)
                    for side in ("U0", "U1", "V0", "V1")
                ],
            )

        self.assertEqual(
            _ship_width_consensus([hatch(20), hatch(20)], np.eye(2), CONFIG,
                                  "V_A")[0], 20
        )
        self.assertIsNone(_ship_width_consensus([hatch(12)], np.eye(2), CONFIG,
                                                "V_A")[0])

    def test_copied_ascii_map_ply_can_be_decoded_without_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.ply"
            path.write_text(
                "ply\nformat ascii 1.0\nelement vertex 2\n"
                "property float x\nproperty float y\nproperty float z\n"
                "end_header\n1 2 3\n4 5 6\n",
                encoding="ascii",
            )
            self.assertTrue(
                np.array_equal(_read_validation_xyz(path), [[1, 2, 3], [4, 5, 6]])
            )

    def test_no_rectangle_cannot_report_ready_for_manual_review(self):
        points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]])
        initial = dict(axes=[[1, 0], [0, 1]], rectangles=[])
        result = refine(points, [], [], initial, CONFIG)
        self.assertEqual(result["status"], "NO_CONFIRMED_HATCH")

    def test_smooth_high_cargo_keeps_ambiguous_sides_at_initial_lines(self):
        points, segments, edges, regions = single_case(cargo=True)
        initial = solve(points, segments, edges, regions, CONFIG)
        result = refine(points, segments, edges, initial, CONFIG)
        sides = result["rectangles"][0]["sides"]
        self.assertTrue(any(row["role"] == "AMBIGUOUS" for row in sides.values()))
        self.assertTrue(
            all(
                row["delta_m"] == 0
                for row in sides.values()
                if row["role"] == "AMBIGUOUS"
            )
        )
        self.assertEqual(result["rectangles"][0]["status"], "REVIEW_REQUIRED")

    def test_forbidden_overflow_cannot_be_reabsorbed_by_neighboring_rho(self):
        points, segments, edges, regions = single_case()
        initial = solve(points, segments, edges, regions, CONFIG)
        segments[0]["reference_role"] = "ROLE_AMBIGUOUS_OVERFLOW"
        result = refine(points, segments, edges, initial, CONFIG)
        side = result["rectangles"][0]["sides"]["V0"]
        self.assertEqual(side["role"], "DISTANT_STRUCTURE")
        self.assertEqual(result["rectangles"][0]["status"], "NO_SAFE_REFINEMENT")


if __name__ == "__main__":
    unittest.main()
