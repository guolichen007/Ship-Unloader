"""B5 forensic counterexamples. Expected failures are the unopened B5 solver contract.

These tests keep the current B4 production behavior visible. Passing forensic
checks do not imply that a measured Hatch corner or inner steel edge is certified.
"""

import unittest

import numpy as np

from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.opening_semantics import _provider_lines, _side_review
from ship_perception.r1_static.r2h_b5_corner_forensics import (
    _corner,
    _side_modes,
    audit_final_side_sources,
    audit_lineage,
)
from ship_perception.tests.test_r1_boundary_topology import CONFIG


RAW = "RAW3D_STRUCTURAL_PROVIDER"
BEV = "BEV_RECTILINEAR_PROVIDER"


def _cloud(inside=-2.0, outside=0.0, shift=0.0, x_start=-4):
    x, y = np.meshgrid(np.arange(x_start, 15, 0.5), np.arange(-4, 15, 0.5))
    z = np.where((x >= 0) & (x <= 10) & (y >= 0) & (y <= 10), inside, outside)
    return np.column_stack((x.ravel(), y.ravel(), (z + shift).ravel()))


def _observation(axis, rho, provider, scope="SAME_VESSEL", start=-2, end=12):
    return dict(
        normal_axis=axis,
        rho=float(rho),
        along=[float(start), float(end)],
        provider_type=provider,
        provider_id=f"{scope}:{provider}:{axis}:{rho}",
        source_vessel_id="V1" if scope == "SAME_VESSEL" else "U0",
        source_support_ids=["S1"],
        source_scope=scope,
        coverage=1.0,
    )


def _line_for_b4(row):
    axis = row["normal_axis"]
    rho = row["rho"]
    endpoints = [[0, rho], [10, rho]] if axis == 1 else [[rho, 0], [rho, 10]]
    return dict(
        geometry_type="AXIAL_LINE_INTERVAL",
        provider_type=row["provider_type"],
        rho_or_region=dict(normal_axis=axis, rho=rho),
        source_geometry=dict(endpoints_xy=endpoints),
        coverage=1.0,
        provider_id=row["provider_id"],
    )


def _fixture_two_hatches_one_proposal():
    def hatch(name, lo, hi):
        return dict(
            hatch_id=name,
            vessel_hypothesis_id="V1",
            hatch_hypothesis_id="V1:H0",
            hatch_hypothesis_iou=0.5,
            axes=[[1, 0], [0, 1]],
            polygon_before=[[lo, 0], [hi, 0], [hi, 10], [lo, 10]],
        )

    b4 = dict(rectangles=[hatch("H000", 0, 10), hatch("H001", 10, 20)])
    b2 = dict(
        scene=dict(
            vessel_hypotheses=[
                dict(vessel_hypothesis_id="V1", local_axes=[[1, 0], [0, 1]])
            ]
        ),
        per_vessel=[
            dict(
                vessel_hypothesis_id="V1",
                hatch_hypotheses=[
                    dict(
                        hatch_hypothesis_id="V1:H0",
                        bounds_axial=[0, 0, 20, 10],
                        state="HATCH_PROPOSAL",
                    )
                ],
            )
        ],
    )
    return b4, b2


class B5ForensicContracts(unittest.TestCase):
    def test_four_true_corners_with_parallel_false_rims_keep_distinct_modes(self):
        observations = [
            _observation(1, rho, provider)
            for rho in (0, 0.2, 10, 10.2)
            for provider in (RAW, BEV)
        ]
        modes = _side_modes(observations, 1, (0, 10), (-1, 11), CONFIG)
        self.assertGreaterEqual(len(modes), 4)
        self.assertEqual(
            sorted(round(row["rho"], 1) for row in modes), [0, 0.2, 10, 10.2]
        )

    @unittest.expectedFailure
    def test_parallel_false_rim_corner_score_must_favor_true_intersection(self):
        observations = [_observation(0, 0, provider) for provider in (RAW, BEV)]
        observations += [
            _observation(1, rho, provider)
            for rho in (0, 0.2)
            for provider in (RAW, BEV)
        ]
        grid = R1HeightGrid.from_points(_cloud(), 0.5)
        true_corner = _corner(
            dict(rho=0), dict(rho=0), observations, grid, [-1], CONFIG, 2
        )
        false_corner = _corner(
            dict(rho=0), dict(rho=0.2), observations, grid, [-1], CONFIG, 2
        )
        # Current contrast/intersection evidence ties physically distinct rims.
        self.assertGreater(
            true_corner["local_height_contrast_m"],
            false_corner["local_height_contrast_m"],
        )

    def test_two_adjacent_hatches_shared_separator_exposes_composite_lineage(self):
        b4, b2 = _fixture_two_hatches_one_proposal()
        report = audit_lineage(b4, b2, CONFIG)
        self.assertEqual(report["composite_b2_proposals"], ["V1:H0"])
        self.assertEqual(len(report["one_to_one_forensic_assignment"]), 2)
        self.assertTrue(
            all(
                row["status"] == "UNMATCHED_OPENING"
                for row in report["one_to_one_forensic_assignment"]
            )
        )

    def test_two_unequal_hatch_lengths_are_not_normalized(self):
        b4, b2 = _fixture_two_hatches_one_proposal()
        b4["rectangles"][0]["polygon_before"] = [[0, 0], [7, 0], [7, 10], [0, 10]]
        b4["rectangles"][1]["polygon_before"] = [[7, 0], [20, 0], [20, 10], [7, 10]]
        report = audit_lineage(b4, b2, CONFIG)
        self.assertFalse(report["algorithm_changed"])
        self.assertEqual(report["composite_b2_proposals"], ["V1:H0"])

    def test_unresolved_selected_edge_is_exposed_in_forensic_audit(self):
        b4 = dict(
            rectangles=[
                dict(
                    hatch_id="H000",
                    sides=dict(V0=dict(selected_source_vessel_ids=["U0"])),
                )
            ]
        )
        b2 = dict(
            scene=dict(
                vessel_hypotheses=[
                    dict(vessel_hypothesis_id="U0", classification_status="UNRESOLVED")
                ]
            )
        )
        report = audit_final_side_sources(b4, b2)
        self.assertTrue(report[0]["unresolved_only"])
        self.assertTrue(report[0]["used_as_formal_b4_side"])

    @unittest.expectedFailure
    def test_two_equal_hatches_wrong_initial_boxes_need_nonlocal_relocalization(self):
        # Current B4 searches only around the old rho. A true measured edge at
        # rho=5 is outside the existing rho=12 support_search_m interval.
        row = _observation(1, 5, RAW)
        groups = _provider_lines(
            [
                dict(
                    vessel_hypothesis_id="V1",
                    vessel_classification_status="VESSEL_HYPOTHESIS",
                    provider_evidence=[_line_for_b4(row)],
                )
            ],
            np.eye(2),
            1,
            (0, 10),
            12,
            CONFIG,
            "V1",
        )
        self.assertTrue(groups)

    def test_outer_hull_l_corner_trap_remains_uncertified(self):
        grid = R1HeightGrid.from_points(_cloud(x_start=0), 0.5)
        modes = [dict(rho=0.0), dict(rho=0.0)]
        observations = [
            _observation(axis, 0, provider)
            for axis in (0, 1)
            for provider in (RAW, BEV)
        ]
        corner = _corner(*modes, observations, grid, [-1.0], CONFIG, 2)
        self.assertTrue(corner["outer_hull_conflict"])
        self.assertNotEqual(corner["role"], "MEASURED_HATCH_CORNER")

    def test_cargo_filled_visible_corners_are_z_translation_invariant(self):
        modes = [dict(rho=0.0), dict(rho=0.0)]
        observations = [
            _observation(axis, 0, provider)
            for axis in (0, 1)
            for provider in (RAW, BEV)
        ]
        first = _corner(
            *modes,
            observations,
            R1HeightGrid.from_points(_cloud(inside=3), 0.5),
            [1],
            CONFIG,
            2,
        )
        shifted = _corner(
            *modes,
            observations,
            R1HeightGrid.from_points(_cloud(inside=3, shift=20), 0.5),
            [21],
            CONFIG,
            2,
        )
        self.assertEqual(first["role"], shifted["role"])
        self.assertAlmostEqual(
            first["local_height_contrast_m"], shifted["local_height_contrast_m"]
        )

    @unittest.expectedFailure
    def test_one_corner_occluded_must_be_partial_not_completed(self):
        observations = [
            _observation(0, 0, RAW),
            _observation(0, 0, BEV),
            _observation(1, 0, RAW),
        ]
        corner = _corner(
            dict(rho=0),
            dict(rho=0),
            observations,
            R1HeightGrid.from_points(_cloud(), 0.5),
            [-1],
            CONFIG,
            2,
        )
        self.assertEqual(corner["role"], "PARTIAL_HATCH")

    def test_height_corner_without_structural_edge_is_position_only(self):
        corner = _corner(
            dict(rho=0),
            dict(rho=0),
            [],
            R1HeightGrid.from_points(_cloud(), 0.5),
            [-1],
            CONFIG,
            2,
        )
        self.assertEqual(corner["structural_support"], 0)
        self.assertNotEqual(corner["role"], "MEASURED_HATCH_CORNER")

    @unittest.expectedFailure
    def test_two_hatches_one_b2_hypothesis_current_formal_lineage_is_invalid(self):
        b4, _ = _fixture_two_hatches_one_proposal()
        ids = [row["hatch_hypothesis_id"] for row in b4["rectangles"]]
        self.assertEqual(len(ids), len(set(ids)))

    @unittest.expectedFailure
    def test_unresolved_support_edge_leak_current_b4_allows_candidate(self):
        row = _observation(1, 10, RAW, scope="CONTEXT_ONLY", start=0, end=10)
        groups = _provider_lines(
            [
                dict(
                    vessel_hypothesis_id="U0",
                    vessel_classification_status="UNRESOLVED",
                    provider_evidence=[_line_for_b4(row)],
                )
            ],
            np.eye(2),
            1,
            (0, 10),
            10,
            CONFIG,
            "V1",
        )
        self.assertEqual(groups, [])


if __name__ == "__main__":
    unittest.main()
