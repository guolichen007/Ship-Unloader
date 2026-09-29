"""B4 counterexamples for relative opening topology and evidence isolation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.opening_semantics import (
    _interior_evidence,
    _provider_lines,
    _reconcile_measured_shared_width,
    _side_review,
)
from ship_perception.r1_static.opening_semantics_review import write_review_ply
from ship_perception.tests.test_r1_boundary_topology import CONFIG


def cloud(interior_z=-2.0, outside_z=0.0, shift=0.0):
    x, y = np.meshgrid(np.arange(-4, 17, 0.5), np.arange(-4, 17, 0.5))
    z = (
        np.where((x >= 0) & (x <= 10) & (y >= 0) & (y <= 10), interior_z, outside_z)
        + shift
    )
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def line(rho, kind, vessel="A", axis=1, start=0, end=10):
    points = [[start, rho], [end, rho]] if axis == 1 else [[rho, start], [rho, end]]
    return dict(
        geometry_type="AXIAL_LINE_INTERVAL",
        provider_type=kind,
        rho_or_region=dict(normal_axis=axis, rho=rho),
        source_geometry=dict(endpoints_xy=points),
        coverage=1.0,
        provider_id="%s:%s:%.2f" % (vessel, kind, rho),
    )


def context(vessel, status, lines):
    return dict(
        vessel_hypothesis_id=vessel,
        vessel_classification_status=status,
        provider_evidence=lines,
    )


def old_side(role="AMBIGUOUS", structural=0.0, raster=0.1, conflict=False):
    return dict(
        role=role,
        structural_coverage=structural,
        raster_coverage=raster,
        mode_conflict=conflict,
        rho_after=12.0,
    )


class OpeningTopology(unittest.TestCase):
    def test_lower_basin_and_four_relative_corners(self):
        grid = R1HeightGrid.from_points(cloud(), 0.5)
        row = _interior_evidence(grid, [0, 0, 10, 10], CONFIG)
        self.assertEqual(row["status"], "LOWER_INTERIOR")
        self.assertEqual(row["corner_support_count"], 4)
        self.assertGreater(row["local_basin_depth_m"], 0)

    def test_corners_disfavor_a_box_outside_the_opening(self):
        grid = R1HeightGrid.from_points(cloud(), 0.5)
        correct = _interior_evidence(grid, [0, 0, 10, 10], CONFIG)
        outer = _interior_evidence(grid, [-2, -2, 12, 12], CONFIG)
        self.assertEqual(correct["corner_support_count"], 4)
        self.assertEqual(outer["corner_support_count"], 0)

    def test_loaded_cargo_and_z_shift_keep_relative_semantics(self):
        first = _interior_evidence(
            R1HeightGrid.from_points(cloud(2.0, 0.0), 0.5), [0, 0, 10, 10], CONFIG
        )
        shifted = _interior_evidence(
            R1HeightGrid.from_points(cloud(2.0, 0.0, 17.0), 0.5), [0, 0, 10, 10], CONFIG
        )
        self.assertEqual(first["status"], "RAISED_OR_CARGO_INTERIOR")
        self.assertEqual(first["status"], shifted["status"])
        self.assertAlmostEqual(
            first["local_basin_depth_m"], shifted["local_basin_depth_m"]
        )

    def test_flat_outer_structure_is_not_a_lower_opening(self):
        flat = cloud(0.0, 0.0)
        row = _interior_evidence(
            R1HeightGrid.from_points(flat, 0.5), [0, 0, 10, 10], CONFIG
        )
        self.assertEqual(row["status"], "HEIGHT_AMBIGUOUS")
        self.assertEqual(row["corner_support_count"], 0)


class SideProvenance(unittest.TestCase):
    def test_other_confirmed_vessel_cannot_supply_edge(self):
        rows = [
            context("A", "VESSEL_HYPOTHESIS", []),
            context(
                "B", "VESSEL_HYPOTHESIS", [line(10, "RAW3D_STRUCTURAL_PROVIDER", "B")]
            ),
        ]
        groups = _provider_lines(rows, np.eye(2), 1, (0, 10), 10, CONFIG, "A")
        self.assertEqual(groups, [])

    def test_neighbor_hatch_nonoverlapping_span_cannot_supply_edge(self):
        rows = [
            context(
                "A",
                "VESSEL_HYPOTHESIS",
                [line(10, "RAW3D_STRUCTURAL_PROVIDER", start=20, end=30)],
            )
        ]
        groups = _provider_lines(rows, np.eye(2), 1, (0, 10), 10, CONFIG, "A")
        self.assertEqual(groups, [])

    def test_weak_outer_side_does_not_become_confirmed_from_geometry(self):
        grid = R1HeightGrid.from_points(cloud(), 0.5)
        row = _side_review(
            "V1", [0, 0, 10, 12], old_side(), [], "A", np.eye(2), grid, CONFIG
        )
        self.assertEqual(row["role"], "OUTER_HULL_EDGE")
        self.assertEqual(row["evidence_status"], "UNRESOLVED_KEEP_R2G_GEOMETRY")

    def test_local_axis_compares_rho_in_one_coordinate_frame(self):
        grid = R1HeightGrid.from_points(cloud(), 0.5)
        old = old_side("INNER_EDGE", 0.8, 0.8)
        old["rho_after"] = 13.5  # Coordinate in the old R2G frame.
        row = _side_review("V1", [0, 0, 10, 12], old, [], "A", np.eye(2), grid, CONFIG)
        self.assertEqual(row["rho_before"], 12)
        self.assertEqual(row["rho_after"], 12)
        self.assertEqual(row["r2g_rho_before"], 13.5)

    def test_b3_multimode_conflict_cannot_jump_to_other_rim(self):
        grid = R1HeightGrid.from_points(cloud(), 0.5)
        rows = [
            context(
                "A",
                "VESSEL_HYPOTHESIS",
                [
                    line(10, "RAW3D_STRUCTURAL_PROVIDER"),
                    line(10, "BEV_RECTILINEAR_PROVIDER"),
                ],
            )
        ]
        row = _side_review(
            "V1",
            [0, 0, 10, 12],
            old_side("INNER_EDGE", 0.7, 0.8, True),
            rows,
            "A",
            np.eye(2),
            grid,
            CONFIG,
        )
        self.assertEqual(row["rho_after"], 12)
        self.assertEqual(row["evidence_status"], "MULTIMODE_CONFLICT_KEEP_R2G")

    def test_equal_width_uses_only_shared_measured_rim(self):
        def side(rho, modes=None, conflict=False):
            return dict(
                rho_before=rho,
                rho_after=rho,
                delta_m=0.0,
                role="UNKNOWN",
                selected_provider_ids=[],
                selected_source_vessel_ids=[],
                evidence_status=(
                    "MULTIMODE_CONFLICT_KEEP_R2G"
                    if conflict
                    else "UNRESOLVED_KEEP_R2G_GEOMETRY"
                ),
                candidate_modes=modes or [],
            )

        def mode(rho, far=0.8):
            return dict(
                rho=rho,
                raw3d_coverage=0.8,
                bev_coverage=0.8,
                opening_relation=dict(
                    status="OPENING_LOWER",
                    support_fraction=0.8,
                    far_support_fraction=far,
                ),
                provider_ids=["raw", "bev"],
                source_vessel_ids=["A"],
            )

        first = dict(
            vessel_hypothesis_id="A",
            axis_source="R2G_SUPPORTED",
            status="UNRESOLVED",
            sides=dict(
                U0=side(10), U1=side(40), V0=side(2, [mode(4.6)], True), V1=side(17)
            ),
        )
        second = dict(
            vessel_hypothesis_id="A",
            axis_source="R2G_SUPPORTED",
            status="UNRESOLVED",
            sides=dict(
                U0=side(40),
                U1=side(70),
                V0=side(4, [mode(4.6, 0.1)], True),
                V1=side(17.1),
            ),
        )
        _reconcile_measured_shared_width([first, second], CONFIG)
        self.assertAlmostEqual(first["sides"]["V0"]["rho_after"], 4.6)
        self.assertAlmostEqual(second["sides"]["V0"]["rho_after"], 4.6)
        self.assertEqual(first["status"], "PARTIAL_HATCH")

        first["sides"]["V0"] = side(2, [mode(4.6)], True)
        second["sides"]["V0"] = side(4, [], True)
        _reconcile_measured_shared_width([first, second], CONFIG)
        self.assertEqual(first["sides"]["V0"]["rho_after"], 2)
        self.assertEqual(second["sides"]["V0"]["rho_after"], 4)

    def test_review_ply_never_claims_verified_steel(self):
        row = dict(
            polygon_after=[[0, 0], [1, 0], [1, 1], [0, 1]],
            sides={
                side_id: dict(role="UNKNOWN") for side_id in ("U0", "U1", "V0", "V1")
            },
        )
        with TemporaryDirectory() as folder:
            path = Path(folder) / "review.ply"
            write_review_ply([row], CONFIG, path)
            header = path.read_text(encoding="ascii").split("end_header")[0]
            self.assertIn("CANDIDATE_NOT_VERIFIED_STEEL", header)
            self.assertIn("element vertex ", header)


if __name__ == "__main__":
    unittest.main()
