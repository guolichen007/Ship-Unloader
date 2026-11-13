"""Current material assignment differs from geometric step/face existence."""

import copy
from types import SimpleNamespace
import unittest
import numpy as np

from ship_perception.perception.measurement import VesselMeasurementContext
from ship_perception.perception.opening_boundary_assignment import (
    OpeningBoundaryAssignment,
)
from ship_perception.perception.opening_rim_source import support_fingerprint
from ship_perception.perception.review_geometry import ReviewGeometryAssembler
from ship_perception.r1_static.run import resolve_config


class OpeningBoundaryAssignmentContracts(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()

    def fixture(self, disconnected=False, interior_face=False, wall=False):
        along = np.arange(0.0, 12.01, 0.1)
        x, y = np.meshgrid(along, np.arange(0.8, 4.01, 0.1))
        floor = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        if disconnected:
            floor = floor[~((floor[:, 1] > 1.0) & (floor[:, 1] < 2.0))]
        if interior_face:
            x, y = np.meshgrid(along, np.arange(-0.5, 0.8, 0.1))
            floor = np.r_[floor, np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]]
        roof = np.c_[along, np.full(len(along), 0.7), np.full(len(along), 2.0)]
        wall_points = np.empty((0, 3))
        if wall:
            x, z = np.meshgrid(along, np.arange(0.05, 2.0, 0.1))
            wall_points = np.c_[x.ravel(), np.full(x.size, 0.7), z.ravel()]
        points = np.vstack([floor, roof, wall_points])
        indexes = np.arange(len(points), dtype=np.int64) * 3 + 17
        own = (np.arange(len(points)) < len(floor)) & (points[:, 1] >= 1.0)
        if interior_face:
            own |= (np.arange(len(points)) < len(floor)) & (points[:, 1] <= 0.6)
        if wall:
            # Enough actual Core line support, fewer than min_plane_points.
            patch = np.flatnonzero(
                (np.arange(len(points)) >= len(floor) + len(roof))
                & (points[:, 0] <= 2.1)
                & (points[:, 2] > 0.9)
                & (points[:, 2] < 1.1)
            )
            own[patch[::2]] = True
        vessel = SimpleNamespace(vessel_id="V", point_indexes=indexes[own])
        y0 = -0.5 if interior_face else 0.0
        region = SimpleNamespace(
            region_id="OPEN",
            vessel_id="V",
            identity_role="CONFIRMED_INDEPENDENT_OPENING",
            conservative_bounds=[0.0, 0.7, 12.0, 4.0],
            identity_evidence=dict(
                member_support_grid=dict(x0=0.0, y0=y0, cell_m=0.5),
                member_support_cells_rc=[
                    [r, c] for r in range(9 if interior_face else 8) for c in range(25)
                ],
            ),
        )
        primary = SimpleNamespace(vessel=vessel, xyz=points[own], config=self.config)
        local = SimpleNamespace(
            vessel=vessel,
            xyz=points[own],
            context_xyz=points,
            config=self.config,
            primary_cache=primary,
            measurement_context=SimpleNamespace(point_indexes=indexes),
            opening_identities=[region],
        )
        measurement = VesselMeasurementContext(local)
        rows = []
        for lo, hi in zip([0.0, 3.0, 6.0, 9.0], [3.0, 6.0, 9.0, 12.0]):
            inside = np.flatnonzero(
                (points[:, 0] >= lo)
                & (points[:, 0] <= hi)
                & (abs(points[:, 1] - 0.8) < 1e-6)
                & (points[:, 2] == 0)
            )
            outside = np.flatnonzero(
                (points[:, 0] >= lo)
                & (points[:, 0] <= hi)
                & (points[:, 1] == 0.7)
                & (points[:, 2] == 2.0)
            )
            rows.append(
                dict(
                    source_pair_provenance=dict(
                        decision_input=False,
                        used_for_detection=False,
                        normal_axis=1,
                        inward_sign=1,
                        section_span_m=[lo, hi],
                        anchor_rho_m=0.75,
                        outside_surface_median_z=2.0,
                        pairs=[
                            dict(
                                inside_point_indexes=indexes[inside].tolist(),
                                outside_point_indexes=indexes[outside].tolist(),
                            )
                        ],
                    )
                )
            )
        return measurement, region, dict(sections=rows)

    def test_connected_context_pair_has_owned_opening_path_without_core_promotion(self):
        measurement, region, row = self.fixture()
        before_core = measurement.local.primary_cache.vessel.point_indexes.copy()
        fingerprint = support_fingerprint(region)
        result = OpeningBoundaryAssignment(measurement, self.config).profile(
            region, row
        )
        self.assertEqual(result["state"], "ASSIGNED_CURRENT_OPENING")
        self.assertEqual(result["supported_section_count"], 4)
        self.assertFalse(result["core_promoted"])
        self.assertFalse(result["corner_z_source"])
        self.assertFalse(result["steel_certified"])
        for section in row["sections"]:
            self.assertTrue(section["source_pair_provenance"]["decision_input"])
        np.testing.assert_array_equal(
            before_core, measurement.local.primary_cache.vessel.point_indexes
        )
        self.assertEqual(fingerprint, support_fingerprint(region))
        original = set(measurement.raw_point_indexes.tolist())
        for section in result["sections"]:
            for path in section["low_material_proof"]["witness_paths"]:
                self.assertTrue(set(path["point_indexes"]).issubset(original))

    def test_disconnected_geometric_step_is_not_this_openings_measurement(self):
        measurement, region, row = self.fixture(disconnected=True)
        result = OpeningBoundaryAssignment(measurement, self.config).profile(
            region, row
        )
        self.assertEqual(result["state"], "UNASSIGNED_CURRENT_OPENING")
        self.assertEqual(result["supported_section_count"], 0)
        self.assertEqual(len(row["sections"]), 4)

    def test_other_opening_material_blocks_attachment(self):
        measurement, region, row = self.fixture()
        other = copy.deepcopy(region)
        other.region_id = "OTHER"
        other.identity_evidence["member_support_cells_rc"] = [
            [r, c] for r in [2, 3] for c in range(25)
        ]
        measurement.local.opening_identities.append(other)
        result = OpeningBoundaryAssignment(measurement, self.config).profile(
            region, row
        )
        self.assertEqual(result["state"], "UNASSIGNED_CURRENT_OPENING")

    def test_missing_members_does_not_claim_assignment(self):
        measurement, region, row = self.fixture()
        region.identity_evidence = {}
        result = OpeningBoundaryAssignment(measurement, self.config).profile(
            region, row
        )
        self.assertEqual(result["state"], "UNASSIGNED_CURRENT_OPENING")

    def test_fit_outlier_cannot_supply_missing_independent_assignment_vote(self):
        measurement, region, row = self.fixture()
        row["fit"] = dict(outlier_section_indexes=[0])
        result = OpeningBoundaryAssignment(measurement, self.config).profile(
            region, row
        )
        self.assertEqual(result["state"], "UNASSIGNED_CURRENT_OPENING")
        self.assertEqual(result["supported_section_count"], 3)
        self.assertFalse(row["sections"][0]["source_pair_provenance"]["decision_input"])
        self.assertEqual(len(row["sections"]), 4)

    def test_repeated_same_physical_section_cannot_create_independent_votes(self):
        measurement, region, row = self.fixture()
        row["sections"] = [copy.deepcopy(row["sections"][0]) for _ in range(4)]
        result = OpeningBoundaryAssignment(measurement, self.config).profile(
            region, row
        )
        self.assertEqual(result["state"], "UNASSIGNED_CURRENT_OPENING")
        self.assertEqual(result["supported_section_count"], 1)

    def test_complete_context_face_accepts_measured_small_core_patch(self):
        measurement, region, _ = self.fixture(wall=True)
        modes, audit = OpeningBoundaryAssignment(
            measurement, self.config
        ).complete_faces(region, "V0", 1, 1, [0.0, 12.0])
        self.assertEqual(len(modes), 1, audit)
        mode = modes[0]
        self.assertAlmostEqual(mode["rho"], 0.7, places=6)
        self.assertFalse(mode["steel_certified"])
        self.assertLess(
            mode["opening_assignment"]["core_attachment"]["core_normal_point_count"],
            self.config["geometry"]["min_plane_points"],
        )
        self.assertTrue(mode["independent_opening_role_gain"])
        self.assertEqual(mode["opening_assignment_state"], "ASSIGNED_CURRENT_OPENING")

    def test_internal_face_with_positive_material_on_both_sides_is_not_boundary(self):
        measurement, region, _ = self.fixture(wall=True, interior_face=True)
        modes, audit = OpeningBoundaryAssignment(
            measurement, self.config
        ).complete_faces(region, "V0", 1, 1, [0.0, 12.0])
        self.assertEqual(modes, [])
        self.assertTrue(
            any(
                f["assignment"]["cuts_positive_opening_material"]
                for f in audit["faces"]
            )
        )

    def test_same_face_role_corroboration_preserves_original_raw_coordinates(self):
        measurement, region, _ = self.fixture(wall=True)
        assignment = OpeningBoundaryAssignment(measurement, self.config)
        complete, _ = assignment.complete_faces(region, "V0", 1, 1, [0.0, 12.0])
        primary = copy.deepcopy(complete[0])
        primary.update(
            mode_id="PRIMARY_FACE",
            rho=0.701,
            independent_opening_role_gain=False,
            source_point_indexes=primary["source_point_indexes"][::2],
        )
        before = copy.deepcopy(primary)
        result = assignment.corroborate_primary_faces([primary], complete)
        self.assertEqual(len(result), 1)
        self.assertEqual(primary, before)
        for key in [
            "mode_id",
            "rho",
            "normal_uv",
            "measured_intervals",
            "source_point_indexes",
            "uncertainty_m",
        ]:
            self.assertEqual(result[0][key], before[key])
        self.assertTrue(result[0]["independent_opening_role_gain"])
        self.assertEqual(
            result[0]["opening_assignment_state"], "ASSIGNED_CURRENT_OPENING"
        )
        self.assertFalse(result[0]["steel_certified"])

    def test_competing_parallel_face_is_not_averaged_or_used_as_same_face_proof(self):
        measurement, region, _ = self.fixture(wall=True)
        assignment = OpeningBoundaryAssignment(measurement, self.config)
        complete, _ = assignment.complete_faces(region, "V0", 1, 1, [0.0, 12.0])
        primary = copy.deepcopy(complete[0])
        primary.update(
            mode_id="OTHER_FACE", rho=-0.7, independent_opening_role_gain=False
        )
        result = assignment.corroborate_primary_faces([primary], complete)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], primary)
        self.assertEqual(result[1], complete[0])

    def test_rejected_mode_stays_diagnostic_and_does_not_move_side(self):
        assembler = ReviewGeometryAssembler(self.config)
        side = dict(candidate_modes=[], rejected_mode_ids=[], final_rho=1.0)
        mode = dict(
            mode_id="QUAY_STEP", ownership_rejected=True, source="MEASURED_RAW", rho=0.0
        )
        assembler._apply_mode(side, mode)
        self.assertEqual(side["final_rho"], 1.0)
        self.assertEqual(side["candidate_modes"], [mode])
        self.assertEqual(side["rejected_mode_ids"], ["QUAY_STEP"])


if __name__ == "__main__":
    unittest.main()
