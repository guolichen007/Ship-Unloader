import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from ship_perception.perception.region_model import OpeningRegion, TopologySupport
from ship_perception.perception.review_geometry import (
    ReviewGeometryAssembler,
    ReviewGeometryEligibility,
    geometry_consistency,
    infer_fourth_corner,
)
from ship_perception.perception.review_height import cross_segment_consensus
from ship_perception.perception.review_runtime import MeasurementTasks
from ship_perception.r1_static.run import resolve_config


class ReviewGeometryContracts(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()

    def test_bounded_keeps_original_pipeline_until_normal6_gate_passes(self):
        from ship_perception.perception import pipeline_v2
        with patch.object(pipeline_v2, 'run_pre_v29_bounded', return_value='ORIGINAL') as original, patch(
            'ship_perception.perception.review_pipeline.run_perception', side_effect=AssertionError('Unaccepted default')
        ):
            self.assertEqual(pipeline_v2.run_perception(None), 'ORIGINAL')
            original.assert_called_once()
            with self.assertRaisesRegex(ValueError, 'OPERATOR_ROI_REQUIRES'):
                pipeline_v2.run_perception(None, operator_vessel_roi='scope.json')

    def test_review_first_requires_explicit_mode(self):
        from ship_perception.perception import pipeline_v2
        with patch.object(pipeline_v2, 'run_pre_v29_bounded', side_effect=AssertionError('Wrong mode')), patch(
            'ship_perception.perception.review_pipeline.run_perception', return_value='REVIEW'
        ) as review:
            self.assertEqual(pipeline_v2.run_perception(None, mode='review-first'), 'REVIEW')
            review.assert_called_once()

    def region(self):
        r = OpeningRegion(
            "R",
            "V",
            (0.0, 0.0, 20.0, 8.0),
            (-4.0, -4.0, 24.0, 12.0),
            (TopologySupport("S4_OPENING_TOPOLOGY", "SUPPORTING"),),
            component_bounds=((0.0, 0.0, 20.0, 8.0),),
        )
        r.identity_role = "CONFIRMED_CONNECTED_OPENING"
        r.side_topology_support["V0"] = TopologySupport(
            "STRUCTURAL_LINE_FAMILY", "SUPPORTING"
        )
        r.identity_evidence = dict(member_support_cells_rc=[[1, 1], [2, 2]])
        return r

    def test_valid_region_without_measured_steel_still_outputs_review_polygon(self):
        result, elig = ReviewGeometryAssembler(self.config).assemble(
            self.region(), np.eye(2)
        )
        self.assertTrue(elig["eligible"])
        self.assertEqual(result["geometry_status"], "COMPLETE_REVIEW_GEOMETRY")
        self.assertEqual(len(result["boundary_xy"]), 4)
        self.assertFalse(result["steel_edge_certified"])
        self.assertTrue(
            all(
                s["final_state"] == "CONSERVATIVE_PRIOR"
                and not s["independent_geometry_support"]
                for s in result["sides"].values()
            )
        )

    def test_unassociated_nomination_not_promoted(self):
        for state in [
            "UNASSOCIATED_NOMINATION",
            "COMPOUND_SEARCH_GROUP",
            "WEAK_UNASSIGNED_REGION",
        ]:
            r = self.region()
            r.identity_role = state
            self.assertIsNone(
                ReviewGeometryAssembler(self.config).assemble(r, np.eye(2))[0]
            )

    def test_open_search_envelope_is_not_a_transverse_extent(self):
        r = self.region()
        r.identity_evidence = {}
        r.component_bounds = ()
        self.assertFalse(ReviewGeometryEligibility.assess(r)["eligible"])

    def mode(self, rho, source="MEASURED_RAW", gain=True):
        return dict(
            mode_id="M" + str(rho) + source,
            rho=rho,
            normal_uv=[1.0, 0.0],
            support_span=[0.0, 8.0],
            source=source,
            state="MEASURED_WEAK" if source == "MEASURED_RAW" else "TOPOLOGY_SUPPORTED",
            uncertainty_m=0.05,
            independent_raw_sections=4 if gain else 0,
            physical_role_improved=gain,
            raw_support_ids=["RAW"],
        )

    def test_lower_priority_source_cannot_override_raw(self):
        r = self.region()
        a = ReviewGeometryAssembler(self.config)
        p, _ = a.assemble(r, np.eye(2))
        q = a.update(r, p, {"U0": [self.mode(0.1)]}, np.eye(2))
        z = a.update(r, q, {"U0": [self.mode(0.2, "TOPOLOGY_SUPPORTED")]}, np.eye(2))
        self.assertEqual(z["sides"]["U0"]["final_source"], "MEASURED_RAW")
        self.assertEqual(z["sides"]["U0"]["final_rho"], 0.1)

    def test_equal_priority_conflict_becomes_multimode(self):
        r = self.region()
        a = ReviewGeometryAssembler(self.config)
        p, _ = a.assemble(r, np.eye(2))
        q = a.update(r, p, {"U0": [self.mode(0.1), self.mode(0.8)]}, np.eye(2))
        self.assertEqual(q["sides"]["U0"]["final_state"], "MULTIMODE")
        self.assertEqual(q["sides"]["U0"]["final_rho"], 0.0)
        self.assertEqual(len(q["sides"]["U0"]["candidate_modes"]), 2)

    def test_large_or_inward_move_without_raw_gain_keeps_conservative(self):
        r = self.region()
        a = ReviewGeometryAssembler(self.config)
        p, _ = a.assemble(r, np.eye(2))
        for value in [-2.0, 2.0]:
            q = a.update(r, p, {"U0": [self.mode(value, gain=False)]}, np.eye(2))
            self.assertEqual(q["sides"]["U0"]["final_rho"], 0.0)

    def test_ill_conditioned_corner_is_downgraded(self):
        r = self.region()
        a = ReviewGeometryAssembler(self.config)
        p, _ = a.assemble(r, np.eye(2))
        mode = self.mode(0.1)
        mode["support_span"] = [3.0, 4.0]
        q = a.update(r, p, {"U0": [mode]}, np.eye(2))
        self.assertEqual(q["sides"]["U0"]["guard_result"], "ILL_CONDITIONED_CORNER")
        self.assertTrue(geometry_consistency(q["sides"], self.config)["valid"])
        self.assertEqual(
            q["boundary_measurements"]["U0"]["final_source"], "MEASURED_RAW"
        )

    def test_inferred_corner_not_claimed_measured(self):
        result = infer_fourth_corner({0: [0.0, 0.0], 1: [20.0, 0.0], 2: [20.0, 8.0]})
        self.assertEqual(result["point_uv"], [0.0, 8.0])
        self.assertFalse(result["observed"])
        self.assertFalse(result["steel_certified"])

    def test_four_line_cartesian_count_zero_and_operation_scaling(self):
        counts = []
        for n in [32, 64, 128, 256]:
            a = ReviewGeometryAssembler(self.config)
            families = [
                dict(
                    family_id="F" + str(i),
                    direction_axis=0,
                    normal_uv=[0.0, 1.0],
                    direction_uv=[1.0, 0.0],
                    rho=8.0 + i * 0.01,
                    support_span=[0.0, 20.0],
                    rho_uncertainty=0.1,
                )
                for i in range(n)
            ]
            result, _ = a.assemble(
                self.region(), np.eye(2), dict(side_families=families)
            )
            self.assertEqual(a.four_line_cartesian_enumeration_count, 0)
            counts.append(a.assembly_operation_count)
        for before, after in zip(counts, counts[1:]):
            self.assertLessEqual(after, 2.5 * before + 64)

    def test_measurement_task_without_new_evidence_cannot_retry(self):
        tasks = MeasurementTasks("INPUT")
        key = tasks.begin("R", "U0", "A")
        self.assertIsNotNone(key)
        self.assertIsNone(tasks.begin("R", "U0", "A"))
        self.assertEqual(tasks.rows[key]["status"], "NO_PROGRESS")

    def test_z_deck_requires_cross_segment_consensus_and_rejects_cargo(self):
        rows = [
            dict(
                segment_id="S" + str(i),
                vessel_id="V",
                plane_role="TRUE_DECK",
                role_confirmed=True,
                source_roles=["TRUE_DECK"],
                normal_uvz=[0.0, 0.0, 1.0],
                offset_m=-3.0,
                fit_residual=0.01,
                source_point_indexes=[i * 10 + j for j in range(10)],
                fit_extent=[[0.0, 0.0], [1.0, 1.0]],
            )
            for i in range(4)
        ]
        self.assertIsNone(
            cross_segment_consensus(rows[:3], np.zeros(2), self.config, "V")
        )
        self.assertEqual(
            cross_segment_consensus(rows, np.zeros(2), self.config, "V")[
                "segment_vote_count"
            ],
            4,
        )
        cargo = copy.deepcopy(rows)
        for row in cargo:
            row["source_roles"] = ["CARGO"]
        self.assertIsNone(cross_segment_consensus(cargo, np.zeros(2), self.config, "V"))


if __name__ == "__main__":
    unittest.main()
