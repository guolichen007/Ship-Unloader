"""Topology persists independently of steel and corner measurement success."""

import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ship_perception.io.frame import PointCloudFrame
from ship_perception.perception.context import VesselEvidence, freeze
from ship_perception.perception.legacy_topology import FrozenLegacyTopologyProvider
from ship_perception.r1_static.run import resolve_config


class LegacyTopologyProviderTest(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()
        self.frame = PointCloudFrame(
            np.array(
                [[0.0, 0.0, 2.0], [8.0, 0.0, 2.0], [8.0, 5.0, 2.0], [0.0, 5.0, 2.0]]
            ),
            None,
            "map",
            7,
            12,
            "offline",
            ("x", "y", "z"),
            4,
            json.dumps(dict(sensor_id="lidar-A")),
        )
        indexes = np.arange(4, dtype="<i8")
        vessel = VesselEvidence(
            freeze(
                dict(
                    vessel_hypothesis_id="V",
                    local_axes=[[1, 0], [0, 1]],
                    bbox_xy=[0, 0, 8, 5],
                    source_support_ids=["S"],
                    classification_status="VESSEL_HYPOTHESIS",
                )
            ),
            indexes,
            hashlib.sha256(indexes.tobytes()).hexdigest(),
        )
        self.context = SimpleNamespace(vessels=(vessel,), validate_frame=lambda f: None)
        polygon = [[0, 0], [8, 0], [8, 5], [0, 5]]
        self.b4 = dict(
            rectangles=[
                dict(
                    vessel_hypothesis_id="V",
                    hatch_id="H",
                    axes=[[1, 0], [0, 1]],
                    polygon_before=polygon,
                    polygon_after=[[1, 0], [8, 0], [8, 5], [1, 5]],
                    opening_interior=dict(status="RAISED_OR_CARGO_INTERIOR"),
                    sides={
                        name: dict(rho_after=rho, candidate_modes=[])
                        for name, rho in (("U0", 1), ("U1", 8), ("V0", 0), ("V1", 5))
                    },
                )
            ]
        )
        self.report = dict(
            vessels=[
                dict(
                    vessel_hypothesis_id="V",
                    rectangles=[
                        dict(
                            hatch_id="H",
                            status="UNRESOLVED",
                            corners=[],
                            side_stage_arbitration={
                                "U0": dict(
                                    final_rho=2,
                                    evidence_stronger_than_replaced_stage=True,
                                )
                            },
                        )
                    ],
                )
            ]
        )

    def collect(self):
        return FrozenLegacyTopologyProvider.collect(
            self.frame, self.context, self.b4, self.report, {}, {}, self.config
        )

    def test_missing_corners_and_high_cargo_do_not_remove_region(self):
        regions = self.collect().regions
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].status.region_status, "REGION_AMBIGUOUS")
        self.assertEqual(regions[0].status.geometry_status, "UNRESOLVED")

    def test_old_side_proposal_cannot_crop_conservative_region(self):
        region = self.collect().regions[0]
        side = next(s for s in region.sides if s.side_id == "U0")
        self.assertEqual(
            (side.rho_before, side.rho_proposed, side.rho_after), (0, 2, 0)
        )
        self.assertEqual(side.max_inward_move, 0)
        self.assertEqual(side.inward_move_proposed_m, 2)
        self.assertEqual(side.movement_decision, "KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY")
        self.assertIsNone(side.support_before)
        self.assertIsNone(side.support_after)

    def test_no_invented_z_or_steel_measurements(self):
        result = self.collect().audit()
        region = result["regions"][0]
        self.assertEqual(region["steel_edge_benchmark_status"], "NOT_CERTIFIED")
        self.assertTrue(all(c["z_observed"] is None for c in region["corners"]))
        self.assertTrue(
            all(c["z_reference_estimate"] is None for c in region["corners"])
        )
        json.dumps(result, allow_nan=False)

    def test_evidence_keeps_frame_and_missing_index_trace_explicit(self):
        region = self.collect().regions[0]
        for side in region.sides:
            self.assertEqual(side.frame_id, "map")
            self.assertEqual(side.sensor_id, "lidar-A")
            self.assertEqual(side.measurement_time, "7.000000012")
            self.assertIsNone(side.source_point_indexes)
            np.testing.assert_array_equal(side.vessel_frame_transform, np.eye(4))
        self.assertIn("UNAVAILABLE", region.ownership["source_point_indexes_status"])

    def test_region_status_survives_partial_and_empty_measurements(self):
        self.report["vessels"][0]["rectangles"][0]["status"] = "PARTIAL_HATCH"
        region = self.collect().regions[0]
        self.assertEqual(region.status.region_status, "REGION_PARTIAL")
        self.assertEqual(region.status.geometry_status, "UNRESOLVED")
        self.assertEqual(region.status.operator_confirmation, "PENDING")

    def test_recompute_calls_frozen_post_b2_chain_on_current_raw(self):
        prefix = "ship_perception.perception.legacy_topology."
        with patch(prefix + "verify_legacy_sources"), patch(
            prefix + "prepare_geometry", return_value=({}, self.b4, {})
        ) as prepare, patch(
            prefix + "review_points", return_value=(self.report, None, None)
        ) as review:
            result = FrozenLegacyTopologyProvider().recompute(
                self.frame, self.context, self.config, {}
            )
        self.assertIs(prepare.call_args.args[0], self.frame.xyz)
        self.assertIs(review.call_args.args[1], self.frame.xyz)
        self.assertIs(review.call_args.args[4], self.b4)
        self.assertFalse(review.call_args.kwargs["height_rim"])
        self.assertEqual(len(result.regions), 1)


if __name__ == "__main__":
    unittest.main()
