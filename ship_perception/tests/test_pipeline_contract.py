"""Verify reuse at the actual pipeline boundary, plus per-hatch axis isolation."""

import unittest
from unittest import mock

import numpy as np

from ship_perception.io.frame import PointCloudFrame
from ship_perception.perception.pipeline import run_perception
from ship_perception.perception.output import auxiliary_review
from ship_perception.perception.context import infer_scene_vessels
from ship_perception.tests.test_hatch_metamorphic import vessel_points


class PipelineContracts(unittest.TestCase):
    def test_split_core_keeps_raw_supported_boundary_context_as_partial(self):
        from ship_perception.perception.final_boundary import FinalBoundaryRefiner

        xyz, _ = vessel_points(1, spacing=0.12)
        frame = PointCloudFrame(xyz, None, "map", 0, 0, "", ("x", "y", "z"), len(xyz))
        masks = [((xyz[:, 1] >= 6) & (xyz[:, 1] < 11)), xyz[:, 1] < 6, xyz[:, 1] >= 11]
        indexes = [np.flatnonzero(mask) for mask in masks]

        def split_discovery(*args):
            rows = []
            for number, ids in enumerate(indexes):
                points = xyz[ids]
                rows.append(
                    dict(
                        vessel_hypothesis_id="V" + str(number),
                        source_support_ids=["S0"],
                        local_axes=np.eye(2).tolist(),
                        bbox_xy=[*points[:, :2].min(0), *points[:, :2].max(0)],
                        point_count=len(ids),
                        classification_status=(
                            "VESSEL_HYPOTHESIS" if number == 0 else "UNRESOLVED"
                        ),
                    )
                )
            return dict(vessel_hypotheses=rows), dict(
                vessel_point_indexes={
                    "V" + str(n): ids for n, ids in enumerate(indexes)
                }
            )

        def comparison(*args):
            low = xyz[xyz[:, 2] < -1, :2]
            return [
                dict(
                    proposal_id="legacy:context",
                    vessel_hypothesis_id="V0",
                    bounds_axial=[*low.min(0), *low.max(0)],
                    axes=np.eye(2).tolist(),
                    proposal_source="SYNTHETIC_CONTEXT_COMPARISON",
                    inherited_review_status="UNRESOLVED",
                )
            ], {}

        original = FinalBoundaryRefiner.refine
        for weak, bridge in ((False, False), (True, False), (False, True)):
            current_frame = frame
            if bridge:
                connected = xyz.copy()
                low_path = (
                    (connected[:, 0] > 11)
                    & (connected[:, 0] < 17)
                    & (connected[:, 1] < 3.5)
                )
                connected[low_path, 2] = -4
                current_frame = PointCloudFrame(
                    connected, None, "map", 0, 0, "", ("x", "y", "z"), len(connected)
                )

            def refine(refiner, proposal, axes):
                result = original(refiner, proposal, axes)
                if weak and proposal["proposal_id"] == "legacy:context":
                    for side in result["sides"].values():
                        side["raw_subvoxel_section_count"] = 0
                return result

            with mock.patch(
                "ship_perception.perception.context.infer_scene_vessels",
                side_effect=split_discovery,
            ) as b1, mock.patch(
                "ship_perception.perception.pipeline._legacy_proposals",
                side_effect=comparison,
            ), mock.patch.object(
                FinalBoundaryRefiner, "refine", new=refine
            ):
                result = run_perception(current_frame, legacy_evidence=True)
            self.assertEqual(b1.call_count, 1)
            kept = [
                h
                for h in result["geometry_only"]["hatches"]
                if h["proposal_id"] == "legacy:context"
            ]
            if weak or bridge:
                self.assertEqual(kept, [])
                if bridge:
                    deferred = [
                        h
                        for v in result["per_vessel_status"]
                        for h in v["unresolved_candidates"]["samples"]
                        if h.get("proposal_id") == "legacy:context"
                    ]
                    self.assertEqual(len(deferred), 1)
                    self.assertGreaterEqual(deferred[0]["review_support_side_count"], 3)
                    self.assertTrue(
                        deferred[0]["exterior"]["known_exterior_low_connected"]
                    )
                continue
            self.assertEqual(len(kept), 1)
            hatch = kept[0]
            self.assertEqual(hatch["status"], "PARTIAL_HATCH")
            self.assertEqual(
                hatch["containment"]["status"], "PARTIAL_BOUNDARY_AMBIGUOUS"
            )
            self.assertFalse(hatch["exterior"]["known_exterior_low_connected"])
            self.assertFalse(hatch["boundary_context_review"]["ownership_confirmed"])
            self.assertEqual(
                hatch["boundary_context_review"]["measurement_containment"]["status"],
                "CONTAINED",
            )
            self.assertIn("BOUNDARY_OWNERSHIP_UNRESOLVED", hatch["candidate_blockers"])
            self.assertGreaterEqual(hatch["review_support_side_count"], 3)

    def test_invalid_final_polygon_is_local_uncertainty_not_a_frame_failure(self):
        from ship_perception.perception.final_boundary import FinalBoundaryRefiner

        xyz, _ = vessel_points(2, spacing=0.12)
        frame = PointCloudFrame(xyz, None, "map", 0, 0, "", ("x", "y", "z"), len(xyz))
        original = FinalBoundaryRefiner.refine
        failed = []

        def invalid_once(refiner, proposal, axes):
            result = original(refiner, proposal, axes)
            measured = sum(
                min(s["raw_subvoxel_section_count"], s["positive_height_section_count"])
                >= 4
                for s in result.get("sides", {}).values()
            )
            if not failed and measured >= 3:
                failed.append(proposal["proposal_id"])
                return dict(
                    status="UNRESOLVED",
                    error_code="DEGENERATE_FINAL_BOUNDARY",
                    sides=result["sides"],
                )
            return result

        with mock.patch.object(FinalBoundaryRefiner, "refine", new=invalid_once):
            result = run_perception(frame, legacy_evidence=False)
        self.assertEqual(len(failed), 1)
        self.assertGreater(result["geometry_only"]["hatch_count"], 0)
        unresolved = [
            r
            for v in result["per_vessel_status"]
            for r in v["unresolved_candidates"]["samples"]
        ]
        self.assertTrue(
            any(r.get("error_code") == "DEGENERATE_FINAL_BOUNDARY" for r in unresolved)
        )

    def test_supported_outer_edges_cannot_resolve_ambiguous_inner_topology(self):
        from ship_perception.perception.recall import validate_and_select

        xyz, _ = vessel_points(1, spacing=0.12)
        frame = PointCloudFrame(xyz, None, "map", 0, 0, "", ("x", "y", "z"), len(xyz))
        baseline = run_perception(frame, legacy_evidence=False)

        def unresolved_split(*args):
            rows, audit = validate_and_select(*args)
            for row in rows:
                row["topology_status"] = "AMBIGUOUS_TOPOLOGY"
            return rows, audit

        with mock.patch(
            "ship_perception.perception.pipeline.validate_and_select",
            side_effect=unresolved_split,
        ):
            result = run_perception(frame, legacy_evidence=False)
        candidates = result["geometry_only"]["hatches"]
        self.assertGreater(len(candidates), 0)
        self.assertEqual(
            [(r["boundary_xy"], r["per_side_uncertainty"]) for r in candidates],
            [
                (r["boundary_xy"], r["per_side_uncertainty"])
                for r in baseline["geometry_only"]["hatches"]
            ],
        )
        self.assertEqual(result["geometry_only"]["hatch_count_status"], "AMBIGUOUS")
        self.assertEqual(
            result["pipeline_health"]["first_failure_stage"],
            "OPENING_TOPOLOGY_VALIDATION",
        )
        self.assertTrue(all(r["status"].startswith("UNRESOLVED") for r in candidates))
        self.assertTrue(all(r["reason"] == "HATCH_COUNT_AMBIGUOUS" for r in candidates))
        self.assertTrue(all(r["individual_hatch_count"] is None for r in candidates))
        self.assertTrue(
            all("HATCH_COUNT_AMBIGUOUS" in r["candidate_blockers"] for r in candidates)
        )
        self.assertTrue(
            all(r["geometry_role"] == "UNRESOLVED_ENCLOSURE_REVIEW" for r in candidates)
        )

    def test_real_b1_and_structural_provider_reuse_with_legacy_enabled(self):
        xyz, _ = vessel_points(1, spacing=0.12)
        frame = PointCloudFrame(xyz, None, "map", 0, 0, "", ("x", "y", "z"), len(xyz))
        with mock.patch(
            "ship_perception.perception.context.infer_scene_vessels",
            wraps=infer_scene_vessels,
        ) as b1, mock.patch(
            "ship_perception.r1_static.corner_topology_review.infer_scene_vessels",
            side_effect=AssertionError("B1 repeated by legacy consumer"),
        ):
            result = run_perception(frame, legacy_evidence=True)
        self.assertEqual(b1.call_count, 1)
        self.assertGreater(result["geometry_only"]["hatch_count"], 0)
        for vessel in result["per_vessel_status"]:
            cache = vessel.get("provider_cache")
            if cache:
                self.assertEqual(cache["raw3d_build_count"], 1)
                self.assertEqual(cache["bev_build_count"], 1)
                self.assertTrue(
                    all(n == 1 for n in cache["grid_build_counts"].values())
                )

    def test_intensity_scorer_is_keyed_by_each_immutable_hatch_axis(self):
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
        frame = PointCloudFrame(
            xyz,
            np.array([1.0, 2.0, 3.0]),
            "map",
            0,
            0,
            "",
            ("x", "y", "z", "intensity"),
            3,
        )
        yaw = 0.3
        rotated = [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]
        identity = np.eye(2).tolist()
        hatches = [
            dict(
                vessel_hypothesis_id="V0",
                vessel_local_axes=a,
                boundary_xy=[[0, 0], [1, 0], [1, 1], [0, 1]],
            )
            for a in (rotated, identity, rotated)
        ]
        payload = dict(
            geometry_only=dict(hatches=hatches),
            intensity_status=dict(status="AVAILABLE"),
        )
        with mock.patch(
            "ship_perception.perception.output.IntensityEvidence"
        ) as scorer:
            scorer.return_value.edge.return_value = dict(
                intensity_edge_score=0.5, intensity_used=True
            )
            output = auxiliary_review(payload, frame, {"V0": np.arange(3)})
        self.assertEqual(scorer.call_count, 2)
        np.testing.assert_array_equal(scorer.call_args_list[0].args[2], rotated)
        np.testing.assert_array_equal(scorer.call_args_list[1].args[2], identity)
        self.assertEqual(
            [h["boundary_xy"] for h in output["hatches"]],
            [h["boundary_xy"] for h in hatches],
        )

    def test_unresolved_legacy_evidence_is_retained_without_promotion(self):
        xyz, _ = vessel_points(1, spacing=0.12)
        frame = PointCloudFrame(xyz, None, "map", 0, 0, "", ("x", "y", "z"), len(xyz))
        from ship_perception.perception.final_boundary import FinalBoundaryRefiner

        original = FinalBoundaryRefiner.refine

        def comparison(frame, context, *args):
            vessel = max(context.vessels, key=lambda v: len(v.point_indexes))
            uv = frame.xyz[frame.xyz[:, 2] < -1, :2] @ vessel.axes.T
            return [
                dict(
                    proposal_id="legacy:weak",
                    vessel_hypothesis_id=vessel.vessel_id,
                    axes=vessel.axes.tolist(),
                    bounds_axial=[*uv.min(0), *uv.max(0)],
                    proposal_source="SYNTHETIC_COMPARISON",
                    inherited_review_status="UNRESOLVED",
                )
            ], {}

        def refine(refiner, proposal, axes):
            result = original(refiner, proposal, axes)
            if proposal["proposal_id"] == "legacy:weak":
                for side in result["sides"].values():
                    side["raw_subvoxel_section_count"] = 0
            return result

        with mock.patch(
            "ship_perception.perception.pipeline._legacy_proposals",
            side_effect=comparison,
        ), mock.patch.object(FinalBoundaryRefiner, "refine", new=refine):
            result = run_perception(frame, legacy_evidence=True)
        unresolved = [
            r
            for v in result["per_vessel_status"]
            for r in v["unresolved_candidates"]["samples"]
            if r.get("proposal_id") == "legacy:weak"
        ]
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["status"], "UNRESOLVED")
        self.assertFalse(
            any(
                r["proposal_id"] == "legacy:weak"
                for r in result["geometry_only"]["hatches"]
            )
        )

    def test_legacy_raw_failure_isolated_from_sibling_and_local_result(self):
        xyz, _ = vessel_points(1, spacing=0.12)
        frame = PointCloudFrame(xyz, None, "map", 0, 0, "", ("x", "y", "z"), len(xyz))
        from ship_perception.perception.final_boundary import FinalBoundaryRefiner

        original = FinalBoundaryRefiner.refine
        visited = []

        def comparison(frame, context, *args):
            vessel = max(context.vessels, key=lambda v: len(v.point_indexes))
            uv = frame.xyz[frame.xyz[:, 2] < -1, :2] @ vessel.axes.T
            bounds = [*uv.min(0), *uv.max(0)]
            rows = [
                dict(
                    proposal_id="legacy:" + name,
                    vessel_hypothesis_id=vessel.vessel_id,
                    axes=vessel.axes.tolist(),
                    bounds_axial=bounds,
                    proposal_source="SYNTHETIC_COMPARISON",
                )
                for name in ("bad", "good")
            ]
            return rows, {}

        def refine(refiner, proposal, axes):
            visited.append(proposal["proposal_id"])
            if proposal["proposal_id"] == "legacy:bad":
                raise ValueError("SYNTHETIC_LOCAL_RAW_FAILURE")
            return original(refiner, proposal, axes)

        with mock.patch(
            "ship_perception.perception.pipeline._legacy_proposals",
            side_effect=comparison,
        ), mock.patch.object(FinalBoundaryRefiner, "refine", new=refine):
            result = run_perception(frame, legacy_evidence=True)
        self.assertIn("legacy:good", visited)
        self.assertGreater(result["geometry_only"]["hatch_count"], 0)
        failures = result["pipeline_health"]["isolated_failures"]
        self.assertTrue(
            any(
                f.get("hatch") == "legacy:bad"
                and f["stage"] == "COMPARISON_BOUNDARY_PRECISION"
                for f in failures
            )
        )


if __name__ == "__main__":
    unittest.main()
