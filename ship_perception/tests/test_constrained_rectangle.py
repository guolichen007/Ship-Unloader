"""The final review model is rectangular without changing its measurements."""

import copy
import math
from types import SimpleNamespace
import unittest
import numpy as np

from ship_perception.perception.constrained_rectangle import (
    ConstrainedRectangleSolver,
    constrain_review_geometry,
    rectangle_audit,
)
from ship_perception.perception.review_geometry import ReviewGeometryAssembler
from ship_perception.r1_static.run import resolve_config


class ConstrainedRectangleContracts(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()

    def fixture(self, deviations=(0.002, -0.001, 0.003, -0.004)):
        theta = 0.2
        eu = np.array([math.cos(theta), math.sin(theta)])
        ev = np.array([-eu[1], eu[0]])
        center = np.array([10.0, -5.0])
        length = 30.0
        width = 9.0
        polygon = np.array(
            [
                center - length / 2 * eu - width / 2 * ev,
                center + length / 2 * eu - width / 2 * ev,
                center + length / 2 * eu + width / 2 * ev,
                center - length / 2 * eu + width / 2 * ev,
            ]
        )
        region = SimpleNamespace(
            region_id="R",
            conservative_bounds=tuple(np.r_[polygon.min(0), polygon.max(0)]),
            identity_evidence={},
        )
        sides = {}
        for i, (name, axis, pair) in enumerate(
            [("U0", 0, [0, 3]), ("U1", 0, [1, 2]), ("V0", 1, [0, 1]), ("V1", 1, [3, 2])]
        ):
            angle = theta + axis * math.pi / 2 + deviations[i]
            n = np.array([math.cos(angle), math.sin(angle)])
            t = np.array([-n[1], n[0]])
            point = polygon[pair].mean(0)
            rho = float(point @ n)
            span = sorted((polygon[pair] @ t).tolist())
            mode = dict(
                mode_id=name,
                source="MEASURED_RAW",
                state="MEASURED_WEAK",
                normal_uv=n.tolist(),
                rho=rho,
                support_span=span,
                uncertainty_m=0.05,
                independent_opening_role_gain=True,
            )
            sides[name] = dict(
                normal_uv=n.tolist(),
                final_rho=rho,
                support_span=span,
                uncertainty_m=0.05,
                final_source="MEASURED_RAW",
                final_state="MEASURED_WEAK",
                selected_mode_id=name,
                candidate_modes=[mode],
                rejected_mode_ids=[],
            )
        return region, sides

    def test_five_parameter_model_is_strictly_rectangular(self):
        region, sides = self.fixture()
        before = copy.deepcopy(sides)
        result = ConstrainedRectangleSolver(self.config).solve(region, sides)
        self.assertTrue(result["solver"]["success"])
        self.assertTrue(result["rectangularity"]["rectangularity_gate_pass"])
        self.assertLess(result["rectangularity"]["max_right_angle_error_deg"], 1e-10)
        self.assertEqual(sides, before)
        self.assertGreater(result["length_m"], result["width_m"])
        self.assertEqual(result["solver"]["four_line_cartesian_enumeration_count"], 0)

    def test_convex_parallelogram_does_not_pass_rectangularity(self):
        self.assertFalse(
            rectangle_audit([[0, 0], [20, 0], [21, 8], [1, 8]])[
                "rectangularity_gate_pass"
            ]
        )
        self.assertTrue(
            rectangle_audit([[0, 0], [20, 0], [20, 8], [0, 8]])[
                "rectangularity_gate_pass"
            ]
        )

    def test_current_raw_conflict_is_preserved_and_not_certified(self):
        region, sides = self.fixture((0.08, -0.08, 0.03, -0.03))
        before = copy.deepcopy(sides)
        result = ConstrainedRectangleSolver(self.config).solve(region, sides)
        self.assertTrue(result["constraint_conflict_sides"])
        self.assertEqual(result["boundary_measurement_status"], "UNRESOLVED")
        self.assertFalse(result["steel_certified"])
        self.assertEqual(sides, before)
        for name in result["constraint_conflict_sides"]:
            self.assertEqual(
                result["sides"][name]["measured_line"],
                sides[name]["candidate_modes"][0],
            )

    def test_finalization_preserves_every_raw_geometry_and_measurement(self):
        from ship_perception.tests.test_review_geometry_v29 import (
            ReviewGeometryContracts,
        )

        fixture = ReviewGeometryContracts()
        fixture.setUp()
        region = fixture.region()
        original, _ = ReviewGeometryAssembler(self.config).assemble(region, np.eye(2))
        before = copy.deepcopy(original)
        result = constrain_review_geometry(region, original, self.config)
        self.assertEqual(original, before)
        self.assertEqual(
            result["boundary_measurements"], before["boundary_measurements"]
        )
        self.assertEqual(result["raw_evidence_geometry"]["sides"], before["sides"])
        self.assertEqual(result["opening_identity"], before["opening_identity"])
        self.assertTrue(result["geometry_consistency"]["valid"])
        self.assertTrue(
            all(
                not c["observed"] and not c["steel_certified"]
                for c in result["corners"]
            )
        )
        self.assertTrue(
            all(not s["measured_intervals"] for s in result["sides"].values())
        )

    def test_corner_export_keeps_both_geometry_layers(self):
        from ship_perception.tests.test_review_geometry_v29 import (
            ReviewGeometryContracts,
        )
        from ship_perception.tests.test_corner_export import payload
        from ship_perception.perception.coordinates import corner_coordinates

        fixture = ReviewGeometryContracts()
        fixture.setUp()
        region = fixture.region()
        original, _ = ReviewGeometryAssembler(self.config).assemble(region, np.eye(2))
        result = constrain_review_geometry(region, original, self.config)
        packet = payload()
        packet["schema"] = "ship_perception.hatch_once.3"
        packet["geometry_only"]["hatches"] = [result]
        exported = corner_coordinates(packet, "e" * 64)["hatches"][0]
        self.assertEqual(
            exported["raw_evidence_geometry"], result["raw_evidence_geometry"]
        )
        self.assertEqual(
            exported["constrained_review_rectangle"],
            result["constrained_review_rectangle"],
        )
        self.assertEqual(exported["boundary_xy_m"], result["boundary_xy"])
        self.assertTrue(all(c["z"] is None for c in exported["corners"]))

    def transform(self, region, sides, angle, translation):
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        changed = copy.deepcopy(sides)
        for side in changed.values():
            for line in [side] + side["candidate_modes"]:
                normal = np.asarray(line["normal_uv"])
                tangent = np.array([-normal[1], normal[0]])
                rho = line.get("rho", line.get("final_rho"))
                points = (
                    rho * normal + np.asarray(line["support_span"])[:, None] * tangent
                )
                points = points @ rotation.T + translation
                normal = rotation @ normal
                tangent = np.array([-normal[1], normal[0]])
                line["normal_uv"] = normal.tolist()
                line["support_span"] = sorted((points @ tangent).tolist())
                line["rho" if "rho" in line else "final_rho"] = float(
                    points[0] @ normal
                )
        b = np.asarray(region.conservative_bounds)
        corners = (
            np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
            @ rotation.T
            + translation
        )
        transformed = SimpleNamespace(
            region_id=region.region_id,
            conservative_bounds=tuple(np.r_[corners.min(0), corners.max(0)]),
            identity_evidence={},
        )
        return transformed, changed, rotation

    def test_joint_fit_commutes_with_rigid_transform(self):
        region, sides = self.fixture()
        solver = ConstrainedRectangleSolver(self.config)
        base = solver.solve(region, sides)
        for angle, translation in [
            (0.0, [1200.0, -2400.0]),
            (0.55, [0.0, 0.0]),
            (0.55, [1200.0, -2400.0]),
        ]:
            r, s, rotation = self.transform(
                region, sides, angle, np.asarray(translation)
            )
            result = solver.solve(r, s)
            recovered = (np.asarray(result["polygon_uv"]) - translation) @ rotation
            np.testing.assert_allclose(recovered, base["polygon_uv"], atol=1e-6, rtol=0)
            self.assertAlmostEqual(
                result["theta_rad"] - angle, base["theta_rad"], places=7
            )
            self.assertAlmostEqual(result["length_m"], base["length_m"], places=6)
            self.assertAlmostEqual(result["width_m"], base["width_m"], places=6)
            self.assertEqual(r.region_id, region.region_id)

    def test_normal_sign_flip_does_not_change_joint_solution(self):
        region, sides = self.fixture()
        solver = ConstrainedRectangleSolver(self.config)
        a = solver.solve(region, sides)
        changed = copy.deepcopy(sides)
        for side in changed.values():
            for line in [side] + side["candidate_modes"]:
                line["normal_uv"] = (-np.asarray(line["normal_uv"])).tolist()
                key = "rho" if "rho" in line else "final_rho"
                line[key] = -line[key]
                line["support_span"] = sorted(
                    (-np.asarray(line["support_span"])).tolist()
                )
        b = solver.solve(region, changed)
        np.testing.assert_allclose(a["polygon_uv"], b["polygon_uv"], atol=1e-6)

    def test_existing_separator_is_hard_constraint_without_identity_change(self):
        region, sides = self.fixture((0, 0, 0, 0))
        region.identity_evidence = {"member_height_nodes": ["LEFT"]}
        bridge = {
            "node_partition_audit": [
                dict(
                    children=[dict(node_id="LEFT", side=0)],
                    separator_faces=[23.0, 24.0],
                )
            ]
        }
        before = copy.deepcopy(bridge)
        result = ConstrainedRectangleSolver(self.config).solve(region, sides, bridge)
        self.assertLessEqual(max(p[0] for p in result["polygon_uv"]), 23.0 + 1e-8)
        self.assertEqual(bridge, before)
        self.assertTrue(result["constraint_conflict_sides"])

    def test_competing_raw_modes_are_not_averaged(self):
        region, sides = self.fixture((0, 0, 0, 0))
        side = sides["U0"]
        side["final_state"] = "MULTIMODE"
        side["selected_mode_id"] = None
        alternative = copy.deepcopy(side["candidate_modes"][0])
        alternative.update(mode_id="OTHER", rho=alternative["rho"] + 5)
        side["candidate_modes"].append(alternative)
        result = ConstrainedRectangleSolver(self.config).solve(region, sides)
        self.assertIsNone(result["sides"]["U0"]["measured_line"])
        self.assertAlmostEqual(
            result["sides"]["U0"]["final_rho"], side["final_rho"], places=7
        )

    def test_rejected_raw_mode_cannot_reenter_rectangle_objective(self):
        region, sides = self.fixture((0, 0, 0, 0))
        side = sides["U0"]
        side["rejected_mode_ids"] = ["U0"]
        side["candidate_modes"][0]["rho"] += 5
        result = ConstrainedRectangleSolver(self.config).solve(region, sides)
        self.assertIsNone(result["sides"]["U0"]["measured_line"])
        self.assertAlmostEqual(
            result["sides"]["U0"]["final_rho"], side["final_rho"], places=7
        )

    def test_unobserved_side_stations_remain_covered(self):
        from ship_perception.perception.constrained_rectangle import line_points

        region, sides = self.fixture()
        for side in sides.values():
            side.update(
                final_source="CONSERVATIVE_REGION_BOUND",
                final_state="CONSERVATIVE_PRIOR",
                selected_mode_id=None,
            )
        result = ConstrainedRectangleSolver(self.config).solve(region, sides)
        for name, sign in [("U0", -1), ("U1", 1), ("V0", -1), ("V1", 1)]:
            model = result["sides"][name]
            normal = np.asarray(model["normal_uv"])
            displacement = (model["final_rho"] - line_points(sides[name]) @ normal) / (
                normal @ sides[name]["normal_uv"]
            )
            self.assertGreaterEqual(min(sign * displacement), -1e-7)


if __name__ == "__main__":
    unittest.main()
