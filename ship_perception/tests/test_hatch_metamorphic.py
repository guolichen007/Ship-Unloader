"""End-to-end transport-free counterexamples; annotations never enter runtime."""

import unittest

import numpy as np

from ship_perception.io.frame import PointCloudFrame
from ship_perception.perception.pipeline import run_perception


def vessel_points(count=3, spacing=0.12, phase=(0.0, 0.0), cargo=None):
    u, v = np.meshgrid(
        np.arange(phase[0], 80, spacing), np.arange(phase[1], 18, spacing)
    )
    z = np.zeros(u.shape)
    boxes = [(8.0, 3.0, 20.0, 14.0), (27.0, 3.0, 44.0, 14.0), (54.0, 3.0, 75.0, 14.0)][
        :count
    ]
    for index, (a, b, c, d) in enumerate(boxes):
        mask = (u > a) & (u < c) & (v > b) & (v < d)
        height = np.full(u.shape, -4.0)
        if cargo is not None:
            # Measured load variation, not a boundary hint passed to runtime.
            height = -4.0 + cargo * np.exp(-(((u - (a + c) / 2) / ((c - a) / 5)) ** 2))
        z[mask] = height[mask]
    return np.column_stack((u.ravel(), v.ravel(), z.ravel())), boxes


def run(points):
    frame = PointCloudFrame(
        points, None, "synthetic", 0, 0, "", ("x", "y", "z"), len(points)
    )
    return run_perception(frame, legacy_evidence=False)


def unresolved_count(result):
    return sum(
        v["unresolved_candidates"]["total_count"] for v in result["per_vessel_status"]
    )


class Metamorphic(unittest.TestCase):
    def check_health(self, result):
        self.assertEqual(result["pipeline_health"]["b1_execution_count"], 1)
        self.assertEqual(result["pipeline_health"]["isolated_failures"], [])
        for row in result["geometry_only"]["hatches"]:
            self.assertFalse(
                any(s["steel_edge_certified"] for s in row["sides"].values())
            )

    def test_single_two_three_unequal_openings(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                result = run(vessel_points(count)[0])
                self.check_health(result)
                self.assertEqual(result["geometry_only"]["hatch_count"], count)

    def test_rigid_transform_order_and_shuffle(self):
        xyz, _ = vessel_points()
        baseline = run(xyz)
        expected = baseline["geometry_only"]["hatches"]
        for yaw, translation, reorder in (
            (0.417, [13.27, -23.41, 19.0], "reverse"),
            (-1.183, [-40.61, 70.33, -5.0], "shuffle"),
            (2.26, [120.0, 30.0, 0.8], "original"),
        ):
            with self.subTest(yaw=yaw, order=reorder):
                rotation = np.array(
                    [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]
                )
                points = (
                    np.column_stack((xyz[:, :2] @ rotation, xyz[:, 2])) + translation
                )
                if reorder == "reverse":
                    points = points[::-1]
                elif reorder == "shuffle":
                    points = points[np.random.default_rng(17).permutation(len(points))]
                result = run(points)
                self.check_health(result)
                self.assertEqual(result["geometry_only"]["hatch_count"], 3)
                from scipy.optimize import linear_sum_assignment

                actual = result["geometry_only"]["hatches"]
                recovered = [
                    (np.asarray(h["boundary_xy"]) - np.asarray(translation)[:2])
                    @ rotation.T
                    for h in actual
                ]
                targets = [np.asarray(h["boundary_xy"]) for h in expected]
                costs = np.array(
                    [
                        [np.linalg.norm(a.mean(0) - b.mean(0)) for b in targets]
                        for a in recovered
                    ]
                )
                ai, bi = linear_sum_assignment(costs)
                for i, j in zip(ai, bi):
                    distances = np.linalg.norm(
                        recovered[i][:, None] - targets[j][None, :], axis=2
                    )
                    ci, cj = linear_sum_assignment(distances)
                    self.assertLessEqual(float(distances[ci, cj].max()), 0.15)
                    self.assertEqual(actual[i]["status"], expected[j]["status"])

    def test_sampling_phase_density_and_deterministic_dropout(self):
        for spacing, phase, dropout in (
            (0.10, (0.023, 0.017), 0.0),
            (0.12, (0.037, 0.041), 0.20),
            (0.15, (0.011, 0.029), 0.0),
        ):
            with self.subTest(spacing=spacing, phase=phase, dropout=dropout):
                xyz, _ = vessel_points(spacing=spacing, phase=phase)
                if dropout:
                    xyz = xyz[np.random.default_rng(23).random(len(xyz)) >= dropout]
                result = run(xyz)
                self.check_health(result)
                self.assertEqual(result["geometry_only"]["hatch_count"], 3)

    def test_cargo_ridge_and_near_deck_load_remain_reviewable(self):
        for cargo in (2.0, 3.65, 4.2):
            with self.subTest(cargo=cargo):
                result = run(vessel_points(cargo=cargo)[0])
                self.check_health(result)
                self.assertGreaterEqual(
                    result["geometry_only"]["hatch_count"] + unresolved_count(result), 3
                )

    def test_deep_hold_and_small_single_basin_are_retained(self):
        xyz, _ = vessel_points(1)
        xyz[xyz[:, 2] < 0, 2] = -16.0
        self.assertEqual(run(xyz)["geometry_only"]["hatch_count"], 1)
        xyz, _ = vessel_points(0)
        small = (xyz[:, 0] > 37) & (xyz[:, 0] < 41) & (xyz[:, 1] > 7) & (xyz[:, 1] < 10)
        xyz[small, 2] = -4.0
        result = run(xyz)
        self.check_health(result)
        self.assertGreater(
            result["geometry_only"]["hatch_count"] + unresolved_count(result), 0
        )

    def test_crop_and_sparse_side_do_not_invent_observed_steel(self):
        xyz, _ = vessel_points()
        for points in (
            xyz[xyz[:, 1] < 12],
            xyz[~((xyz[:, 1] > 13.5) & (xyz[:, 0] > 25))],
        ):
            result = run(points)
            self.check_health(result)
            self.assertTrue(
                unresolved_count(result) > 0
                or any(
                    h["status"] in ("PARTIAL_HATCH", "UNRESOLVED_VESSEL_REVIEW")
                    for h in result["geometry_only"]["hatches"]
                )
            )

    def test_multiple_vessels_keep_independent_ownership(self):
        first, _ = vessel_points(2)
        second, _ = vessel_points(3)
        result = run(np.vstack((first, second + [3.7, 40.3, -2.0])))
        self.check_health(result)
        self.assertEqual(result["geometry_only"]["hatch_count"], 5)
        self.assertGreaterEqual(
            len(
                {r["vessel_hypothesis_id"] for r in result["geometry_only"]["hatches"]}
            ),
            2,
        )

    def test_wharf_crane_water_and_outliers_never_create_exterior_hatches(self):
        boat, _ = vessel_points()
        x, y = np.meshgrid(np.arange(0, 80, 0.2), np.arange(-15, -4, 0.2))
        water = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -6.0)))
        quay = np.column_stack((x.ravel(), y.ravel() - 15.0, np.full(x.size, 2.0)))
        crane = np.array(
            [
                [u, -20.0 + v, 10.0 + z]
                for u in np.arange(30, 32, 0.2)
                for v in np.arange(1, 3, 0.2)
                for z in np.arange(0, 4, 0.5)
            ]
        )
        outliers = np.array([[100, 25, 20], [-20, -45, -10], [93, -55, 3.0]])
        result = run(np.vstack((boat, water, quay, crane, outliers)))
        self.check_health(result)
        for hatch in result["geometry_only"]["hatches"]:
            corners = np.asarray(hatch["boundary_xy"])
            self.assertTrue(np.all(corners[:, 1] > 0))
        self.assertEqual(result["geometry_only"]["hatch_count"], 3)


if __name__ == "__main__":
    unittest.main()
