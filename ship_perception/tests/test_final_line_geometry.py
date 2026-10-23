"""A fitted opening line must survive into corners and coverage uncertainty."""

import unittest

import numpy as np

from ship_perception.perception.final_boundary import (
    FinalBoundaryRefiner,
    guard_line_move,
)
from ship_perception.perception.line_geometry import intersect_final_sides
from ship_perception.r1_static.run import resolve_config


class FinalLineGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()
        angle = np.radians(0.4)
        rotation = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        u, v = np.meshgrid(np.arange(-3, 83, 0.1), np.arange(-3, 19, 0.1))
        z = np.where((u > 0) & (u < 80) & (v > 0) & (v < 16), -4.0, 0.0)
        cls.xyz = np.column_stack(
            (np.column_stack((u.ravel(), v.ravel())) @ rotation, z.ravel())
        )
        cls.truth = np.array([[0, 0], [80, 0], [80, 16], [0, 16]]) @ rotation
        cls.bounds = [*cls.truth.min(0), *cls.truth.max(0)]

    def result(self, corroborated):
        # Controlled provider evidence isolates final geometry from recall.
        proposal = dict(bounds_axial=self.bounds, vessel_hypothesis_id="synthetic")
        if corroborated:
            proposal["deck_ring"] = dict(
                sides={
                    name: dict(raw3d_support=True, bev_support=True)
                    for name in ("U0", "U1", "V0", "V1")
                }
            )
        return FinalBoundaryRefiner(self.xyz, [], self.config).refine(
            proposal, np.eye(2)
        )

    def test_supported_raw_line_slope_reaches_final_intersections(self):
        result = self.result(True)
        for side in result["sides"].values():
            self.assertGreaterEqual(
                side["support_section_count"],
                self.config["boundary"]["profile_min_sections"],
            )
        error = np.linalg.norm(np.asarray(result["boundary_xy"]) - self.truth, axis=1)
        self.assertLessEqual(
            np.percentile(error, 95), self.config["evaluation"]["corner_p95_m"]
        )

    def test_conservative_line_covers_discarded_endpoint_motion(self):
        result = self.result(False)
        error = np.linalg.norm(np.asarray(result["boundary_xy"]) - self.truth, axis=1)
        reported = np.array([c["uncertainty_m"] for c in result["corners"]])
        self.assertTrue(
            np.all(error <= reported + self.config["geometry"]["plane_inlier_m"]),
            (error.tolist(), reported.tolist()),
        )

    def test_midpoint_cannot_authorize_unsupported_endpoint_crop(self):
        kept = guard_line_move(0, 0, 0.01, 40, (0, 80), 1, 4, False, self.config)
        self.assertEqual(kept, (0, 0, "KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY", True))
        supported = guard_line_move(0, 0, 0.01, 40, (0, 80), 1, 4, True, self.config)
        self.assertEqual(supported, (0, 0.01, "RAW_GEOMETRY_REVIEW", False))

    def test_parallel_adjacent_lines_return_no_corner(self):
        sides = {
            name: dict(
                line=dict(slope=1.0, rho_at_reference_m=0.0, reference_along_m=0.0),
                boundary_uncertainty_m=0.05,
            )
            for name in ("U0", "U1", "V0", "V1")
        }
        with self.assertRaisesRegex(ValueError, "UNSTABLE_FINAL_LINE_INTERSECTION"):
            intersect_final_sides(sides, self.config)


if __name__ == "__main__":
    unittest.main()
