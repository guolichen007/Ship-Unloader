"""Physical counterexamples for the proposed local covariance qualifier."""

import unittest
import numpy as np

from ship_perception.r1_static.hatch_proposal import _raw3d_provider
from ship_perception.r1_static.run import resolve_config


class RawSurfaceNormalsContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()
        cls.vessel = dict(
            vessel_hypothesis_id="synthetic",
            local_axes=np.eye(2).tolist(),
            source_support_ids=["synthetic_raw"],
        )

    def test_smooth_curved_cargo_is_never_a_verified_vertical_face(self):
        u, v = np.meshgrid(np.arange(0, 12, 0.05), np.arange(0, 20, 0.05))
        points = np.column_stack(
            (u.ravel(), v.ravel(), (6 * np.exp(-(((v - 10) / 3) ** 2))).ravel())
        )
        providers = _raw3d_provider(points, self.vessel, self.config)
        self.assertTrue(providers)
        self.assertFalse(
            any(
                "RAW3D_VERIFIED_VERTICAL_FACE" in p["evidence_types"] for p in providers
            )
        )

    def test_true_vertical_wall_keeps_qualification_under_order_and_translation(self):
        u, v = np.meshgrid(np.arange(0, 12, 0.1), np.arange(0, 20, 0.1))
        floor = np.column_stack(
            (u.ravel(), v.ravel(), np.where(u < 6, -4.0, 0.0).ravel())
        )
        along, z = np.meshgrid(np.arange(0, 20, 0.1), np.arange(-4, 0, 0.1))
        wall = np.column_stack((np.full(along.size, 6.03), along.ravel(), z.ravel()))
        points = np.vstack((floor, wall))
        baseline = None
        for variant, offset in (
            (points, np.zeros(3)),
            (points[::-1], np.array([1e6, -2e6, 3e5])),
        ):
            providers = _raw3d_provider(variant + offset, self.vessel, self.config)
            faces = [
                p
                for p in providers
                if "RAW3D_VERIFIED_VERTICAL_FACE" in p["evidence_types"]
            ]
            self.assertTrue(faces)
            for p in faces:
                validation = p["source_geometry"]["vertical_face_validation"]
                self.assertEqual(validation["method"], "ISOTROPIC_3D_LOCAL_COVARIANCE")
                self.assertEqual(validation["physical_role"], "UNKNOWN")
                self.assertGreaterEqual(
                    validation["vertical_normal_point_count"],
                    self.config["geometry"]["min_plane_points"],
                )
            signature = [
                (
                    p["rho_or_region"]["normal_axis"],
                    round(
                        p["rho_or_region"]["rho"]
                        - offset[p["rho_or_region"]["normal_axis"]],
                        5,
                    ),
                    p["source_geometry"]["vertical_face_validation"][
                        "vertical_normal_point_count"
                    ],
                )
                for p in faces
            ]
            if baseline is None:
                baseline = signature
            else:
                self.assertEqual(signature, baseline)

    def test_duplicate_points_cannot_manufacture_a_plane(self):
        from ship_perception.r1_static.raw_surface_normals import LocalSurfaceNormals

        points = np.repeat([[0.0, 0.0, 0.0]], 100, axis=0)
        result = LocalSurfaceNormals(points, self.config).face_evidence(
            0, 0.0, (-1.0, 1.0)
        )
        self.assertEqual(result["vertical_normal_point_count"], 0)
        self.assertEqual(result["status"], "UNVERIFIED_VERTICAL_FACE")


if __name__ == "__main__":
    unittest.main()
