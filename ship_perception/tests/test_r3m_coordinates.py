import copy
import unittest

import numpy as np

from ship_perception.perception.discovery_frame import conditioned_discovery
from ship_perception.perception.side_evidence import bounded_side_modes
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG
from ship_perception.r1_static.run import resolve_config


class ReviewCoordinateContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json

        cls.config, _ = resolve_config()
        cls.research = json.loads(RESEARCH_CONFIG.read_bytes())

    def test_raw_conditioner_commutes_with_rigid_transform_and_preserves_indexes(self):
        x, y = np.meshgrid(np.arange(0.0, 65.0, 0.5), np.arange(0.0, 16.0, 0.5))
        z = np.where((x > 8) & (x < 51) & (y > 3) & (y < 13), -3.0, 0.0)
        z[(x > 59) & (y < 2)] = 2.0
        raw = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        frozen = raw.copy()
        a, ap = conditioned_discovery(raw, self.config, self.research)
        angle = 0.417
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        shift = np.array([217.314, -63.771, 7.31])
        transformed = raw.copy()
        transformed[:, :2] = raw[:, :2] @ rotation.T + shift[:2]
        transformed[:, 2] += shift[2]
        b, bp = conditioned_discovery(transformed, self.config, self.research)
        self.assertTrue(np.array_equal(raw, frozen))
        self.assertEqual(a["vessel_hypothesis_count"], b["vessel_hypothesis_count"])
        self.assertGreater(a["vessel_hypothesis_count"], 0)
        for first, second in zip(a["vessel_hypotheses"], b["vessel_hypotheses"]):
            self.assertTrue(
                np.array_equal(
                    ap["vessel_point_indexes"][first["vessel_hypothesis_id"]],
                    bp["vessel_point_indexes"][second["vessel_hypothesis_id"]],
                )
            )
            np.testing.assert_allclose(
                np.asarray(second["local_axes"]) @ rotation,
                first["local_axes"],
                rtol=0,
                atol=1e-7,
            )
        frame = b["discovery_lattice_frame"]
        restored = (
            bp["membership_xyz"][:, :2] @ np.asarray(frame["axes_xy"])
            + frame["origin_xyz"][:2]
        )
        np.testing.assert_allclose(restored, transformed[:, :2], rtol=0, atol=1e-12)

    def test_exact_provider_positions_are_translation_covariant_without_rounding(self):
        rows = [
            dict(
                normal_axis=0,
                rho=1.1234567,
                along=[2.0, 18.0],
                provider_type=p,
                provider_id=p,
                source_support_ids=["S"],
                source_vessel_id="V",
                source_scope="SAME_VESSEL",
                coverage=1.0,
            )
            for p in ("RAW3D_STRUCTURAL_PROVIDER", "BEV_RECTILINEAR_PROVIDER")
        ]
        frozen = copy.deepcopy(rows)
        first = bounded_side_modes(
            rows, 0, (3.0, 16.0), (0.0, 4.0), self.config, exact_positions=True
        )
        translation = 0.007314159
        moved = [dict(r, rho=r["rho"] + translation) for r in rows]
        second = bounded_side_modes(
            moved,
            0,
            (3.0, 16.0),
            (translation, 4.0 + translation),
            self.config,
            exact_positions=True,
        )
        self.assertEqual(first[0]["rho"], rows[0]["rho"])
        self.assertAlmostEqual(
            second[0]["rho"] - translation, first[0]["rho"], places=14
        )
        self.assertEqual(first[0]["provider_ids"], second[0]["provider_ids"])
        self.assertEqual(rows, frozen)
        legacy = bounded_side_modes(rows, 0, (3.0, 16.0), (0.0, 4.0), self.config)
        self.assertEqual(legacy[0]["rho"], round(rows[0]["rho"], 2))
