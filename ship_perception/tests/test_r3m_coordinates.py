import copy
import unittest


from ship_perception.perception.side_evidence import bounded_side_modes
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG
from ship_perception.r1_static.run import resolve_config


class ReviewCoordinateContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json

        cls.config, _ = resolve_config()
        cls.research = json.loads(RESEARCH_CONFIG.read_bytes())

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
