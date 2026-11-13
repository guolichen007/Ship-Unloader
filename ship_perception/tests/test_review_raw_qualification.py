"""Review does not turn an unsigned surface pair into an opening side."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ship_perception.perception.raw_boundary_probe import SignedRoleResolver
from ship_perception.r1_static.run import resolve_config


class ReviewRawQualification(unittest.TestCase):
    def resolver(self, record):
        config, _ = resolve_config()
        profile = SimpleNamespace(
            points=np.zeros((100, 3)),
            window=lambda *a: np.zeros((100, 3)),
            height=lambda *a: {"transition": False},
        )
        measurement = SimpleNamespace(
            raw_uvz=np.zeros((100, 3)),
            local=SimpleNamespace(structural_providers=lambda: []),
            side_measurements=SimpleNamespace(profile=lambda *a: profile),
        )
        refiner = SimpleNamespace(
            config=config,
            budget=None,
            _raw_section=lambda *a: copy.deepcopy(record),
        )
        return SignedRoleResolver(measurement, refiner)

    def test_unsigned_pair_is_not_review_measurement(self):
        resolver = self.resolver(dict(status="INSUFFICIENT_RAW_SUPPORT"))
        with patch(
            "ship_perception.r1_static.rectangle_refinement._raw_edge_rho",
            return_value=(0.0, 100),
        ):
            result = resolver.side(0, 0.0, (0.0, 16.0), 1, [], measurement_only=True)
        self.assertNotEqual(result["status"], "STRONG_UNIQUE_EDGE")
        self.assertEqual(result["sections"], [])

    def test_signed_raw_survives_absent_separate_height_provider(self):
        record = dict(
            status="UNIQUE_SIGNED_RAW_STEP",
            continuous=[0.0, 0.02, 100, True],
            height=dict(transition=True),
        )
        resolver = self.resolver(record)
        with patch(
            "ship_perception.r1_static.rectangle_refinement._raw_edge_rho",
            return_value=(None, 0),
        ):
            result = resolver.side(0, 0.0, (0.0, 16.0), 1, [], measurement_only=True)
        self.assertEqual(result["status"], "STRONG_UNIQUE_EDGE")
        self.assertTrue(
            all(
                r["raw_self_qualification"] == "SIGNED_SURFACE_PAIR"
                for r in result["sections"]
            )
        )


if __name__ == "__main__":
    unittest.main()
