"""End-to-end near-deck cargo evidence must survive without certification."""

import unittest
import json
from pathlib import Path

import jsonschema
import numpy as np

from ship_perception.tests.test_hatch_metamorphic import vessel_points, run


def measured_coaming(load):
    xyz, boxes = vessel_points(1, spacing=0.12)
    xyz[xyz[:, 2] < 0, 2] = load
    a, b, c, d = boxes[0]
    walls = []
    for axis, rho, lo, hi in ((0, a, b, d), (0, c, b, d), (1, b, a, c), (1, d, a, c)):
        along, z = np.meshgrid(np.arange(lo, hi, 0.08), np.arange(0.0, 0.65, 0.08))
        wall = np.empty((along.size, 3))
        wall[:, axis], wall[:, 1 - axis], wall[:, 2] = rho, along.ravel(), z.ravel()
        walls.append(wall)
    return np.vstack((xyz, *walls)), np.array([[a, b], [c, b], [c, d], [a, d]])


class LoadedProposalRetention(unittest.TestCase):
    def test_visible_enclosure_without_low_basin_survives_as_unresolved_proposal(self):
        for load in (-0.1, 0.3):
            with self.subTest(load=load):
                points, target = measured_coaming(load)
                result = run(points)
                schema = json.loads(
                    (
                        Path(__file__).parents[1] / "config/hatch_once.schema.json"
                    ).read_text(encoding="utf-8")
                )
                jsonschema.validate(
                    result["per_vessel_status"],
                    schema["properties"]["per_vessel_status"],
                )
                proposals = [
                    p
                    for v in result["per_vessel_status"]
                    for p in v.get("unresolved_review_proposals", {}).get("samples", [])
                ]
                self.assertTrue(proposals)
                target_low, target_high = target.min(0), target.max(0)
                matches = []
                for p in proposals:
                    self.assertEqual(
                        p["geometry_role"], "UNREFINED_REVIEW_PROPOSAL_NOT_STEEL"
                    )
                    self.assertNotIn("hatch_id", p)
                    self.assertNotIn("best_estimate_boundary_xy", p)
                    poly = np.asarray(p["proposal_boundary_xy"])
                    low, high = poly.min(0), poly.max(0)
                    intersection = np.maximum(
                        0, np.minimum(target_high, high) - np.maximum(target_low, low)
                    ).prod()
                    union = (
                        np.prod(high - low)
                        + np.prod(target_high - target_low)
                        - intersection
                    )
                    if intersection / union >= 0.8:
                        matches.append(p)
                self.assertTrue(
                    matches,
                    "No surviving proposal covers the measured enclosure; fragments are insufficient.",
                )
                self.assertTrue(all(p["status"] == "UNRESOLVED" for p in matches))
                self.assertTrue(
                    all(
                        p["reason"]
                        in (
                            "UNRESOLVED_NO_OBSERVED_OPENING_INTERIOR",
                            "RAISED_INTERIOR_ROLE_AMBIGUOUS",
                        )
                        for p in matches
                    )
                )
                self.assertFalse(
                    any(
                        h["status"] == "HATCH_REVIEW_CANDIDATE"
                        for h in result["geometry_only"]["hatches"]
                    )
                )
                self.assertEqual(result["pipeline_health"]["isolated_failures"], [])


if __name__ == "__main__":
    unittest.main()
