"""An exterior shell/profile line must not outrank the observed opening rim."""

import copy
import unittest
from unittest import mock

import numpy as np

from ship_perception.perception.final_boundary import FinalBoundaryRefiner
from ship_perception.r1_static.run import resolve_config


class InnerEdgeRole(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()
        u, v = np.meshgrid(np.arange(0, 36, 0.1), np.arange(0, 24, 0.1))
        z = np.where((u > 6) & (u < 30) & (v > 4) & (v < 16), -4.0, 0.0)
        z[v > 19] = -7.0
        self.xyz = np.column_stack((u.ravel(), v.ravel(), z.ravel()))
        self.proposal = dict(
            vessel_hypothesis_id="synthetic_vessel",
            bounds_axial=[6, 4, 30, 19],
            measured_side_candidates={
                "V1": [
                    dict(
                        rho=16.0,
                        primary_raw3d=1.0,
                        primary_bev=1.0,
                        provider_ids=["measured_raw", "measured_bev"],
                    )
                ]
            },
        )

    def refine(self, proposal=None, observations=()):
        return FinalBoundaryRefiner(self.xyz, observations, self.config).refine(
            proposal or self.proposal, np.eye(2)
        )

    def test_inner_rim_replaces_wrong_polarity_shell_with_corroboration(self):
        result = self.refine()
        side = result["sides"]["V1"]
        self.assertEqual(
            side["opening_role_reanchor"]["status"], "CORROBORATED_INNER_STEP"
        )
        self.assertAlmostEqual(side["rho_after_m"], 16.0, delta=0.15)
        self.assertGreater(side["side_inward_move_m"], 2.5)
        self.assertGreaterEqual(
            side["positive_height_section_count"],
            self.config["boundary"]["profile_min_sections"],
        )
        self.assertFalse(side["steel_edge_certified"])
        # Coverage is deliberately separate from the best physical line.
        self.assertGreater(
            np.asarray(result["review_coverage_boundary_xy"])[:, 1].max(), 18.9
        )

    def test_profile_on_shell_alone_does_not_claim_an_inner_rim(self):
        proposal = copy.deepcopy(self.proposal)
        proposal["measured_side_candidates"] = {}
        observations = [
            dict(
                xy=[float(u), 19.0],
                reference_role="PERIMETER_REFERENCE",
                observation_id="shell" + str(u),
            )
            for u in np.arange(6, 30, 0.1)
        ]
        result = self.refine(proposal, observations)
        side = result["sides"]["V1"]
        self.assertEqual(side["support_section_count"], 0)
        self.assertTrue(side["undercoverage_risk"])
        self.assertEqual(side["rho_after_m"], 19.0)

    def test_one_provider_cannot_authorize_large_inward_move(self):
        proposal = copy.deepcopy(self.proposal)
        proposal["measured_side_candidates"]["V1"][0]["primary_bev"] = 0.0
        result = self.refine(proposal)
        self.assertEqual(result["sides"]["V1"]["rho_after_m"], 19.0)
        self.assertNotEqual(
            result["sides"]["V1"]["opening_role_reanchor"]["status"],
            "CORROBORATED_INNER_STEP",
        )

    def test_competing_measured_positive_steps_remain_unresolved(self):
        proposal = copy.deepcopy(self.proposal)
        self.xyz[
            (self.xyz[:, 0] > 6)
            & (self.xyz[:, 0] < 30)
            & (self.xyz[:, 1] > 17)
            & (self.xyz[:, 1] < 18),
            2,
        ] = -4.0
        alternate = copy.deepcopy(proposal["measured_side_candidates"]["V1"][0])
        alternate["rho"] = 18.0
        proposal["measured_side_candidates"]["V1"].append(alternate)
        result = self.refine(proposal)
        self.assertEqual(
            result["sides"]["V1"]["opening_role_reanchor"]["status"],
            "MULTIMODE_BOUNDARY_CONFLICT",
        )
        self.assertEqual(result["sides"]["V1"]["rho_after_m"], 19.0)

    def test_identical_geometry_query_reuses_evidence_without_aliasing(self):
        refiner = FinalBoundaryRefiner(self.xyz, [], self.config)
        with mock.patch.object(
            refiner, "_refine_uncached", wraps=refiner._refine_uncached
        ) as measured:
            first = refiner.refine(self.proposal, np.eye(2))
            first["sides"]["V1"]["rho_after_m"] = 100.0
            second = refiner.refine(
                dict(self.proposal, proposal_id="another_path"), np.eye(2)
            )
            self.assertEqual(measured.call_count, 1)
        self.assertAlmostEqual(second["sides"]["V1"]["rho_after_m"], 16.0, delta=0.15)


if __name__ == "__main__":
    unittest.main()
