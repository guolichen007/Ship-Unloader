"""Opening role, internal topology and per-opening ownership contracts."""

import copy
import subprocess
import sys
import unittest
import numpy as np
from ship_perception.perception.opening_identity_eligibility import (
    opening_role_eligibility,
    owned_enclosure_bounds,
)
from ship_perception.perception.side_evidence import (
    project_provider_observations,
    bounded_side_modes,
)
from ship_perception.perception.review_geometry import ReviewGeometryAssembler
from ship_perception.r1_static.run import resolve_config


class OpeningRoleContracts(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()

    def entry(self):
        return dict(
            nomination=dict(polarity="LOW", side_height_rises_m=[-0.5, 0.8, 0.6, -0.5]),
            current_interior_corner_evidence=dict(
                status="RAISED_OR_CARGO_INTERIOR", corner_support_count=2
            ),
        )

    def test_verified_end_geometry_does_not_replace_orthogonal_opening_role(self):
        e = self.entry()
        self.assertTrue(opening_role_eligibility(e, self.config)[0])
        e["nomination"]["side_height_rises_m"] = [0.8, 0.8, -1.0, -0.2]
        e["verified_end_faces"] = True
        self.assertFalse(opening_role_eligibility(e, self.config)[0])

    def test_ambiguous_interior_is_not_rescued_by_closed_walls(self):
        e = self.entry()
        e["current_interior_corner_evidence"]["status"] = "HEIGHT_AMBIGUOUS"
        self.assertFalse(opening_role_eligibility(e, self.config)[0])

    def test_raised_nomination_keeps_its_separate_enclosure_qualification(self):
        e = self.entry()
        e["nomination"].update(polarity="RAISED", side_height_rises_m=[-1.0] * 4)
        self.assertTrue(opening_role_eligibility(e, self.config)[0])

    def test_owned_extent_does_not_rewrite_observed_witnesses(self):
        e = dict(
            sides={
                n: dict(conservative_rho=x)
                for n, x in [("U0", 0), ("V0", 0), ("U1", 20), ("V1", 8)]
            },
            side_ownership_constraints={"U1": dict(upper_u=18.5)},
        )
        original = copy.deepcopy(e)
        self.assertEqual(owned_enclosure_bounds(e), [0, 0, 18.5, 8])
        self.assertEqual(e, original)

    def test_foreign_raw_mode_cannot_conflict_with_valid_owned_raw(self):
        from ship_perception.tests.test_review_geometry_v29 import (
            ReviewGeometryContracts,
        )

        fixture = ReviewGeometryContracts()
        fixture.setUp()
        r = fixture.region()
        r.identity_evidence["side_ownership_constraints"] = {"U1": dict(upper_u=20.0)}
        a = ReviewGeometryAssembler(self.config)
        result, _ = a.assemble(r, np.eye(2))
        foreign = fixture.mode(21.0)
        valid = fixture.mode(19.9)
        after = a.update(r, result, {"U1": [foreign, valid]}, np.eye(2))
        side = after["sides"]["U1"]
        self.assertEqual(side["selected_mode_id"], valid["mode_id"])
        self.assertNotEqual(side["final_state"], "MULTIMODE")
        self.assertTrue(
            any(m.get("ownership_rejected") for m in side["candidate_modes"])
        )
        self.assertLessEqual(max(np.array(after["boundary_uv"])[:, 0]), 20.0)

    def test_public_provider_helpers_are_equivalent_without_diagnostic_imports(self):
        from ship_perception.r1_static.r2h_b5_corner_forensics import (
            _observations,
            _side_modes,
        )

        providers = []
        for k, rho in enumerate(np.linspace(-1, 1, 19)):
            for source in ["BEV_RECTILINEAR_PROVIDER", "RAW3D_STRUCTURAL_PROVIDER"]:
                providers.append(
                    dict(
                        geometry_type="AXIAL_LINE_INTERVAL",
                        provider_type=source,
                        provider_id=source + str(k),
                        source_geometry=dict(endpoints_xy=[[rho, 0], [rho, 8]]),
                        rho_or_region=dict(normal_axis=0, rho=rho),
                        coverage=0.8,
                        source_support_ids=["S"],
                    )
                )
        data = dict(
            scene=dict(
                vessel_hypotheses=[
                    dict(
                        vessel_hypothesis_id="V",
                        classification_status="VESSEL_HYPOTHESIS",
                    )
                ]
            ),
            per_vessel=[dict(vessel_hypothesis_id="V", provider_evidence=providers)],
        )
        old = _observations(data, "V", np.eye(2), [-2, -1, 2, 9])
        new = project_provider_observations(data, "V", np.eye(2), [-2, -1, 2, 9])
        self.assertEqual(new, old)
        self.assertEqual(
            bounded_side_modes(new, 0, (0, 8), (-2, 2), self.config),
            _side_modes(old, 0, (0, 8), (-2, 2), self.config),
        )
        code = "import sys; import ship_perception.perception.local_evidence, ship_perception.perception.current_separator; assert 'ship_perception.r1_static.r2h_b5_corner_forensics' not in sys.modules"
        subprocess.run(
            [sys.executable, "-B", "-c", code], check=True, capture_output=True
        )

    def test_internal_partial_topology_stays_compound_and_attributes_children(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from ship_perception.perception import opening_identity_eligibility as module

        points = np.array([[9.5, y, 0.0] for y in [1.0, 3.0, 5.0, 7.0]])
        cache = SimpleNamespace(
            config=self.config,
            xyz=points,
            vessel=SimpleNamespace(
                vessel_id="V",
                point_indexes=np.arange(4),
                public=dict(
                    vessel_hypothesis_id="V", classification_status="VESSEL_HYPOTHESIS"
                ),
            ),
            structural_providers=lambda: [],
            grid=lambda *a: None,
        )
        parent = self.entry()
        parent.update(qualified=True)
        parent["nomination"].update(proposal_id="P", bounds_axial=[0.0, 0.0, 20.0, 8.0])
        parent["current_interior_corner_evidence"]["status"] = "LOWER_INTERIOR"
        left = self.entry()
        left.update(
            qualified=True,
            sides={
                n: dict(conservative_rho=x)
                for n, x in [("U0", 0.0), ("V0", 0.0), ("U1", 11.0), ("V1", 8.0)]
            },
        )
        left["nomination"].update(proposal_id="L", bounds_axial=[0.0, 0.0, 9.8, 8.0])
        right = copy.deepcopy(left)
        right["nomination"].update(proposal_id="R", bounds_axial=[9.2, 0.0, 20.0, 8.0])
        right["sides"]["U0"]["conservative_rho"] = 8.0
        right["sides"]["U1"]["conservative_rho"] = 20.0
        sample = dict(status="MEASURED", signed_contrast_m=1.0, transition=False)
        with patch.object(
            module,
            "_separator_split",
            return_value=([[0.0, 0.0, 9.0, 8.0], [10.0, 0.0, 20.0, 8.0]], [9.0, 10.0]),
        ), patch.object(module, "robust_height_support", return_value=sample):
            module.compile_opening_roles(cache, [parent, left, right])
        self.assertFalse(parent["qualified"])
        self.assertEqual(parent["identity_role_hint"], "COMPOUND_SEARCH_GROUP")
        self.assertFalse(
            parent["internal_partition_review"]["measured_cross_deck_certified"]
        )
        self.assertTrue(left["qualified"] and right["qualified"])
        self.assertEqual(owned_enclosure_bounds(left)[2], 9.0)
        self.assertEqual(owned_enclosure_bounds(right)[0], 10.0)
        self.assertEqual(left["sides"]["U1"]["conservative_rho"], 11.0)

    def test_unobserved_foreign_corner_does_not_retract_other_raw_side(self):
        from ship_perception.tests.test_review_geometry_v29 import (
            ReviewGeometryContracts,
        )

        fixture = ReviewGeometryContracts()
        fixture.setUp()
        r = fixture.region()
        r.identity_evidence["side_ownership_constraints"] = {"U1": dict(upper_u=20.0)}
        a = ReviewGeometryAssembler(self.config)
        result, _ = a.assemble(r, np.eye(2))
        u = fixture.mode(20.0)
        length = np.linalg.norm([1.0, 0.01])
        u.update(
            normal_uv=(np.array([1.0, 0.01]) / length).tolist(),
            rho=20.0 / length,
            support_span=[-0.3, 8.0],
        )
        v = fixture.mode(-0.5)
        v.update(mode_id="RAWV0", normal_uv=[0.0, 1.0], support_span=[-20.0, 0.0])
        after = a.update(r, result, {"U1": [u], "V0": [v]}, np.eye(2))
        self.assertEqual(after["sides"]["V0"]["final_source"], "MEASURED_RAW")
        self.assertEqual(
            after["boundary_measurements"]["U1"]["final_source"], "MEASURED_RAW"
        )
        self.assertEqual(
            after["sides"]["U1"]["guard_result"],
            "CURRENT_SEPARATOR_CORNER_OWNERSHIP_CONFLICT",
        )
        self.assertLessEqual(max(np.array(after["boundary_uv"])[:, 0]), 20.0)
        self.assertEqual(after["geometry_status"], "COMPLETE_REVIEW_GEOMETRY")


if __name__ == "__main__":
    unittest.main()
