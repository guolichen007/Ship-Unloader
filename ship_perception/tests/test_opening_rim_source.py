"""Physical rim association cannot create core points, hatch IDs or steel."""

import copy
from types import SimpleNamespace
import unittest
import numpy as np
from ship_perception.perception.measurement import VesselMeasurementContext
from ship_perception.perception.review_height import ReviewCornerHeight
from ship_perception.perception.opening_rim_source import (
    member_mask,
    support_fingerprint,
)
from ship_perception.r1_static.run import resolve_config


class OpeningRimSourceContracts(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()

    def fixture(self, ridge=False, quay=False, neighbour=False, visible_length=None):
        along = np.linspace(0, 12, 121)

        def patch(ys, z):
            x, y = np.meshgrid(along, ys)
            return np.c_[x.ravel(), y.ravel(), np.full(x.size, z)]

        floor = patch(np.arange(0.8, 7.91, 0.1), 0.0)
        if ridge:
            floor = floor[~((floor[:, 1] >= 2) & (floor[:, 1] <= 2.7))]
        roof = patch(np.linspace(2 if ridge else 0, 2.7 if ridge else 0.7, 15), 2.0)
        upper = patch(np.linspace(8, 8.7, 15), 2.0)
        if visible_length is not None:
            roof = roof[roof[:, 0] <= visible_length]
            upper = upper[upper[:, 0] <= visible_length]
        deck = patch(np.linspace(-0.8, -0.1, 12), 1.0)
        outer_deck = patch(np.linspace(8.8, 9.5, 12), 1.0)
        extra = []
        if quay:
            extra.extend(
                (
                    patch(np.linspace(-1.2, -0.5, 15), 4.0),
                    patch(np.linspace(-2, -1.3, 12), 1.0),
                )
            )
        if neighbour:
            extra.extend(
                (
                    patch(np.linspace(-1.1, -0.4, 15), 2.0),
                    patch(np.linspace(-1.9, -1.2, 12), 0.0),
                )
            )
        arrays = [floor, roof, upper, deck, outer_deck] + extra
        points = np.vstack(arrays)
        indexes = np.arange(len(points), dtype=np.int64)
        owned = indexes[: len(floor)][(floor[:, 1] >= 2) & (floor[:, 1] <= 6)]
        vessel = SimpleNamespace(vessel_id="VESSEL", point_indexes=owned)
        evidence = dict(
            member_support_grid=dict(x0=0.0, y0=0.0, cell_m=0.5),
            member_support_cells_rc=[[r, c] for r in range(1, 16) for c in range(25)],
        )
        region = SimpleNamespace(
            region_id="OPEN",
            vessel_id="VESSEL",
            identity_role="CONFIRMED_INDEPENDENT_OPENING",
            identity_evidence=evidence,
        )
        primary = SimpleNamespace(vessel=vessel, xyz=points[owned], config=self.config)
        local = SimpleNamespace(
            vessel=vessel,
            xyz=primary.xyz,
            context_xyz=points,
            config=self.config,
            measurement_context=SimpleNamespace(point_indexes=indexes),
            primary_cache=primary,
            opening_identities=[region],
        )
        measurement = VesselMeasurementContext(local)
        lower = 2.15 if ridge else 0.15
        result = dict(
            opening_identity="OPEN",
            boundary_uv=[[0, lower], [12, lower], [12, 8], [0, 8]],
            corners=[{} for _ in range(4)],
            raw_evidence_geometry=dict(sides={}),
            constrained_review_rectangle=dict(
                sides={
                    "V0": dict(normal_uv=[0.0, 1.0], final_rho=lower),
                    "V1": dict(normal_uv=[0.0, 1.0], final_rho=8.0),
                }
            ),
        )
        extra_start = sum(len(a) for a in arrays[:5])
        return measurement, result, region, np.arange(extra_start, len(points))

    def test_observed_wide_crest_with_attached_low_side_restores_estimated_z(self):
        measurement, result, region, _ = self.fixture()
        before = copy.deepcopy(result)
        core = measurement.local.primary_cache.vessel.point_indexes.copy()
        fingerprint = support_fingerprint(region)
        estimator = ReviewCornerHeight(measurement, self.config)
        heights = estimator.measure(result)
        self.assertTrue(all(z["z"] is not None for z in heights))
        for z in heights:
            self.assertAlmostEqual(z["z"], 2.0, places=6)
            self.assertEqual(z["z_state"], "ESTIMATED")
            self.assertFalse(z["steel_certified"])
            self.assertEqual(
                z["source_ownership_state"], "CURRENT_OPENING_ATTACHED_RIM_ESTIMATED"
            )
            self.assertFalse(z["source_ownership_evidence"]["core_promoted"])
        self.assertEqual(result, before)
        np.testing.assert_array_equal(
            core, measurement.local.primary_cache.vessel.point_indexes
        )
        self.assertEqual(fingerprint, support_fingerprint(region))
        self.assertFalse(estimator.audit["opening_rim_association"]["core_modified"])

    def test_internal_cargo_crest_does_not_gain_rim_ownership(self):
        measurement, result, _, _ = self.fixture(ridge=True)
        estimator = ReviewCornerHeight(measurement, self.config)
        heights = estimator.measure(result)
        self.assertIsNone(heights[0]["z"])
        self.assertIsNone(heights[1]["z"])
        family = next(
            v
            for v in estimator.audit["opening_rim_association"]["side_families"]
            if v["side"] == "V0"
        )
        self.assertEqual(family["seed_sections"], [])

    def test_quay_crest_cannot_cross_the_inner_ship_crest_to_borrow_cargo(self):
        measurement, result, _, foreign = self.fixture(quay=True)
        heights = ReviewCornerHeight(measurement, self.config).measure(result)
        for z in heights:
            self.assertNotEqual(z["z"], 4.0)
            self.assertFalse(set(z["source_point_indexes"]).intersection(foreign))

    def test_neighbour_roof_does_not_supply_original_points(self):
        measurement, result, _, foreign = self.fixture(neighbour=True)
        estimator = ReviewCornerHeight(measurement, self.config)
        heights = estimator.measure(result)
        self.assertTrue(any(z["z"] is not None for z in heights))
        for source in estimator.audit["role_qualified_sources"]:
            self.assertFalse(set(source["source_point_indexes"]).intersection(foreign))

    def test_one_local_structure_cannot_supply_four_independent_sections(self):
        measurement, result, _, _ = self.fixture(visible_length=2.8)
        heights = ReviewCornerHeight(measurement, self.config).measure(result)
        self.assertTrue(all(z["z"] is None for z in heights))

    def test_missing_frozen_members_or_unassociated_identity_are_not_promoted(self):
        for change in ("members", "identity", "vessel"):
            measurement, result, region, _ = self.fixture()
            if change == "members":
                region.identity_evidence = {}
            elif change == "identity":
                region.identity_role = "UNASSOCIATED_NOMINATION"
            else:
                region.vessel_id = "OTHER"
            heights = ReviewCornerHeight(measurement, self.config).measure(result)
            self.assertTrue(all(z["z"] is None for z in heights))

    def test_copied_or_modified_association_cannot_forge_point_ownership(self):
        measurement, result, _, _ = self.fixture()
        estimator = ReviewCornerHeight(measurement, self.config)
        estimator.measure(result)
        source = next(
            s
            for s in estimator.audit["role_qualified_sources"]
            if s.get("source_ownership_state")
        )
        owned = measurement.local.primary_cache.vessel.point_indexes
        self.assertTrue(estimator._eligible(source, owned, {"RIM_TOP"}))
        self.assertFalse(estimator._eligible(copy.deepcopy(source), owned, {"RIM_TOP"}))
        source["source_point_indexes"] = source["source_point_indexes"][:-1]
        self.assertFalse(estimator._eligible(source, owned, {"RIM_TOP"}))

    def test_membership_uses_grid_origin_and_keeps_negative_cells_distinct(self):
        region = SimpleNamespace(
            identity_evidence=dict(
                member_support_grid=dict(x0=100.0, y0=-50.0, cell_m=0.5),
                member_support_cells_rc=[[0, 0], [1, 1]],
            )
        )
        points = np.array(
            [[100.1, -49.9, 0], [100.6, -49.4, 0], [99.9, -50.1, 0], [100.1, -49.4, 0]]
        )
        self.assertEqual(
            member_mask(points, region).tolist(), [True, True, False, False]
        )

    def test_actual_low_attachment_witnesses_end_in_owned_member_support(self):
        measurement, result, region, _ = self.fixture()
        estimator = ReviewCornerHeight(measurement, self.config)
        estimator.measure(result)
        owned = set(measurement.local.primary_cache.vessel.point_indexes)
        members = member_mask(measurement.raw_uvz, region)
        for source in estimator.audit["role_qualified_sources"]:
            proof = source.get("source_ownership_evidence", {})
            if "root_low_attachment" not in proof:
                continue
            root = proof["root_upper_plane_uvz"]
            normal = np.array(root["normal"])
            for witness in proof["root_low_attachment"]["witness_paths"]:
                ids = witness["point_indexes"]
                self.assertIn(ids[-1], owned)
                self.assertTrue(members[ids[-1]])
                raw = measurement.raw_uvz[ids]
                self.assertLessEqual(
                    float(np.linalg.norm(np.diff(raw, axis=0), axis=1).max()),
                    self.config["boundary"]["max_support_gap_m"],
                )
                self.assertTrue(
                    (
                        -(raw @ normal + root["offset_m"]) / normal[2]
                        >= self.config["roi"]["opening_drop_m"]
                    ).all()
                )

    def test_rim_height_does_not_depend_on_map_zero_or_world_origin(self):
        measurement, result, region, _ = self.fixture()
        shift = np.array([100.0, -50.0, 7.0])
        points = measurement.raw_uvz + shift
        measurement.local.context_xyz = points
        measurement.local.primary_cache.xyz = points[
            measurement.local.primary_cache.vessel.point_indexes
        ]
        measurement.local.xyz = measurement.local.primary_cache.xyz
        region.identity_evidence["member_support_grid"].update(x0=100.0, y0=-50.0)
        result["boundary_uv"] = (np.array(result["boundary_uv"]) + shift[:2]).tolist()
        for side in result["constrained_review_rectangle"]["sides"].values():
            side["final_rho"] += shift[1]
        moved = VesselMeasurementContext(measurement.local)
        heights = ReviewCornerHeight(moved, self.config).measure(result)
        self.assertTrue(all(z["z"] is not None for z in heights))
        for z in heights:
            self.assertAlmostEqual(z["z"], 9.0, places=6)

    def test_incompatible_rim_sections_are_not_joined_by_fringe_plane_inliers(self):
        measurement, result, region, _ = self.fixture()
        points = measurement.raw_uvz.copy()
        lower = (points[:, 2] == 2.0) & (points[:, 1] < 1.0)
        section = np.minimum((points[lower, 0] / 3).astype(int), 3)
        points[lower, 2] = np.array([2.0, 2.3, 1.7, 2.2])[section]
        measurement.local.context_xyz = points
        measurement.local.primary_cache.xyz = points[
            measurement.local.primary_cache.vessel.point_indexes
        ]
        measurement.local.xyz = measurement.local.primary_cache.xyz
        moved = VesselMeasurementContext(measurement.local)
        estimator = ReviewCornerHeight(moved, self.config)
        heights = estimator.measure(result)
        persistent = [
            s
            for s in estimator.audit["role_qualified_sources"]
            if s.get("side_id") == "V0"
            and s.get("source_ownership_state")
            and s["tier"] == "SAME_SIDE_RIM_PLANE"
        ]
        self.assertEqual(persistent, [])
        self.assertAlmostEqual(heights[0]["z"], 2.0, places=6)
        self.assertAlmostEqual(heights[1]["z"], 2.2, places=6)

    def test_ambiguous_sibling_member_support_cannot_supply_attachment_seeds(self):
        measurement, result, region, _ = self.fixture()
        other = copy.deepcopy(region)
        other.region_id = "OTHER_OPENING"
        measurement.local.opening_identities.append(other)
        heights = ReviewCornerHeight(measurement, self.config).measure(result)
        self.assertTrue(all(z["z"] is None for z in heights))


if __name__ == "__main__":
    unittest.main()
