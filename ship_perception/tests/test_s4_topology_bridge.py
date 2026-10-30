"""Topology requires shared observed structure; unknown is not no separator."""

import unittest
from types import SimpleNamespace
import numpy as np

from ship_perception.perception.s4_topology_bridge import (
    S4TopologyBridge,
    member_cell_assignment,
    topology_relation,
    topology_components,
)
from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.run import resolve_config


class S4TopologyContracts(unittest.TestCase):
    def test_incomplete_continuity_keeps_a_compound_search_identity(self):
        config, _ = resolve_config()
        x, y = np.meshgrid(np.arange(10.0), np.arange(5.0))
        bridge = S4TopologyBridge.__new__(S4TopologyBridge)
        bridge.config, bridge.vessel_id = config, "V"
        bridge.axes = list(np.eye(2))
        bridge.axis = dict(status="AXIS_ESTIMATED", axis_uncertainty_deg=1.0)
        bridge.grid = SimpleNamespace(cell_m=1.0, x0=-0.5, y0=-0.5)
        bridge.xy = np.stack((x, y), axis=-1)
        bridge.projected = bridge.xy
        bridge.cache = SimpleNamespace(vessel=SimpleNamespace(axes=np.eye(2)))
        masks = [x < 4, x > 5]
        bridge.nodes = {
            n: SimpleNamespace(
                mask=m, bbox_xy=tuple(np.r_[bridge.xy[m].min(0), bridge.xy[m].max(0)])
            )
            for n, m in zip(("A", "B"), masks)
        }
        bridge.families = [
            dict(
                family_id="F" + str(i),
                direction_axis=0,
                axis_role="LONGITUDINAL",
                rho=rho,
                rho_uncertainty=0.1,
                support_span=[0.0, 9.0],
            )
            for i, rho in enumerate((0.0, 4.0))
        ]
        bridge.vertices = [
            dict(node_id=n, vessel_id="V", side_family_ids=["F0", "F1"])
            for n in ("A", "B")
        ]
        bridge.relations = [
            topology_relation(
                *bridge.vertices, ["F0", "F1"], dict(observed_continuity=False), []
            )
        ]
        regions = bridge.compile_regions([])
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].relationship, "COMPOUND_OPENING_GROUP")
        self.assertEqual(regions[0].status, "REGION_AMBIGUOUS")
        self.assertEqual(regions[0].precision_status, "NOT_CERTIFIED")
        self.assertEqual(bridge.relations[0]["relation_decision"], "AMBIGUOUS")

    def test_membership_includes_connected_higher_layer_but_not_empty_bbox_overlap(
        self,
    ):
        x, y = np.meshgrid(np.arange(8.0), np.arange(4.0))
        xy = np.stack((x, y), axis=-1)
        low, high = x < 3, x < 7
        low[:, 3] = False
        high[:, 3] = False
        members = {"LOW": low, "HIGH": high}
        mapped = member_cell_assignment(
            xy, members, (0.0, 0.0, 7.0, 3.0), (5.0, 1.0, 6.0, 2.0)
        )
        self.assertEqual(mapped["member_cell_counts"], {"HIGH": 4})
        empty = member_cell_assignment(
            xy, members, (0.0, 0.0, 7.0, 3.0), (3.0, 1.0, 3.5, 2.0)
        )
        self.assertEqual(empty["member_cell_counts"], {})

    def test_family_identity_is_order_independent_and_observed_gaps_stay_missing(self):
        config, _ = resolve_config()
        segments = []
        for name, lo in (("A", 0.0), ("B", 5.0)):
            segments.append(
                dict(
                    node_id=name,
                    seed_id=name + ":seed",
                    segment_id=name + ":side",
                    profile_executed_after=True,
                    profile_executed_before=True,
                    profile_break_positions=[
                        [float(x), 0.0] for x in np.linspace(lo, lo + 2, 12)
                    ],
                    face_positions=[],
                    reference_role="BROAD_PERIMETER_SUPPORT",
                )
            )
        families = []
        for rows in (segments, list(reversed(segments))):
            bridge = S4TopologyBridge.__new__(S4TopologyBridge)
            bridge.config, bridge.vessel_id, bridge.axes = config, "V", list(np.eye(2))
            bridge.s4 = dict(segment_resolutions=rows, edges=[])
            bridge.families = []
            bridge._families()
            families.append(bridge.families)
        self.assertEqual(families[0], families[1])
        self.assertEqual(len(families[0]), 1)
        intervals = families[0][0]["observed_intervals"]
        np.testing.assert_allclose(intervals, [[0.0, 2.0], [5.0, 7.0]])

    def test_topology_merge_requires_shared_boundary_evidence(self):
        a = dict(node_id="A", vessel_id="V", bounds=[0, 0, 10, 5])
        b = dict(node_id="B", vessel_id="V", bounds=[0, 0, 10, 5])
        r = topology_relation(a, b, [], dict(observed_continuity=True), [])
        self.assertEqual(r["relation_decision"], "AMBIGUOUS")
        self.assertEqual(len(topology_components([a, b], [r])), 2)

    def test_unobserved_separator_without_boundary_continuity_is_ambiguous(self):
        a, b = [dict(node_id=n, vessel_id="V") for n in ("A", "B")]
        r = topology_relation(
            a,
            b,
            [],
            dict(observed_continuity=False),
            [dict(kind="CROSS_DECK", qualified=False, status="UNOBSERVED")],
        )
        self.assertEqual(r["relation_decision"], "AMBIGUOUS")
        r = topology_relation(a, b, ["shared"], dict(observed_continuity=False), [])
        self.assertEqual(r["relation_decision"], "AMBIGUOUS")

    def test_raw_cross_deck_splits_nodes_even_with_shared_longitudinal_families(self):
        config, _ = resolve_config()
        x, y = np.meshgrid(np.arange(0, 10.01, 0.1), np.arange(0, 4.01, 0.1))
        floor = ((x < 4) | (x > 6)) & (y > 0.5) & (y < 3.5)
        grid = R1HeightGrid.from_points(
            np.c_[x.ravel(), y.ravel(), np.where(floor, -2.0, 0.0).ravel()], 0.1
        )
        u, v = grid.centers()
        masks = [(u < 4) & (v > 0.5) & (v < 3.5), (u > 6) & (v > 0.5) & (v < 3.5)]
        providers = [
            dict(
                provider_id="face" + str(i),
                rho_or_region=dict(normal_axis=0, rho=rho),
                support_interval=(0.0, 4.0),
                evidence_types=["RAW3D_VERIFIED_VERTICAL_FACE"],
            )
            for i, rho in enumerate((4.1, 5.9))
        ]
        cache = SimpleNamespace(
            config=config,
            context_grid=lambda _: grid,
            structural_providers=lambda: providers,
        )
        bridge = S4TopologyBridge.__new__(S4TopologyBridge)
        bridge.config, bridge.cache, bridge.families = config, cache, []
        bridge.axes = list(np.eye(2))
        bridge.xy = np.stack((u, v), axis=-1)
        bridge.projected = bridge.xy
        bridge.nodes = {
            n: SimpleNamespace(
                mask=m, bbox_xy=tuple(np.r_[bridge.xy[m].min(0), bridge.xy[m].max(0)])
            )
            for n, m in zip(("A", "B"), masks)
        }
        vertices = [
            dict(
                node_id=n,
                vessel_id="V",
                centroid=bridge.xy[m].mean(0).tolist(),
                height_level_m=0.0,
            )
            for n, m in zip(("A", "B"), masks)
        ]
        split = bridge._split_evidence(*vertices)
        self.assertTrue(
            next(e for e in split if e["kind"] == "CROSS_DECK")["qualified"]
        )
        relation = topology_relation(
            *vertices, ["port", "starboard"], dict(observed_continuity=True), split
        )
        self.assertEqual(relation["relation_decision"], "SPLIT")

    def test_transitive_connect_cannot_bypass_a_measured_split(self):
        vertices = [dict(node_id=n, vessel_id="V") for n in "ABC"]
        relations = [
            dict(node_ids=[a, b], relation_decision=d)
            for a, b, d in [
                ("A", "B", "CONNECT"),
                ("B", "C", "CONNECT"),
                ("A", "C", "SPLIT"),
            ]
        ]
        groups = topology_components(vertices, relations)
        self.assertFalse(any({"A", "C"}.issubset(g) for g in groups))

    def test_different_vessel_never_connects(self):
        a, b = dict(node_id="A", vessel_id="V1"), dict(node_id="B", vessel_id="V2")
        r = topology_relation(a, b, ["shared"], dict(observed_continuity=True), [])
        self.assertEqual(r["relation_decision"], "SPLIT")


if __name__ == "__main__":
    unittest.main()
