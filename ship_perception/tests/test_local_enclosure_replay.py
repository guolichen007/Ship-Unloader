"""Topology mechanisms; no scene names, reference coordinates or counts as input."""

import copy
from types import SimpleNamespace
import unittest

import numpy as np

from ship_perception.perception.local_enclosure_replay import (
    enclosure_admission,
    resolve_local_identities,
)
from ship_perception.perception.region_model import OpeningRegion, TopologySupport
from ship_perception.r1_static.run import resolve_config


def vector(box):
    sides = {}
    for name, pos in (("U0", 0), ("U1", 2), ("V0", 1), ("V1", 3)):
        sides[name] = dict(
            selected_rho=box[pos],
            conservative_rho=box[pos],
            selected_band_rhos=[box[pos]],
            observed_intervals=[[0, 20]],
            provider_ids=["RAW:" + name, "BEV:" + name],
            verified_face_provider_ids=["FACE:" + name],
            raw3d_coverage=0.8,
            bev_coverage=0.8,
        )
    return sides


def identity(name, box):
    r = OpeningRegion(
        name,
        "V",
        tuple(box),
        tuple(np.r_[np.asarray(box[:2]) - 4, np.asarray(box[2:]) + 4]),
        (TopologySupport("S4_OPENING_TOPOLOGY", "SUPPORTING"),),
        component_bounds=(tuple(box),),
    )
    r.identity_role = "CONFIRMED_CONNECTED_OPENING"
    for n in ("U0", "U1", "V0", "V1"):
        r.side_topology_support[n] = TopologySupport(
            "STRUCTURAL_LINE_FAMILY", "SUPPORTING"
        )
    return r


def evidence(name, box):
    return dict(
        qualified=True,
        nomination=dict(
            proposal_id=name, bounds_axial=list(box), occupied_cells=100, polarity="LOW"
        ),
        sides=vector(box),
        member_support_cells_rc=[[0, 0]],
        member_support_grid=dict(x0=box[0], y0=box[1], cell_m=0.5),
    )


class LocalEnclosureContracts(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()
        self.cache = SimpleNamespace(
            config=self.config,
            vessel=SimpleNamespace(vessel_id="V"),
            side_candidates=lambda box: {n: [] for n in ("U0", "U1", "V0", "V1")},
        )

    def test_weak_s4_strong_raw_bev_without_b2_lineage_has_one_review_identity(self):
        e = evidence("height-only-nomination", [0, 0, 20, 8])
        result = resolve_local_identities([], [e], self.cache, None)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].identity_role.startswith("CONFIRMED"))
        self.assertFalse(any("B2" in s.source_type for s in result[0].topology_support))

    def test_structural_child_without_its_own_partition_remains_parent_evidence(self):
        parent = identity("parent", [0, 0, 20, 8])
        child = evidence("child", [4, 2, 8, 5])
        result = resolve_local_identities([parent], [child], self.cache, None)
        self.assertEqual([r.region_id for r in result], ["parent"])

    def test_compound_nomination_cannot_merge_two_separator_owned_identities(self):
        first = identity("first", [0, 0, 12, 8])
        second = identity("second", [14, 0, 26, 8])
        result = resolve_local_identities(
            [first, second], [evidence("compound", [0, 0, 26, 8])], self.cache, None
        )
        self.assertEqual([r.region_id for r in result], ["first", "second"])

    def test_three_independent_basins_stay_independent(self):
        regions = [identity(str(i), [i * 24, 0, i * 24 + 20, 8]) for i in range(3)]
        result = resolve_local_identities(regions, [], self.cache, None)
        self.assertEqual([r.region_id for r in result], ["0", "1", "2"])

    def test_one_face_reused_as_two_ends_is_not_an_independent_raised_opening(self):
        sides = vector([0, 0, 20, 8])
        sides["U1"]["verified_face_provider_ids"] = sides["U0"][
            "verified_face_provider_ids"
        ]
        ok, reason = enclosure_admission(dict(polarity="RAISED"), sides, self.config)
        self.assertFalse(ok)
        self.assertEqual(reason, "RAISED_STRUCTURE_WITHOUT_INDEPENDENT_END_ENCLOSURE")

    def test_terminal_only_water_scope_is_downgraded_before_main_output(self):
        r = identity("unbounded-terminal-scope", [0, 0, 20, 20])
        r.identity_evidence["opposite_terminal_family_ids"] = ["F0", "F1"]
        self.cache.s4_topology_bridge = dict(
            side_families=[
                dict(family_id="F" + str(i), support_span=[0, 8], direction_uv=[0, 1])
                for i in range(2)
            ]
        )
        for n in ("V0", "V1"):
            r.side_topology_support[n] = TopologySupport(
                "STRUCTURAL_LINE_FAMILY", "UNOBSERVED"
            )
        result = resolve_local_identities([r], [], self.cache, None)
        self.assertEqual(result[0].identity_role, "WEAK_UNASSIGNED_REGION")
        self.assertEqual(
            result[0].identity_evidence["structural_child_role"],
            "UNRESOLVED_CHILD_STRUCTURE",
        )

    def test_missing_measurements_without_scope_contradiction_do_not_remove_identity(
        self,
    ):
        r = identity("partial-longitudinal-measurement", [0, 0, 20, 8])
        r.identity_evidence["opposite_terminal_family_ids"] = ["F0", "F1"]
        self.cache.s4_topology_bridge = dict(
            side_families=[
                dict(family_id="F" + str(i), support_span=[0, 8], direction_uv=[0, 1])
                for i in range(2)
            ]
        )
        for n in ("V0", "V1"):
            r.side_topology_support[n] = TopologySupport(
                "STRUCTURAL_LINE_FAMILY", "UNOBSERVED"
            )
        self.assertTrue(
            resolve_local_identities([r], [], self.cache, None)[
                0
            ].identity_role.startswith("CONFIRMED")
        )

    def test_corner_contrast_does_not_invent_opposite_end_opening_role(self):
        sides = vector([0, 0, 20, 8])
        sides["U0"]["verified_face_provider_ids"] = []
        ok, _ = enclosure_admission(
            dict(
                polarity="LOW",
                side_height_rises_m=[-0.5, 2.0, -1.0, 0.4],
                corner_support_count=4,
            ),
            sides,
            self.config,
        )
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
