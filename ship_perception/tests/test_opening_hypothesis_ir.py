"""Lossless lifetime, physical persistence, separator and grid counterexamples."""

from dataclasses import FrozenInstanceError
import json
import unittest

import numpy as np

from ship_perception.perception.candidate_union import (
    CandidateProposal,
    OpeningObservation as Observation,
    OpeningHypothesisIR as IR,
)
from tools.run_opening_ir_shadow import (
    cell_overlap,
    recover_height_members,
    NativeObserver,
    source_inventory,
    restored_inventory,
)


def height(name, cells, polarity="LOW", level=0, vessel="v", **extra):
    features = dict(
        origin="LOCAL_HEIGHT_RECALL",
        grid_id="grid",
        polarity=polarity,
        level_m=level,
        bounds=[0, 0, 10, 10],
    )
    features.update(extra)
    return Observation.from_evidence(
        "HEIGHT_LEVEL_COMPONENT",
        vessel,
        name,
        dict(proposal_id=name, measured_strong_side_count=0),
        cells,
        **features,
    )


def node(name, cells):
    return Observation.from_evidence(
        "HEIGHT_LEVEL_COMPONENT",
        "v",
        name,
        {},
        cells,
        origin="S4_HIERARCHY_AFTER_PARTITION",
    )


def local(name, **extra):
    features = dict(
        height_source_id=name,
        complete_enclosure=True,
        enclosure_bounds=[-0.5, -0.5, 10.5, 10.5],
        interior_role="LOWER_INTERIOR",
    )
    features.update(extra)
    return Observation.from_evidence(
        "LOCAL_ENCLOSURE_WITNESS",
        "v",
        "LOCAL:" + name,
        dict(qualified=False, reason="COMPOUND_OR_CORROBORATION"),
        **features,
    )


def build(rows, proposals=()):
    return IR.compile(proposals, rows, min_support_cells=1, support_ratio=0.5)


class IRTest(unittest.TestCase):
    def test_independent_source_inventory_detects_a_missing_source_record(self):
        bundle = {
            k: []
            for k in (
                "height",
                "proposals",
                "providers",
                "s4_nodes_before",
                "s4_nodes_after",
                "height_regions",
                "s4_regions",
                "local_regions",
                "local_before_role",
                "local_after_role",
            )
        }
        bundle["height"] = [dict(raw="must remain")]
        bundle["s4"] = {
            k: []
            for k in (
                "raw_observations",
                "side_families",
                "topology_relations",
                "topology_vertices",
                "node_partition_audit",
            )
        }
        bundle["s4"]["structural_axis"] = {}
        expected = source_inventory(bundle)
        self.assertIn("/height/0", expected)
        self.assertNotEqual(expected, restored_inventory([]))

    def test_source_inventory_detects_payload_change_with_same_observation_id(self):
        row = dict(
            observation_id="same",
            evidence=dict(raw="original"),
            features=dict(
                source_record_refs=[dict(path="/source/0", payload_field=None)]
            ),
        )
        expected = restored_inventory([row])
        row["evidence"]["raw"] = "altered"
        self.assertNotEqual(expected, restored_inventory([row]))

    def test_observer_failure_preserves_original_call_arguments_and_return_value(self):
        calls = []

        class Native:
            def run(self, value):
                calls.append(value)
                return value

        def broken(*args):
            raise TypeError("observation only")

        value = []
        observer = NativeObserver()
        with observer.watch(Native, "run", before=broken, after=broken):
            self.assertIs(Native().run(value), value)
        self.assertEqual(calls, [value])
        self.assertEqual(len(observer.bundle["errors"]), 2)

    def test_three_nested_height_observations_are_one_persistent_interpretation(self):
        observations = [
            height("a", [(0, 0)], level=0),
            height("b", [(0, 0), (0, 1)], level=1),
            height("c", [(0, 0), (0, 1), (0, 2)], level=2),
        ]
        ir = build(observations)
        self.assertEqual(len(ir.hypotheses), 1)
        self.assertEqual(ir.hypotheses[0].height_component_lineage, ("a", "b", "c"))
        self.assertEqual(len(ir.observations), 3)
        self.assertEqual(build(list(reversed(observations))).audit(), ir.audit())

    def test_s4_fragment_cannot_replace_complete_basin(self):
        basin = height("whole", [(0, c) for c in range(8)])
        before = build([basin])
        fragment = Observation.from_evidence(
            "S4_REGION_FRAGMENT", "v", "fragment", {}, [(0, 0)]
        )
        after = build([basin, fragment])
        self.assertEqual(
            before.hypotheses[0].hypothesis_id, after.hypotheses[0].hypothesis_id
        )
        self.assertEqual(
            before.hypotheses[0].conservative_bounds,
            after.hypotheses[0].conservative_bounds,
        )
        self.assertEqual(
            before.hypotheses[0].member_support_cells,
            after.hypotheses[0].member_support_cells,
        )
        self.assertIn(fragment.observation_id, after.hypotheses[0].observation_ids)

    def test_compound_local_candidate_keeps_its_complete_interpretation(self):
        witness = local("whole")
        ir = build([height("whole", [(0, 0)]), witness])
        result = ir.hypotheses[0].audit()
        explanation = result["interior_evidence"]["local_enclosure_interpretations"][0]
        self.assertTrue(explanation["complete_enclosure"])
        self.assertEqual(explanation["bounds"], [-0.5, -0.5, 10.5, 10.5])
        self.assertIn(witness.observation_id, result["observation_ids"])

    def test_separator_preserves_parent_and_independent_children(self):
        a, b = height("left", [(0, 0)]), height("right", [(0, 2)])
        parent = height("parent", [(0, c) for c in range(3)])
        sep = Observation.from_evidence(
            "SEPARATOR_FACE",
            "v",
            "separator",
            {},
            qualified=True,
            node_ids=["nl", "nr"],
            association_source_ids=["nl", "nr"],
        )
        ir = build([a, b, parent, node("nl", [(0, 0)]), node("nr", [(0, 2)]), sep])
        by_source = {s: h for h in ir.hypotheses for s in h.height_component_lineage}
        self.assertNotEqual(
            by_source["left"].hypothesis_id, by_source["right"].hypothesis_id
        )
        self.assertEqual(by_source["parent"].status, "COMPOUND_INTERPRETATION")
        self.assertEqual(len(ir.hypotheses), 3)
        rel = json.loads(ir.relations_json)
        self.assertEqual(
            {r["interpretation"] for r in rel},
            {"COMPOUND_INTERPRETATION", "SPLIT_INTERPRETATION"},
        )
        self.assertTrue(all(not r["parent_deleted"] for r in rel))

    def test_shared_s4_family_has_multiple_owners_without_extra_opening(self):
        a, b = height("left", [(0, 0)]), height("right", [(0, 2)])
        family = Observation.from_evidence(
            "S4_FAMILY",
            "v",
            "family",
            {},
            axis_role="TRANSVERSE",
            association_source_ids=["nl", "nr"],
        )
        ir = build([a, b, node("nl", [(0, 0)]), node("nr", [(0, 2)]), family])
        self.assertEqual(len(ir.hypotheses), 2)
        shared = [
            r
            for r in json.loads(ir.relations_json)
            if r["relationship"] == "SHARED_BOUNDARY_EVIDENCE"
        ]
        self.assertEqual(len(shared), 1)
        self.assertEqual(len(shared[0]["hypothesis_ids"]), 2)
        self.assertFalse(shared[0]["exclusive_opening_ownership"])

    def test_different_polarities_or_vessels_do_not_canonicalize_together(self):
        cells = [(0, 0)]
        ir = build(
            [
                height("same", cells),
                height("raised", cells, polarity="RAISED"),
                height("same", cells, vessel="other"),
            ]
        )
        self.assertEqual(len(ir.hypotheses), 3)

    def test_zero_strong_sides_cannot_remove_a_hypothesis_or_proposal(self):
        proposal = CandidateProposal.from_evidence(
            "HEIGHT_BASIN",
            "v",
            dict(
                proposal_id="whole",
                bounds_axial=[0, 0, 10, 10],
                measured_strong_side_count=0,
            ),
        )
        ir = build([height("whole", [(0, 0)])], [proposal])
        self.assertEqual(ir.source_proposals, (proposal,))
        self.assertEqual(ir.hypotheses[0].source_proposal_ids, (proposal.proposal_id,))
        self.assertEqual(len(ir.hypotheses), 1)

    def test_unassigned_context_observation_is_fully_retained_not_a_fake_opening(self):
        context = Observation.from_evidence(
            "RAW_FACE", "v", "unassigned", {"raw": [1, 2, 3]}
        )
        ir = build([height("whole", [(0, 0)]), context])
        self.assertIn(context, ir.observations)
        row = next(
            r
            for r in json.loads(ir.observation_mapping_json)
            if r["observation_id"] == context.observation_id
        )
        self.assertEqual(row["hypothesis_ids"], [])
        self.assertEqual(len(ir.hypotheses), 1)
        with self.assertRaises(FrozenInstanceError):
            ir.hypotheses = ()

    def test_duplicate_observation_identity_fails_closed(self):
        obs = height("same", [(0, 0)])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_IR_OBSERVATION_ID"):
            build([obs, obs])

    def test_exact_cell_intersection_preserves_area_on_offset_b1_grids(self):
        source = dict(x0=0.25, y0=0.25, cell_m=0.5)
        target = dict(x0=0.0, y0=0.0, cell_m=0.5)
        result = cell_overlap([(0, 0)], source, target)
        self.assertEqual(
            result, [[0, 0, 0.25], [0, 1, 0.25], [1, 0, 0.25], [1, 1, 0.25]]
        )
        self.assertEqual(source, dict(x0=0.25, y0=0.25, cell_m=0.5))

    def test_height_provenance_recovers_original_connected_component(self):
        class Grid:
            occupancy = np.ones((2, 3), bool)
            median = np.array([[0.0, 0.0, 3.0], [3.0, 3.0, 3.0]])

            def centers(self):
                return np.meshgrid(np.arange(3) + 0.5, np.arange(2) + 0.5)

        rows = [
            dict(
                proposal_id="x",
                level_m=2.0,
                polarity="LOW",
                occupied_cells=2,
                bounds_axial=[0, 0, 2, 1],
            )
        ]
        self.assertEqual(
            recover_height_members(Grid(), rows, 1.0)[0]["cells"], [[0, 0], [0, 1]]
        )


if __name__ == "__main__":
    unittest.main()
