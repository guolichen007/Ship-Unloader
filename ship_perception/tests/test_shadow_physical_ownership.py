import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from run_physical_opening_shadow import (
    retain_review_partials, select, structure_adjacency, weighted_membership,
)


class ShadowPhysicalOwnership(unittest.TestCase):
    def test_signed_adjacent_chain_does_not_require_a_clique(self):
        for count in (2, 3, 4):
            scopes = [
                dict(observation_id=str(i), vessel_id="V", cells=[[0, i]])
                for i in range(count)
            ]
            pairs = {frozenset((str(i - 1), str(i))) for i in range(1, count)}
            blocked, graph = structure_adjacency(scopes, pairs)
            self.assertFalse(blocked)
            self.assertEqual(len(graph), count - 1)
            self.assertTrue(all(edge["relation"] == "ADJACENT_DISTINCT" for edge in graph))

    def test_missing_separator_and_shared_interior_remain_unresolved(self):
        scopes = [
            dict(observation_id="a", vessel_id="V", cells=[[0, 0], [0, 1]]),
            dict(observation_id="b", vessel_id="V", cells=[[0, 1], [0, 2]]),
        ]
        blocked, graph = structure_adjacency(scopes, {frozenset(("a", "b"))})
        self.assertEqual(blocked, {"a", "b"})
        self.assertEqual(graph[0]["relation"], "COMPETING_INTERIOR")
        scopes[0]["cells"] = [[0, 0]]
        blocked, graph = structure_adjacency(scopes, set())
        self.assertEqual(blocked, {"a", "b"})
        self.assertEqual(graph[0]["relation"], "AMBIGUOUS")

    def test_canonical_competitors_survive_as_independent_review_records(self):
        scopes = []
        records = []
        for i in range(2):
            key = "S" + str(i)
            cells = [[0, i], [0, i + 1]]
            scopes.append(dict(observation_id=key, cells=cells, end_evidence=[[key]]))
            records.append(dict(
                hypothesis_id=str(i), role="PHYSICAL_OPENING_HYPOTHESIS",
                basin_witness_ids=[key], structure_scope_ids=[],
                structure_associations=[], member_cells=cells,
            ))
        result = select(dict(scopes=scopes, records=records, grounded_scope_ids=["S0", "S1"],
                             separator_evidence=[], separator_support_audit=[],
                             grid_contract=dict(canonical=dict(x0=0, y0=0, cell_m=1))))
        self.assertEqual(len(result["unresolved_openings"]), 2)
        self.assertTrue(all(r["review_state"] == "COMPETING_INTERIOR_REVIEW" for r in records))

    def test_weighted_alias_retains_mass_without_four_member_votes(self):
        grid = dict(x0=0., y0=0., cell_m=.5)
        for shift in (1e-9, 1.3483073502129628e-6, .24):
            shifted = dict(grid, x0=shift, y0=shift)
            for source, target in ((shifted, grid), (grid, shifted)):
                cells, links = weighted_membership([(1, 1)], source, target)
                self.assertEqual(cells, {(1, 1)})
                self.assertEqual(len(links), 4)
                self.assertAlmostEqual(sum(r["source_fraction"] for r in links), 1.)
                self.assertAlmostEqual(sum(r["target_fraction"] for r in links), 1.)

    def test_partial_raw_evidence_never_promotes_identity_or_unknown_sides(self):
        mode = dict(
            mode_id="M", source="MEASURED_RAW", source_point_indexes=[5, 6],
            independent_opening_role_gain=True, opening_assignment={},
            opening_assignment_state="ASSIGNED_CURRENT_OPENING",
        )
        candidate = dict(opening_identity="R", sides={
            "U0": {"candidate_modes": [mode]}, "U1": {"candidate_modes": []},
        })
        region = dict(region_id="R", vessel_id="V",
                      identity_evidence={"member_support_cells_rc": [[0, 1]]})
        bundle = dict(vessel_id="V", native_geometry=dict(
            hatches=[candidate], opening_regions=[region],
        ))
        review = dict(scopes=[], selected_structure_scope_ids=[], records=[], substructure_relations=[])
        output = retain_review_partials(review, {"hypotheses": []}, bundle)
        partial, = output["review_only_partials"]
        self.assertEqual(partial["status"], "REVIEW_ONLY_PARTIAL")
        self.assertFalse(partial["physical_identity_verified"])
        self.assertEqual(partial["sides"]["U1"]["status"], "UNKNOWN_UNOBSERVED")
        self.assertEqual(partial["corner_z_m"], [None] * 4)
        mode["opening_assignment_state"] = "UNASSIGNED_CURRENT_OPENING"
        self.assertEqual(
            retain_review_partials(review, {"hypotheses": []}, bundle)["review_only_partials"], []
        )
