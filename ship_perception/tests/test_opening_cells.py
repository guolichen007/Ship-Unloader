"""Ownership requires source support in addition to footprint containment."""

from types import SimpleNamespace
import unittest

import numpy as np

from ship_perception.perception.opening_cells import coarse_review, opening_cells


def region(key, bounds, members, *, vessel="A", ends=(), terminal_pair=False):
    return SimpleNamespace(
        region_id=key, vessel_id=vessel, conservative_bounds=bounds,
        source_proposal_ids=(key,),
        side_topology_support={
            "U0": SimpleNamespace(provider_ids=()),
            "U1": SimpleNamespace(provider_ids=ends),
        },
        identity_evidence=dict(
            member_height_nodes=[key],
            member_support_grid=dict(x0=0., y0=0., cell_m=1.),
            member_support_cells_rc=members,
            opposite_terminal_family_ids=list(ends) if terminal_pair else [],
        ),
    )


class OpeningCellsTests(unittest.TestCase):
    def test_source_subset_collapses_one_fragment_but_keeps_distinct_neighbors(self):
        first = region("a", (0, 0, 10, 8), [[2, 2], [3, 3]], ends=("a-end",), terminal_pair=True)
        fragment = region("slice", (1, 1, 8, 3), [[2, 2]])
        independent = region("parallel", (1, 5, 8, 7), [[5, 2]], ends=("p-end",), terminal_pair=True)
        orphan = region("orphan", (2, 4, 8, 5), [[4, 2]])
        other_ship = region("other", (1, 1, 8, 3), [[2, 2]], vessel="B")
        cells, links = opening_cells([first, fragment, independent, orphan, other_ship], {}, [], 1.)
        self.assertEqual([c["primary_region_id"] for c in cells], ["a", "parallel", "orphan", "other"])
        self.assertEqual(cells[0]["candidate_ids"], ["a", "slice"])
        self.assertEqual((links[0]["first"], links[0]["second"]), ("slice", "a"))

    def test_neighbor_chain_keeps_only_actual_neighbors_and_deduplicates_families(self):
        regions = [region(key, (u, 0, u + 8, 5), [[2, u + 1]],
                          ends=(key + "-end",), terminal_pair=True)
                   for key, u in (("a", 0), ("b", 10), ("c", 20))]
        relations = [dict(node_ids=[a,b], separator_evidence=[
            dict(family_id=a + b, qualified=True), dict(family_id=a + b, qualified=True),
        ]) for a,b in (("a","b"),("b","c"),("a","c"))]
        cells, links = opening_cells(regions, {"topology_relations": relations}, [], 1.)
        self.assertEqual(len(cells), 3)
        self.assertEqual([(r["first"],r["second"]) for r in links], [("a","b"),("b","c")])
        self.assertEqual(cells[1]["adjacent_separator_family_ids"], ["ab", "bc"])

    def test_review_estimate_is_orthogonal_clipped_to_separator_and_unmeasured(self):
        cell = dict(cell_id="OC:a", primary_region_id="a", vessel_id="A",
                    bounds_uv=[0, 0, 11, 5], separator_constraints=[dict(bound="upper_u", value=10)])
        measurement = SimpleNamespace(spatial=SimpleNamespace(query_indexes=lambda bounds: [1, 0]),
                                      raw_point_indexes=np.array([40, 10]))
        result = coarse_review(cell, np.array([[0., 1.], [-1., 0.]]), measurement, "INFEASIBLE")
        self.assertEqual(result["corners_xyz"], [[0., 0., None], [0., 10., None],
                                                [-5., 10., None], [-5., 0., None]])
        self.assertEqual(result["strict_rectangle_state"], "INFEASIBLE")
        self.assertFalse(result["steel_certified"])
        self.assertEqual(result["raw_occupancy_finite_index_count"], 2)


if __name__ == "__main__":
    unittest.main()
