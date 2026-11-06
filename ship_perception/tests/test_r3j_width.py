import copy
import json
import unittest
import numpy as np

from ship_perception.r1_static.run import resolve_config
from ship_perception.perception.width_consistency import (
    width_consistency,
    side_line,
    line_separation,
)


def cell(key, lo, hi, low, high, vessel="A", observed=False):
    p = np.array([[lo, low], [hi, low], [hi, high], [lo, high]], float)
    evidence = {}
    if observed:
        for side in ("V0", "V1"):
            line = side_line(p, side)
            n = np.array(line["normal_uv"])
            span = (np.array(line["points_uv"]) @ np.array([-n[1], n[0]])).tolist()
            mode = dict(
                qualified_for_shadow=True,
                source_point_indexes=[key + ":raw"],
                normal_uv=n.tolist(),
                rho=line["rho"],
                support_span=sorted(span),
                uncertainty_m=0.05,
            )
            evidence[side] = dict(side_candidates=[mode])
    return dict(cell_id=key, vessel_id=vessel, local_rim_evidence=evidence), p.tolist()


class WidthCorridorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = resolve_config()[0]

    def run_model(self, items):
        cells = [x[0] for x in items]
        polygons = {x[0]["cell_id"]: x[1] for x in items}
        before = copy.deepcopy((cells, polygons))
        result = width_consistency(cells, polygons, self.cfg)
        self.assertEqual((cells, polygons), before)
        json.dumps(result[0], allow_nan=False)
        return result

    def test_single_opening_skips_cross_cell(self):
        audit, requests = self.run_model([cell("one", 0, 20, 0, 14)])
        self.assertEqual(audit["A"]["status"], "SINGLE_OPENING_CROSS_CELL_SKIPPED")
        self.assertEqual(requests, {"one": []})

    def test_two_widths_only_nominate_both_sides_for_own_raw(self):
        audit, requests = self.run_model(
            [cell("one", 0, 20, 0, 14), cell("two", 23, 43, 4, 14)]
        )
        self.assertEqual(requests, {"one": ["V0"], "two": ["V0"]})
        self.assertIsNone(audit["A"]["common_width_m"])
        self.assertFalse(audit["A"]["two_width_vote_allowed"])
        self.assertFalse(audit["A"]["geometry_changed"])

    def test_observed_real_unequal_widths_are_not_forced_equal(self):
        audit, requests = self.run_model(
            [
                cell("one", 0, 20, 0, 14, observed=True),
                cell("two", 23, 43, 3, 14, observed=True),
                cell("three", 46, 66, 4, 14, observed=True),
            ]
        )
        self.assertTrue(all(not value for value in requests.values()))
        self.assertIsNone(audit["A"]["common_width_m"])
        self.assertEqual(len(audit["A"]["independently_supported_openings"]), 3)

    def test_three_observed_widths_can_describe_a_family(self):
        audit, _ = self.run_model(
            [cell(str(i), i * 23, i * 23 + 20, 2, 14, observed=True) for i in range(3)]
        )
        self.assertAlmostEqual(audit["A"]["common_width_m"], 12.0)
        self.assertFalse(audit["A"]["corridors"]["V0"]["measurement_transfer_allowed"])

    def test_other_vessel_never_supplies_width_prior(self):
        audit, requests = self.run_model(
            [cell("one", 0, 20, 0, 14), cell("two", 23, 43, 4, 14, vessel="B")]
        )
        self.assertEqual(requests, {"one": [], "two": []})
        self.assertTrue(
            all(
                a["status"] == "SINGLE_OPENING_CROSS_CELL_SKIPPED"
                for a in audit.values()
            )
        )

    def test_normal_projection_is_equivariant_to_input_frame(self):
        items = [cell("one", 0, 20, 0, 14), cell("two", 23, 43, 4, 14)]
        a = self.run_model(items)
        theta = 0.43
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        axes = rotation.T
        shift = np.array([-1200.0, 47.0])
        transformed = [
            (c, ((np.array(p) @ rotation.T + shift) @ axes.T).tolist())
            for c, p in items
        ]
        b = self.run_model(transformed)
        self.assertEqual(a[1], b[1])
        for key in ("one", "two"):
            self.assertAlmostEqual(
                a[0]["A"]["widths_m"][key], b[0]["A"]["widths_m"][key]
            )
        for side in ("V0", "V1"):
            self.assertAlmostEqual(
                a[0]["A"]["relations"][0]["side_distance_m"][side],
                b[0]["A"]["relations"][0]["side_distance_m"][side],
            )

    def test_identically_wrong_hull_edges_do_not_invent_an_inner_edge(self):
        audit, requests = self.run_model(
            [cell("one", 0, 20, 0, 14), cell("two", 23, 43, 0, 14)]
        )
        self.assertEqual(requests, {"one": [], "two": []})
        self.assertIsNone(audit["A"]["common_width_m"])
        self.assertFalse(audit["A"]["corridors"]["V0"]["observed"])


if __name__ == "__main__":
    unittest.main()
