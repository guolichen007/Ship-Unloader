"""R0 measurement safety and attribution tests (no real customer data)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ship_perception.measurement.proposal import Candidate, evaluate_instances, nms
from ship_perception.measurement.oracle_local_geometry import _profile, diagnose
from ship_perception.measurement.run_oracle import write_xyz_cache, score_physical_edges, run as run_oracle
from scipy.spatial import cKDTree


def rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def c(name, box, score=0.9):
    return Candidate("TEST", name, score, box, "test")


class ProposalTests(unittest.TestCase):
    def test_duplicate_candidate_and_nms_reason(self):
        rows = nms([c("a", (0, 0, 10, 10), 0.9), c("b", (0.1, 0, 10.1, 10), 0.8)])
        self.assertEqual([r.rank_post_nms for r in rows], [1, None])
        self.assertEqual(rows[1].rejection_reason, "NMS_SUPPRESSED_BY:a")

    def test_two_adjacent_hatches_require_two_candidates(self):
        gt = [{"local_id": "h1", "rough_polygon_raw_xy": rect(0, 0, 10, 10)},
              {"local_id": "h2", "rough_polygon_raw_xy": rect(11, 0, 21, 10)}]
        result = evaluate_instances([c("a", (0, 0, 10, 10)), c("b", (11, 0, 21, 10))], gt)
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["merged_candidate_ids"], [])

    def test_oversized_candidate_cannot_claim_two_hatches(self):
        gt = [{"local_id": "h1", "rough_polygon_raw_xy": rect(0, 0, 10, 10)},
              {"local_id": "h2", "rough_polygon_raw_xy": rect(11, 0, 21, 10)}]
        result = evaluate_instances([c("merged", (0, 0, 21, 10))], gt)
        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["merged_candidate_ids"], ["merged"])

    def test_three_hatch_one_to_one(self):
        gt = [{"local_id": f"h{i}", "rough_polygon_raw_xy": rect(11*i, 0, 11*i+10, 10)}
              for i in range(3)]
        rows = [c(f"c{i}", (11*i, 0, 11*i+10, 10)) for i in range(3)]
        self.assertEqual(evaluate_instances(rows, gt)["matched"], 3)

    def test_partial_observation_missing_roi_cannot_be_scored(self):
        with self.assertRaisesRegex(ValueError, "MISSING_ADJUDICATED_ROUGH_ROI"):
            evaluate_instances([], [{"local_id": "partial", "rough_polygon_raw_xy": None}])


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).resolve().parents[1] / "config/v15.json").read_text())

    def test_profile_break_requires_measured_deck_and_lower_return(self):
        xs = np.arange(-1.0, 1.01, 0.05)
        points = np.array([[x, y, 0.0 if x < 0 else -1.0]
                           for x in xs for y in (-0.1, 0.0, 0.1)], dtype=float)
        row = _profile(points, cKDTree(points[:, :2]), np.array([0., 0.]),
                       np.array([0., 1.]), np.array([1., 0.]),
                       (np.array([0., 0., 1.]), 0.), self.config, "test")
        self.assertTrue(row["valid"], row)
        self.assertLess(abs(row["break_position"][0]), 0.15)

    def test_local_deck_fail_closed(self):
        points = np.array([[0.0, 0.0, -1.0], [0.2, 0.2, -1.0]], dtype=float)
        result = diagnose(points, rect(0, 0, 1, 1), rect(-1, -1, 2, 2), self.config)
        self.assertEqual(result["failure_layer"], "LOCAL_DECK")
        self.assertEqual(result["local_deck"]["status"], "LOCAL_DECK_UNRESOLVED")

    def test_partial_observation_does_not_infer_missing_edge(self):
        axis = np.arange(-2, 12.01, 0.1)
        xx, yy = np.meshgrid(axis, axis)
        zz = np.where((xx > 0) & (xx < 10) & (yy > 0) & (yy < 10), -1.0, 0.0)
        points = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)
        points = points[np.abs(points[:, 0]) >= 1.7]  # one opening edge has no returns
        result = diagnose(points, rect(0, 0, 10, 10), rect(-2, -2, 12, 12), self.config)
        self.assertEqual(result["local_deck"]["status"], "RESOLVED")
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["observed_edge_count"], 3)

    def test_oracle_code_not_referenced_by_product(self):
        root = Path(__file__).resolve().parents[1]
        product = list((root / "src").rglob("*.cpp")) + list((root / "include").rglob("*.hpp"))
        self.assertTrue(product)
        self.assertFalse(any("ship_perception.measurement" in p.read_text(encoding="utf-8") or
                             "oracle_local_geometry" in p.read_text(encoding="utf-8") for p in product))

    def test_oracle_cache_is_exactly_existing_v15_format(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "cloud.sxyz"
            write_xyz_cache(target, np.array([[1, 2, 3], [4, 5, 6]], dtype=float))
            raw = target.read_bytes()
            self.assertEqual(raw[:8], b"SXYZV15\0")
            self.assertEqual(int.from_bytes(raw[8:16], "little"), 2)
            self.assertEqual(np.frombuffer(raw[16:], dtype="<f4").reshape(-1, 3).tolist(),
                             [[1, 2, 3], [4, 5, 6]])

    def test_missing_adjudication_reports_null_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_oracle(root, root / "validation", root / "report.json", None, None)
            self.assertIsNone(report["physical_edge_success_count"])
            self.assertEqual(report["status"], "NOT_MEASURABLE_NO_ADJUDICATED_ROI")

    def test_physical_edge_scorer_never_uses_rough_roi_as_truth(self):
        corners = rect(0, 0, 10, 10)
        edges = [{"edge_local_id": str(i), "visibility": "VISIBLE",
                  "opening_side_status": "OBSERVABLE", "observed_support": [a, b]}
                 for i, (a, b) in enumerate(zip(corners, corners[1:]+corners[:1]))]
        hatch = {"rough_polygon_raw_xy": rect(-1, -1, 11, 11), "edges": edges}
        boundaries = [{"a": [*a, 0], "b": [*b, 0], "side": "INNER_OPENING_FACE",
                       "visibility": "VISIBLE", "evidence_flags": 1}
                      for a, b in zip(corners, corners[1:]+corners[:1])]
        model = {"hatches": [{"boundaries": boundaries}]}
        self.assertTrue(score_physical_edges(hatch, model, 0.15)["success"])
        model["hatches"][0]["boundaries"][0] = dict(boundaries[0], a=[0, 0.3, 0], b=[10, 0.3, 0])
        self.assertFalse(score_physical_edges(hatch, model, 0.15)["success"])
        hatch["edges"] = []
        self.assertIsNone(score_physical_edges(hatch, model, 0.15)["success"])


if __name__ == "__main__":
    unittest.main()
