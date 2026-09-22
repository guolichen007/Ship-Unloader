"""Synthetic-only governance tests; these coordinates are not real annotations."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from ship_perception.measurement.adjudicate_scene_manifest import (
    EXPECTED, edge_difference, polygon_difference, review_pair, validate_review,
    require_audit, verify_final)
from ship_perception.measurement.blind_review_pack import RENDER, NORMAL, review_template, sha, verify_render_config
from ship_perception.measurement.run_proposals import scoring_rois
from ship_perception.measurement.run_oracle import score_physical_edges
from ship_perception.tools.validate_v15r_scene_manifest import DATASET, read_json


def rect(x=0):
    return [[x, 0], [x+4, 0], [x+4, 3], [x, 3]]


class AdjudicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scan = next(s for s in read_json(DATASET)["scans"] if s["scan_id"] == "2026-08-01-05-09-32")

    def review(self, role, count=2):
        row = review_template(self.scan, "a"*64, role)
        row.update(review_status="COMPLETE", reviewer_id=f"reviewer_{role}", repo_access=role == "A",
                   review_a_prior_access=False, algorithm_output_access=False, raw_render_only=True,
                   vessel_roi_raw_xy=rect(-2), fov_coverage="PARTIAL_FOV",
                   visible_in_fov_hatch_count=count,
                   hatches=[{"local_id": f"{role}{i}", "rough_polygon_raw_xy": rect(i*6),
                             "observation_status": "VISIBLE", "reason": "synthetic fixture",
                             "edges": []} for i in range(count)])
        return row

    def test_review_binding_and_sha_mismatch(self):
        row = self.review("A")
        validate_review(row, role="A", scan=self.scan, index_sha="a"*64)
        row["source_pcd_sha256"] = "b"*64
        with self.assertRaisesRegex(ValueError, "REVIEW_BINDING_MISMATCH"):
            validate_review(row, role="A", scan=self.scan, index_sha="a"*64)

    def test_count_disagreement_and_one_to_one_suggestion(self):
        pair = review_pair(self.review("A"), self.review("B"), self.scan, "a"*64)
        self.assertEqual(len(pair["suggested_pairs_not_decisions"]), 2)
        self.assertFalse(any(p["human_pairing_confirmed"] for p in pair["suggested_pairs_not_decisions"]))
        pair = review_pair(self.review("A"), self.review("B", 1), self.scan, "a"*64)
        self.assertEqual(pair["status"], "COUNT_DISAGREEMENT")
        self.assertEqual(len(pair["unpaired_A"]), 1)

    def test_polygon_missing_and_independence(self):
        row = self.review("A")
        row["hatches"][0]["rough_polygon_raw_xy"] = None
        with self.assertRaisesRegex(ValueError, "REVIEW_SCHEMA_INVALID"):
            validate_review(row, role="A", scan=self.scan, index_sha="a"*64)
        b = self.review("B")
        b["review_a_prior_access"] = True
        with self.assertRaisesRegex(ValueError, "REVIEW_B_NOT_INDEPENDENT"):
            validate_review(b, role="B", scan=self.scan, index_sha="a"*64)

    def test_edge_missing_and_five_cm_disagreement(self):
        row = self.review("A")
        row["hatches"][0]["edges"] = [{"edge_local_id": "e", "visibility": "VISIBLE",
                                         "opening_side_status": "OBSERVABLE", "observed_support": None,
                                         "reason": "fixture"}]
        with self.assertRaisesRegex(ValueError, "VISIBLE_EDGE_WITHOUT_SUPPORT"):
            validate_review(row, role="A", scan=self.scan, index_sha="a"*64)
        diff = edge_difference([[0, 0], [4, 0]], [[0, .1], [4, .1]])
        self.assertGreater(diff["p95_m"], .05)
        self.assertAlmostEqual(polygon_difference(rect(), rect())["polygon_iou_approx"], 1)

    def test_overlay_leakage_and_unscored_edge(self):
        bad = copy.deepcopy(RENDER)
        bad["layers"].append("WEAK_BOX")
        with self.assertRaisesRegex(ValueError, "ALGORITHM_OVERLAY_LEAKAGE_MARKER"):
            verify_render_config(bad)
        edge = {"edge_local_id": "e", "visibility": "VISIBLE", "opening_side_status": "OBSERVABLE",
                "observed_support": [[0, 0], [4, 0]]}
        row = score_physical_edges({"edges": [edge]}, {"hatches": []}, .15)
        self.assertEqual(row["status"], "PHYSICAL_EDGE_UNSCORED")
        self.assertIsNone(row["success"])

    def test_manifest_without_signed_audit_cannot_score(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp)/"scene_manifest.json"
            manifest.write_text(json.dumps({"annotation_status": "ADJUDICATED"}), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                require_audit(manifest)
            with self.assertRaises(FileNotFoundError):
                scoring_rois(manifest)

    def test_synthetic_full_provenance_gate_and_stale_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = {s["scan_id"]: s for s in read_json(DATASET)["scans"]}
            index = root/"blind_index.json"
            index.write_text(json.dumps({"annotation_inputs_read": False, "algorithm_outputs_read": False,
                                         "scans": [{"scan_id": sid, "pcd_sha256": dataset[sid]["pcd_sha256"],
                                                    "renders": []} for sid in NORMAL]}), encoding="utf-8")
            index_sha = sha(index)
            reviews = root/"reviews"
            scans, decisions = [], []
            for sid in NORMAL:
                source = dataset[sid]
                items = {}
                for role in ("A", "B"):
                    row = review_template(source, index_sha, role)
                    row.update(review_status="COMPLETE", reviewer_id=f"person_{role}", repo_access=role == "A",
                               review_a_prior_access=False, algorithm_output_access=False, raw_render_only=True,
                               vessel_roi_raw_xy=rect(-2), fov_coverage="PARTIAL_FOV",
                               visible_in_fov_hatch_count=EXPECTED[sid],
                               hatches=[{"local_id": f"{role}{i}", "rough_polygon_raw_xy": rect(i*6),
                                         "observation_status": "VISIBLE", "edges": [], "reason": "synthetic"}
                                        for i in range(EXPECTED[sid])])
                    path = reviews/f"review_{role}"/f"{sid}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(row), encoding="utf-8")
                    items[role] = (row, sha(path))
                hatches = [{"local_id": f"h{i}", "rough_polygon_raw_xy": rect(i*6),
                            "observation_status": "VISIBLE", "edges": [],
                            "partition_relation": "UNKNOWN", "reason": "synthetic"}
                           for i in range(EXPECTED[sid])]
                scans.append({"scan_id": sid, "source_pcd_sha256": source["pcd_sha256"],
                              "split_role": "DEVELOPMENT", "scene_review_status": "ADJUDICATED",
                              "reviewers": ["person_A", "person_B"],
                              "independent_reviews": [{"reviewer_id": f"person_{role}",
                                                       "review_file_sha256": items[role][1]}
                                                      for role in ("A", "B")],
                              "vessels": [{"vessel_local_id": "target", "vessel_role": "PRIMARY_DEVELOPMENT",
                                           "roi_raw_xy": rect(-2), "fov_coverage": "PARTIAL_FOV",
                                           "visible_in_fov_hatch_count": EXPECTED[sid],
                                           "whole_vessel_hatch_count": None, "count_certainty": "CONFIRMED",
                                           "subdivision_evidence": [], "hatches": hatches, "reason": "synthetic"}],
                              "reason": "synthetic"})
                decisions.append({"scan_id": sid, "review_A_sha256": items["A"][1],
                                  "review_B_sha256": items["B"][1], "roi_confirmed": True,
                                  "hatch_links": [{"a": f"A{i}", "b": f"B{i}", "final": f"h{i}",
                                                   "human_confirmed": True} for i in range(EXPECTED[sid])],
                                  "edge_links": []})
            manifest = root/"scene_manifest.json"
            manifest.write_text(json.dumps({"schema_version": "ship_perception.v15r.scene_manifest.2",
                                            "annotation_status": "ADJUDICATED", "scans": scans}), encoding="utf-8")
            decision = root/"decision.json"
            decision.write_text(json.dumps({"schema": "ship_perception.r0a.human_decision.1",
                                            "adjudicator_id": "synthetic_fixture", "scans": decisions}), encoding="utf-8")
            audit_path = root/"adjudication_audit.json"
            audit = verify_final(manifest, decision, reviews, index, audit_path)
            self.assertEqual(audit["adjudicated_hatch_count"], 9)
            self.assertEqual(sum(len(v) for v in scoring_rois(manifest)[0].values()), 9)
            first = reviews/"review_A"/f"{NORMAL[0]}.json"
            first.write_text(first.read_text(encoding="utf-8")+" ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "REVIEW_FILE_SHA_MISMATCH"):
                require_audit(manifest)


if __name__ == "__main__":
    unittest.main()
