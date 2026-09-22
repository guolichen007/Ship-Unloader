"""The R0 worksheet is valid input but cannot be silently used as scoring GT."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_v15r_visible_hatch_manifest import validate


class VisibleHatchManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads((ROOT / "datasets/v15r_visible_hatch_manifest.schema.json").read_text(encoding="utf-8"))
        cls.dataset = json.loads((ROOT / "datasets/v15_dataset_manifest.json").read_text(encoding="utf-8"))
        cls.draft = {"schema_version": "ship_perception.v15r.visible_hatch_manifest.1",
                     "annotation_status": "DRAFT", "scans": []}
        for source in cls.dataset["scans"]:
            if source["split_role"] == "DEVELOPMENT":
                cls.draft["scans"].append(dict(scan_id=source["scan_id"],
                    source_pcd_sha256=source["pcd_sha256"], split_role="DEVELOPMENT",
                    target_vessel_status="UNRESOLVED", target_vessel_roi_raw_xy=None,
                    target_selection_source="UNSET", fov_coverage="UNKNOWN",
                    visible_in_fov_hatch_count=None, whole_vessel_hatch_count=None,
                    count_certainty="UNREVIEWED", hatches=[], review_status="NEEDS_REVIEW",
                    reviewers=[], independent_reviews=[], reason="DRAFT_NO_GT"))

    def test_draft_valid_but_not_scoreable(self):
        errors, ready = validate(self.draft, self.dataset, self.schema)
        self.assertEqual(errors, [])
        self.assertEqual(ready, [])
        errors, _ = validate(self.draft, self.dataset, self.schema, require_scoring_ready=True)
        self.assertTrue(any("SCORING_NOT_READY" in error for error in errors))

    def test_split_and_hash_tampering_fail(self):
        labels = copy.deepcopy(self.draft)
        labels["scans"][0]["split_role"] = "SEALED_WEAK_HOLDOUT"
        labels["scans"][1]["source_pcd_sha256"] = "0" * 64
        errors, _ = validate(labels, self.dataset, self.schema)
        self.assertTrue(any("SPLIT_MISMATCH" in error for error in errors))
        self.assertTrue(any("PCD_HASH_MISMATCH" in error for error in errors))

    def test_confirmed_count_and_reviewers_required(self):
        labels = copy.deepcopy(self.draft)
        scan = labels["scans"][0]
        scan.update(target_vessel_status="RESOLVED", target_selection_source="OPERATOR",
                    target_vessel_roi_raw_xy=[[0, 0], [10, 0], [10, 10], [0, 10]],
                    fov_coverage="WHOLE_VESSEL_VISIBLE", visible_in_fov_hatch_count=2,
                    whole_vessel_hatch_count=2, count_certainty="CONFIRMED",
                    review_status="ADJUDICATED", reviewers=["same_person", "same_person"])
        labels["annotation_status"] = "INDEPENDENT_REVIEW"
        errors, _ = validate(labels, self.dataset, self.schema)
        self.assertTrue(any("VISIBLE_COUNT_MISMATCH" in error for error in errors))
        self.assertTrue(any("NEEDS_TWO_INDEPENDENT_REVIEWERS" in error for error in errors))

    def test_observed_boundary_requires_measured_support(self):
        labels = copy.deepcopy(self.draft)
        labels["scans"][0]["hatches"] = [{"local_id": "h1", "rough_polygon_raw_xy": None,
            "observation_status": "PARTIAL", "edges": [{"edge_local_id": "e1",
                "visibility": "VISIBLE", "evidence_type": "OBSERVED_PROFILE_BREAK",
                "opening_side_status": "OBSERVABLE", "observed_support": None, "reason": ""}],
            "partition_relation": "UNKNOWN", "reason": ""}]
        errors, _ = validate(labels, self.dataset, self.schema)
        self.assertTrue(any("OBSERVED_WITHOUT_SUPPORT" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
