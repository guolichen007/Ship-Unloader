"""Validate R0 human labels without promoting draft or weak boxes to ground truth."""
import argparse
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "datasets/v15r_visible_hatch_manifest.schema.json"
DATASET = ROOT / "datasets/v15_dataset_manifest.json"


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f"NON_FINITE_JSON_NUMBER:{value}")
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid_constant)
    def check_finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("NON_FINITE_JSON_NUMBER")
        if isinstance(item, dict):
            for child in item.values():
                check_finite(child)
        elif isinstance(item, list):
            for child in item:
                check_finite(child)
    check_finite(value)
    return value


def signed_area(polygon):
    return sum(a[0] * b[1] - b[0] * a[1]
               for a, b in zip(polygon, polygon[1:] + polygon[:1])) / 2


def validate(labels, dataset, schema, require_scoring_ready=False):
    errors = [f"SCHEMA:{'/'.join(map(str, issue.path))}:{issue.message}"
              for issue in Draft202012Validator(schema).iter_errors(labels)]
    if errors:
        return sorted(errors), []

    canonical = {row["scan_id"]: row for row in dataset["scans"]}
    seen = set()
    scoring_ready = []
    for scan in labels["scans"]:
        scan_error_count = len(errors)
        sid = scan["scan_id"]
        if sid in seen:
            errors.append(f"DUPLICATE_SCAN:{sid}")
        seen.add(sid)
        source = canonical.get(sid)
        if source is None:
            errors.append(f"UNKNOWN_SCAN:{sid}")
            continue
        if scan["split_role"] != source["split_role"]:
            errors.append(f"SPLIT_MISMATCH:{sid}")
        if scan["source_pcd_sha256"] != source["pcd_sha256"]:
            errors.append(f"PCD_HASH_MISMATCH:{sid}")
        if labels["annotation_status"] == "DRAFT" and scan["review_status"] == "ADJUDICATED":
            errors.append(f"DRAFT_CANNOT_BE_ADJUDICATED:{sid}")
        if scan["review_status"] == "ADJUDICATED":
            if len(set(scan["reviewers"])) < 2:
                errors.append(f"NEEDS_TWO_INDEPENDENT_REVIEWERS:{sid}")
            independent = scan["independent_reviews"]
            if (len({item["reviewer_id"] for item in independent}) < 2 or
                    len({item["review_file_sha256"] for item in independent}) < 2):
                errors.append(f"NEEDS_TWO_REVIEW_RECORDS:{sid}")
            if scan["count_certainty"] == "UNREVIEWED":
                errors.append(f"ADJUDICATED_UNREVIEWED_COUNT:{sid}")
        if scan["target_vessel_status"] == "RESOLVED":
            if scan["target_vessel_roi_raw_xy"] is None or scan["target_selection_source"] == "UNSET":
                errors.append(f"RESOLVED_TARGET_NEEDS_ROI_AND_SOURCE:{sid}")
        for name, polygon in [("target_vessel_roi", scan["target_vessel_roi_raw_xy"])] + [
            (f"hatch:{hatch['local_id']}", hatch["rough_polygon_raw_xy"])
            for hatch in scan["hatches"]]:
            if polygon is not None and abs(signed_area(polygon)) < 1e-4:
                errors.append(f"DEGENERATE_POLYGON:{sid}:{name}")
        hatch_ids = [hatch["local_id"] for hatch in scan["hatches"]]
        if len(set(hatch_ids)) != len(hatch_ids):
            errors.append(f"DUPLICATE_HATCH_ID:{sid}")
        for hatch in scan["hatches"]:
            edge_ids = [edge["edge_local_id"] for edge in hatch["edges"]]
            if len(set(edge_ids)) != len(edge_ids):
                errors.append(f"DUPLICATE_EDGE_ID:{sid}:{hatch['local_id']}")
            for edge in hatch["edges"]:
                if edge["evidence_type"].startswith("OBSERVED") and edge["observed_support"] is None:
                    errors.append(f"OBSERVED_WITHOUT_SUPPORT:{sid}:{hatch['local_id']}:{edge['edge_local_id']}")
                if edge["evidence_type"] in {"INFERRED", "HISTORICAL", "UNKNOWN"} and edge["observed_support"] is not None:
                    errors.append(f"NON_OBSERVED_WITH_SUPPORT:{sid}:{hatch['local_id']}:{edge['edge_local_id']}")
                if edge["evidence_type"] == "HISTORICAL" and source["dataset_id"] == "hold_detector_static_xyz_v1":
                    errors.append(f"STATIC_SCAN_CANNOT_PROVE_HISTORY:{sid}:{hatch['local_id']}:{edge['edge_local_id']}")
                support = edge["observed_support"]
                if support is not None and len({tuple(point) for point in support}) < 2:
                    errors.append(f"DEGENERATE_EDGE_SUPPORT:{sid}:{hatch['local_id']}:{edge['edge_local_id']}")
        confirmed = scan["count_certainty"] == "CONFIRMED"
        if confirmed:
            if scan["target_vessel_status"] != "RESOLVED" or scan["fov_coverage"] == "UNKNOWN":
                errors.append(f"CONFIRMED_NEEDS_TARGET_AND_FOV:{sid}")
            if scan["visible_in_fov_hatch_count"] != len(scan["hatches"]):
                errors.append(f"VISIBLE_COUNT_MISMATCH:{sid}")
            if any(hatch["rough_polygon_raw_xy"] is None for hatch in scan["hatches"]):
                errors.append(f"CONFIRMED_HATCH_NEEDS_ROUGH_POLYGON:{sid}")
            if scan["fov_coverage"] == "WHOLE_VESSEL_VISIBLE" and scan["whole_vessel_hatch_count"] != scan["visible_in_fov_hatch_count"]:
                errors.append(f"WHOLE_COUNT_MISMATCH:{sid}")
            if (scan["review_status"] == "ADJUDICATED" and
                    len(set(scan["reviewers"])) >= 2 and len(errors) == scan_error_count):
                scoring_ready.append(sid)
    if labels["annotation_status"] == "ADJUDICATED" and any(
            row["review_status"] != "ADJUDICATED" for row in labels["scans"]):
        errors.append("MANIFEST_ADJUDICATED_WITH_UNREVIEWED_SCAN")
    if require_scoring_ready and (errors or len(scoring_ready) != len(labels["scans"])):
        errors.append(f"SCORING_NOT_READY:{len(scoring_ready)}/{len(labels['scans'])}")
    return sorted(errors), scoring_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-scoring-ready", action="store_true")
    args = parser.parse_args()
    errors, ready = validate(read_json(args.manifest), read_json(DATASET), read_json(SCHEMA),
                             args.require_scoring_ready)
    print(json.dumps({"valid": not errors, "scoring_ready_scan_count": len(ready),
                      "scoring_ready_scan_ids": ready, "errors": errors}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
