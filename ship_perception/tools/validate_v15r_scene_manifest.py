"""Validate V1.5-R scene manifest v2 (multi-vessel). Never promotes draft / weak
boxes / TRACE_ONLY subdivision to ground truth."""
import argparse
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "datasets/v15r_scene_manifest.schema.json"
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
        before = len(errors)
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
        if labels["annotation_status"] == "DRAFT" and scan["scene_review_status"] == "ADJUDICATED":
            errors.append(f"DRAFT_CANNOT_BE_ADJUDICATED:{sid}")

        vessel_ids = [v["vessel_local_id"] for v in scan["vessels"]]
        if len(set(vessel_ids)) != len(vessel_ids):
            errors.append(f"DUPLICATE_VESSEL_ID:{sid}")
        for vessel in scan["vessels"]:
            vid = vessel["vessel_local_id"]
            if vessel["vessel_role"] == "MULTI_VESSEL_STRESS" and len(scan["vessels"]) < 2:
                errors.append(f"MULTI_VESSEL_STRESS_NEEDS_MULTIPLE_VESSELS:{sid}:{vid}")
            for name, polygon in ([("vessel_roi", vessel.get("roi_raw_xy"))] +
                                  [(f"hatch:{h['local_id']}", h["rough_polygon_raw_xy"]) for h in vessel["hatches"]]):
                if polygon is not None and abs(signed_area(polygon)) < 1e-4:
                    errors.append(f"DEGENERATE_POLYGON:{sid}:{vid}:{name}")
            hatch_ids = [h["local_id"] for h in vessel["hatches"]]
            if len(set(hatch_ids)) != len(hatch_ids):
                errors.append(f"DUPLICATE_HATCH_ID:{sid}:{vid}")
            for hatch in vessel["hatches"]:
                edge_ids = [e["edge_local_id"] for e in hatch["edges"]]
                if len(set(edge_ids)) != len(edge_ids):
                    errors.append(f"DUPLICATE_EDGE_ID:{sid}:{vid}:{hatch['local_id']}")
                for edge in hatch["edges"]:
                    if edge["evidence_type"].startswith("OBSERVED") and edge["observed_support"] is None:
                        errors.append(f"OBSERVED_WITHOUT_SUPPORT:{sid}:{vid}:{hatch['local_id']}:{edge['edge_local_id']}")
                    if edge["evidence_type"] in {"INFERRED_GEOMETRY", "INFERRED_VESSEL_PRIOR", "HISTORICAL_MODEL", "UNKNOWN"} and edge["observed_support"] is not None:
                        errors.append(f"NON_OBSERVED_WITH_SUPPORT:{sid}:{vid}:{hatch['local_id']}:{edge['edge_local_id']}")
                    # OPERATOR_CONFIRMED is a separate human provenance, not an OBSERVED_* sensor claim.
                    if edge["evidence_type"] == "HISTORICAL_MODEL" and source["dataset_id"] == "hold_detector_static_xyz_v1":
                        errors.append(f"STATIC_SCAN_CANNOT_PROVE_HISTORY:{sid}:{vid}:{hatch['local_id']}:{edge['edge_local_id']}")
            # Subdivision semantics: TRACE_ONLY must never be promoted to two hatches.
            for sub in vessel["subdivision_evidence"]:
                if sub["status"] == "TRACE_ONLY" and sub["physical_partition_confirmed"]:
                    errors.append(f"TRACE_ONLY_CANNOT_CONFIRM_PARTITION:{sid}:{vid}")
                if sub["status"] == "TRACE_ONLY" and sub.get("operational_hatch_count", 1) != 1:
                    errors.append(f"TRACE_ONLY_OPERATIONAL_COUNT_NOT_ONE:{sid}:{vid}")
                if sub["status"] == "CONFIRMED_PARTITION" and not sub["physical_partition_confirmed"]:
                    errors.append(f"CONFIRMED_PARTITION_NEEDS_FLAG:{sid}:{vid}")
            if vessel["count_certainty"] == "CONFIRMED":
                if vessel["fov_coverage"] == "UNKNOWN":
                    errors.append(f"CONFIRMED_NEEDS_FOV:{sid}:{vid}")
                if vessel["visible_in_fov_hatch_count"] != len(vessel["hatches"]):
                    errors.append(f"VISIBLE_COUNT_MISMATCH:{sid}:{vid}")
                if any(h["rough_polygon_raw_xy"] is None for h in vessel["hatches"]):
                    errors.append(f"CONFIRMED_HATCH_NEEDS_ROUGH_POLYGON:{sid}:{vid}")
                if (vessel["fov_coverage"] == "WHOLE_VESSEL_VISIBLE" and
                        vessel["whole_vessel_hatch_count"] != vessel["visible_in_fov_hatch_count"]):
                    errors.append(f"WHOLE_COUNT_MISMATCH:{sid}:{vid}")

        if scan["scene_review_status"] == "ADJUDICATED":
            reviewers = scan.get("reviewers")
            independent = scan.get("independent_reviews")
            if reviewers is None:
                errors.append(f"MISSING_REVIEWERS:{sid}")
            elif len(set(reviewers)) < 2:
                errors.append(f"NEEDS_TWO_INDEPENDENT_REVIEWERS:{sid}")
            if independent is None:
                errors.append(f"MISSING_INDEPENDENT_REVIEWS:{sid}")
            elif (len({i["reviewer_id"] for i in independent}) < 2 or
                    len({i["review_file_sha256"] for i in independent}) < 2):
                errors.append(f"NEEDS_TWO_REVIEW_RECORDS:{sid}")
            if len(errors) == before:
                scoring_ready.append(sid)

    if labels["annotation_status"] == "ADJUDICATED" and any(
            r["scene_review_status"] != "ADJUDICATED" for r in labels["scans"]):
        errors.append("MANIFEST_ADJUDICATED_WITH_UNREVIEWED_SCAN")
    if require_scoring_ready and labels["annotation_status"] != "ADJUDICATED":
        errors.append("MANIFEST_NOT_ADJUDICATED")
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
