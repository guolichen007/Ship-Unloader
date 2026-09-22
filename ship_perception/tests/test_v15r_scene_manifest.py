"""scene_manifest.2 场景契约测试：positive / multi-vessel / partial-FOV / TRACE_ONLY / 非法误分 / scoring-ready 拒绝。"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_v15r_scene_manifest import validate, read_json  # noqa: E402

SCHEMA = read_json(ROOT / "datasets/v15r_scene_manifest.schema.json")

SHA = "0" * 64
WHOLE_COUNT_UNSET = object()
# fake dataset manifest，scan_id 与 source_pcd_sha256 一一对应
SCANS = [
    {"scan_id": "final_map_4_16_2d_005", "pcd_sha256": "1" * 64, "split_role": "DEVELOPMENT", "dataset_id": "hold_detector_static_xyz_v1"},
    {"scan_id": "final_map_5_29_2d_005", "pcd_sha256": "2" * 64, "split_role": "DEVELOPMENT", "dataset_id": "hold_detector_static_xyz_v1"},
    {"scan_id": "final_map_7_2_005", "pcd_sha256": "3" * 64, "split_role": "DEVELOPMENT", "dataset_id": "hold_detector_static_xyz_v1"},
    {"scan_id": "final_8_17_005", "pcd_sha256": "4" * 64, "split_role": "DEVELOPMENT", "dataset_id": "hold_detector_static_xyz_v1"},
    {"scan_id": "final_map_8_22_cut", "pcd_sha256": "5" * 64, "split_role": "DEVELOPMENT", "dataset_id": "hold_detector_static_xyz_v1"},
]
DATASET = {"scans": SCANS}
HASH = {s["scan_id"]: s["pcd_sha256"] for s in SCANS}


def hatch(lid, n_edges=4):
    edges = [{"edge_local_id": f"{lid}_e{i}", "visibility": "VISIBLE", "evidence_type": "OBSERVED_PROFILE_BREAK",
              "opening_side_status": "OBSERVABLE", "observed_support": [[0.0, 0.0], [1.0, 0.0]], "reason": ""}
             for i in range(n_edges)]
    return {"local_id": lid, "rough_polygon_raw_xy": [[0, 0], [10, 0], [10, 5], [0, 5]],
            "observation_status": "VISIBLE", "edges": edges, "partition_relation": "NO_PARTITION_EVIDENCE", "reason": ""}


def vessel(vid, n_hatch, role="PRIMARY_DEVELOPMENT", sub=None, fov="WHOLE_VESSEL_VISIBLE",
           whole_count=WHOLE_COUNT_UNSET, certainty="CONFIRMED"):
    hs = [hatch(f"{vid}_h{i}") for i in range(n_hatch)]
    return {"vessel_local_id": vid, "roi_raw_xy": [[0, 0], [10, 0], [10, 5], [0, 5]],
            "vessel_role": role, "fov_coverage": fov,
            "visible_in_fov_hatch_count": n_hatch,
            "whole_vessel_hatch_count": (n_hatch if whole_count is WHOLE_COUNT_UNSET else whole_count),
            "count_certainty": certainty, "subdivision_evidence": sub or [],
            "hatches": hs, "reason": ""}


def scan(sid, vessels, status="NEEDS_REVIEW"):
    return {"scan_id": sid, "source_pcd_sha256": HASH[sid], "split_role": "DEVELOPMENT",
            "scene_review_status": status, "vessels": vessels, "reason": ""}


def manifest(status, scans):
    return {"schema_version": "ship_perception.v15r.scene_manifest.2",
            "annotation_status": status, "scans": scans}


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print(f"  {name}=PASS")


# 1. positive：正常单船 1/2/3 舱
labels = manifest("DRAFT", [scan("final_map_8_22_cut", [vessel("v1", 3)])])
errs, _ = validate(labels, DATASET, SCHEMA)
check("POSITIVE_3_HATCH", not errs)

# 2. multi-vessel stress：5-29 三条船
labels = manifest("DRAFT", [scan("final_map_5_29_2d_005", [vessel("v1", 1, "MULTI_VESSEL_STRESS"), vessel("v2", 1, "MULTI_VESSEL_STRESS"), vessel("v3", 1, "MULTI_VESSEL_STRESS")])])
errs, _ = validate(labels, DATASET, SCHEMA)
check("MULTI_VESSEL_3", not errs)
# multi-vessel 只有一条船 → 拒绝
labels = manifest("DRAFT", [scan("final_map_5_29_2d_005", [vessel("v1", 1, "MULTI_VESSEL_STRESS")])])
errs, _ = validate(labels, DATASET, SCHEMA)
check("MULTI_VESSEL_SINGLE_REJECTED", any("MULTI_VESSEL_STRESS_NEEDS_MULTIPLE_VESSELS" in e for e in errs))

# 3. partial-FOV：8-17 整船舱数 UNKNOWN
labels = manifest("DRAFT", [scan("final_8_17_005", [vessel("v1", 1, "PARTIAL_FOV", fov="PARTIAL_FOV", whole_count=None, certainty="UNREVIEWED")])])
check("PARTIAL_FOV_MANIFEST_CONTAINS_NULL",
      labels["scans"][0]["vessels"][0]["whole_vessel_hatch_count"] is None and
      '"whole_vessel_hatch_count": null' in json.dumps(labels))
errs, _ = validate(labels, DATASET, SCHEMA)
check("PARTIAL_FOV_WHOLE_COUNT_NULL_PASS", not errs)

# 4. TRACE_ONLY：4-16 一个舱 + 疑似隔断痕迹，physical_partition_confirmed=false
sub_trace = [{"status": "TRACE_ONLY", "physical_partition_confirmed": False, "operational_hatch_count": 1, "reason": "疑似隔断痕迹"}]
labels = manifest("DRAFT", [scan("final_map_4_16_2d_005", [vessel("v1", 1, sub=sub_trace)])])
errs, _ = validate(labels, DATASET, SCHEMA)
check("TRACE_ONLY_VALID", not errs)

# 5. 非法误分契约：TRACE_ONLY 却 physical_partition_confirmed=true → 拒绝
sub_bad = [{"status": "TRACE_ONLY", "physical_partition_confirmed": True, "operational_hatch_count": 1, "reason": ""}]
labels = manifest("DRAFT", [scan("final_map_4_16_2d_005", [vessel("v1", 1, sub=sub_bad)])])
errs, _ = validate(labels, DATASET, SCHEMA)
check("TRACE_ONLY_CONFIRMED_REJECTED", any("TRACE_ONLY_CANNOT_CONFIRM_PARTITION" in e for e in errs))
# TRACE_ONLY 却 operational_hatch_count=2 → 拒绝
sub_two = [{"status": "TRACE_ONLY", "physical_partition_confirmed": False, "operational_hatch_count": 2, "reason": ""}]
labels = manifest("DRAFT", [scan("final_map_4_16_2d_005", [vessel("v1", 1, sub=sub_two)])])
errs, _ = validate(labels, DATASET, SCHEMA)
check("TRACE_ONLY_TWO_HATCH_REJECTED", any("TRACE_ONLY_OPERATIONAL_COUNT_NOT_ONE" in e for e in errs))
check("TRACE_ONLY_FALSE_SPLIT_GATE", any("TRACE_ONLY_OPERATIONAL_COUNT_NOT_ONE" in e for e in errs))

# 旧枚举必须拒绝；新枚举保持人工、推断与传感器观测来源分离。
def edge_case(evidence, support=None):
    labels = manifest("DRAFT", [scan("final_map_4_16_2d_005", [vessel("v1", 1)])])
    edge = labels["scans"][0]["vessels"][0]["hatches"][0]["edges"][0]
    edge["evidence_type"] = evidence
    edge["observed_support"] = support
    return labels


old_rejected = []
for old in ("INFERRED", "HISTORICAL"):
    errs, _ = validate(edge_case(old), DATASET, SCHEMA)
    old_rejected.append(any(e.startswith("SCHEMA:") for e in errs))
    check(f"OLD_EVIDENCE_{old}_REJECTED", old_rejected[-1])
check("OLD_EVIDENCE_ENUM_REJECTED", all(old_rejected))

measured = [[0.0, 0.0], [1.0, 0.0]]
new_accepted = []
for evidence in ("OBSERVED_PROFILE_BREAK", "OBSERVED_3D_FACE", "OBSERVED_BEV_CONTOUR",
                 "INFERRED_GEOMETRY", "INFERRED_VESSEL_PRIOR", "OPERATOR_CONFIRMED", "UNKNOWN"):
    support = measured if evidence.startswith("OBSERVED_") else None
    errs, _ = validate(edge_case(evidence, support), DATASET, SCHEMA)
    new_accepted.append(not errs)
    check(f"NEW_EVIDENCE_{evidence}_ACCEPTED", new_accepted[-1])
temporal_dataset = copy.deepcopy(DATASET)
temporal_dataset["scans"][0]["dataset_id"] = "time_resolved_sequence"
historical_labels = edge_case("HISTORICAL_MODEL")
errs, _ = validate(historical_labels, temporal_dataset, SCHEMA)
check("HISTORICAL_MODEL_NEEDS_SOURCE_AND_AGE", any(e.startswith("SCHEMA:") for e in errs))
historical_edge = historical_labels["scans"][0]["vessels"][0]["hatches"][0]["edges"][0]
historical_edge["historical_source_frame_id"] = "frame_17"
historical_edge["historical_age_s"] = 0.8
errs, _ = validate(historical_labels, temporal_dataset, SCHEMA)
new_accepted.append(not errs)
check("NEW_EVIDENCE_HISTORICAL_MODEL_ACCEPTED_WITH_TEMPORAL_SOURCE", new_accepted[-1])
check("NEW_EVIDENCE_ENUM_ACCEPTED", all(new_accepted))
for evidence in ("INFERRED_GEOMETRY", "INFERRED_VESSEL_PRIOR", "HISTORICAL_MODEL", "UNKNOWN"):
    labels = edge_case(evidence, measured)
    if evidence == "HISTORICAL_MODEL":
        edge = labels["scans"][0]["vessels"][0]["hatches"][0]["edges"][0]
        edge["historical_source_frame_id"] = "frame_17"
        edge["historical_age_s"] = 0.8
    errs, _ = validate(labels, DATASET, SCHEMA)
    check(f"NON_OBSERVED_{evidence}_SUPPORT_REJECTED", any("NON_OBSERVED_WITH_SUPPORT" in e for e in errs))
errs, _ = validate(edge_case("OBSERVED_BEV_CONTOUR"), DATASET, SCHEMA)
check("OBSERVED_BEV_NEEDS_SUPPORT", any("OBSERVED_WITHOUT_SUPPORT" in e for e in errs))
errs, _ = validate(historical_labels, DATASET, SCHEMA)
check("STATIC_HISTORY_REJECTED", any("STATIC_SCAN_CANNOT_PROVE_HISTORY" in e for e in errs))

# 6. scoring-ready 拒绝
# DRAFT 不能 adjudicated
labels = manifest("DRAFT", [scan("final_map_8_22_cut", [vessel("v1", 3)], status="ADJUDICATED")])
errs, _ = validate(labels, DATASET, SCHEMA)
check("DRAFT_CANNOT_ADJUDICATED", any("DRAFT_CANNOT_BE_ADJUDICATED" in e for e in errs))
# adjudicated 但缺独立 reviewer → 拒绝
labels = manifest("ADJUDICATED", [scan("final_map_8_22_cut", [vessel("v1", 3)], status="ADJUDICATED")])
labels["scans"][0]["independent_reviews"] = [
    {"reviewer_id": "A", "review_file_sha256": "a" * 64},
    {"reviewer_id": "B", "review_file_sha256": "b" * 64}]
errs, _ = validate(labels, DATASET, SCHEMA)
check("ADJUDICATED_MISSING_REVIEWERS_REJECTED", any("MISSING_REVIEWERS:" in e for e in errs))
labels = manifest("ADJUDICATED", [scan("final_map_8_22_cut", [vessel("v1", 3)], status="ADJUDICATED")])
labels["scans"][0]["reviewers"] = ["A", "B"]
errs, _ = validate(labels, DATASET, SCHEMA)
check("ADJUDICATED_MISSING_REVIEW_RECORDS_REJECTED", any("MISSING_INDEPENDENT_REVIEWS:" in e for e in errs))
# 完整双 reviewer + 双记录 → 通过并 scoring_ready
s = scan("final_map_8_22_cut", [vessel("v1", 3)], status="ADJUDICATED")
s["reviewers"] = ["A", "B"]
s["independent_reviews"] = [{"reviewer_id": "A", "review_file_sha256": "a" * 64}, {"reviewer_id": "B", "review_file_sha256": "b" * 64}]
labels = manifest("ADJUDICATED", [s])
errs, ready = validate(labels, DATASET, SCHEMA, require_scoring_ready=True)
check("SCORING_READY_AFTER_DUAL_REVIEW", not errs and ready == ["final_map_8_22_cut"])
duplicate_reviewer = copy.deepcopy(labels)
duplicate_reviewer["scans"][0]["reviewers"] = ["A", "A"]
errs, _ = validate(duplicate_reviewer, DATASET, SCHEMA)
check("DUPLICATE_REVIEWER_REJECTED", any("NEEDS_TWO_INDEPENDENT_REVIEWERS" in e for e in errs))
duplicate_record = copy.deepcopy(labels)
duplicate_record["scans"][0]["independent_reviews"][1]["review_file_sha256"] = "a" * 64
errs, _ = validate(duplicate_record, DATASET, SCHEMA)
check("DUPLICATE_REVIEW_SHA_REJECTED", any("NEEDS_TWO_REVIEW_RECORDS" in e for e in errs))
labels["annotation_status"] = "INDEPENDENT_REVIEW"
errs, _ = validate(labels, DATASET, SCHEMA, require_scoring_ready=True)
check("SCORING_REQUIRES_ADJUDICATED_MANIFEST", any("MANIFEST_NOT_ADJUDICATED" in e for e in errs))

print("scene_manifest.2 测试全部通过")
