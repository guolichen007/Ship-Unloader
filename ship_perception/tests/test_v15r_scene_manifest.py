"""scene_manifest.2 场景契约测试：positive / multi-vessel / partial-FOV / TRACE_ONLY / 非法误分 / scoring-ready 拒绝。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_v15r_scene_manifest import validate, read_json  # noqa: E402

SCHEMA = read_json(ROOT / "datasets/v15r_scene_manifest.schema.json")

SHA = "0" * 64
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
           whole_count=None, certainty="CONFIRMED"):
    hs = [hatch(f"{vid}_h{i}") for i in range(n_hatch)]
    return {"vessel_local_id": vid, "roi_raw_xy": [[0, 0], [10, 0], [10, 5], [0, 5]],
            "vessel_role": role, "fov_coverage": fov,
            "visible_in_fov_hatch_count": n_hatch if fov != "PARTIAL_FOV" else n_hatch,
            "whole_vessel_hatch_count": (n_hatch if whole_count is None else whole_count),
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
errs, _ = validate(labels, DATASET, SCHEMA)
check("PARTIAL_FOV_WHOLE_UNKNOWN", not errs)

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

# 6. scoring-ready 拒绝
# DRAFT 不能 adjudicated
labels = manifest("DRAFT", [scan("final_map_8_22_cut", [vessel("v1", 3)], status="ADJUDICATED")])
errs, _ = validate(labels, DATASET, SCHEMA)
check("DRAFT_CANNOT_ADJUDICATED", any("DRAFT_CANNOT_BE_ADJUDICATED" in e for e in errs))
# adjudicated 但缺独立 reviewer → 拒绝
labels = manifest("ADJUDICATED", [scan("final_map_8_22_cut", [vessel("v1", 3)], status="ADJUDICATED")])
errs, _ = validate(labels, DATASET, SCHEMA)
check("ADJUDICATED_NEEDS_REVIEWERS", any("NEEDS_TWO_INDEPENDENT_REVIEWERS" in e for e in errs))
# 完整双 reviewer + 双记录 → 通过并 scoring_ready
s = scan("final_map_8_22_cut", [vessel("v1", 3)], status="ADJUDICATED")
s["reviewers"] = ["A", "B"]
s["independent_reviews"] = [{"reviewer_id": "A", "review_file_sha256": "a" * 64}, {"reviewer_id": "B", "review_file_sha256": "b" * 64}]
labels = manifest("ADJUDICATED", [s])
errs, ready = validate(labels, DATASET, SCHEMA, require_scoring_ready=True)
check("SCORING_READY_AFTER_DUAL_REVIEW", not errs and ready == ["final_map_8_22_cut"])

print("scene_manifest.2 测试全部通过")
