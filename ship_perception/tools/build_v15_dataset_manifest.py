"""只读扫描资产，生成相对路径数据契约。文件审计不等同于算法评分访问。"""
import argparse
import hashlib
import json
from pathlib import Path
from v15_pcd import decode, digest

HOLDOUT = {"final_map_4_22_2d_005", "final_map_5_21_empty_boat",
           "final_map_7_15_2d_005", "2026-08-02-13-10-59"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root):
    root = Path(root)
    contracts = {}
    def decoded(path):
        xyz, header = decode(path)
        if "AUDITED_ZERO_TAIL_BYTES" in header:
            identity = header["AUDITED_FILE_SHA256"][0]
            contracts[identity] = dict(expected_file_sha256=identity,
                expected_point_payload_bytes=int(header["POINTS"][0])*sum(int(a)*int(b) for a,b in zip(header["SIZE"],header["COUNT"])),
                expected_trailing_zero_bytes=int(header["AUDITED_ZERO_TAIL_BYTES"][0]),
                audit_basis="EXISTING_MANIFEST_FILE_HASH_AND_ZERO_TAIL_VERIFIED")
        return xyz, header
    scans = []
    xyz_to_scan = {}
    for p in sorted((root / "detect/test").glob("*.pcd")):
        if "_hatch_red" in p.stem:
            continue
        xyz, header = decoded(p)
        info = digest(xyz)
        if info["ordered_xyz_sha256"] in xyz_to_scan:
            raise ValueError("规范扫描重复")
        record = dict(dataset_id="hold_detector_static_xyz_v1", scan_id=p.stem,
                      pcd_filename=p.name, pcd_path=p.relative_to(root).as_posix(),
                      pcd_sha256=sha(p), fields=header["FIELDS"], **info,
                      annotation_files=[], annotation_sha256=[], aliases=[],
                      duplicate_group=p.stem, annotation_scope="PARTIAL_SINGLE_HATCH",
                      source_family="dated_scan" if p.stem.startswith("2026-") else "final_map",
                      background_contamination="UNASSESSED_GEOMETRY_ONLY",
                      alignment_class="UNASSESSED_GEOMETRY_ONLY",
                      split_role="SEALED_WEAK_HOLDOUT" if p.stem in HOLDOUT else "DEVELOPMENT",
                      golden_candidate=p.stem in HOLDOUT, physical_vessel_id="UNKNOWN",
                      vessel_relationship="LIKELY_DIFFERENT" if p.stem in HOLDOUT else "UNKNOWN",
                      notes="静态XYZ；无可信射线轨迹；文件名和标注不得参与产品几何判定")
        xyz_to_scan[info["ordered_xyz_sha256"]] = record
        scans.append(record)
    file_cache = {}
    for p in sorted((root / "detect/pcd").glob("*.pcd")):
        h = sha(p)
        if h not in file_cache:
            xyz, _ = decoded(p)
            file_cache[h] = hashlib.sha256(xyz.tobytes()).hexdigest()
        record = xyz_to_scan[file_cache[h]]
        record["aliases"].append(dict(path=p.relative_to(root).as_posix(), sha256=h))
        annotation = p.with_name(p.stem + "_annotations.json")
        if annotation.exists():
            j = json.loads(annotation.read_text(encoding="utf-8"))
            if j["point_count"] != record["point_count"] or len(j["annotations"]) != 1:
                raise ValueError("标注契约不一致")
            record["annotation_files"].append(annotation.relative_to(root).as_posix())
            record["annotation_sha256"].append(sha(annotation))
    variant = root / "annotator/build/pcd/2026-08-08-02-33-57.pcd"
    xyz, _ = decoded(variant)
    rec = xyz_to_scan[hashlib.sha256(xyz.tobytes()).hexdigest()]
    rec["aliases"].append(dict(path=variant.relative_to(root).as_posix(), sha256=sha(variant)))
    for r in scans:
        if not r["annotation_files"]:
            r["annotation_scope"] = "UNLABELED"
    count = sum(len(r["annotation_files"]) for r in scans)
    labeled = sum(bool(r["annotation_files"]) for r in scans)
    if (len(scans), labeled, count, count-labeled) != (13, 12, 23, 11):
        raise ValueError("13/12/23/11 数据契约冲突")
    return dict(schema_version="ship_perception.v15.dataset.1", independent_scans=13,
                labeled_scans=12, annotation_count=23, extra_annotations=11,
                audit_purpose="FILE_AND_DUPLICATE_AUDIT_NOT_ALGORITHM_SCORING", scans=scans,
                pcd_padding_contracts=sorted(contracts.values(),key=lambda x:x["expected_file_sha256"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build(args.data_root)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
