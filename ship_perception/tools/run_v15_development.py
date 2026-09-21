"""运行九份 Development；失败仍归档，不能把运行完整性当能力通过。"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

from v15_pcd import decode, digest, write_cache
from v15_artifacts import write_artifacts,write_csv
from v15_config_identity import config_file_hash
from v15_holdout_ledger import current_identity,append_access,DEFAULT_LEDGER
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.v15_partial import evaluate_annotation, disagreement


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def operational(records, config):
    # Count scans, never annotation versions. Missing results cannot disappear.
    if len(records) != 9 or len({r["scan_id"] for r in records}) != 9:
        raise ValueError("DEVELOPMENT_SCAN_COMPLETENESS")
    labeled = [r for r in records if r["annotation_count"]]
    if len(labeled) != 8:
        raise ValueError("DEVELOPMENT_ANNOTATION_COMPLETENESS")
    counts = dict(FRAME_RESOLVED_COUNT=sum(r["frame_resolved"] for r in records),
                  DECK_RESOLVED_COUNT=sum(r["deck_resolved"] for r in records),
                  JOINT_FRAME_AND_DECK_RESOLVED_COUNT=sum(r["frame_resolved"] and r["deck_resolved"] for r in records),
                  NONAMBIGUOUS_LABEL_ASSOCIATION_COUNT=sum(r["consistent_unique_association"] for r in labeled),
                  NONEMPTY_STRUCTURE_COUNT=sum(r["nondeck_primitive_count"] > 0 for r in records),
                  NONEMPTY_HATCH_COUNT=sum(r["hatch_count"] > 0 for r in records))
    passed = (counts["FRAME_RESOLVED_COUNT"] >= config["real_frame_min"] and
              counts["DECK_RESOLVED_COUNT"] >= config["real_deck_min"] and
              counts["JOINT_FRAME_AND_DECK_RESOLVED_COUNT"] >= config["real_joint_min"] and
              counts["NONAMBIGUOUS_LABEL_ASSOCIATION_COUNT"] >= config["real_association_min"])
    return dict(counts=counts, REAL_DEVELOPMENT_OPERATIONAL_GATE="PASS" if passed else "FAIL",
                RUN_COMPLETENESS="PASS" if all(r["execution_status"] == "COMPLETED" for r in records) else "FAIL",
                REAL_FORMAL_15CM="PENDING_GOLDEN", SITE_ACCURACY="SITE_PENDING")


def write_forensic(directory, model, annotations, evaluations, points):
    """只读法证输出：不改任何 association 阈值，不改 evaluator 逻辑。
    用于区分“坐标系错误”与“识别到错误舱口”：candidate center 与 annotation
    center 在输入坐标中的距离，加上 T_B_input round-trip 自检。"""
    import numpy as np
    T = model.get("T_B_input")
    roundtrip = None
    if T is not None and points is not None and len(points):
        t = np.asarray(T, float)
        tinv = np.linalg.inv(t)
        n = len(points)
        sample = points[np.linspace(0, n - 1, min(n, 512)).astype(int)]
        pB = (t @ np.c_[sample, np.ones(len(sample))].T).T[:, :3]
        pback = (tinv @ np.c_[pB, np.ones(len(pB))].T).T[:, :3]
        roundtrip = float(np.abs(sample - pback).max())
    rows = []
    for annotation, evaluation in zip(annotations, evaluations):
        poly = np.asarray(annotation["corners"], float)
        ann_center = poly.mean(axis=0)
        scores = sorted((float(c["association_score"]) for c in evaluation["candidates"]), reverse=True)
        top1 = scores[0] if scores else 0.0
        top2 = scores[1] if len(scores) > 1 else None
        gap = (top1 - top2) if top2 is not None else None
        candidates = []
        for c in evaluation["candidates"]:
            hatch = next((h for h in model["hatches"] if h["candidate_id"] == c["candidate_id"]), None)
            center_input = None
            dist = None
            if hatch is not None and T is not None:
                center_B = np.asarray(hatch["center"], float)
                center_input = (np.linalg.inv(np.asarray(T, float)) @ np.r_[center_B, 1.0])[:2]
                dist = float(np.linalg.norm(center_input - ann_center))
            observed = c.get("observed_error", {}) or {}
            candidates.append(dict(candidate_id=c["candidate_id"],
                                   candidate_center_input_xy=center_input.tolist() if center_input is not None else None,
                                   candidate_to_annotation_center_distance_m=dist,
                                   association_score=float(c["association_score"]),
                                   status=c["status"],
                                   observed_P95_m=(float(observed["P95"]) if observed.get("P95") is not None else None)))
        rows.append(dict(annotation_path=evaluation.get("annotation_path"),
                         annotation_center_input_xy=ann_center.tolist(),
                         top1_score=top1,
                         top2_score=top2,
                         top2_gap=gap,
                         candidates=candidates))
    write(directory / "forensic.json", dict(frame_resolved=bool(model.get("frame_resolved")),
                                             deck_resolved=bool((model.get("deck") or {}).get("valid")),
                                             hatch_count=len(model.get("hatches", [])),
                                             transform_roundtrip_max_error_m=roundtrip,
                                             annotations=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke",action="store_true",help="固定两份 Development Quick smoke；不计算或代替九份 operational 门禁")
    parser.add_argument("--include-sealed-holdout", action="store_true",help="仅在精确配置冻结记录有效时评估全部13份扫描")
    parser.add_argument("--freeze-record",type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    manifest_path = source/"datasets/v15_dataset_manifest.json"
    config_path = source/"config/v15.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scans = [s for s in manifest["scans"] if s["split_role"] == "DEVELOPMENT"]
    if args.smoke:
        if args.include_sealed_holdout:raise ValueError("QUICK_MUST_NOT_ACCESS_HOLDOUT")
        smoke_ids={"final_map_4_16_2d_005","2026-08-08-02-33-57"}
        scans=[s for s in scans if s['scan_id'] in smoke_ids]
        if {s['scan_id'] for s in scans}!=smoke_ids:raise ValueError("QUICK_DEVELOPMENT_MANIFEST_MISMATCH")
    identity=current_identity()
    if args.include_sealed_holdout:
        if not args.freeze_record:raise ValueError("HOLDOUT_REQUIRES_FROZEN_CONFIGURATION_RECORD")
        freeze=json.loads(args.freeze_record.read_text(encoding="utf-8"))
        if freeze.get("status")!="FROZEN_FOR_HOLDOUT" or any(freeze.get(k)!=v for k,v in identity.items()):
            raise ValueError("HOLDOUT_FREEZE_IDENTITY_MISMATCH")
        for name,entry in freeze["prerequisite_reports"].items():
            if sha(entry["path"])!=entry["sha256"]:raise ValueError("HOLDOUT_PREREQUISITE_REPORT_CHANGED:"+name)
        from freeze_v15_evaluation import freeze as validate_freeze
        prerequisites={name:json.loads(Path(entry["path"]).read_text(encoding="utf-8")) for name,entry in freeze["prerequisite_reports"].items()}
        clean=not subprocess.check_output(["git","status","--porcelain"],cwd=source,text=True).strip()
        validate_freeze(prerequisites,identity,clean)
        scans += [s for s in manifest["scans"] if s["split_role"] == "SEALED_WEAK_HOLDOUT"]
    args.output.mkdir(parents=True, exist_ok=True)
    trace = dict(dataset_manifest_hash=config_file_hash(manifest_path), manifest_file_sha256=sha(manifest_path), config_hash=config_file_hash(config_path),
                 config_hash_algorithm="JSON_CANONICAL_SORTED_V1",
                 algorithm_fingerprint=identity["algorithm_fingerprint"], executable_sha256=sha(args.executable),
                 git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
                 worktree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()),
                 holdout_accessed=False, access_purpose="DEVELOPMENT_ALGORITHM_SCORING")
    records = []
    held_access=False
    for scan in scans:
        directory = args.output/scan["scan_id"]
        directory.mkdir(exist_ok=True)
        record = dict(scan_id=scan["scan_id"], annotation_count=len(scan["annotation_files"]),
                      execution_status="FAILED", frame_resolved=False, deck_resolved=False,
                      consistent_unique_association=False, hatch_count=0, nondeck_primitive_count=0)
        start = time.monotonic()
        try:
            if scan["split_role"]=="SEALED_WEAK_HOLDOUT" and not held_access:
                development_records=records[:9]
                if operational(development_records,config["evaluation"])["REAL_DEVELOPMENT_OPERATIONAL_GATE"]!="PASS":
                    raise ValueError("HOLDOUT_NOT_ACCESSED_DEVELOPMENT_GATE_FAILED")
                receipt=append_access(DEFAULT_LEDGER,"ALGORITHM_SCORING",identity)
                write(args.output/"holdout_access_receipt.json",dict(ledger_path=str(DEFAULT_LEDGER),entry=receipt))
                held_access=True;trace["holdout_accessed"]=True;trace["access_purpose"]="FULL_REAL_ALGORITHM_SCORING"
            pcd = args.data_root/scan["pcd_path"]
            if sha(pcd) != scan["pcd_sha256"]:
                raise ValueError("PCD_FILE_HASH_MISMATCH")
            points, header = decode(pcd)
            actual = digest(points)
            if any(actual[k] != scan[k] for k in actual):
                raise ValueError("CANONICAL_XYZ_MISMATCH")
            write(directory/"decoder.json", dict(**actual, header=header))
            cache = directory/"input.xyzbin"
            write_cache(points, cache)
            completed = subprocess.run([str(args.executable.resolve()), "offline", str(cache), str(directory/"model.json")],
                                       capture_output=True, text=True, timeout=600)
            (directory/"process.log").write_text(completed.stdout+completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise ValueError("PRODUCT_EXIT_"+str(completed.returncode))
            model = json.loads((directory/"model.json").read_text(encoding="utf-8"))
            if model["config_hash"] != trace["config_hash"]:
                raise ValueError("STALE_CONFIG_BINARY")
            write_artifacts(directory/"model.json",cache,"offline",product_trace=dict(source_dataset_id=scan['dataset_id'],
                dataset_manifest_hash=trace['dataset_manifest_hash'],input_pcd_sha256=scan['pcd_sha256']))
            annotations, evaluations = [], []
            for relative, expected in zip(scan["annotation_files"], scan["annotation_sha256"]):
                path = args.data_root/relative
                if sha(path) != expected:
                    raise ValueError("ANNOTATION_HASH_MISMATCH")
                content = json.loads(path.read_text(encoding="utf-8"))
                if len(content["annotations"]) != 1:
                    raise ValueError("PARTIAL_ANNOTATION_CARDINALITY")
                annotation = content["annotations"][0]
                result = evaluate_annotation(model, annotation, config["evaluation"])
                result.update(annotation_path=relative, annotation_sha256=expected)
                annotations.append(annotation)
                evaluations.append(result)
            write(directory/"evaluation.json", evaluations)
            write_forensic(directory, model, annotations, evaluations, points)
            differences=[dict(a=i,b=j,**disagreement(annotations[i],annotations[j])) for i,j in itertools.combinations(range(len(annotations)),2)]
            write(directory/"disagreement.json",differences)
            write_csv(directory/"evaluation.csv",['annotation_path','annotation_sha256','annotation_status','candidate_id','candidate_status','association_score','observed_P95_m','inferred_P95_m'],
                [[e['annotation_path'],e['annotation_sha256'],e['status'],r.get('candidate_id'),r.get('status'),r.get('association_score'),r.get('observed_error',{}).get('P95'),r.get('inferred_error',{}).get('P95')]
                 for e in evaluations for r in (e['candidates'] or [{}])])
            write_csv(directory/"label_disagreement.csv",['annotation_a','annotation_b','classification','x0_m','x1_m','y0_m','y1_m','corner_P95_m'],
                [[d['a'],d['b'],d['classification'],d['x0'],d['x1'],d['y0'],d['y1'],d['corner']['P95']] for d in differences])
            good = [e["status"] == "MATCHED" and not e["local_duplicate_count"] and
                    not any(c["status"] == "LOCAL_ASSOCIATION_UNCERTAIN" for c in e["candidates"])
                    for e in evaluations]
            consistent = bool(evaluations) and all(good) and len({e["matched_candidate_id"] for e in evaluations}) == 1
            record.update(execution_status="COMPLETED", frame_resolved=model["frame_resolved"],
                          deck_resolved=model["deck"]["valid"], hatch_count=len(model["hatches"]),
                          nondeck_primitive_count=len(model["structures"]), consistent_unique_association=consistent,
                          warnings=model["warnings"])
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            record["failure_reason"] = str(error)
        record["elapsed_s"] = time.monotonic()-start
        records.append(record)
        write(directory/"status.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    if args.smoke:
        passed=len(records)==2 and all(r['execution_status']=='COMPLETED' and r['frame_resolved'] and r['deck_resolved'] for r in records)
        summary=dict(**trace,profile='quick',QUICK_REAL_SMOKE='PASS' if passed else 'FAIL',
            REAL_DEVELOPMENT_OPERATIONAL_GATE='NOT_RUN_QUICK_IS_NOT_OPERATIONAL_ACCEPTANCE',scans=records)
        write(args.output/'report.json',summary)
        return 0 if passed else 1
    summary = dict(**trace, profile='full', **operational(records[:9], config["evaluation"]), scans=records)
    if args.include_sealed_holdout:
        complete=len(records)==13 and sum(r["annotation_count"] for r in records)==23 and all(r["execution_status"]=="COMPLETED" for r in records)
        summary.update(FULL_REAL_RUN_COMPLETENESS="PASS" if complete else "FAIL",
                       SEALED_WEAK_HOLDOUT_STATUS="SCORED" if held_access else "NOT_ACCESSED_PREREQUISITE_FAILED")
    write(args.output/"report.json", summary)
    print(json.dumps(summary["counts"], ensure_ascii=False))
    return 0 if summary["REAL_DEVELOPMENT_OPERATIONAL_GATE"] == "PASS" and summary["RUN_COMPLETENESS"] == "PASS" and summary.get("FULL_REAL_RUN_COMPLETENESS","PASS")=="PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
