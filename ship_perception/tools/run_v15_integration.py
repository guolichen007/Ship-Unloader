"""两个真实后端分别闭环；同一个后端必须覆盖全部场景和种子。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
from evaluation.v15_synthetic import fixture,evaluate
from v15_pcd import write_cache
from v15_artifacts import write_artifacts,file_hash
from v15_config_identity import config_hash


def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable",type=Path);parser.add_argument("profile",choices=["quick","full"]);parser.add_argument("output",type=Path)
    parser.add_argument("--rescore-existing",action="store_true",help="按原配置重评已有位姿/模型产物；不会宣称重新执行后端")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    original_report=json.loads((args.output/"report.json").read_text()) if args.rescore_existing else None
    hash_algorithm=original_report.get("config_hash_algorithm","LEGACY_RAW_FILE") if original_report else "JSON_CANONICAL_SORTED_V1"
    identity=lambda data:config_hash(data) if hash_algorithm=="JSON_CANONICAL_SORTED_V1" else hashlib.sha256(data).hexdigest()
    config_bytes=(SOURCE/"config/v15.json").read_bytes()
    if original_report:
        snapshot=args.output/"v15_config_snapshot.json"
        config_bytes=snapshot.read_bytes() if snapshot.exists() else subprocess.check_output(["git","show",original_report["compiled_sha"]+":ship_perception/config/v15.json"],cwd=SOURCE)
        if identity(config_bytes)!=original_report["config_hash"] and hash_algorithm=="LEGACY_RAW_FILE":
            # Git stores LF while the old Windows development run hashed its
            # CRLF working-tree bytes. Only an exact recorded digest can match.
            config_bytes=config_bytes.replace(b"\r\n",b"\n").replace(b"\n",b"\r\n")
        if identity(config_bytes)!=original_report["config_hash"]:raise ValueError("ORIGINAL_CONFIG_UNAVAILABLE")
    (args.output/"v15_config_snapshot.json").write_bytes(config_bytes)
    config=json.loads(config_bytes)
    tracking=json.loads((SOURCE/"config/v14.json").read_text())
    thresholds=dict(**config["evaluation"],high_quality_threshold=config["boundary"]["high_quality_threshold"])
    frames=config["evaluation"][args.profile+"_frames"];seeds=[42] if args.profile=="quick" else [42,1337,2026]
    report=dict(schema="ship_perception.v15.integration_report.1",profile=args.profile,status="FAIL",
                compiled_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=SOURCE,text=True).strip(),
                config_hash=identity(config_bytes),config_hash_algorithm=hash_algorithm,
                v14_config_hash=hashlib.sha256((SOURCE/"config/v14.json").read_bytes()).hexdigest(),
                worktree_dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=SOURCE,text=True).strip()),
                dynamic_points_filtered=False,offline_rebootstrap=False,reused_binary_outputs=args.rescore_existing,
                executable_sha256=file_hash(args.executable),series=[],qualified_backends=[])
    if original_report:
        report["compiled_sha"]=original_report["compiled_sha"]
        report["original_report_sha256"]=file_hash(args.output/"report.json")
        report["original_run_worktree_dirty"]=original_report["worktree_dirty"]
    for seed in seeds:
        reference=fixture("MULTI_HATCH_BEAM",seed);reference_file=args.output/f"reference_{seed}.xyzbin";write_cache(reference.runtime,reference_file)
        for scene in ("CLEAN","DYNAMIC"):
            for method in ("GICP","VGICP"):
                directory=args.output/f"{scene}_{seed}_{method}";directory.mkdir(exist_ok=True)
                result=dict(scene=scene,seed=seed,method=method,status="FAIL")
                try:
                    if not args.rescore_existing:
                        with (directory/"execution.log").open("w",encoding="utf-8") as log:
                            process=subprocess.run([str(args.executable.resolve()),str(reference_file),method,str(frames),str(seed),scene,str(directory),directory.name],stdout=log,stderr=subprocess.STDOUT,timeout=3600)
                        if process.returncode:raise ValueError("INTEGRATION_EXECUTION_FAILED")
                    meta=json.loads((directory/"integration.json").read_text())
                    model=json.loads((directory/"model.json").read_text())
                    write_artifacts(directory/"model.json",directory/"accumulated.xyzbin","V14_ESTIMATED_SHIP_FRAME",report["config_hash"])
                    rows=list(csv.DictReader((directory/"poses.csv").open(encoding="utf-8")))
                    if len(rows)!=frames or [int(r["frame"]) for r in rows]!=list(range(frames)):raise ValueError("FRAME_COMPLETENESS")
                    if meta["compiled_sha"]!=report["compiled_sha"] or model["config_hash"]!=report["config_hash"]:raise ValueError("STALE_BINARY")
                    if model["software_git_sha"]!=meta["compiled_sha"] or (meta["method"],meta["scene"],meta["frames"],meta["seed"])!=(method,scene,frames,seed):raise ValueError("SERIES_PROVENANCE_MISMATCH")
                    if meta["dynamic_points_filtered"] or meta["offline_rebootstrap"]:raise ValueError("INTEGRATION_CONTRACT")
                    truth=fixture("MULTI_HATCH_BEAM",seed);truth.T_input_fixture=np.array(meta["T_B_S_initial"])
                    geometry=evaluate(model,truth,thresholds)
                    errors=[float(r["translation_error_m"]) for r in rows[1:] if r["valid"]=="1"]
                    angles=[float(r["rotation_error_deg"]) for r in rows[1:] if r["valid"]=="1"]
                    valid_ratio=len(errors)/(frames-1)
                    required=tracking["acceptance"]
                    tracking_ok=(meta["gauge_initialized"] and meta["backend_calls"]==frames-1 and
                                 valid_ratio>=required["min_valid_ratio"] and meta["max_consecutive_failures"]<=required["max_consecutive_failures"] and
                                 bool(errors) and np.isfinite(errors+angles).all() and
                                 np.quantile(errors,.95)<=required["translation_p95_m"] and np.quantile(angles,.95)<=required["rotation_p95_deg"] and
                                 rows[-1]["valid"]=="1" and errors[-1]<=required["translation_p95_m"] and angles[-1]<=required["rotation_p95_deg"])
                    for row in rows:
                        matrix=np.array([[float(row[f"T_B_C_{r}{c}"]) for c in range(4)] for r in range(4)])
                        if not np.isfinite(matrix).all():raise ValueError("NONFINITE_POSE")
                        rotation=matrix[:3,:3]
                        if (not np.allclose(matrix[3],[0,0,0,1],atol=1e-9,rtol=0) or
                            abs(np.linalg.det(rotation)-1)>1e-6 or np.linalg.norm(rotation.T@rotation-np.eye(3))>1e-6):
                            raise ValueError("ILLEGAL_SE3_POSE")
                        if row["valid"]=="0" and (row["translation_error_m"] or row["rotation_error_deg"]):raise ValueError("INVALID_FRAME_FAKE_ERROR")
                    result.update(status="PASS" if tracking_ok and geometry["status"]=="PASS" else "FAIL",
                                  tracking_gate="PASS" if tracking_ok else "FAIL",valid_ratio=valid_ratio,
                                  translation_P95_m=float(np.quantile(errors,.95)) if errors else None,
                                  rotation_P95_deg=float(np.quantile(angles,.95)) if angles else None,geometry=geometry,metadata=meta)
                except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:result["error"]=str(error)
                write(directory/"evaluation.json",result);report["series"].append(result)
                print(scene,seed,method,result["status"],flush=True)
    for method in ("GICP","VGICP"):
        series=[r for r in report["series"] if r["method"]==method]
        if len(series)==2*len(seeds) and all(r["status"]=="PASS" for r in series):report["qualified_backends"].append(method)
    integrity=(len(report["series"])==4*len(seeds) and all("error" not in r for r in report["series"]))
    report["both_backend_execution_integrity"]="PASS" if integrity else "FAIL"
    if report["qualified_backends"] and integrity:report["status"]="PASS"
    write(args.output/"report.json",report)
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
