"""结构 Synthetic 开发矩阵；完整版本放行还需其他各条 lane。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
from evaluation.v15_synthetic import fixture,evaluate,QUICK_SCENARIOS,FULL_SCENARIOS
from v15_pcd import write_cache
from v15_artifacts import write_artifacts
import hashlib
from v15_config_identity import config_file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("executable",type=Path);p.add_argument("mode",choices=["quick","full"]);p.add_argument("output",type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads((SOURCE/"config/v15.json").read_text());thresholds=dict(**config["evaluation"],high_quality_threshold=config["boundary"]["high_quality_threshold"])
    config_hash=config_file_hash(SOURCE/"config/v15.json")
    from v15_holdout_ledger import current_identity
    identity=current_identity()
    identity.update(executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                    worktree_dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=SOURCE,text=True).strip()))
    results=[]
    for scenario in QUICK_SCENARIOS if args.mode=="quick" else FULL_SCENARIOS:
        for seed in (42,) if args.mode=="quick" else (42,1337,2026):
            directory=args.output/f"{scenario}_{seed}";directory.mkdir(exist_ok=True)
            truth=fixture(scenario,seed);write_cache(truth.runtime,directory/"runtime.xyzbin")
            mode="ship-frame" if scenario=="MISSING_TWO_EDGES" else "offline"
            run=subprocess.run([str(args.executable.resolve()),mode,str(directory/"runtime.xyzbin"),str(directory/"model.json")],capture_output=True,text=True,timeout=600)
            (directory/"process.log").write_text(run.stdout+run.stderr,encoding="utf-8")
            if run.returncode:result=dict(status="FAIL",failures=["PRODUCT_EXECUTION"])
            else:
                try:
                    write_artifacts(directory/"model.json",directory/"runtime.xyzbin",mode,config_hash,product_trace=dict(source_dataset_id='V15_SYNTHETIC'))
                    result=evaluate(json.loads((directory/"model.json").read_text()),truth,thresholds)
                except (ValueError,KeyError) as error:result=dict(status="FAIL",failures=["ARTIFACT_CONTRACT:"+str(error)])
            result.update(scenario=scenario,seed=seed,input_path=mode);results.append(result)
            (directory/"evaluation.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
            print(scenario,seed,result["status"],result["failures"],flush=True)
    report=dict(status="PASS" if all(r["status"]=="PASS" for r in results) else "FAIL",mode=args.mode,results=results,**identity,
                scenarios=list(QUICK_SCENARIOS if args.mode=="quick" else FULL_SCENARIOS),seeds=[42] if args.mode=="quick" else [42,1337,2026],
                acceptance_coverage="DEVELOPMENT_SUBSET_NOT_FULL_V15_ACCEPTANCE")
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
