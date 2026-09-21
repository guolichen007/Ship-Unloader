"""固定产品输入、修改私有评分通道，验证产品输出不受隐藏真值影响。"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
from evaluation.v15_synthetic import fixture,evaluate
from v15_pcd import write_cache
from v15_artifacts import dump,validate_model,file_hash


def same_output(producer, private_file):
    outputs=[]
    for truth in ({"GT_pose":[0,0,0],"dynamic_labels":[False],"source_indices":[1]},
                  {"GT_pose":[400,-900,200],"dynamic_labels":[True],"source_indices":[999]}):
        dump(private_file,truth);outputs.append(producer())
    return outputs[0]==outputs[1]


def run(executable, directory):
    directory.mkdir(parents=True,exist_ok=True)
    truth=fixture("NONRECTANGULAR",42);runtime=directory/"runtime.xyzbin";private=directory/"scorer-private.json"
    write_cache(truth.runtime,runtime);original_hash=file_hash(runtime)
    model_path=directory/"model.json"
    def product():
        result=subprocess.run([str(executable.resolve()),"offline",str(runtime.resolve()),str(model_path.resolve())],capture_output=True,text=True,timeout=600)
        if result.returncode:raise ValueError("PRODUCT_FAILED:"+result.stderr)
        model=json.loads(model_path.read_text());validate_model(model);return model
    isolated=same_output(product,private)
    # A deliberately cheating producer must fail this test; geometrically
    # perfect GT output alone is never sufficient acceptance evidence.
    cheat_detected=not same_output(lambda:json.loads(private.read_text()),private)
    baseline=product()
    runtime_unchanged=original_hash==file_hash(runtime)
    shuffled=truth.runtime[np.random.default_rng(1337).permutation(len(truth.runtime))]
    write_cache(shuffled,runtime);after_shuffle=product()
    config=json.loads((SOURCE/"config/v15.json").read_text())
    thresholds=dict(**config["evaluation"],high_quality_threshold=config["boundary"]["high_quality_threshold"])
    before=evaluate(baseline,truth,thresholds);after=evaluate(after_shuffle,truth,thresholds)
    passed=isolated and cheat_detected and runtime_unchanged and before["status"]==after["status"]=="PASS"
    report=dict(status="PASS" if passed else "FAIL",GT_CHANNEL_ISOLATION=isolated,
                CHEATING_PRODUCER_REJECTED=cheat_detected,RUNTIME_INPUT_UNCHANGED=runtime_unchanged,
                SHUFFLE_GEOMETRY_GATE=after,baseline_geometry=before,input_sha256=original_hash,
                config_hash=baseline["config_hash"],software_git_sha=baseline["software_git_sha"])
    dump(directory/"isolation_report.json",report)
    print(json.dumps(report,ensure_ascii=False))
    return 0 if passed else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("executable",type=Path);parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.output:raise SystemExit(run(args.executable,args.output))
    with tempfile.TemporaryDirectory(prefix="v15-isolation-") as directory:raise SystemExit(run(args.executable,Path(directory)))
