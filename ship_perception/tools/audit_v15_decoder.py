"""PCD 编码交叉审计；只解码文件，不运行结构识别或查看 holdout 评分。"""
import argparse
import datetime
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
from v15_pcd import decode, digest
from v15_artifacts import dump, file_hash
from v15_config_identity import config_file_hash
from v15_holdout_ledger import append_access,DEFAULT_LEDGER


def read_cache(path):
    data=Path(path).read_bytes()
    if data[:8]!=b"SXYZV15\0" or len(data)<16:raise ValueError("INVALID_PCL_CACHE")
    count=struct.unpack("<Q",data[8:16])[0]
    if len(data)!=16+count*12:raise ValueError("INVALID_PCL_CACHE_LENGTH")
    return np.frombuffer(data,dtype="<f4",offset=16).reshape(-1,3)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--pcl-executable",type=Path);p.add_argument("--require-pcl",action="store_true")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[1];manifest_path=source/"datasets/v15_dataset_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    report=dict(access_purpose="FILE_CODEC_AUDIT_NOT_ALGORITHM_SCORING",holdout_algorithm_scored=False,
                dataset_manifest_hash=config_file_hash(manifest_path),manifest_file_sha256=file_hash(manifest_path),timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scans=[],internal_decoder_gate="FAIL",pcl_crosscheck="NOT_RUN",alias_0808_gate="FAIL")
    alias_checked=False
    for scan in manifest["scans"]:
        record=dict(scan_id=scan["scan_id"],status="FAIL",pcl="NOT_RUN")
        try:
            path=args.data_root/scan["pcd_path"]
            if file_hash(path)!=scan["pcd_sha256"]:raise ValueError("PCD_FILE_HASH")
            points,header=decode(path);canonical=digest(points)
            if any(canonical[k]!=scan[k] for k in canonical):raise ValueError("CANONICAL_XYZ_HASH")
            record.update(canonical=canonical,header=header,status="PASS")
            if scan["scan_id"]=="2026-08-08-02-33-57":
                record["encoding_variants"]=[]
                for alias in scan["aliases"]:
                    alias_path=args.data_root/alias["path"]
                    if alias_path.suffix.lower()!=".pcd":continue
                    alternate,alternate_header=decode(alias_path)
                    same=np.array_equal(points,alternate)
                    record["encoding_variants"].append(dict(path=alias["path"],raw_sha256=file_hash(alias_path),xyz_identical=same))
                    if not same:raise ValueError("ENCODING_VARIANT_XYZ_MISMATCH")
                    if file_hash(alias_path)!=scan["pcd_sha256"]:alias_checked=True
            if args.pcl_executable:
                output=args.output/(scan["scan_id"]+".pcl.xyzbin")
                process=subprocess.run([str(args.pcl_executable.resolve()),"pcd-cache",str(path.resolve()),str(output.resolve())],capture_output=True,text=True,timeout=600)
                (args.output/(scan["scan_id"]+".pcl.log")).write_text(process.stdout+process.stderr,encoding="utf-8")
                if process.returncode:raise ValueError("PCL_LOADER_FAILED")
                pcl=read_cache(output);pcl_digest=digest(pcl)
                if pcl_digest!=canonical or not np.array_equal(points,pcl):raise ValueError("PCL_INTERNAL_DECODER_MISMATCH")
                record["pcl"]="PASS"
        except (ValueError,OSError,KeyError,subprocess.SubprocessError) as error:
            record.update(status="FAIL",error=str(error))
        report["scans"].append(record)
    if len(report["scans"])==13 and all(r["status"]=="PASS" for r in report["scans"]):report["internal_decoder_gate"]="PASS"
    if alias_checked:report["alias_0808_gate"]="PASS"
    if args.pcl_executable:report["pcl_crosscheck"]="PASS" if all(r["pcl"]=="PASS" for r in report["scans"]) else "FAIL"
    passed=report["internal_decoder_gate"]==report["alias_0808_gate"]=="PASS" and (not args.require_pcl or report["pcl_crosscheck"]=="PASS")
    report["status"]="PASS" if passed else "FAIL";dump(args.output/"decoder_report.json",report)
    # All runs share a persistent dataset ledger. A new output directory must
    # never reset prior holdout exposure or contamination history.
    receipt=append_access(DEFAULT_LEDGER,report["access_purpose"])
    dump(args.output/"holdout_access_receipt.json",dict(ledger_path=str(DEFAULT_LEDGER),entry=receipt))
    print(json.dumps({k:v for k,v in report.items() if k!="scans"},ensure_ascii=False))
    return 0 if passed else 1


if __name__=="__main__":raise SystemExit(main())
