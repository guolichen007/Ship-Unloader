"""封存弱标注集访问记录；污染状态只升不降，文件审计与算法评分分开。"""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from v15_config_identity import canonical_bytes,config_file_hash
from v15_artifacts import file_hash

SOURCE=Path(__file__).resolve().parents[1]
DEFAULT_LEDGER=SOURCE.parent/"validation/v15_holdout_access.jsonl"


def algorithm_fingerprint():
    paths=list((SOURCE/"src/structure").glob("*"))+list((SOURCE/"include/ship_perception/structure").glob("*"))
    paths += [SOURCE/name for name in ("CMakeLists.txt","cmake/V15.cmake","tools/v15_main.cpp",
                                     "tools/v15_pcd.py","tools/configure_v15.py","tools/v15_config_identity.py")]
    digest=hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():digest.update(path.relative_to(SOURCE).as_posix().encode()+b"\0"+path.read_bytes().replace(b"\r\n",b"\n")+b"\0")
    return digest.hexdigest()


def current_identity():
    return dict(git_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=SOURCE,text=True).strip(),
                algorithm_fingerprint=algorithm_fingerprint(),config_hash=config_file_hash(SOURCE/"config/v15.json"),
                dataset_manifest_hash=config_file_hash(SOURCE/"datasets/v15_dataset_manifest.json"))


def read_ledger(path):
    if not Path(path).exists():return []
    result=[];previous="0"*64
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item=json.loads(line);digest=item.pop("entry_sha256")
        if item["previous_entry_sha256"]!=previous or hashlib.sha256(canonical_bytes(item)).hexdigest()!=digest:
            raise ValueError("HOLDOUT_LEDGER_INTEGRITY")
        item["entry_sha256"]=digest;result.append(item);previous=digest
    return result


def append_access(path,purpose,identity=None):
    if purpose not in ("FILE_CODEC_AUDIT_NOT_ALGORITHM_SCORING","ALGORITHM_SCORING","IDENTITY_CHECK"):
        raise ValueError("INVALID_HOLDOUT_ACCESS_PURPOSE")
    identity=current_identity() if identity is None else identity
    required={"git_sha","algorithm_fingerprint","config_hash","dataset_manifest_hash"}
    if set(identity)!=required:raise ValueError("HOLDOUT_IDENTITY_FIELDS")
    for value in identity.values():
        if not isinstance(value,str) or not value:raise ValueError("HOLDOUT_IDENTITY_VALUE")
    history=read_ledger(path)
    scored=[r for r in history if r["access_purpose"]=="ALGORITHM_SCORING"]
    contaminated=any(r["HOLDOUT_CONTAMINATED"]=="YES" for r in history)
    if scored and any(identity[key]!=scored[0][key] for key in ("algorithm_fingerprint","config_hash","dataset_manifest_hash")):
        contaminated=True
    entry=dict(**identity,timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               access_purpose=purpose,holdout_accessed=purpose!="IDENTITY_CHECK",
               holdout_algorithm_scored=bool(scored) or purpose=="ALGORITHM_SCORING",
               HOLDOUT_CONTAMINATED="YES" if contaminated else "NO",
               split_role="SEALED_WEAK_HOLDOUT",golden_candidate=True,physical_vessel_id="UNKNOWN",
               previous_entry_sha256=history[-1]["entry_sha256"] if history else "0"*64)
    entry["entry_sha256"]=hashlib.sha256(canonical_bytes(entry)).hexdigest()
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open("a",encoding="utf-8",newline="\n") as out:out.write(json.dumps(entry,ensure_ascii=False,sort_keys=True)+"\n")
    return entry
