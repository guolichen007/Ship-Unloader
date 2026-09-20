"""候选模型契约与可审计产物；只读取产品输出，不接触 GT 或 annotation。"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def schema_check(value, rule, location="model"):
    # The committed model schema uses this deliberately small JSON Schema
    # subset. Unknown keywords fail, so adding schema rules cannot silently
    # outrun the offline validator. No pip/network dependency on Ubuntu20.
    supported = {"$schema", "title", "type", "const", "enum", "anyOf", "properties", "required",
                 "additionalProperties", "items", "minItems", "maxItems", "minimum", "maximum", "pattern"}
    if set(rule)-supported: raise ValueError("UNIMPLEMENTED_SCHEMA_KEYWORD:"+location)
    if "anyOf" in rule:
        for option in rule["anyOf"]:
            try: schema_check(value, option, location); return
            except ValueError: pass
        raise ValueError("SCHEMA_UNION:"+location)
    if "const" in rule and (type(value) is not type(rule["const"]) or value != rule["const"]):
        raise ValueError("SCHEMA_CONST:"+location)
    t = rule.get("type")
    valid = {"object": type(value) is dict, "array": type(value) is list,
             "string": type(value) is str, "boolean": type(value) is bool, "null": value is None,
             "integer": type(value) is int, "number": type(value) in (int, float)}
    if t and not valid.get(t, False): raise ValueError("SCHEMA_TYPE:"+location)
    if t == "object":
        if set(rule["required"])-set(value): raise ValueError("SCHEMA_MISSING:"+location)
        if rule.get("additionalProperties") is False and set(value)-set(rule["properties"]):
            raise ValueError("SCHEMA_EXTRA:"+location)
        for key, item in value.items(): schema_check(item, rule["properties"][key], location+"."+key)
    elif t == "array":
        if not rule.get("minItems", 0) <= len(value) <= rule.get("maxItems", math.inf):
            raise ValueError("SCHEMA_LENGTH:"+location)
        for i, item in enumerate(value): schema_check(item, rule["items"], location+f"[{i}]")
    elif t in ("integer", "number"):
        if not math.isfinite(value) or not rule.get("minimum", -math.inf) <= value <= rule.get("maximum", math.inf):
            raise ValueError("SCHEMA_NUMBER:"+location)
    if "enum" in rule and value not in rule["enum"]: raise ValueError("SCHEMA_ENUM:"+location)
    if "pattern" in rule and not re.fullmatch(rule["pattern"], value): raise ValueError("SCHEMA_PATTERN:"+location)


def validate_model(model, expected_config_hash=None):
    schema_check(model, json.loads((SOURCE/"config/v15_model.schema.json").read_text(encoding="utf-8")))
    if expected_config_hash and model["config_hash"] != expected_config_hash: raise ValueError("MODEL_CONFIG_HASH")
    if model["frame_resolved"]:
        t = np.asarray(model["T_B_input"])
        if t.shape != (4,4) or not np.isfinite(t).all() or not np.allclose(t[3], [0,0,0,1], atol=1e-9, rtol=0):
            raise ValueError("MODEL_FRAME")
        r = t[:3,:3]
        if abs(np.linalg.det(r)-1)>1e-6 or np.linalg.norm(r.T@r-np.eye(3))>1e-6: raise ValueError("MODEL_SE3")
    elif model["T_B_input"] is not None or model["hatches"] or model["structures"]:
        raise ValueError("UNRESOLVED_FRAME_WITH_GEOMETRY")
    deck = model["deck"]
    if deck["valid"]:
        if deck["normal"] is None or abs(np.linalg.norm(deck["normal"])-1)>1e-6 or deck["offset"] is None:
            raise ValueError("DECK_PLANE")
    elif any(deck[k] is not None for k in ("normal", "offset", "residual_p50_m", "residual_p95_m")):
        raise ValueError("INVALID_DECK_FAKE_METRIC")
    ids = [h["candidate_id"] for h in model["hatches"]]
    if len(ids) != len(set(ids)): raise ValueError("DUPLICATE_LOCAL_ID")
    def boundary(edge):
        flags = edge["evidence_flags"]
        if flags & 1 and flags & 6: raise ValueError("OBSERVED_INFERRED_CONFLATION")
        if flags & 1 and (edge["visibility"] != "VISIBLE" or not edge["support_count"]):
            raise ValueError("OBSERVED_WITHOUT_SUPPORT")
        length = float(np.linalg.norm(np.array(edge["b"])-edge["a"]))
        end = 0.; total = 0.
        for lo, hi in edge["support_intervals"]:
            if not end-1e-8 <= lo <= hi <= length+1e-5: raise ValueError("SUPPORT_INTERVAL_ORDER")
            total += hi-lo; end = hi
        if abs(total-edge["observed_support_length"])>1e-5: raise ValueError("SUPPORT_LENGTH")
        for key in ("start", "end"):
            e = edge[key]
            if e["endpoint_bounded"] != (e["endpoint_uncertainty_m"] is not None): raise ValueError("ENDPOINT_BOUNDING")
    for hatch in model["hatches"]:
        for edge in hatch["boundaries"]: boundary(edge)
        observed = sum(bool(e["evidence_flags"]&1) and e["side"]=="INNER_OPENING_FACE" for e in hatch["boundaries"])
        inferred = sum(bool(e["evidence_flags"]&6) for e in hatch["boundaries"])
        if observed != hatch["observed_edge_count"] or inferred != hatch["inferred_edge_count"]:
            raise ValueError("HATCH_EVIDENCE_COUNTS")
        if hatch["status"] == "PARTIAL":
            if hatch["nominal_polygon"] is not None: raise ValueError("PARTIAL_FORCED_POLYGON")
        else:
            if not hatch["nominal_polygon"] or len(hatch["nominal_polygon"])<3 or inferred>1:
                raise ValueError("COMPLETE_TOPOLOGY")
            if hatch["status"]=="COMPLETE_OBSERVED" and (inferred or observed!=len(hatch["boundaries"])):
                raise ValueError("FALSE_COMPLETE_OBSERVED")
    for primitive in model["structures"]: boundary(primitive["segment"])
    return dict(schema="PASS", mathematical_health="PASS", evidence_consistency="PASS")


def write_csv(path, columns, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer=csv.writer(stream);writer.writerow(columns);writer.writerows(rows)


def write_artifacts(model_path, input_path, mode, expected_config_hash=None):
    model_path, input_path = Path(model_path), Path(input_path)
    directory = model_path.parent
    model = json.loads(model_path.read_text(encoding="utf-8"))
    audit = validate_model(model, expected_config_hash)
    dump(directory/"deck.json", model["deck"])
    write_csv(directory/"polygon.csv", ["candidate_id","status","vertex","x_B","y_B","z_B"],
              [[h["candidate_id"],h["status"],i,*p] for h in model["hatches"] for i,p in enumerate(h["nominal_polygon"] or [])])
    records = [(h["candidate_id"],str(i),e) for h in model["hatches"] for i,e in enumerate(h["boundaries"])]
    write_csv(directory/"boundaries.csv", ["candidate_id","edge","side","evidence_flags","mechanism","support_count","observed_length_m","extrapolation_m","start_bounded","start_uncertainty_m","end_bounded","end_uncertainty_m"],
              [[h,i,e["side"],e["evidence_flags"],e["evidence_mechanism"],e["support_count"],e["observed_support_length"],e["extrapolation_length"],e["start"]["endpoint_bounded"],e["start"]["endpoint_uncertainty_m"],e["end"]["endpoint_bounded"],e["end"]["endpoint_uncertainty_m"]] for h,i,e in records])
    write_csv(directory/"primitives.csv", ["primitive_index","kind","reason","side","flags","ax","ay","az","bx","by","bz"],
              [[i,p["kind"],p["reason"],p["segment"]["side"],p["segment"]["evidence_flags"],*p["segment"]["a"],*p["segment"]["b"]] for i,p in enumerate(model["structures"])])
    # These are rendered model primitives, NOT pointwise semantic truth.
    vertices=[]
    for _,_,edge in records:
        a,b=np.asarray(edge["a"]),np.asarray(edge["b"]);length=np.linalg.norm(b-a)
        intervals=edge["support_intervals"] if edge["evidence_flags"]&1 else [(0,length)]
        color=(40,180,80) if edge["evidence_flags"]&1 else (240,160,30)
        for lo,hi in intervals:
            for s in np.linspace(lo,hi,max(2,int((hi-lo)/.05)+1)):
                vertices.append([*(a+(b-a)*s/max(length,1e-12)),*color])
    with (directory/"debug_semantics.ply").open("w",encoding="ascii") as out:
        out.write("ply\nformat ascii 1.0\ncomment MODEL_PRIMITIVES_NOT_POINTWISE_LABELS\nelement vertex "+str(len(vertices))+"\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for vertex in vertices:out.write(" ".join(map(str,vertex))+"\n")
    timing_file=Path(str(model_path)+".timing.json")
    timing=json.loads(timing_file.read_text()) if timing_file.exists() else {"stage_timing_status":"NOT_RECORDED_IN_THIS_PRODUCER"}
    dump(directory/"timing.json",timing)
    metadata=dict(schema="ship_perception.v15.product_artifacts.1",software_git_sha=model["software_git_sha"],
                  config_hash=model["config_hash"],input_sha256=file_hash(input_path),input_mode=mode,
                  annotation_access=False,debug_semantics="MODEL_PRIMITIVES_NOT_POINTWISE_LABELS",
                  review_status="UNREVIEWED",control_ready=False,canonical=False,real_formal_15cm="PENDING_GOLDEN",site_accuracy="SITE_PENDING")
    dump(directory/"metadata.json",metadata)
    dump(directory/"artifact_audit.json",audit)
    (directory/"report.md").write_text("# 结构候选输出\n\n"+
        f"坐标解析：{model['frame_resolved']}；Deck：{model['deck']['valid']}；Hatch 候选：{len(model['hatches'])}。\n\n"+
        "模型未审查，不能用于控制，也未成为 Canonical Model。真实 15 cm 精度等待 Golden Final；现场状态为 SITE_PENDING。\n\n"+
        "debug_semantics.ply 仅显示候选边界的实际支持区间及推断段，不是逐点语义真值。评估数据与本产品元数据分离。\n",encoding="utf-8")
    names=[model_path.name,"deck.json","polygon.csv","boundaries.csv","primitives.csv","debug_semantics.ply","timing.json","metadata.json","artifact_audit.json","report.md"]
    dump(directory/"artifact_index.json",{name:file_hash(directory/name) for name in names})
    return audit
