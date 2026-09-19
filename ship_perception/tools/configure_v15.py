"""从显式 schema 生成 V1.5 独立配置；验收参数不属于 V1.4。"""
import hashlib
import json
import math
from pathlib import Path
import sys


def generate(path, output):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    schema = json.loads((Path(__file__).parents[1]/"config/v15.schema.json").read_text(encoding="utf-8"))
    def check(value, rule):
        t = rule["type"]
        if t == "object":
            if type(value) is not dict or set(value) != set(rule["required"]):
                raise ValueError("V15_CONFIG_FIELDS")
            for k,v in value.items(): check(v,rule["properties"][k])
        elif t == "string":
            if value != rule["const"]: raise ValueError("V15_CONFIG_SCHEMA")
        else:
            if type(value) not in ((int,) if t == "integer" else (int,float)) or not math.isfinite(value):
                raise ValueError("V15_CONFIG_NUMBER")
            if not rule["minimum"] <= value <= rule["maximum"]: raise ValueError("V15_CONFIG_RANGE")
    check(data,schema)
    g=data["geometry"]
    if not 0<g["refine_voxel_m"]<=g["candidate_voxel_m"]<=g["coarse_voxel_m"]:
        raise ValueError("V15_RESOLUTION_ORDER")
    if not 3<=g["normal_min_points"]<=g["normal_k"]: raise ValueError("V15_NEIGHBORS")
    if abs(sum(v for k,v in data["roi"].items() if k.endswith("_weight"))-1)>1e-9:
        raise ValueError("V15_SCORE_WEIGHTS")
    lines=["#pragma once", "#include <cstdint>", "namespace ship { namespace v15 {", "struct Config {"]
    for section,fields in data.items():
        if not isinstance(fields,dict):continue
        lines.append("struct "+section.title()+" {")
        for name,value in fields.items():
            lines.append(("std::int64_t" if type(value) is int else "double")+" "+name+" = "+str(value)+";")
        lines.append("} "+section+";")
    lines += ["};", 'constexpr const char* config_hash = "'+hashlib.sha256(raw).hexdigest()+'";', "}}"]
    Path(output).write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__": generate(*sys.argv[1:])
