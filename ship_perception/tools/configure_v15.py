"""从显式 schema 生成 V1.5 独立配置；验收参数不属于 V1.4。"""
import hashlib
import json
import math
from pathlib import Path
import sys
from v15_config_identity import config_hash


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
    for section in ('geometry','frame','roi','boundary'):
        for key,value in data[section].items():
            if key.startswith('grid_phase_') or key.endswith('_weight') or key in ('max_inferred_edges','profile_lower_quantile'):continue
            if value<=0:raise ValueError('V15_NONPOSITIVE_GEOMETRY_PARAMETER:'+section+'.'+key)
    if not 0<=data['boundary']['max_inferred_edges']<=1:raise ValueError('V15_AT_MOST_ONE_INFERRED_EDGE')
    for section,key in (('geometry','planarity_min'),('geometry','normal_min_secondary_ratio'),('frame','candidate_score_gap'),
                        ('roi','min_enclosure_ratio'),('roi','deck_min_enclosure_ratio'),('roi','deck_patch_fill_min'),
                        ('roi','deck_normal_mode_support_ratio'),('boundary','profile_lower_quantile'),
                        ('boundary','min_edge_coverage'),('boundary','high_quality_threshold')):
        if not 0<=data[section][key]<=1:raise ValueError('V15_FRACTION_PARAMETER:'+section+'.'+key)
    # These are the approved version contract, not tuning knobs. Stronger
    # gates are allowed; a configuration edit cannot weaken them to get PASS.
    e=data['evaluation']
    maxima={'boundary_p95_m':.15,'corner_p95_m':.15,'deck_offset_max_m':.05,'deck_tilt_budget_m':.05,
            'deck_angle_max_deg':.25,'deck_distance_p95_m':.10}
    minima={'coverage_total_min':.9,'coverage_edge_min':.8,'real_frame_min':7,'real_deck_min':7,
            'real_joint_min':7,'real_association_min':6}
    if any(e[k]<=0 or e[k]>v for k,v in maxima.items()) or any(e[k]<v for k,v in minima.items()):
        raise ValueError('V15_FROZEN_ACCEPTANCE_WEAKENED')
    if e['quick_frames']!=50 or e['full_frames']!=200:raise ValueError('V15_FROZEN_FRAME_COUNTS')
    lines=["#pragma once", "#include <cstdint>", "namespace ship { namespace v15 {", "struct Config {"]
    for section,fields in data.items():
        if not isinstance(fields,dict):continue
        lines.append("struct "+section.title()+" {")
        for name,value in fields.items():
            lines.append(("std::int64_t" if type(value) is int else "double")+" "+name+" = "+str(value)+";")
        lines.append("} "+section+";")
    lines += ["};", 'constexpr const char* config_hash = "'+config_hash(raw)+'";', "}}"]
    Path(output).write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__": generate(*sys.argv[1:])
