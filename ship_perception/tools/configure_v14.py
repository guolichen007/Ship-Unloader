"""V1.4 配置：独立 schema、无运行时隐式参数、原始字节 SHA256。"""
import hashlib
import json
import math
import pathlib
import sys


def validate(value, rule, path="config"):
    typ = rule["type"]
    allowed = {"object": (dict,), "array": (list,), "integer": (int,), "number": (int, float), "string": (str,)}
    if type(value) not in allowed[typ]:
        raise ValueError("配置类型错误: " + path)
    if "const" in rule and value != rule["const"]:
        raise ValueError("配置版本不支持: " + path)
    if typ == "object":
        if set(value) != set(rule["required"]):
            raise ValueError("配置字段不匹配: " + path)
        for key, item in value.items():
            validate(item, rule["properties"][key], path + "." + key)
    elif typ == "array":
        if not rule["minItems"] <= len(value) <= rule["maxItems"]:
            raise ValueError("配置数组长度错误: " + path)
        for item in value:
            validate(item, rule["items"], path)
    elif typ in ("integer", "number"):
        if not math.isfinite(value) or value < rule.get("minimum", -1e20) or value > rule.get("maximum", 1e20):
            raise ValueError("配置超出范围: " + path)


def generate(path, output):
    raw = pathlib.Path(path).read_bytes()
    data = json.loads(raw)
    schema = json.loads((pathlib.Path(__file__).parents[1] / "config/v14.schema.json").read_text(encoding="utf-8"))
    validate(data, schema)
    m = data["tracking_map"]
    if not m["suspect_min_conflict_frames"] < m["quarantine_min_conflict_frames"] <= m["conflict_window_frames"]:
        raise ValueError("冲突窗口与隔离门槛不一致")
    lines = ["#pragma once", "#include <cstdint>", "#include <vector>", "namespace ship { namespace v14 {", "struct Config {"]
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        lines.append("  struct " + section.title().replace("_", "") + " {")
        for key, value in values.items():
            rule = schema["properties"][section]["properties"][key]
            if isinstance(value, list):
                typ = "std::uint32_t" if rule["items"]["type"] == "integer" else "double"
                lines.append("    std::vector<%s> %s{%s};" % (typ, key, ",".join(map(str, value))))
            else:
                typ = "std::int64_t" if rule["type"] == "integer" else "double"
                lines.append("    %s %s = %s;" % (typ, key, value))
        lines.append("  } " + section + ";")
    lines += ["};", 'constexpr const char* v14_config_hash = "%s";' % hashlib.sha256(raw).hexdigest(), "}}"]
    pathlib.Path(output).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


if __name__ == "__main__":
    generate(*sys.argv[1:])
