"""Validate the versioned configuration and generate immutable build metadata.

M0 configuration is selected at CMake configure time; no hidden runtime overrides.
The SHA-256 is over the exact JSON bytes (including thresholds).
"""
import hashlib
import json
import math
import pathlib
import re
import sys


def validate_config(c):
    schema = json.loads((pathlib.Path(__file__).parents[1] / "config/m0.schema.json").read_text())
    if not isinstance(c, dict) or set(c) != set(schema["required"]):
        raise ValueError("configuration keys do not match schema version 1")
    for key, value in c.items():
        rule = schema["properties"][key]
        allowed = {"string": (str,), "array": (list,), "integer": (int,), "number": (int, float)}
        if type(value) not in allowed[rule["type"]]:
            raise ValueError("incorrect type: " + key)
        if "const" in rule and value != rule["const"]:
            raise ValueError("unsupported schema/version/mode: " + key)
        if type(value) in (int, float):
            if not math.isfinite(value):
                raise ValueError("nonfinite: " + key)
            if "minimum" in rule and value < rule["minimum"]:
                raise ValueError("below minimum: " + key)
            if "maximum" in rule and value > rule["maximum"]:
                raise ValueError("above maximum: " + key)
            if "exclusiveMinimum" in rule and value <= rule["exclusiveMinimum"]:
                raise ValueError("not positive: " + key)
        if isinstance(value, list) and (len(value) != rule["minItems"] or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
            raise ValueError("invalid vector: " + key)
        if isinstance(value, str) and not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError("unsafe identifier: " + key)
    if any(lo > hi for lo, hi in zip(c["occlusion_min_m"], c["occlusion_max_m"])):
        raise ValueError("inverted occlusion ROI")
    # Remaining integer parameters must also fit the generated int64 constants.
    if any(not -(2**63) <= v < 2**63 for v in c.values() if type(v) is int):
        raise ValueError("integer outside int64 range")


def generate(config_path, output_path, git_sha):
    raw = pathlib.Path(config_path).read_bytes()
    c = json.loads(raw)
    validate_config(c)
    properties = json.loads((pathlib.Path(__file__).parents[1] / "config/m0.schema.json").read_text())["properties"]
    if not re.fullmatch(r"[a-f0-9]{40}|UNCOMMITTED", git_sha):
        raise ValueError("invalid git SHA")
    lines = ["#pragma once", "#include <cstdint>", "namespace ship { namespace cfg {"]
    for key, value in c.items():
        if isinstance(value, str):
            lines.append("constexpr const char* %s = %s;" % (key, json.dumps(value)))
        elif isinstance(value, list):
            lines.append("constexpr double %s[] = {%s};" % (key, ",".join(map(str, value))))
        else:
            # Schema number fields stay double even when JSON spells the value as 0.
            rule = properties[key]
            typ = "std::int64_t" if rule["type"] == "integer" else "double"
            lines.append("constexpr %s %s = %s;" % (typ, key, value))
    lines += ['constexpr const char* config_hash = "%s";' % hashlib.sha256(raw).hexdigest(),
              'constexpr const char* git_sha = "%s";' % git_sha, "}}"]
    pathlib.Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate(*sys.argv[1:])
