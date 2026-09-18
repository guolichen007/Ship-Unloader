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


def generate(config_path, output_path, git_sha):
    raw = pathlib.Path(config_path).read_bytes()
    c = json.loads(raw)
    reference = json.loads((pathlib.Path(__file__).parents[1] / "config/m0.json").read_text())
    if set(c) != set(reference):
        raise ValueError("configuration keys do not match schema version 1")
    if c["schema"] != "ship_perception.m0" or c["version"] != 1:
        raise ValueError("unsupported schema/version")
    if c["mode"] != "EVALUATION_MODE":
        raise ValueError("M0 gate configuration must use EVALUATION_MODE")
    for key, value in c.items():
        if type(value) is not type(reference[key]):
            raise ValueError("incorrect type: " + key)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite: " + key)
        if isinstance(value, list) and (len(value) != len(reference[key]) or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
            raise ValueError("invalid vector: " + key)
        if isinstance(value, str) and not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError("unsafe identifier: " + key)
    positive = ["frames", "frame_period_ns", "scan_duration_ns", "frequency_hz",
                "decimation", "queue_capacity", "latency_samples", "math_tolerance",
                "point_tolerance_m", "noisy_rms_limit_m", "fault_min_rms_m",
                "local_precision_limit_m", "absolute_float_min_loss_m"]
    for key in positive:
        if c[key] <= 0:
            raise ValueError("must be positive: " + key)
    for key in ("drop_probability", "outlier_probability"):
        if not 0 <= c[key] <= 1:
            raise ValueError("probability outside [0,1]: " + key)
    for key in ("seed", "noise_sigma_m", "timestamp_jitter_ns", "dynamic_points",
                "outlier_scale_m", "smooth_random_m", "heave_m"):
        if c[key] < 0:
            raise ValueError("must be nonnegative: " + key)
    if not re.fullmatch(r"[a-f0-9]{40}|UNCOMMITTED", git_sha):
        raise ValueError("invalid git SHA")
    lines = ["#pragma once", "#include <cstdint>", "namespace ship { namespace cfg {"]
    for key, value in c.items():
        if isinstance(value, str):
            lines.append("constexpr const char* %s = %s;" % (key, json.dumps(value)))
        elif isinstance(value, list):
            lines.append("constexpr double %s[] = {%s};" % (key, ",".join(map(str, value))))
        else:
            typ = "std::int64_t" if isinstance(value, int) else "double"
            lines.append("constexpr %s %s = %s;" % (typ, key, value))
    lines += ['constexpr const char* config_hash = "%s";' % hashlib.sha256(raw).hexdigest(),
              'constexpr const char* git_sha = "%s";' % git_sha, "}}"]
    pathlib.Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate(*sys.argv[1:])
