"""执行并审计 V1.4 矩阵；同一个后端必须通过全部必测场景/种子/通道。"""
import csv
import hashlib
import json
import math
import datetime
import platform
from pathlib import Path
import subprocess
import sys

NORMAL = ["NORMAL_6DOF", "TRANSLATION", "YAW", "ROLL_PITCH", "COMBINED", "NOISE",
          "LOW_DENSITY", "OCCLUSION", "TRANSIENT", "INITIAL_OFFSET", "INDEPENDENT_VIEW_SAMPLING"]
DIAGNOSTIC = ["FLAT_DECK", "TIME_DAMAGE", "EXTRINSIC_DAMAGE", "LOST", "REPEATED_HATCH", "MOTION_80MS"]
QUICK = ["NORMAL_6DOF", "NOISE", "TRANSIENT", "INITIAL_OFFSET", "INDEPENDENT_VIEW_SAMPLING"]
METHODS = ["GICP", "VGICP"]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(output, profile, config):
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if metadata["filtered"] or metadata["profile"] != profile or not metadata["gt_isolation"]:
        raise ValueError("验收入口被过滤或 GT 隔离未通过")
    thresholds = config["acceptance"]
    scenes = NORMAL + DIAGNOSTIC if profile == "full" else QUICK
    seeds = thresholds["seeds"] if profile == "full" else [42]
    frames = thresholds[profile + "_frames"]
    expected = {(scene, str(seed), method, lane) for scene in scenes for seed in seeds
                for method in METHODS for lane in ("closed_loop", "benchmark")}
    series = read_csv(output / "series.csv")
    actual = {(r["scenario"], r["seed"], r["method"], r["lane"]) for r in series}
    if actual != expected or len(series) != len(expected):
        raise ValueError("矩阵缺失或存在重复证据")
    quality = read_csv(output / "quality.csv")
    for row in quality:
        for field in ("overlap", "rmse", "fitness", "raw_objective") + tuple("H%d%d" % (i, j) for i in range(6) for j in range(6)):
            if not math.isfinite(float(row[field])):
                raise ValueError("输出包含非有限数值")
        if row["valid"] == "1" and (row["mathematical_failure"] == "1" or row["backend_executed"] != "1"):
            raise ValueError("非法结果被接受或实际后端调用证据缺失")
    comparison = {}
    for method in METHODS:
        failures = []
        for row in series:
            if row["method"] != method:
                continue
            if int(row["frames"]) != frames - 1:
                raise ValueError("帧数不完整")
            if row["scenario"] not in NORMAL:
                continue
            checks = {
                "initialized": row["initialized"] == "1",
                "backend_calls": int(row["backend_calls"]) == frames - 1,
                "valid_ratio": float(row["valid_ratio"]) >= thresholds["min_valid_ratio"],
                "max_consecutive_failures": int(row["max_consecutive_failures"]) <= thresholds["max_consecutive_failures"],
                "translation_p95": float(row["translation_p95_m"]) <= thresholds["translation_p95_m"],
                "rotation_p95": float(row["rotation_p95_deg"]) <= thresholds["rotation_p95_deg"],
                "last_valid": row["last_valid"] == "1",
                "last_translation": float(row["last_translation_m"]) <= thresholds["translation_p95_m"],
                "last_rotation": float(row["last_rotation_deg"]) <= thresholds["rotation_p95_deg"],
            }
            if row["lane"] == "closed_loop":
                checks.update(static_samples=int(row["static_samples"]) > 0,
                              static_overall=float(row["static_p95_m"]) <= thresholds["static_p95_m"],
                              static_each_frame=float(row["worst_frame_static_p95_m"]) <= thresholds["static_p95_m"])
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                failures.append(dict(scenario=row["scenario"], seed=int(row["seed"]), lane=row["lane"], failed=failed))
        comparison[method] = dict(qualified=not failures, failures=failures)
    poses = read_csv(output / "pose.csv")
    pose_keys = {(r["scenario"], r["seed"], r["frame"], r["method"], r["lane"]) for r in poses}
    wanted = {(scene, str(seed), str(frame), method, lane) for scene in scenes for seed in seeds for method in METHODS
              for lane in ("closed_loop", "benchmark") for frame in range(0 if lane == "closed_loop" else 1, frames)}
    if not wanted.issubset(pose_keys) or len(pose_keys) != len(poses):
        raise ValueError("逐帧证据缺失或重复")
    keys = ("scenario", "seed", "frame", "method", "lane")
    quality_by_key = {tuple(r[k] for k in keys): r for r in quality}
    if not wanted.issubset(quality_by_key) or len(quality_by_key) != len(quality):
        raise ValueError("逐帧质量证据缺失或重复")
    pcl_quality = {key: row for key, row in quality_by_key.items() if key[3] == "PCL_GICP" or key[4] == "pcl_comparison"}
    pcl_pose_keys = {key for key in pose_keys if key[3] == "PCL_GICP" or key[4] == "pcl_comparison"}
    pcl_expected = {(scene, str(seed), "1", "PCL_GICP", "pcl_comparison") for scene in scenes for seed in seeds} if metadata.get("pcl_available") else set()
    if set(pcl_quality) != pcl_expected or pcl_pose_keys != pcl_expected:
        raise ValueError("PCL 对照结果缺失或与构建可用性不一致")
    for row in pcl_quality.values():
        if row["backend_executed"] != "1" or row["mathematical_failure"] != "0":
            raise ValueError("PCL 对照未真实执行或出现数学非法输出")
    for scene in scenes:
        for seed in seeds:
            for frame in range(1, frames):
                requests = [quality_by_key[(scene, str(seed), str(frame), m, "benchmark")]["request_fingerprint"] for m in METHODS]
                if not requests[0] or requests[0] != requests[1]:
                    raise ValueError("benchmark 后端输入不一致")
    poses_by_key = {tuple(r[k] for k in keys): r for r in poses}
    timing = read_csv(output / "timing.csv")
    timing_keys = {tuple(r[k] for k in keys) for r in timing}
    if not wanted.issubset(timing_keys) or len(timing) != len(timing_keys):
        raise ValueError("逐阶段耗时证据缺失或重复")
    for row in timing:
        if any(not math.isfinite(float(row[k])) or float(row[k]) < 0 for k in ("prepare_ms", "registration_ms", "map_ms", "total_ms")):
            raise ValueError("非法耗时指标")
    maps = read_csv(output / "map.csv")
    map_by_key = {tuple(r[k] for k in keys): r for r in maps}
    map_keys = {key for key in wanted if key[-1] == "closed_loop"}
    if set(map_by_key) != map_keys or len(maps) != len(map_keys):
        raise ValueError("地图证据缺失或重复")
    for scene in scenes:
        for seed in seeds:
            for method in METHODS:
                for frame in range(1, frames):
                    key = (scene, str(seed), str(frame), method, "closed_loop")
                    row, previous = map_by_key[key], map_by_key[(scene, str(seed), str(frame-1), method, "closed_loop")]
                    if row["tracking_target_revision"] != previous["tracking_map_revision"]:
                        raise ValueError("当前帧没有使用上一帧提交的不可变目标")
                    if poses_by_key[key]["valid"] == "0":
                        if any(row[k] != previous[k] for k in ("tracking_map_revision", "candidate_revision", "active", "candidates", "quarantined", "suspect", "stable")):
                            raise ValueError("拒绝帧污染地图")
    for row in poses:
        if row["scenario"] == "INDEPENDENT_VIEW_SAMPLING" and int(row["frame"]) > 0:
            if not thresholds["view_change_min"] <= float(row["view_change_ratio"]) <= thresholds["view_change_max"]:
                raise ValueError("独立视角变化率不合格")
        if row["valid"] == "0" and any(row[k] for k in ("static_p50_m", "static_p95_m", "static_max_m")):
            raise ValueError("无效帧伪造静态距离")
    qualified = [method for method in METHODS if comparison[method]["qualified"]]
    return metadata, comparison, qualified


def main():
    executable, profile, destination = sys.argv[1:]
    if profile not in ("quick", "full"):
        raise ValueError("未知验收级别")
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    config_path = root / "ship_perception/config/v14.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"], text=True)
    report = dict(gate="G5", schema="ship_perception.v14.report", implementation_sha=sha, profile=profile,
                  worktree="DIRTY" if status.strip() else "CLEAN", config_hash=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  site_status="SITE_PENDING", status="FAIL", qualified_backends=[],
                  timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), environment=platform.platform())
    try:
        with (output / "execution.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([str(Path(executable).resolve()), profile, str(output)], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("验收程序失败，见 execution.log")
        metadata, comparison, qualified = audit(output, profile, config)
        if metadata["config_hash"] != report["config_hash"] or metadata["compiled_sha"] != sha:
            raise ValueError("程序编译配置或 SHA 不匹配，请重新配置构建")
        report.update(metadata=metadata, backend_comparison=comparison, qualified_backends=qualified,
                      status="PASS_SYNTHETIC" if qualified else "FAIL", g5_scope="REGISTRATION_AND_CLOSED_LOOP")
    except (ValueError, RuntimeError, KeyError) as error:
        report["error"] = str(error)
    after_status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"], text=True)
    if after_status.strip():
        report["worktree"] = "DIRTY"
    dependencies = Path(executable).resolve().parent / "build_dependencies.txt"
    if not dependencies.exists():
        dependencies = dependencies.parent.parent / "build_dependencies.txt"
    report["build_dependencies"] = dependencies.read_text(encoding="utf-8") if dependencies.exists() else "MISSING"
    report["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())
                                 if p.is_file() and p.name != "report.json"}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "profile", "qualified_backends")}, ensure_ascii=False))
    return 0 if report["status"] == "PASS_SYNTHETIC" else 1


if __name__ == "__main__":
    sys.exit(main())
