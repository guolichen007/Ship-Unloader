#!/usr/bin/env python3
"""Clean-build an exact SHA on Ubuntu 20.04. Never edits product code or tags a baseline."""
import argparse
import datetime
import hashlib
import json
import math
import pathlib
import platform
import re
import subprocess
import sys
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECT = ROOT / "ship_perception"
GATES = ["g%d" % i for i in range(1, 8)]
RUN_ORDER = ["validation_boundary", "replay_selfcheck", "pcd_selfcheck"] + GATES
LAB_STATUS = dict(g1="PASS_LAB", g2="PASS_SYNTHETIC", g3="PASS_SYNTHETIC",
                  g4="PASS_SYNTHETIC", g5="PASS_LAB", g6="PASS_LAB", g7="PASS_LAB_LOCAL")


def capture(command, cwd=ROOT):
    try:
        r = subprocess.run(command, cwd=str(cwd), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True,
                           encoding="utf-8", errors="replace")
        return r.returncode, r.stdout
    except OSError as exc:
        return 127, str(exc) + "\n"


def git(*args):
    code, output = capture(["git"] + list(args))
    if code:
        raise RuntimeError(output)
    return output.strip()


def ubuntu20(os_release):
    values = {}
    for line in os_release.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            values[k] = v.strip('"')
    return values.get("ID") == "ubuntu" and values.get("VERSION_ID") == "20.04"


def read_build_dependencies(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    for key in ("CMAKE_COMMAND", "CMAKE_VERSION", "CXX_COMPILER", "CXX_COMPILER_VERSION",
                "EIGEN_VERSION", "EIGEN_CONFIG_DIR", "EIGEN_INCLUDE_DIRS",
                "PCL_VERSION", "PCL_CONFIG_DIR", "PCL_INCLUDE_DIRS"):
        if not values.get(key) or values[key].endswith("NOTFOUND"):
            raise ValueError("missing resolved build dependency: " + key)
    if values["PCL_VERSION"] == "DISABLED_DEVELOPER_ONLY":
        raise ValueError("formal validation requires PCL ON")
    return values


def check_artifact(data, gate, sha, config_hash):
    expected_name = gate.upper()
    if data.get("gate") != expected_name or data.get("git_sha") != sha:
        raise ValueError("artifact gate/SHA mismatch: " + gate)
    if data.get("config_hash") != config_hash or data.get("mode") != "EVALUATION_MODE":
        raise ValueError("artifact config/mode mismatch: " + gate)
    dataset = "local_xyz_fixture_v1" if gate == "pcd_selfcheck" else "synthetic_ship_grid_v1"
    if data.get("dataset_id") != dataset or not data.get("timestamp"):
        raise ValueError("artifact dataset/timestamp missing or incorrect: " + gate)
    if data.get("status") != "PASS_SYNTHETIC" or data.get("error"):
        raise ValueError("gate did not pass: " + gate)
    if not isinstance(data.get("metrics"), dict) or not data["metrics"]:
        raise ValueError("missing numeric evidence: " + gate)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in data["metrics"].values()):
        raise ValueError("nonfinite or nonnumeric evidence: " + gate)


def write_report(directory, report):
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    labels = [
        ("验证目标SHA", "VALIDATION_SHA"), ("验证基线SHA", "BASE_SHA"), ("验证分支", "BRANCH"),
        ("OS", "OS"), ("KERNEL", "KERNEL"), ("GCC", "GCC"), ("GXX", "GXX"), ("CMAKE", "CMAKE"),
        ("CMAKE_EXECUTABLE", "CMAKE_EXECUTABLE"), ("CTEST_EXECUTABLE", "CTEST_EXECUTABLE"),
        ("PCL", "PCL"), ("EIGEN", "EIGEN"), ("SMALL_GICP", "SMALL_GICP"), ("工作树", "WORKTREE"),
        ("构建门禁", "BUILD_GATE")]
    labels += [(g.upper(), g.upper()) for g in GATES]
    labels += [("G2_SITE", "G2_SITE"), ("G3_SITE", "G3_SITE"), ("G4_SITE", "G4_SITE"), ("G7_SITE", "G7_SITE"),
               ("首个失败项", "FIRST_FAIL"), ("失败类型", "FAIL_TYPE"),
               ("失败分类需独立复核", "FAIL_TYPE_REVIEW_REQUIRED"),
               ("产品代码被Claude修改", "PRODUCT_CODE_MODIFIED_BY_CLAUDE"), ("Claude补丁SHA", "CLAUDE_PATCH_SHA"),
               ("日志路径", "LOG_DIR"), ("测试产物路径", "ARTIFACT_DIR"),
               ("M0_CODE", "M0_CODE"), ("M0_SYNTHETIC", "M0_SYNTHETIC"),
               ("M0_LINUX20_BUILD", "M0_LINUX20_BUILD"), ("M0_SITE", "M0_SITE"),
               ("本SHA最终结论", "FINAL_DECISION"), ("允许进入下一阶段", "ALLOW_NEXT_STAGE")]
    lines = ["# M0 machine evidence — independent ClaudeCLI review required", "", "```text"]
    lines += ["%s=%s" % (label, report.get(key) if report.get(key) is not None else "NOT_RUN") for label, key in labels]
    lines += ["```", "", "G5_SCOPE=HARNESS_ONLY; registration/EKF/observability/NEES not implemented.",
              "BASELINE_M0_SHA=PENDING (this script never creates a baseline).", "",
              "Machine evidence is not the independent review. ClaudeCLI must inspect logs,",
              "confirm failure classification and return the final report. SITE_PENDING remains."]
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha", required=True, help="full 40-character commit SHA, already checked out")
    parser.add_argument("--base-sha", default="NONE")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cmake", default="/usr/bin/cmake", help="absolute CMake executable; defaults to Ubuntu system CMake")
    parser.add_argument("--ctest", default="/usr/bin/ctest", help="absolute CTest executable; defaults to Ubuntu system CTest")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.sha) or args.jobs < 1:
        parser.error("--sha must be a full lowercase SHA and --jobs must be positive")
    if args.base_sha != "NONE" and not re.fullmatch(r"[a-f0-9]{40}", args.base_sha):
        parser.error("--base-sha must be a full SHA or NONE")
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = ROOT / "validation" / args.sha / run_id
    # New directory for every run: no deletion, no cache reuse, no stale PASS.
    directory.mkdir(parents=True, exist_ok=False)
    build = directory / "build"
    config_path = PROJECT / "config/m0.json"
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    report = dict(VALIDATION_SHA=args.sha, BASE_SHA=args.base_sha, BRANCH="UNKNOWN",
                  TARGET_OS="Ubuntu 20.04", OS=platform.platform(), KERNEL=platform.release(),
                  GCC="UNKNOWN", GXX="UNKNOWN", CMAKE="UNKNOWN", PCL="UNKNOWN", EIGEN="UNKNOWN",
                  SMALL_GICP="NOT_INTEGRATED_M0", WORKTREE="UNKNOWN", BUILD_GATE="NOT_RUN",
                  FIRST_FAIL="NONE", FAIL_TYPE="NONE", PRODUCT_CODE_MODIFIED_BY_CLAUDE="NO",
                  CLAUDE_PATCH_SHA="NONE", LOG_DIR=str(directory), ARTIFACT_DIR=str(build / "artifacts"),
                  M0_CODE="NOT_RUN", M0_SYNTHETIC="NOT_RUN", M0_LINUX20_BUILD="NOT_RUN", M0_SITE="PENDING",
                  FINAL_DECISION="NOT_VALIDATED", ALLOW_NEXT_STAGE="NO", config_hash=config_hash,
                  dataset_id="synthetic_ship_grid_v1", mode="EVALUATION_MODE", timestamp=run_id,
                  calibration_version="synthetic_v1_SITE_PENDING", independent_review="REQUIRED",
                  G5_SCOPE="HARNESS_ONLY", execution={}, BASELINE_M0_SHA=None,
                  CMAKE_EXECUTABLE=args.cmake, CTEST_EXECUTABLE=args.ctest,
                  FAIL_TYPE_REVIEW_REQUIRED=False)
    report.update({g.upper(): None for g in GATES})
    report.update({g + "_SITE": "SITE_PENDING" for g in ("G2", "G3", "G4", "G7")})
    failure_type = "ENV_FAIL"
    step = "audit"
    def run_log(command, filename, cwd=ROOT):
        code, text = capture(command, cwd)
        with (directory / filename).open("a", encoding="utf-8") as log:
            log.write("COMMAND=" + json.dumps(command) + "\n" + text + "\nEXIT_CODE=" + str(code) + "\n")
        print(text, end="", flush=True)
        return code
    try:
        status = git("status", "--porcelain", "--untracked-files=normal")
        report["WORKTREE"] = "CLEAN" if not status else "NOT_CLEAN"
        report["BRANCH"] = git("branch", "--show-current") or "DETACHED"
        audit = "HEAD=" + git("rev-parse", "HEAD") + "\nSTATUS=\n" + status + "\n" + git("log", "-1", "--oneline")
        (directory / "audit.txt").write_text(audit + "\n", encoding="utf-8")
        if status:
            raise RuntimeError("dirty worktree; no checkout/reset performed")
        if git("rev-parse", "HEAD") != args.sha:
            raise RuntimeError("HEAD differs from requested exact SHA")
        os_path = pathlib.Path("/etc/os-release")
        os_text = os_path.read_text() if os_path.exists() else "OS=" + platform.platform()
        with (directory / "environment.txt").open("w", encoding="utf-8") as log:
            log.write(os_text + "\n" + platform.platform() + "\n")
            commands = {"GCC": ["gcc", "--version"], "GXX": ["g++", "--version"],
                        "CMAKE": [args.cmake, "--version"], "CTEST": [args.ctest, "--version"],
                        "PYTHON": [sys.executable, "--version"], "GIT": ["git", "--version"],
                        "PCL_DPKG": ["dpkg-query", "-W", "libpcl-dev"],
                        "EIGEN_DPKG": ["dpkg-query", "-W", "libeigen3-dev"],
                        "EIGEN_PKGCONFIG": ["pkg-config", "--modversion", "eigen3"],
                        "PACKAGES": ["dpkg-query", "-W", "libeigen3-dev", "libpcl-dev"]}
            for key, command in commands.items():
                code, output = capture(command)
                report[key] = output.strip().splitlines()[0] if output.strip() else "UNAVAILABLE"
                log.write("\n" + key + " (exit " + str(code) + ")\n" + output)
        step = "environment"
        if not ubuntu20(os_text):
            raise RuntimeError("formal validation requires Ubuntu 20.04; current OS is not accepted")
        report["OS"] = "Ubuntu 20.04"
        step = "build_tools"
        for tool in (args.cmake, args.ctest):
            if not pathlib.Path(tool).is_absolute() or not pathlib.Path(tool).is_file():
                raise RuntimeError("build tool must be an existing absolute executable: " + tool)
            if run_log([tool, "--version"], "build_tools.log"):
                raise RuntimeError("cannot execute build tool: " + tool)
        step = "cross_platform_audit"
        failure_type = "CODE_FAIL"
        for path in git("ls-files", "ship_perception").splitlines():
            file = ROOT / path
            if file.suffix in (".sh", ".py", ".hpp", ".cpp", ".cmake", ".json") and b"\r\n" in file.read_bytes():
                raise RuntimeError("CRLF found: " + path)
        mode = git("ls-files", "-s", "ship_perception/scripts/validate_ubuntu20.sh").split()[0]
        if mode != "100755":
            raise RuntimeError("validation shell script is not executable in Git")
        step = "configure"
        # A configure failure is not evidence that dependencies are missing.
        # Provisional code/integration failure; ClaudeCLI must inspect the log and
        # reclassify ENV_FAIL if missing/broken environment dependencies are proven.
        failure_type = "CODE_FAIL"
        if run_log([args.cmake, "-S", str(PROJECT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DM0_WITH_PCL=ON", "-DBUILD_TESTING=ON"], "configure.log"):
            raise RuntimeError("configure failed; inspect configure.log for dependency/compatibility evidence")
        step = "dependency_evidence"
        dependencies = read_build_dependencies(build / "build_dependencies.txt")
        report["BUILD_DEPENDENCIES"] = dependencies
        report["PCL"] = dependencies["PCL_VERSION"]
        report["EIGEN"] = dependencies["EIGEN_VERSION"]
        report["GXX_ENVIRONMENT"] = report["GXX"]
        report["GXX"] = dependencies["CXX_COMPILER"] + " " + dependencies["CXX_COMPILER_VERSION"]
        with (directory / "environment.txt").open("a", encoding="utf-8") as log:
            log.write("\nCMAKE_SELECTED_DEPENDENCIES\n" + (build / "build_dependencies.txt").read_text(encoding="utf-8"))
        step = "build"
        failure_type = "CODE_FAIL"
        if run_log([args.cmake, "--build", str(build), "--parallel", str(args.jobs)], "build.log"):
            report["BUILD_GATE"] = "FAIL"
            raise RuntimeError("clean build failed; inspect build.log")
        report["BUILD_GATE"] = "PASS"
        report["M0_LINUX20_BUILD"] = "PASS"
        failure_type = "TEST_FAIL"
        # CTest 3.16 compatible: cwd=build, never --test-dir.
        if run_log([args.ctest, "-N"], "ctest_inventory.log", build):
            raise RuntimeError("test inventory failed")
        for gate in RUN_ORDER:
            step = gate
            report["execution"][gate] = "RUNNING"
            if run_log([args.ctest, "-R", "^" + gate + "$", "--output-on-failure", "-V"], "ctest.log", build):
                report["execution"][gate] = "FAIL"
                if gate in GATES:
                    report[gate.upper()] = "FAIL"
                raise RuntimeError("gate failed; preserve evidence and classify CODE_FAIL vs TEST_FAIL")
            if gate == "validation_boundary":
                report["execution"][gate] = "COMPLETE"
                continue
            artifact = build / "artifacts" / gate / "report.json"
            check_artifact(json.loads(artifact.read_text()), gate, args.sha, config_hash)
            report["execution"][gate] = "COMPLETE"
            if gate in GATES:
                report[gate.upper()] = LAB_STATUS[gate]
        step = "replay_export"
        if run_log([sys.executable, str(PROJECT / "tools/replay.py"), "--executable", str(build / "synthetic_replay"),
                    "--input", str(PROJECT / "tests/fixtures/local_xyz.pcd"),
                    "--output", str(directory / "pcd_replay")], "replay.log"):
            raise RuntimeError("replay export failed")
        exported = json.loads((directory / "pcd_replay/metadata.json").read_text())
        if exported["git_sha"] != args.sha or exported["config_hash"] != config_hash:
            raise RuntimeError("replay provenance mismatch")
        step = "final_integrity"
        if git("status", "--porcelain", "--untracked-files=normal") or git("rev-parse", "HEAD") != args.sha:
            report["WORKTREE"] = "NOT_CLEAN"
            raise RuntimeError("repository changed during validation")
        report.update(M0_CODE="PASS", M0_SYNTHETIC="PASS",
                      FINAL_DECISION="M0_MACHINE_CHECKS_PASS_AWAITING_CLAUDECLI_REVIEW")
    except Exception as exc:
        report.update(FIRST_FAIL=step, FAIL_TYPE=failure_type, error=str(exc), FINAL_DECISION="FAIL",
                      FAIL_TYPE_REVIEW_REQUIRED=True)
        if report["BUILD_GATE"] == "NOT_RUN" and step in ("configure", "build"):
            report["BUILD_GATE"] = "FAIL"
        print("VALIDATION_FAILURE: " + str(exc), file=sys.stderr)
    finally:
        write_report(directory, report)
        print("\nSTRUCTURED_REPORT=" + str(directory / "report.json"))
        print("SUMMARY=" + str(directory / "summary.md"))
    return 0 if report["FAIL_TYPE"] == "NONE" else 1


if __name__ == "__main__":
    sys.exit(main())
