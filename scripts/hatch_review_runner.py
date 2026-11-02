"""One-command review delivery through the unchanged public PCD launcher.

This runner manages inputs, isolation and review artifacts. It has no hatch
recall, geometry, threshold, reference-coordinate or certification decisions.
"""

import argparse
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("PRIMARY_ORIGINAL", "DERIVED_DIAGNOSTIC_CROP")
FIELDS = {
    "scene",
    "pcd",
    "frame_id",
    "input_role",
    "input_sha256",
    "operator_vessel_roi",
    "source_scene",
    "crop_provenance",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_scenes(args):
    base = Path.cwd()
    if args.manifest:
        if args.operator_vessel_roi:
            raise ValueError("MANIFEST_OPERATOR_ROI_MUST_BE_SPECIFIED_PER_SCENE")
        base = args.manifest.resolve().parent
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
        if (
            set(data) != {"schema", "scenes"}
            or data["schema"] != "ship_perception.review_batch.1"
        ):
            raise ValueError("INVALID_REVIEW_BATCH_MANIFEST")
        rows = data["scenes"]
    else:
        paths = (
            [args.pcd]
            if args.pcd
            else sorted(
                p
                for p in args.pcd_dir.rglob("*")
                if p.is_file() and p.suffix.lower() == ".pcd"
            )
        )
        rows = [
            dict(
                scene=p.stem,
                pcd=str(p.resolve()),
                frame_id=args.frame_id,
                input_role="PRIMARY_ORIGINAL",
            )
            for p in paths
        ]
        if args.operator_vessel_roi:
            if len(rows) != 1:
                raise ValueError("ONE_OPERATOR_ROI_REQUIRES_ONE_PCD")
            rows[0]["operator_vessel_roi"] = str(args.operator_vessel_roi.resolve())
    if not isinstance(rows, list) or not rows:
        raise ValueError("EMPTY_REVIEW_INPUTS")
    result, names = [], set()
    for source in rows:
        if not isinstance(source, dict) or set(source) - FIELDS:
            raise ValueError("UNSUPPORTED_REVIEW_INPUT_FIELD")
        row = dict(source)
        name = row.get("scene")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("INVALID_OR_DUPLICATE_SCENE_ID")
        names.add(name)
        row.setdefault("frame_id", args.frame_id)
        row.setdefault("input_role", "PRIMARY_ORIGINAL")
        if row["input_role"] not in ROLES:
            raise ValueError("INVALID_REVIEW_INPUT_ROLE")
        if row["input_role"] == ROLES[1] and not (
            row.get("source_scene") and row.get("crop_provenance")
        ):
            raise ValueError("DERIVED_INPUT_REQUIRES_PARENT_AND_CROP_PROVENANCE")
        for field in ("pcd", "operator_vessel_roi"):
            if row.get(field):
                path = Path(row[field])
                row[field] = str(
                    (base / path).resolve()
                    if not path.is_absolute()
                    else path.resolve()
                )
        if not row.get("pcd"):
            raise ValueError("REVIEW_INPUT_REQUIRES_PCD")
        result.append(row)
    return result


def output_record(row, folder, exit_code, elapsed, command):
    record = dict(
        row,
        exit_code=exit_code,
        elapsed_s=elapsed,
        command=command,
        log=str(folder / "terminal.log"),
        watchdog_hit=False,
        review_polygon_count=None,
        complete_review_polygon_count=None,
        status="EXECUTION_FAILED",
        hatches=[],
    )
    timeouts = sorted(folder.glob(".review_run_*/RUN_TIMEOUT.json"))
    if timeouts or exit_code == 124:
        record.update(
            status="RUN_TIMEOUT",
            watchdog_hit=True,
            timeout_diagnostics=[str(p) for p in timeouts],
        )
    elif exit_code == 130:
        record["status"] = "INTERRUPTED"
    packets = sorted(folder.glob("*/03_results.json"))
    if not packets:
        record["failure_diagnostics"] = [
            str(p) for p in folder.glob("failure_*/failure.json")
        ]
        return record
    path = packets[-1]
    packet = json.loads(path.read_text(encoding="utf-8"))
    export_path = path.with_name("05_hatch_corners.json")
    export = json.loads(export_path.read_text(encoding="utf-8"))
    if export["source_result_sha256"] != digest(path):
        raise ValueError("REVIEW_EXPORT_RESULT_HASH_MISMATCH")
    objects = packet["geometry_only"]["hatches"]
    complete = sum(
        h.get("geometry_status") == "COMPLETE_REVIEW_GEOMETRY" for h in objects
    )
    record.update(
        status=(
            (
                "REVIEW_ARTIFACTS_READY_UNCERTIFIED"
                if objects
                else "UNRESOLVED_NO_COMPLETE_REVIEW"
            )
            if exit_code == 0
            else record["status"]
        ),
        result=str(path),
        result_sha256=digest(path),
        corners_json=str(export_path),
        image=str(path.with_name("01_geometry_only.png")),
        source_identity=packet["software_source_identity"],
        config_hash=packet["config_hash"],
        input_sha256=packet["input_sha256"],
        review_polygon_count=len(objects),
        complete_review_polygon_count=complete,
        soft_runtime_exceeded=packet.get("performance_budget", {}).get(
            "soft_runtime_exceeded", elapsed > 120
        ),
        hatch_count_status=packet["geometry_only"]["hatch_count_status"],
        stage_elapsed_s=packet.get("stage_elapsed", {}),
        algorithm_total_elapsed_s=packet.get("performance_budget", {}).get(
            "total_elapsed_s"
        ),
        selection_status=packet["berth_selection"]["status"],
        precision_certified=False,
        steel_edge_certified=False,
        hatches=export["hatches"],
    )
    return record


def terminate_child(process):
    """An interrupted batch must not leave its isolated detector running."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait()


def run_scene(row, folder, mode, save_snapshot=False):
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    command = [
        "bash",
        str(ROOT / "scripts/run_ship_hatch_pcd.sh"),
        "--pcd",
        row["pcd"],
        "--frame-id",
        row["frame_id"],
        "--mode",
        mode,
        "--output-dir",
        str(folder),
    ]
    if row.get("operator_vessel_roi"):
        command += ["--operator-vessel-roi", row["operator_vessel_roi"]]
    if save_snapshot:
        command.append("--save-snapshot")
    try:
        actual = digest(row["pcd"])
        if row.get("input_sha256") and actual != row["input_sha256"]:
            raise ValueError("REVIEW_INPUT_HASH_MISMATCH")
        row = dict(row, input_sha256=actual)
        row["input_preflight_elapsed_s"] = time.monotonic() - started
        environment = dict(
            os.environ,
            PYTHON_BIN=sys.executable,
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            PYTHONNOUSERSITE="1",
            MPLBACKEND="Agg",
        )
        with (folder / "terminal.log").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
            try:
                code = process.wait(timeout=315)
            except subprocess.TimeoutExpired:
                terminate_child(process)
                code = 124
                write_json(
                    folder / "RUNNER_TIMEOUT.json",
                    dict(
                        status="RUN_TIMEOUT",
                        outer_process_limit_s=315,
                        core_hard_process_watchdog_s=300,
                        no_hatch_verdict=False,
                        command=command,
                    ),
                )
            except KeyboardInterrupt:
                terminate_child(process)
                code = 130
        return output_record(row, folder, code, time.monotonic() - started, command)
    except (OSError, ValueError, KeyError) as error:
        record = dict(
            row,
            status="INPUT_OR_OUTPUT_CONTRACT_FAILED",
            error=str(error),
            exit_code=1,
            elapsed_s=time.monotonic() - started,
            review_polygon_count=None,
            complete_review_polygon_count=None,
            watchdog_hit=False,
            hatches=[],
            command=command,
        )
        write_json(folder / "runner_failure.json", record)
        return record


def link(path, root, label):
    relative = Path(os.path.relpath(path, root)).as_posix()
    return (
        '<a href="'
        + html.escape(relative, quote=True)
        + '">'
        + html.escape(label)
        + "</a>"
    )


def write_gallery(root, rows, planned_count=None):
    cards = []
    for role, title in zip(ROLES, ("原始输入", "派生靠岸裁剪（独立统计）")):
        cards.append("<h2>" + title + "</h2>")
        for r in rows:
            if r["input_role"] != role:
                continue
            card = (
                "<article><h3>"
                + html.escape(r["scene"])
                + "</h3><p>"
                + html.escape(r["status"])
            )
            card += (
                " · 完整复核框："
                + str(r.get("complete_review_polygon_count"))
                + " · "
                + f"{r.get('elapsed_s',0):.2f}"
                + " 秒</p>"
            )
            if r.get("image"):
                image = Path(os.path.relpath(r["image"], root)).as_posix()
                card += (
                    '<a href="'
                    + html.escape(image, quote=True)
                    + '"><img loading="lazy" src="'
                    + html.escape(image, quote=True)
                    + '" alt="'
                    + html.escape(r["scene"], quote=True)
                    + ' 标定图"></a>'
                )
                card += (
                    "<p>"
                    + link(r["corners_json"], root, "四角 XYZ 与来源 JSON")
                    + " · "
                    + link(r["result"], root, "完整结果")
                    + " · "
                    + link(r["log"], root, "运行日志")
                    + "</p>"
                )
            elif r.get("log"):
                card += "<p>" + link(r["log"], root, "运行日志") + "</p>"
            if r.get("error"):
                card += "<p>" + html.escape(r["error"]) + "</p>"
            for hatch in r.get("hatches", []):
                card += (
                    "<p>"
                    + html.escape(str(hatch.get("hatch_id", "")))
                    + "</p><table><tr><th>角点</th><th>X（米）</th><th>Y（米）</th><th>Z（米）</th><th>Z 状态</th></tr>"
                )
                for i, c in enumerate(hatch["corners"]):
                    values = [c.get(k, c.get(k + "_m")) for k in ["x", "y", "z"]]
                    cells = [f"{v:.3f}" if v is not None else "未观测" for v in values]
                    if (
                        values[2] is not None
                        and c.get("z_state", c.get("z_status")) == "ESTIMATED"
                    ):
                        cells[2] = "≈" + cells[2]
                    card += (
                        "<tr><td>P"
                        + str(i + 1)
                        + "</td>"
                        + "".join("<td>" + v + "</td>" for v in cells)
                        + "<td>"
                        + html.escape(str(c.get("z_state", c.get("z_status", ""))))
                        + "</td></tr>"
                    )
                card += "</table>"
            identity = r.get("source_identity", {})
            card += (
                "<details><summary>输入与代码身份</summary><pre>"
                + html.escape(
                    json.dumps(
                        dict(
                            source_identity=identity,
                            config_hash=r.get("config_hash"),
                            input_sha256=r.get("input_sha256"),
                            crop_provenance=r.get("crop_provenance"),
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                + "</pre></details></article>"
            )
            cards.append(card)
    text = (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>船舱标定人工审查</title><style>body{font:16px system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;background:#f3f5f8;color:#172438}article{background:white;padding:20px;margin:20px 0;border-radius:8px}img{width:100%;height:auto}table{border-collapse:collapse}td,th{padding:6px 14px;border:1px solid #ccd4df;text-align:right}a{color:#075eb3}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>船舱标定人工审查</h1><p>当前固定代码实跑结果，未宣称钢内沿、精度或现场验收通过。Z 为舱口围板/甲板上沿标高；估计值与缺测保留原状态，不能用料面高度补值。点击图像可查看原图。</p>"""
        + "<p>已记录 "
        + str(len(rows))
        + " 项，计划 "
        + str(planned_count or len(rows))
        + " 项。</p>"
        + "".join(cards)
        + "</html>"
    )
    (root / "index.html").write_text(text, encoding="utf-8")


def save_progress(root, rows, identity, mode, planned_count=None):
    for role, name in zip(
        ROLES, ("PRIMARY_SCENE_MATRIX.json", "DERIVED_CROP_MATRIX.json")
    ):
        write_json(root / name, [r for r in rows if r["input_role"] == role])
    write_json(
        root / "batch_result.json",
        dict(
            schema="ship_perception.review_batch_result.1",
            source_identity=identity,
            review_mode=mode,
            acceptance_status="HUMAN_REVIEW_PENDING",
            precision_certified=False,
            planned_input_count=planned_count or len(rows),
            pending_input_count=max(0, (planned_count or len(rows)) - len(rows)),
            primary_scene_count=sum(r["input_role"] == ROLES[0] for r in rows),
            independent_primary_input_hash_count=len(
                {
                    r["input_sha256"]
                    for r in rows
                    if r["input_role"] == ROLES[0]
                    and r.get("input_sha256")
                    and r.get("exit_code") == 0
                }
            ),
            derived_input_count=sum(r["input_role"] == ROLES[1] for r in rows),
            scenes=rows,
        ),
    )
    write_gallery(root, rows, planned_count)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run fixed-code PCD reviews and an XYZ review gallery"
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--pcd", type=Path)
    inputs.add_argument("--pcd-dir", type=Path)
    inputs.add_argument("--manifest", type=Path)
    parser.add_argument("--frame-id", default="pcd")
    parser.add_argument("--operator-vessel-roi", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/review"))
    parser.add_argument(
        "--mode", choices=("review-first", "bounded"), default="review-first"
    )
    parser.add_argument("--save-snapshot", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        scenes = load_scenes(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if args.list_only:
        print(json.dumps(scenes, ensure_ascii=False, indent=2))
        return 0
    sys.path.insert(0, str(ROOT))
    from ship_perception.perception.pipeline import (
        source_identity,
        numerical_environment,
    )
    from ship_perception.perception.review_labels import chinese_plot_style
    from ship_perception.perception.legacy_expert import verify_legacy_b2_core

    chinese_plot_style()
    verify_legacy_b2_core()
    identity = source_identity()
    root = args.output_dir.resolve() / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    )
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "batch_input.json",
        dict(schema="ship_perception.review_batch.1", scenes=scenes),
    )
    write_json(
        root / "environment.json",
        dict(
            source_identity=identity,
            numerical_environment=numerical_environment(),
            platform=sys.platform,
            python_executable=sys.executable,
            algorithm_default_unchanged=True,
            explicit_review_mode=args.mode,
            soft_runtime_limit_s=120,
            hard_process_watchdog_s=300,
            launcher_sha256=digest(__file__),
            launcher_shell_sha256=digest(ROOT / "scripts/run_ship_hatch_review.sh"),
        ),
    )
    rows = []
    primary_timeouts = 0
    breaker = False
    for i, row in enumerate(scenes):
        print("[REVIEW] START " + row["scene"], flush=True)
        if breaker:
            result = dict(
                row,
                status="NOT_RUN_PRIMARY_WATCHDOG_BREAKER",
                exit_code=None,
                elapsed_s=0,
                review_polygon_count=None,
                complete_review_polygon_count=None,
                watchdog_hit=False,
                hatches=[],
            )
        else:
            token = hashlib.sha256(row["scene"].encode()).hexdigest()[:12]
            result = run_scene(
                row, root / (f"{i+1:02d}_" + token), args.mode, args.save_snapshot
            )
            if result.get("source_identity") and any(
                result["source_identity"][field] != identity[field]
                for field in [
                    "runtime_source_sha256",
                    "git_head_sha",
                    "git_worktree_dirty",
                ]
            ):
                raise RuntimeError("REVIEW_SOURCE_CHANGED_DURING_BATCH")
            primary_timeouts += int(
                row["input_role"] == ROLES[0] and result["watchdog_hit"]
            )
            breaker = primary_timeouts >= 3
        rows.append(result)
        save_progress(root, rows, identity, args.mode, len(scenes))
        print("[REVIEW] " + result["status"] + " " + row["scene"], flush=True)
        if result["exit_code"] == 130:
            print("REVIEW_INTERRUPTED_INDEX=" + str(root / "index.html"), flush=True)
            return 130
    print("REVIEW_INDEX=" + str(root / "index.html"), flush=True)
    return 0 if all(r["exit_code"] == 0 for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
