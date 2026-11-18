"""Run one PCD or a directory and verify every published review artifact."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_output(folder, expected_sha, native=None):
    required = (
        "opening_cells.json",
        "05_hatch_corners.json",
        "hatches_compact.json",
        "03_results.json",
        "01_geometry_only.png",
        "船舱四角长宽_简表.csv",
    )
    for name in required:
        if not (folder / name).is_file():
            raise ValueError("MISSING_OUTPUT:" + name)
    compact = json.loads((folder / "hatches_compact.json").read_text(encoding="utf-8"))
    if compact["scene_source_sha256"] != expected_sha:
        raise ValueError("OUTPUT_INPUT_SHA_MISMATCH")
    cells = json.loads((folder / "opening_cells.json").read_text(encoding="utf-8"))
    ids = {c["cell_id"] for c in cells if c.get("geometry") is not None}
    seen = {h["opening_id"] for v in compact["vessels"] for h in v["hatches"]}
    if ids != seen:
        raise ValueError("CELL_GEOMETRY_EXPORT_MISMATCH")
    if native is None:
        native = json.loads((folder / "03_results.json").read_text(encoding="utf-8"))
    if (
        native.get("geometry_only", {}).get("opening_review_level")
        == "PRECISION_RECTANGLE_REVIEW"
    ):
        review_path = folder / "review_hatches_compact.json"
        if not review_path.is_file():
            raise ValueError("MISSING_REVIEW_LAYER")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if {r["cell_id"] for r in review} != ids:
            raise ValueError("REVIEW_LAYER_IDENTITY_MISMATCH")
        for row in review:
            if row["control_eligible"] or row["steel_certified"]:
                raise ValueError("UNCERTIFIED_REVIEW_CANNOT_CONTROL")
            if row["adopted_for_review"] and any(
                p[2] is not None for p in row["corners_xyz"]
            ):
                raise ValueError("REVIEW_MOVED_XY_CANNOT_INHERIT_OLD_Z")
    return compact


def worker(args):
    sys.path.insert(0, str(ROOT))
    from ship_perception.io.pcd_source import read_pcd
    from ship_perception.perception.cli import execute
    from ship_perception.tools.stage_progress import StageProgress
    from ship_perception.perception.runtime_metrics import peak_rss_bytes
    from ship_perception.perception.e2e_timing import ReviewTiming

    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    read_start = time.perf_counter()
    frame = read_pcd(args.input, args.frame_id)
    read_end = time.perf_counter()
    if frame.provenance["wire_sha256"] != args.expected_sha:
        raise ValueError("INPUT_CHANGED_DURING_RUN")
    modes = ("preview", "full") if args.mode == "staged" else (args.mode,)
    receipts = []
    decode = time.perf_counter() - start
    for mode in modes:
        folder = args.output / mode
        folder.mkdir()
        observer = ReviewTiming(
            start,
            read_start,
            read_end,
            mode,
            args.expected_sha,
            args.input.stat().st_size,
        )
        options = SimpleNamespace(
            pipeline="v2",
            berth_reference=None,
            operator_vessel_roi=None,
            mode="review-first",
            opening_topology_v2=True,
            opening_preview=mode == "preview",
            local_rim_review=True,
            review_snapshot=args.snapshot if args.mode == "staged" else None,
            review_process_deadline=start + 300.0,
            no_legacy_evidence=False,
            output_dir=folder,
            save_snapshot=False,
            review_timing_observer=observer,
        )
        tick, cpu = time.perf_counter(), time.process_time()
        code = execute(frame, options, StageProgress())
        result_files = list(folder.glob("*/03_results.json"))
        if code or len(result_files) != 1:
            raise ValueError("NATIVE_REVIEW_FAILED")
        result_file, = result_files
        result_dir = result_file.parent
        observer.published()
        atomic_json(result_dir / "e2e_timing.json", observer.report())
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        compact = verify_output(result_dir, args.expected_sha, native=payload)
        report = dict(
            mode=mode,
            exit_code=code,
            wall_s=time.perf_counter() - tick,
            cpu_s=time.process_time() - cpu,
            published_perf_counter=time.perf_counter(),
            process_peak_rss_bytes=peak_rss_bytes(),
            result_dir=str(result_dir),
            recall_snapshot=payload["recall_snapshot"],
            stage_wall_s=payload["stage_elapsed"],
            stage_resources=payload.get("stage_resources", {}),
            visible_count=sum(len(v["hatches"]) for v in compact["vessels"]),
            runtime_source_sha256=compact["runtime_source_sha256"],
            e2e_timing=observer.report(),
        )
        metrics = result_dir / "runtime_io_metrics.json"
        report["output_resources"] = (
            json.loads(metrics.read_text(encoding="utf-8"))
            if metrics.is_file()
            else None
        )
        receipts.append(report)
        atomic_json(
            args.output / "worker_receipt.json",
            dict(
                read_pcd_s=decode,
                pcd_decode_count=1,
                modes=receipts,
                total_wall_s=time.perf_counter() - start,
                status=(
                    "COMPLETED"
                    if len(receipts) == len(modes)
                    else "PREVIEW_COMPLETED_FULL_PENDING"
                ),
                deadline_scope="SHARED_PREVIEW_AND_FULL_PROCESS",
            ),
        )
    print("R3H_WORKER_STATUS=COMPLETED", flush=True)
    return 0


def read_receipt(folder, mode):
    try:
        receipt = json.loads((folder / "worker_receipt.json").read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            return {}, None
        if receipt.get("status") != "COMPLETED":
            return receipt, None
        expected = ["preview", "full"] if mode == "staged" else [mode]
        modes = receipt["modes"]
        if [entry["mode"] for entry in modes] != expected:
            return receipt, None
        for entry in modes:
            result_dir = Path(entry["result_dir"]).resolve()
            if (
                entry["exit_code"] != 0
                or result_dir.parent != folder / entry["mode"]
                or not (result_dir / "hatches_compact.json").is_file()
                or not (result_dir / "01_geometry_only.png").is_file()
            ):
                return receipt, None
        return receipt, result_dir
    except (OSError, KeyError, TypeError, ValueError):
        return {}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode", choices=("preview", "full", "staged"), default="staged"
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--frame-id", default="pcd")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-sha", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if args.worker:
        return worker(args)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    files = sorted(args.input.glob("*.pcd")) if args.input.is_dir() else [args.input]
    if not files or any(not p.is_file() or p.suffix.lower() != ".pcd" for p in files):
        parser.error("Input must be a PCD file or a nonempty PCD directory")
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_")
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=args.output))
    state = dict(run_id=run_dir.name, status="RUNNING", result_count=0, index=None)
    state_path = run_dir / "运行状态.json"
    atomic_json(state_path, state)
    print(f"RUN_ID={run_dir.name}\nRUN_OUTPUT_DIR={run_dir}", flush=True)
    rows = []
    try:
        code = run_batch(args, files, run_dir, rows)
    except KeyboardInterrupt:
        code = 130
    status = {0: "COMPLETED", 130: "INTERRUPTED"}.get(code, "FAILED")
    write_tables(run_dir, rows)
    index = write_index(run_dir, rows, status)
    state.update(status=status, result_count=len(rows), index=index.name, exit_code=code)
    atomic_json(state_path, state)
    print(f"RUN_STATUS={status}\nREVIEW_INDEX={index}", flush=True)
    return code


def run_batch(args, files, run_dir, rows):
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        PYTHONIOENCODING="utf-8",
    )
    for source in files:
        digest = file_sha(source)
        for repeat in range(1, args.rounds + 1):
            folder = (
                run_dir
                / (source.stem + "_" + digest[:12])
                / ("round_" + str(repeat))
            )
            folder.mkdir(parents=True)
            snapshot = folder / "recall.snapshot"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                str(source),
                str(folder),
                "--worker",
                "--mode",
                args.mode,
                "--frame-id",
                args.frame_id,
                "--expected-sha",
                digest,
                "--snapshot",
                str(snapshot),
            ]
            tick = time.perf_counter()
            with (folder / "run.stdout.log").open("wb") as stdout, (
                folder / "run.stderr.log"
            ).open("wb") as stderr:
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=str(ROOT),
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=300,
                    )
                    code = result.returncode
                    reason = "NATIVE_OR_OUTPUT_CONTRACT_FAILED" if code else None
                except subprocess.TimeoutExpired:
                    code, reason = 124, "WATCHDOG_300S"
                except KeyboardInterrupt:
                    code, reason = 130, "INTERRUPTED"
                except OSError as error:
                    stderr.write(str(error).encode("utf-8", errors="replace"))
                    code, reason = 1, "WORKER_START_FAILED"
            receipt, result_dir = read_receipt(folder, args.mode)
            if code == 0 and result_dir is None:
                code, reason = 1, "WORKER_RECEIPT_INVALID"
            status = "FAIL" if code else "PASS"
            if reason == "INTERRUPTED":
                status = "INTERRUPTED"
            row = dict(
                run_id=run_dir.name,
                pcd=str(source),
                input_sha256=digest,
                round=repeat,
                status=status,
                exit_code=code,
                failure_reason=reason,
                cli_wall_s=time.perf_counter() - tick,
                folder=str(folder),
                result_dir=str(result_dir) if status == "PASS" else None,
                command=cmd,
                worker=receipt,
            )
            rows.append(row)
            write_tables(run_dir, rows)
            print(
                "PCD=" + str(source) + " STATUS=" + status + " EXIT_CODE=" + str(code),
                flush=True,
            )
            if status == "INTERRUPTED":
                return 130
            if status == "PASS":
                from r3h_export_dimensions import print_compact

                review_path = result_dir / "review_hatches_compact.json"
                if review_path.is_file():
                    print(
                        "本轮复核建议（非认证，禁止用于控制）：" + str(review_path),
                        flush=True,
                    )
                print("原严格几何结果：", flush=True)
                print_compact(
                    json.loads(
                        (result_dir / "hatches_compact.json").read_text(
                            encoding="utf-8"
                        )
                    )
                )
    return int(any(row["status"] != "PASS" for row in rows))


def write_tables(run_dir, rows):
    atomic_json(run_dir / "运行总表.json", rows)
    fields = (
        "run_id", "pcd", "input_sha256", "round", "status", "exit_code",
        "failure_reason", "cli_wall_s", "folder", "result_dir",
    )
    csv_path = run_dir / "运行总表.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)


def write_index(run_dir, rows, status):
    links = []
    for row in rows:
        if row["status"] == "PASS":
            result_dir = Path(row["result_dir"])
            review = result_dir / "船舱复核总览.html"
            href = review.relative_to(run_dir).as_posix()
            label = "复核输出" if review.is_file() else "预览输出，未运行完整标定"
            if not review.is_file():
                href = (
                    (result_dir / "01_geometry_only.png")
                    .relative_to(run_dir)
                    .as_posix()
                )
            links.append(
                '<li><a href="'
                + html.escape(quote(href), quote=True)
                + '">'
                + html.escape(Path(row["pcd"]).stem)
                + "｜第" + str(row["round"]) + "轮"
                + "｜"
                + label
                + "</a></li>"
            )
        else:
            links.append(
                "<li>"
                + html.escape(Path(row["pcd"]).stem)
                + "｜第" + str(row["round"]) + "轮｜"
                + html.escape(row["failure_reason"]) + "</li>"
            )
    labels = dict(COMPLETED="已完成", FAILED="有失败场景", INTERRUPTED="已中断")
    index = run_dir / "全部场景复核.html"
    tmp = index.with_suffix(".html.tmp")
    tmp.write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>全部场景船舱复核</title><h1>全部场景船舱复核</h1>'
        f'<p>{run_dir.name}｜{labels[status]}。钢边精度未认证，人工审核待签。'
        '<a href="运行总表.csv">运行性能总表</a></p><ul>'
        + "".join(links)
        + "</ul></html>",
        encoding="utf-8",
    )
    tmp.replace(index)
    return index


if __name__ == "__main__":
    raise SystemExit(main())
