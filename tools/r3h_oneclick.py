"""Run one PCD or a directory and verify every published review artifact."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

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


def verify_output(folder, expected_sha):
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
    return compact


def worker(args):
    sys.path.insert(0, str(ROOT))
    from ship_perception.io.pcd_source import read_pcd
    from ship_perception.perception.cli import execute
    from ship_perception.tools.stage_progress import StageProgress
    from ship_perception.perception.runtime_metrics import peak_rss_bytes

    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    frame = read_pcd(args.input, args.frame_id)
    if frame.provenance["wire_sha256"] != args.expected_sha:
        raise ValueError("INPUT_CHANGED_DURING_RUN")
    modes = ("preview", "full") if args.mode == "staged" else (args.mode,)
    receipts = []
    decode = time.perf_counter() - start
    for mode in modes:
        folder = args.output / mode
        options = SimpleNamespace(
            pipeline="v2",
            berth_reference=None,
            operator_vessel_roi=None,
            mode="review-first",
            opening_topology_v2=True,
            opening_preview=mode == "preview",
            local_rim_review=True,
            review_snapshot=args.snapshot if args.mode == "staged" else None,
            no_legacy_evidence=False,
            output_dir=folder,
            save_snapshot=False,
        )
        tick, cpu = time.perf_counter(), time.process_time()
        code = execute(frame, options, StageProgress())
        result_files = sorted(folder.glob("*/03_results.json"))
        if code or not result_files:
            raise ValueError("NATIVE_REVIEW_FAILED")
        result_dir = result_files[-1].parent
        compact = verify_output(result_dir, args.expected_sha)
        payload = json.loads(result_files[-1].read_text(encoding="utf-8"))
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
                status="COMPLETED",
            ),
        )
    print("R3H_WORKER_STATUS=COMPLETED", flush=True)
    return 0


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
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        PYTHONIOENCODING="utf-8",
    )
    rows = []
    failure = False
    for source in files:
        digest = file_sha(source)
        for repeat in range(1, args.rounds + 1):
            folder = (
                args.output
                / (source.stem + "_" + digest[:12])
                / ("round_" + str(repeat))
            )
            folder.mkdir(parents=True, exist_ok=True)
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
            reason = None
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
                except subprocess.TimeoutExpired:
                    code = 124
            receipt_path = folder / "worker_receipt.json"
            receipt = (
                json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt_path.is_file()
                else {}
            )
            status = (
                "PASS" if code == 0 and receipt.get("status") == "COMPLETED" else "FAIL"
            )
            if status == "FAIL":
                reason = (
                    "WATCHDOG_300S"
                    if code == 124
                    else "NATIVE_OR_OUTPUT_CONTRACT_FAILED"
                )
                failure = True
            row = dict(
                pcd=str(source),
                input_sha256=digest,
                round=repeat,
                status=status,
                exit_code=code,
                failure_reason=reason,
                cli_wall_s=time.perf_counter() - tick,
                folder=str(folder),
                command=cmd,
                worker=receipt,
            )
            rows.append(row)
            atomic_json(args.output / "运行总表.json", rows)
            fields = (
                "pcd",
                "input_sha256",
                "round",
                "status",
                "exit_code",
                "failure_reason",
                "cli_wall_s",
                "folder",
            )
            with (args.output / "运行总表.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=fields, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(rows)
            print(
                "PCD=" + str(source) + " STATUS=" + status + " EXIT_CODE=" + str(code),
                flush=True,
            )
            if status == "PASS":
                from r3h_export_dimensions import print_compact

                final = receipt["modes"][-1]["result_dir"]
                print_compact(
                    json.loads(
                        (Path(final) / "hatches_compact.json").read_text(
                            encoding="utf-8"
                        )
                    )
                )
    import html

    links = []
    for row in rows:
        if row["status"] == "PASS":
            result_dir = Path(row["worker"]["modes"][-1]["result_dir"])
            review = result_dir / "船舱复核总览.html"
            href = review.relative_to(args.output).as_posix()
            label = "复核输出" if review.is_file() else "预览输出，未运行完整标定"
            if not review.is_file():
                href = (
                    (result_dir / "01_geometry_only.png")
                    .relative_to(args.output)
                    .as_posix()
                )
            links.append(
                '<li><a href="'
                + html.escape(href, quote=True)
                + '">'
                + html.escape(Path(row["pcd"]).stem)
                + "｜"
                + label
                + "</a></li>"
            )
        else:
            links.append(
                "<li>"
                + html.escape(Path(row["pcd"]).stem)
                + "｜运行失败，已记录并继续下一景</li>"
            )
    (args.output / "全部场景复核.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>全部场景船舱复核</title><h1>全部场景船舱复核</h1><p>钢边精度未认证，人工审核待签。<a href="运行总表.csv">运行性能总表</a></p><ul>'
        + "".join(links)
        + "</ul></html>",
        encoding="utf-8",
    )
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
