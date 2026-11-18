from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "r3h_oneclick.py"
SPEC = importlib.util.spec_from_file_location("oneclick_test", SCRIPT)
ONECLICK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ONECLICK)


class OneClickIsolation(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.input_dir = self.root / "inputs"
        self.input_dir.mkdir()
        self.source = self.input_dir / "scene #1.pcd"
        self.source.write_bytes(b"fixture")
        self.output = self.root / "results with spaces"

    def invoke(self, simulate, source=None, mode="full", rounds=1):
        args = [str(SCRIPT), str(source or self.source), str(self.output),
                "--mode", mode, "--rounds", str(rounds)]
        exporter = SimpleNamespace(print_compact=lambda payload: None)
        output = io.StringIO()
        with patch.object(sys, "argv", args), patch.object(
            ONECLICK.subprocess, "run", side_effect=simulate
        ) as worker, patch.dict(sys.modules, {"r3h_export_dimensions": exporter}), redirect_stdout(output):
            code = ONECLICK.main()
        lines = dict(line.split("=", 1) for line in output.getvalue().splitlines() if "=" in line)
        run_dir = Path(lines["RUN_OUTPUT_DIR"])
        index = Path(lines["REVIEW_INDEX"])
        self.assertEqual(index.parent, run_dir)
        self.assertTrue(index.is_file())
        self.assertEqual(run_dir.name, lines["RUN_ID"])
        state = self.read_json(run_dir / "运行状态.json")
        self.assertEqual(state["exit_code"], code)
        rows = self.read_json(run_dir / "运行总表.json")
        self.assertEqual(state["result_count"], len(rows))
        return code, run_dir, state, rows, worker.call_count

    @staticmethod
    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def publish(self, command, **kwargs):
        folder = Path(command[3])
        requested = command[command.index("--mode") + 1]
        modes = ("preview", "full") if requested == "staged" else (requested,)
        results = []
        for mode in modes:
            result = folder / mode / "result"
            result.mkdir(parents=True)
            (result / "01_geometry_only.png").touch()
            (result / "hatches_compact.json").write_text("{}", encoding="utf-8")
            if mode == "full":
                (result / "船舱复核总览.html").touch()
            results.append(dict(mode=mode, exit_code=0, result_dir=str(result)))
        ONECLICK.atomic_json(folder / "worker_receipt.json", dict(status="COMPLETED", modes=results))
        return subprocess.CompletedProcess(command, 0)

    def test_same_output_and_repeated_rounds_have_independent_paths(self):
        self.output.mkdir()
        legacy = self.output / "全部场景复核.html"
        legacy.write_text("previous output", encoding="utf-8")
        first = self.invoke(self.publish, mode="staged", rounds=2)
        second = self.invoke(self.publish, mode="staged", rounds=2)
        self.assertNotEqual(first[1], second[1])
        self.assertEqual(legacy.read_text(encoding="utf-8"), "previous output")
        for code, run_dir, state, rows, calls in (first, second):
            self.assertEqual((code, state["status"], calls), (0, "COMPLETED", 2))
            results = [Path(row["result_dir"]) for row in rows]
            self.assertEqual(len(set(results)), 2)
            for repeat, result in enumerate(results, start=1):
                self.assertEqual(result.parents[1].name, f"round_{repeat}")
                self.assertIn(run_dir, result.parents)
            index = (run_dir / "全部场景复核.html").read_text(encoding="utf-8")
            self.assertIn("%231", index)

    def test_nonzero_exit_after_receipt_is_failure_and_next_scene_runs(self):
        (self.input_dir / "z.pcd").touch()

        def simulate(command, **kwargs):
            result = self.publish(command, **kwargs)
            if Path(command[2]) == self.source:
                result.returncode = 3
            return result

        code, run_dir, state, rows, calls = self.invoke(simulate, self.input_dir)
        self.assertEqual((code, state["status"], calls), (1, "FAILED", 2))
        self.assertEqual([row["status"] for row in rows], ["FAIL", "PASS"])
        self.assertIsNone(rows[0]["result_dir"])
        self.assertNotIn("scene%20%231", (run_dir / "全部场景复核.html").read_text(encoding="utf-8"))

    def test_receipt_from_preceding_run_is_rejected(self):
        first = self.invoke(self.publish)
        old_receipt = first[3][0]["worker"]

        def simulate(command, **kwargs):
            ONECLICK.atomic_json(Path(command[3]) / "worker_receipt.json", old_receipt)
            return subprocess.CompletedProcess(command, 0)

        code, _, state, rows, _ = self.invoke(simulate)
        self.assertEqual((code, state["status"]), (1, "FAILED"))
        self.assertEqual(rows[0]["failure_reason"], "WORKER_RECEIPT_INVALID")
        self.assertIsNone(rows[0]["result_dir"])

    def test_missing_or_damaged_receipt_is_a_failed_scene(self):
        def simulate(command, **kwargs):
            (Path(command[3]) / "worker_receipt.json").write_text("{", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        code, _, state, rows, _ = self.invoke(simulate)
        self.assertEqual((code, state["status"]), (1, "FAILED"))
        self.assertEqual(rows[0]["failure_reason"], "WORKER_RECEIPT_INVALID")

    def test_keyboard_interrupt_preserves_completed_rows_and_stops(self):
        (self.input_dir / "z.pcd").touch()
        (self.input_dir / "zz.pcd").touch()

        def simulate(command, **kwargs):
            if Path(command[2]).stem == "z":
                raise KeyboardInterrupt
            return self.publish(command, **kwargs)

        code, run_dir, state, rows, calls = self.invoke(simulate, self.input_dir)
        self.assertEqual((code, state["status"], calls), (130, "INTERRUPTED", 2))
        self.assertEqual([row["status"] for row in rows], ["PASS", "INTERRUPTED"])
        self.assertIsNone(rows[1]["result_dir"])
        self.assertIn("已中断", (run_dir / "全部场景复核.html").read_text(encoding="utf-8"))

    def test_timeout_keeps_partial_staged_receipt_without_linking_preview(self):
        def simulate(command, **kwargs):
            folder = Path(command[3])
            ONECLICK.atomic_json(folder / "worker_receipt.json", dict(
                status="PREVIEW_COMPLETED_FULL_PENDING", modes=[dict(mode="preview")]
            ))
            raise subprocess.TimeoutExpired(command, 300)

        code, _, state, rows, _ = self.invoke(simulate, mode="staged")
        self.assertEqual((code, state["status"]), (1, "FAILED"))
        self.assertEqual((rows[0]["exit_code"], rows[0]["failure_reason"]), (124, "WATCHDOG_300S"))
        self.assertEqual(rows[0]["worker"]["status"], "PREVIEW_COMPLETED_FULL_PENDING")
        self.assertIsNone(rows[0]["result_dir"])
