"""Delivery entry preserves input identity, explicit mode and honest failures."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "hatch_review_runner", ROOT / "scripts/hatch_review_runner.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReviewBatchLauncherContracts(unittest.TestCase):
    @unittest.skipUnless(
        os.name == "posix" and shutil.which("bash"), "POSIX shell entry test"
    )
    def test_shell_relative_inputs_resolve_from_callers_directory(self):
        with tempfile.TemporaryDirectory(prefix="review caller ") as tmp:
            root = Path(tmp)
            folder = root / "input spaced"
            folder.mkdir()
            pcd = folder / "sample.PCD"
            pcd.write_bytes(b"input")
            manifest = root / "batch.json"
            manifest.write_text(
                json.dumps(
                    dict(
                        schema="ship_perception.review_batch.1",
                        scenes=[dict(scene="S", pcd="input spaced/sample.PCD")],
                    )
                ),
                encoding="utf-8",
            )
            for option, value in [
                ("--pcd", "input spaced/sample.PCD"),
                ("--pcd-dir", "input spaced"),
                ("--manifest", "batch.json"),
            ]:
                result = subprocess.run(
                    [
                        "bash",
                        str(ROOT / "scripts/run_ship_hatch_review.sh"),
                        option,
                        value,
                        "--list-only",
                    ],
                    cwd=root,
                    env=dict(os.environ, PYTHON_BIN=sys.executable),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)[0]["pcd"], str(pcd))

    def args(self, **kw):
        values = dict(
            manifest=None,
            pcd=None,
            pcd_dir=None,
            frame_id="pcd",
            operator_vessel_roi=None,
        )
        values.update(kw)
        return argparse.Namespace(**values)

    def test_manifest_cannot_pass_reference_coordinates_or_expected_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inputs.json"
            for extra in ["evaluation_reference", "expected_hatch_count", "command"]:
                path.write_text(
                    json.dumps(
                        dict(
                            schema="ship_perception.review_batch.1",
                            scenes=[
                                dict(scene="S", pcd="map.pcd", **{extra: "forbidden"})
                            ],
                        )
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError, "UNSUPPORTED_REVIEW_INPUT_FIELD"
                ):
                    runner.load_scenes(self.args(manifest=path))

    def test_derived_input_requires_explicit_parent_and_crop_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inputs.json"
            path.write_text(
                json.dumps(
                    dict(
                        schema="ship_perception.review_batch.1",
                        scenes=[
                            dict(
                                scene="crop",
                                pcd="crop.pcd",
                                input_role="DERIVED_DIAGNOSTIC_CROP",
                            )
                        ],
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "DERIVED_INPUT_REQUIRES_PARENT"):
                runner.load_scenes(self.args(manifest=path))

    def test_directory_discovers_uppercase_pcd_and_does_not_implicitly_share_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["a.pcd", "b.PCD", "notes.txt"]:
                (root / name).write_text("", encoding="utf-8")
            self.assertEqual(len(runner.load_scenes(self.args(pcd_dir=root))), 2)
            with self.assertRaisesRegex(
                ValueError, "ONE_OPERATOR_ROI_REQUIRES_ONE_PCD"
            ):
                runner.load_scenes(
                    self.args(pcd_dir=root, operator_vessel_roi=root / "roi.json")
                )

    def test_input_hash_mismatch_prevents_detector_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pcd = root / "map.pcd"
            pcd.write_text("different input", encoding="utf-8")
            result = runner.run_scene(
                dict(
                    scene="S",
                    pcd=str(pcd),
                    frame_id="pcd",
                    input_role="PRIMARY_ORIGINAL",
                    input_sha256="0" * 64,
                ),
                root / "result",
                "review-first",
            )
            self.assertEqual(result["status"], "INPUT_OR_OUTPUT_CONTRACT_FAILED")
            self.assertIn("REVIEW_INPUT_HASH_MISMATCH", result["error"])
            self.assertIsNone(result["review_polygon_count"])
            self.assertFalse((root / "result/terminal.log").exists())

    def test_watchdog_does_not_become_zero_hatch_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / ".review_run_test"
            folder.mkdir()
            (folder / "RUN_TIMEOUT.json").write_text("{}", encoding="utf-8")
            result = runner.output_record(dict(scene="S"), root, -9, 300.0, [])
            self.assertEqual(result["status"], "RUN_TIMEOUT")
            self.assertTrue(result["watchdog_hit"])
            self.assertIsNone(result["review_polygon_count"])

    def test_interrupt_terminates_isolated_child_and_preserves_interrupted_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pcd = root / "map.pcd"
            pcd.write_bytes(b"input")
            worker = MagicMock()
            worker.wait.side_effect = KeyboardInterrupt
            with patch.object(
                runner.subprocess, "Popen", return_value=worker
            ), patch.object(runner, "terminate_child") as stop:
                result = runner.run_scene(
                    dict(
                        scene="S",
                        pcd=str(pcd),
                        frame_id="pcd",
                        input_role="PRIMARY_ORIGINAL",
                    ),
                    root / "run",
                    "review-first",
                )
            stop.assert_called_once_with(worker)
            self.assertEqual(result["status"], "INTERRUPTED")
            self.assertEqual(result["exit_code"], 130)
            self.assertIsNone(result["review_polygon_count"])

    def test_no_complete_review_is_not_reported_as_recognition_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "run"
            folder.mkdir()
            packet = folder / "03_results.json"
            packet.write_text(
                json.dumps(
                    dict(
                        geometry_only=dict(
                            hatches=[],
                            hatch_count_status="REVIEW_ONLY_NOT_TOTAL_HATCH_COUNT",
                        ),
                        software_source_identity={},
                        config_hash="c",
                        input_sha256="i",
                        berth_selection=dict(status="MULTI_VESSEL_BERTH_AMBIGUOUS"),
                    )
                ),
                encoding="utf-8",
            )
            (folder / "05_hatch_corners.json").write_text(
                json.dumps(
                    dict(source_result_sha256=runner.digest(packet), hatches=[])
                ),
                encoding="utf-8",
            )
            result = runner.output_record(dict(scene="S"), root, 0, 1.0, [])
            self.assertEqual(result["status"], "UNRESOLVED_NO_COMPLETE_REVIEW")
            self.assertEqual(result["review_polygon_count"], 0)
            self.assertFalse(result["precision_certified"])
            self.assertEqual(result["selection_status"], "MULTI_VESSEL_BERTH_AMBIGUOUS")

    def test_primary_and_derived_stay_separate_and_gallery_escapes_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                dict(
                    scene="<S>",
                    input_role="PRIMARY_ORIGINAL",
                    status="REVIEW_ARTIFACTS_READY_UNCERTIFIED",
                    elapsed_s=1.0,
                    hatches=[],
                ),
                dict(
                    scene="crop",
                    input_role="DERIVED_DIAGNOSTIC_CROP",
                    status="RUN_TIMEOUT",
                    elapsed_s=300.0,
                    hatches=[],
                ),
            ]
            runner.save_progress(root, rows, {}, "review-first")
            self.assertEqual(
                len(
                    json.loads(
                        (root / "PRIMARY_SCENE_MATRIX.json").read_text(encoding="utf-8")
                    )
                ),
                1,
            )
            self.assertEqual(
                len(
                    json.loads(
                        (root / "DERIVED_CROP_MATRIX.json").read_text(encoding="utf-8")
                    )
                ),
                1,
            )
            text = (root / "index.html").read_text(encoding="utf-8")
            self.assertIn("&lt;S&gt;", text)
            self.assertNotIn("<h3><S>", text)
            self.assertIn("RUN_TIMEOUT", text)


if __name__ == "__main__":
    unittest.main()
