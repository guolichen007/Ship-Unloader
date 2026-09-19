"""验收器自身的反例：后端拼接、缺帧、伪造合法、目标自包含、拒绝帧污染。"""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v14_runner", PROJECT / "tools/run_v14.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.NORMAL = runner.QUICK = ["A", "B"]


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.config = json.loads((PROJECT / "config/v14.json").read_text())
        self.config["acceptance"]["quick_frames"] = 3
        self.metadata = dict(filtered=False, profile="quick", gt_isolation=True, pcl_available=False)
        self.series, self.quality, self.poses, self.maps, self.timing = [], [], [], [], []
        for scene in runner.QUICK:
            for method in runner.METHODS:
                for lane in ("closed_loop", "benchmark"):
                    self.series.append(dict(scenario=scene, seed=42, method=method, lane=lane, frames=2, initialized=1,
                        backend_calls=2, valid_ratio=1, max_consecutive_failures=0, translation_p95_m=.01,
                        rotation_p95_deg=.1, last_valid=1, last_translation_m=.01, last_rotation_deg=.1,
                        static_samples=100, static_p95_m=.01, worst_frame_static_p95_m=.01))
                    for frame in range(0 if lane == "closed_loop" else 1, 3):
                        key = dict(scenario=scene, seed=42, frame=frame, method=method, lane=lane)
                        self.quality.append(dict(key, overlap=1, rmse=.01, fitness=.0001, raw_objective=1,
                            valid=1, mathematical_failure=0, backend_executed=1, request_fingerprint="same-input",
                            **{"H%d%d" % (i,j): 0 for i in range(6) for j in range(6)}))
                        self.poses.append(dict(key, valid=1, static_p50_m=.01, static_p95_m=.01, static_max_m=.01))
                        self.timing.append(dict(key, prepare_ms=1, registration_ms=1, map_ms=1, total_ms=3))
                        if lane == "closed_loop":
                            self.maps.append(dict(key, tracking_target_revision=1 if frame else 0, tracking_map_revision=1,
                                candidate_revision=0, active=100, candidates=0, quarantined=0, suspect=0, stable=100))

    def audit(self):
        (self.output / "metadata.json").write_text(json.dumps(self.metadata))
        for filename, rows in [("series.csv", self.series), ("quality.csv", self.quality), ("pose.csv", self.poses),
                               ("map.csv", self.maps), ("timing.csv", self.timing)]:
            with (self.output / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return runner.audit(self.output, "quick", self.config)[2]

    def test_same_backend_must_pass_every_scene(self):
        self.assertEqual(self.audit(), ["GICP", "VGICP"])
        for row in self.series:
            if (row["scenario"], row["method"]) in (("A", "GICP"), ("B", "VGICP")):
                row["translation_p95_m"] = .06
        self.assertEqual(self.audit(), [])

    def test_missing_pose_cannot_pass(self):
        self.poses.pop()
        with self.assertRaises(ValueError): self.audit()

    def test_missing_quality_cannot_pass(self):
        self.quality.pop()
        with self.assertRaises(ValueError): self.audit()

    def test_invalid_accepted_cannot_pass(self):
        self.quality[0]["mathematical_failure"] = 1
        with self.assertRaises(ValueError): self.audit()

    def test_benchmark_requires_identical_request(self):
        next(row for row in self.quality if row["lane"] == "benchmark")["request_fingerprint"] = "different"
        with self.assertRaises(ValueError): self.audit()

    def test_current_frame_cannot_match_its_own_updated_target(self):
        self.maps[1]["tracking_target_revision"] = 2
        with self.assertRaises(ValueError): self.audit()

    def test_rejected_frame_cannot_change_candidates(self):
        self.poses[1].update(valid=0, static_p50_m="", static_p95_m="", static_max_m="")
        self.maps[1]["candidate_revision"] = 1
        with self.assertRaises(ValueError): self.audit()

    def test_nonfinite_metrics_cannot_pass(self):
        self.quality[0]["rmse"] = float("nan")
        with self.assertRaises(ValueError): self.audit()

    def add_pcl_results(self):
        self.metadata["pcl_available"] = True
        for scene in runner.QUICK:
            for source, destination in ((self.quality, self.quality), (self.poses, self.poses)):
                row = next(r for r in source if r["scenario"] == scene and r["frame"] == 1 and r["lane"] == "benchmark")
                destination.append(dict(row, method="PCL_GICP", lane="pcl_comparison"))

    def test_pcl_comparison_required_when_enabled(self):
        self.metadata["pcl_available"] = True
        with self.assertRaisesRegex(ValueError, "PCL"): self.audit()
        self.add_pcl_results()
        self.assertEqual(self.audit(), ["GICP", "VGICP"])
        self.quality.pop()
        with self.assertRaisesRegex(ValueError, "PCL"): self.audit()

    def test_pcl_comparison_must_execute(self):
        self.add_pcl_results()
        self.quality[-1].update(valid=0, backend_executed=0)
        with self.assertRaisesRegex(ValueError, "PCL"): self.audit()

    def test_pcl_illegal_output_blocked_even_when_rejected(self):
        self.add_pcl_results()
        self.quality[-1].update(valid=0, mathematical_failure=1)
        with self.assertRaisesRegex(ValueError, "PCL"): self.audit()

    def test_pcl_cannot_report_evidence_when_disabled(self):
        self.add_pcl_results()
        self.metadata["pcl_available"] = False
        with self.assertRaisesRegex(ValueError, "PCL"): self.audit()


if __name__ == "__main__":
    unittest.main()
