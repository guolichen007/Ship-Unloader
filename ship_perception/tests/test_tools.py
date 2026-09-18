"""Guard the validation boundary: bad provenance/OS/config cannot become PASS."""
import copy
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest

PROJECT = pathlib.Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


configure = module("configure", PROJECT / "tools/configure.py")
validate = module("validate", PROJECT / "scripts/validate_ubuntu20.py")


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((PROJECT / "config/m0.json").read_text())

    def test_config_schema_rejects_bad_values(self):
        for key, value in [("version", 2), ("mode", "REALTIME_MODE"), ("scan_duration_ns", 2**63),
                           ("drop_probability", 1.1), ("noise_sigma_m", float("nan")),
                           ("seed", -1), ("decimation", 0), ("queue_capacity", True),
                           ("drift_mps", [0, 0]), ("local_origin_world_m", [0, 0, float("inf")])]:
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                configure.validate_config(config)
        self.config["untracked_override"] = 1
        with self.assertRaises(ValueError):
            configure.validate_config(self.config)

    def test_generated_config_has_exact_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            output = pathlib.Path(temp) / "config.hpp"
            path = PROJECT / "config/m0.json"
            configure.generate(path, output, "a" * 40)
            text = output.read_text()
            self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(), text)
            self.assertIn("a" * 40, text)

    def test_only_ubuntu20_is_formal(self):
        self.assertTrue(validate.ubuntu20('ID=ubuntu\nVERSION_ID="20.04"\n'))
        self.assertFalse(validate.ubuntu20('ID=ubuntu\nVERSION_ID="22.04"\n'))
        self.assertFalse(validate.ubuntu20('ID=debian\nVERSION_ID="20.04"\n'))
        self.assertFalse(validate.ubuntu20("Windows"))

    def test_missing_or_stale_evidence_cannot_pass(self):
        valid = dict(gate="G1", git_sha="a" * 40, config_hash="b" * 64,
                     mode="EVALUATION_MODE", dataset_id="synthetic_ship_grid_v1",
                     timestamp=1, status="PASS_SYNTHETIC", error="", metrics={"rms": 0.001})
        validate.check_artifact(valid, "g1", "a" * 40, "b" * 64)
        for field in valid:
            if field == "error":
                continue
            broken = dict(valid)
            broken.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate.check_artifact(broken, "g1", "a" * 40, "b" * 64)
        for field, value in [("git_sha", "c" * 40), ("config_hash", "d" * 64),
                             ("status", "FAIL"), ("error", "assertion failed"),
                             ("metrics", {"rms": float("nan")})]:
            broken = dict(valid)
            broken[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate.check_artifact(broken, "g1", "a" * 40, "b" * 64)


if __name__ == "__main__":
    unittest.main()
