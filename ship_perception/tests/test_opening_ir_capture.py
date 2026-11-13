import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from run_opening_ir_shadow import NativeObserver


class CaptureLifecycle(unittest.TestCase):
    def record(self, active, status, vessel_count):
        observer = NativeObserver()
        observer.bundle.update(input_sha256="input", source_identity={"head": "sha"})
        observer.native_result(
            {
                "geometry_only": {"review_candidate_count": 0},
                "berth_selection": {"status": status, "active_vessel_id": active},
                "scene": {"vessel_hypothesis_count": vessel_count},
                "config_sha256": "config",
            }
        )
        return observer.bundle

    def test_missing_or_ambiguous_target_has_explicit_terminal_state(self):
        for count in (0, 2):
            with self.subTest(vessel_count=count):
                bundle = self.record(None, "MULTI_VESSEL_BERTH_AMBIGUOUS", count)
                terminal = bundle["capture_terminal"]
                self.assertEqual(terminal["status"], "NO_ACTIVE_VESSEL")
                self.assertEqual(terminal["ir_status"], "IR_NOT_COMPILED_NO_ACTIVE_VESSEL")
                self.assertEqual(terminal["selector_status"], "SELECTOR_NOT_RUN")
                self.assertIsNone(terminal["berth_selection"]["active_vessel_id"])
                self.assertEqual(terminal["scene"]["vessel_hypothesis_count"], count)
                self.assertNotIn("vessel_id", bundle)

    def test_active_target_keeps_native_capture_contract(self):
        bundle = self.record("V0000", "SINGLE_VESSEL_REVIEW_NO_BERTH_CERTIFICATION", 1)
        self.assertEqual(bundle["native_geometry"], {"review_candidate_count": 0})
        self.assertNotIn("capture_terminal", bundle)
