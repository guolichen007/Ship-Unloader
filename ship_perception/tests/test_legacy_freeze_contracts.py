"""B2 integrity and the full-chain reference have different contracts."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from ship_perception.perception.legacy_expert import (
    verify_legacy_b2_core,
    legacy_full_chain_reference,
)


class LegacyFreezeContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        package = Path(__file__).resolve().parents[1]
        names = ("legacy_b2_core_freeze.json", "legacy_proposal_freeze.json")
        paths = set("config/" + n for n in names)
        for name in names:
            paths.update(
                json.loads((package / "config" / name).read_text(encoding="utf-8"))[
                    "files"
                ]
            )
        for rel in paths:
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(package / rel, target)

    def test_changing_a_real_b2_dependency_fails(self):
        self.assertEqual(verify_legacy_b2_core(self.root)["status"], "PASS")
        p = self.root / "r1_static/hatch_proposal.py"
        p.write_bytes(p.read_bytes() + b"\n# changed B2 dependency\n")
        with self.assertRaisesRegex(ValueError, "LEGACY_EXPERT_BEHAVIOR_CHANGED"):
            verify_legacy_b2_core(self.root)

    def test_approved_s4_changes_leave_b2_core_frozen_and_reference_changed(self):
        manifest = self.root / "config/legacy_proposal_freeze.json"
        before = manifest.read_bytes()
        p = self.root / "r1_static/perimeter_deck.py"
        p.write_bytes(p.read_bytes() + b"\n# authorized S4-only edit\n")
        self.assertEqual(verify_legacy_b2_core(self.root)["status"], "PASS")
        report = legacy_full_chain_reference(self.root)
        self.assertEqual(report["classification"], "CHANGED_EXPECTED_S4_ONLY")
        self.assertFalse(report["full_chain_frozen_pass"])
        self.assertEqual(manifest.read_bytes(), before)
        self.assertTrue(
            all(
                r["reference_sha256"] != r["current_sha256"]
                for r in report["changed_files"]
            )
        )


if __name__ == "__main__":
    unittest.main()
