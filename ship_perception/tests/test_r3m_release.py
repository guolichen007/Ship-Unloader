import copy
import csv
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

from ship_perception.perception.review_ambiguity import classify_edge


class MaterialAmbiguityContracts(unittest.TestCase):
    def test_neighbor_context_cannot_certify_an_ambiguous_parallel_edge(self):
        a = dict(mode_id="near", holdout_evidence=[])
        b = dict(
            mode_id="far", holdout_evidence=[dict(passed=True, outside_plateau=True)]
        )
        original = copy.deepcopy(b)
        result = classify_edge({}, b, None, [a, b])
        self.assertEqual(result["state"], "REVIEW_AMBIGUOUS")
        self.assertEqual(result["own_only_alternative_mode_ids"], ["near"])
        self.assertTrue(result["neighbor_context_changes_selection"])
        self.assertFalse(result["control_eligible"])
        self.assertFalse(result["steel_certified"])
        self.assertEqual(b, original)

    def test_missing_platform_is_explicit_without_inventing_a_veto_or_material(self):
        a = dict(
            mode_id="only", holdout_evidence=[dict(passed=True, outside_plateau=False)]
        )
        r = classify_edge({}, a, a, [])
        self.assertEqual(r["state"], "REVIEW_ESTIMATED")
        self.assertEqual(r["outside_platform_witness_sections"], 0)
        self.assertEqual(r["material_role"], "NOT_IDENTIFIED_AS_STEEL")
        self.assertTrue(r["platform_absence_is_not_veto"])


class ReleaseGateContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "release_gates",
            Path(__file__).resolve().parents[2] / "tools/r3m_release_gates.py",
        )
        cls.gates = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gates)

    def test_missing_human_signoff_or_changed_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "receipt.json"
            p.write_bytes(b"verified receipt")
            gate = dict(
                status="PASS",
                source_sha="a" * 40,
                evidence=[
                    dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                ],
            )
            manifest = dict(
                source_sha="a" * 40,
                gates={k: copy.deepcopy(gate) for k in self.gates.REQUIRED},
            )
            table = Path(directory) / "synthetic-review.csv"

            def write_review(decision):
                with table.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=["开口ID", "几何摘要", "二维结论", "审核人"]
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "开口ID": "SYNTHETIC_UNIT_ONLY",
                            "几何摘要": "test-geometry",
                            "二维结论": decision,
                            "审核人": "SYNTHETIC_UNIT_REVIEWER",
                        }
                    )

            write_review("ACCEPT")
            manifest["human_review"] = dict(
                path=table.name,
                opening_geometry_sha256={"SYNTHETIC_UNIT_ONLY": "test-geometry"},
            )

            def sign_table_digest():
                manifest["gates"]["human_2d"]["evidence"] = [
                    dict(
                        path=table.name,
                        sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                    )
                ]

            sign_table_digest()
            r = self.gates.evaluate(manifest, directory)
            self.assertEqual(r["decision"], "GO_OFFLINE_REVIEW_APPLICATION_ONLY")
            self.assertFalse(r["device_control_allowed"])
            manifest["gates"]["human_2d"]["status"] = "PENDING"
            self.assertEqual(
                self.gates.evaluate(manifest, directory)["decision"], "NO_GO"
            )
            manifest["gates"]["human_2d"]["status"] = "PASS"
            write_review("PENDING")
            sign_table_digest()
            self.assertEqual(
                self.gates.evaluate(manifest, directory)["decision"], "NO_GO"
            )
            write_review("ACCEPT")
            sign_table_digest()
            p.write_bytes(b"changed receipt")
            self.assertEqual(
                self.gates.evaluate(manifest, directory)["decision"], "NO_GO"
            )

    def test_a_different_ci_sha_is_not_same_sha_ci(self):
        manifest = dict(
            source_sha="a" * 40,
            gates={
                "same_sha_ci": dict(status="PASS", source_sha="b" * 40, evidence=[])
            },
        )
        r = self.gates.evaluate(manifest, ".")
        self.assertIn("SOURCE_SHA_MISMATCH", r["gates"]["same_sha_ci"]["reasons"])
        self.assertEqual(r["decision"], "NO_GO")
