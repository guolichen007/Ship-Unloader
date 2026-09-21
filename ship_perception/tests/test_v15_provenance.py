import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from v15_config_identity import config_hash
from v15_holdout_ledger import append_access,read_ledger
from freeze_v15_evaluation import freeze


class Provenance(unittest.TestCase):
    def test_freeze_rejects_dirty_stale_failed_or_partial_evidence(self):
        identity=dict(git_sha='sha',algorithm_fingerprint='algo',config_hash='cfg',dataset_manifest_hash='data')
        common=dict(git_sha='sha',config_hash='cfg',worktree_dirty=False)
        reports=dict(development=dict(common,REAL_DEVELOPMENT_OPERATIONAL_GATE='PASS',RUN_COMPLETENESS='PASS'),
                     synthetic=dict(common,status='PASS',mode='full',acceptance_coverage='FULL_V15_ACCEPTANCE'),
                     integration=dict(common,status='PASS',profile='full',reused_binary_outputs=False,qualified_backends=['GICP'],both_backend_execution_integrity='PASS'))
        self.assertEqual(freeze(reports,identity,True)['status'],'FROZEN_FOR_HOLDOUT')
        with self.assertRaises(ValueError):freeze(reports,identity,False)
        for lane,key,value in [('development','REAL_DEVELOPMENT_OPERATIONAL_GATE','FAIL'),
                               ('synthetic','acceptance_coverage','DEVELOPMENT_SUBSET_NOT_FULL_V15_ACCEPTANCE'),
                               ('integration','reused_binary_outputs',True),('integration','both_backend_execution_integrity','FAIL'),
                               ('synthetic','git_sha','old'),('development','config_hash','old')]:
            bad=copy.deepcopy(reports);bad[lane][key]=value
            with self.subTest(lane=lane,key=key),self.assertRaises(ValueError):freeze(bad,identity,True)

    def test_config_line_endings_and_key_order(self):
        self.assertEqual(config_hash(b'{\r\n"a": 1, "b": 0.5\r\n}'),config_hash(b'{"b":0.5,"a":1}'))
        self.assertNotEqual(config_hash('{"a":1}'),config_hash('{"a":2}'))

    def test_audit_is_not_scoring_and_contamination_is_permanent(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=Path(directory)/"ledger.jsonl"
            identity=dict(git_sha="a",algorithm_fingerprint="b",config_hash="c",dataset_manifest_hash="d")
            audit=append_access(ledger,"FILE_CODEC_AUDIT_NOT_ALGORITHM_SCORING",identity)
            self.assertFalse(audit["holdout_algorithm_scored"])
            first=append_access(ledger,"ALGORITHM_SCORING",identity);self.assertEqual(first["HOLDOUT_CONTAMINATED"],"NO")
            docs=copy.deepcopy(identity);docs["git_sha"]="documentation-only"
            self.assertEqual(append_access(ledger,"IDENTITY_CHECK",docs)["HOLDOUT_CONTAMINATED"],"NO")
            changed=copy.deepcopy(identity);changed["config_hash"]="changed-output-parameter"
            self.assertEqual(append_access(ledger,"IDENTITY_CHECK",changed)["HOLDOUT_CONTAMINATED"],"YES")
            self.assertEqual(append_access(ledger,"ALGORITHM_SCORING",identity)["HOLDOUT_CONTAMINATED"],"YES")

    def test_tampered_access_history_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=Path(directory)/"ledger.jsonl"
            append_access(ledger,"ALGORITHM_SCORING",dict(git_sha="a",algorithm_fingerprint="b",config_hash="c",dataset_manifest_hash="d"))
            ledger.write_text(ledger.read_text().replace('"config_hash": "c"','"config_hash": "changed"'))
            with self.assertRaises(ValueError):read_ledger(ledger)


if __name__=="__main__":unittest.main()
