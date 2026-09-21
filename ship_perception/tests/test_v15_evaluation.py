"""弱标注评估反例：不得隐去重复预测、无效扫描或推断误差。"""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE/"tools"))
from evaluation.v15_partial import evaluate_annotation
from run_v15_development import operational

CONFIG = json.loads((SOURCE/"config/v15.json").read_text())["evaluation"]
LABEL = dict(corners=[[-4,-2],[4,-2],[4,2],[-4,2]])


def hatch(name, dx=0, flag=1, status="COMPLETE_OBSERVED"):
    p = np.asarray(LABEL["corners"], float)+[dx,0]
    return dict(candidate_id=name, status=status, boundaries=[dict(a=[*p[i],0],b=[*p[(i+1)%4],0],evidence_flags=flag,
        support_intervals=[[0,float(np.linalg.norm(p[(i+1)%4]-p[i]))]]) for i in range(4)])


def model(*hatches):
    return dict(T_B_input=np.eye(4).tolist(), hatches=list(hatches))


class EvaluationTests(unittest.TestCase):
    def test_missing_support_cannot_be_filled_by_segment_envelope(self):
        short=hatch("short")
        for edge in short["boundaries"]:edge["support_intervals"]=[[0,.1]]
        short["boundaries"]=short["boundaries"][:1]
        result=evaluate_annotation(model(short),LABEL,CONFIG)
        self.assertEqual(result["status"],"UNMATCHED_LABEL")
        self.assertLess(result["candidates"][0]["label_coverage"],.6)
        del short["boundaries"][0]["support_intervals"]
        with self.assertRaises(ValueError):evaluate_annotation(model(short),LABEL,CONFIG)

    def test_unlabeled_neighbor_is_not_false_positive(self):
        result = evaluate_annotation(model(hatch("a"),hatch("neighbor",20)), LABEL, CONFIG)
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["local_duplicate_count"], 0)
        self.assertEqual(result["candidates"][1]["status"], "UNVERIFIED_CANDIDATE")

    def test_equal_duplicates_are_ambiguous(self):
        result = evaluate_annotation(model(hatch("a"),hatch("b")), LABEL, CONFIG)
        self.assertEqual(result["status"], "AMBIGUOUS_MATCH")
        self.assertEqual(result["local_duplicate_count"], 1)

    def test_partial_fragment_is_audited(self):
        fragment = hatch("fragment",status="PARTIAL")
        fragment["boundaries"] = fragment["boundaries"][:2]
        result = evaluate_annotation(model(hatch("a"),fragment), LABEL, CONFIG)
        self.assertTrue(result["candidates"][1]["duplicate"])

    def test_inferred_error_does_not_include_observed(self):
        actual = hatch("a")
        actual["boundaries"].append(dict(a=[-4,3,0],b=[4,3,0],evidence_flags=4))
        result = evaluate_annotation(model(actual), LABEL, CONFIG)["candidates"][0]
        self.assertAlmostEqual(result["observed_error"]["P95"],0)
        self.assertAlmostEqual(result["inferred_error"]["P95"],1)

    def test_empty_runtime_cannot_pass_operational(self):
        records = [dict(scan_id=str(i),annotation_count=int(i<8),frame_resolved=False,deck_resolved=False,
                        consistent_unique_association=False,nondeck_primitive_count=0,hatch_count=0,
                        execution_status="COMPLETED") for i in range(9)]
        result = operational(records, CONFIG)
        self.assertEqual(result["RUN_COMPLETENESS"],"PASS")
        self.assertEqual(result["REAL_DEVELOPMENT_OPERATIONAL_GATE"],"FAIL")
        with self.assertRaises(ValueError):
            operational(records[:-1], CONFIG)

    def test_duplicate_annotations_never_add_scan_credit(self):
        records = [dict(scan_id=str(i),annotation_count=3 if i<8 else 0,frame_resolved=i<7,deck_resolved=i<7,
                        consistent_unique_association=i<5,nondeck_primitive_count=4,hatch_count=1,
                        execution_status="COMPLETED") for i in range(9)]
        self.assertEqual(operational(records, CONFIG)["REAL_DEVELOPMENT_OPERATIONAL_GATE"],"FAIL")
        records[5]["consistent_unique_association"]=True
        self.assertEqual(operational(records, CONFIG)["REAL_DEVELOPMENT_OPERATIONAL_GATE"],"PASS")


if __name__ == "__main__":unittest.main()
