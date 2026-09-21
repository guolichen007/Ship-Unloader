"""反作弊与序列化契约测试：oracle 仅用于攻击评分器，从不传入产品。"""
import copy
import json
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(SOURCE),str(SOURCE/"tools")]
from evaluation.v15_synthetic import fixture,evaluate
from evaluation.v15_partial import polygon_segments
from v15_artifacts import validate_model,write_debug_cloud
from v15_pcd import write_cache

C=json.loads((SOURCE/"config/v15.json").read_text())
THRESHOLDS=dict(**C["evaluation"],high_quality_threshold=C["boundary"]["high_quality_threshold"])


def oracle_model(truth):
    # Test-only fabrication: this deliberately knows GT. A model like this is
    # insufficient evidence without the separate process-level isolation gate.
    model=dict(schema_version="ship_perception.v15.model.1",software_git_sha="0"*40,config_hash="0"*64,
        review_status="UNREVIEWED",control_ready=False,canonical=False,heading_semantics="UNRESOLVED",
        candidate_id_semantics="LOCAL_TO_THIS_MODEL",frame_resolved=True,input_alignment="ALIGNED",T_B_input=np.eye(4).tolist(),
        deck=dict(valid=True,normal=[0,0,1],offset=0,support_region=[],support_count=len(truth.visible_deck),
                  support_ratio=1.,residual_p50_m=0.,residual_p95_m=0.,quality_score=1.),hatches=[],structures=[],warnings=[])
    for i,poly in enumerate(truth.polygons):
        edges=[]
        for a,b in polygon_segments(poly):
            length=float(np.linalg.norm(b-a));endpoint=dict(endpoint_bounded=True,endpoint_uncertainty_m=.05,termination_evidence="OBSERVED_EDGE_INTERSECTION")
            edges.append(dict(a=[*a,0],b=[*b,0],side="INNER_OPENING_FACE",visibility="VISIBLE",evidence_flags=1,
                              evidence_mechanism="TEST_ORACLE",support_count=100,observed_support_length=length,
                              extrapolation_length=0.,fit_residual_p50_m=0.,fit_residual_p95_m=0.,normal_uncertainty_rad=0.,
                              quality_score=1.,start=copy.deepcopy(endpoint),end=copy.deepcopy(endpoint),last_observed_ns=None,support_intervals=[[0.,length]]))
        model["hatches"].append(dict(candidate_id=f"candidate_{i}",status="COMPLETE_OBSERVED",nominal_polygon=[[*q,0] for q in poly],
                                    center=[*poly.mean(axis=0),0],extent=[*np.ptp(poly,axis=0),0],observed_edge_count=len(edges),
                                    inferred_edge_count=0,completeness=1.,quality_score=1.,boundaries=edges,warnings=[]))
    return json.loads(json.dumps(model))


class Contracts(unittest.TestCase):
    def test_hidden_side_still_requires_accurate_deck(self):
        truth=fixture("PAIRED_040_OUTER",42);actual=oracle_model(truth)
        actual["hatches"][0].update(status="PARTIAL",nominal_polygon=None,boundaries=[])
        self.assertEqual(evaluate(actual,truth,THRESHOLDS)["status"],"PASS")
        actual["deck"]["offset"]=.2
        self.assertIn("DECK_OFFSET",evaluate(actual,truth,THRESHOLDS)["failures"])

    def test_hidden_inner_fixture_has_no_dense_top_or_inner_wall(self):
        for name in ("OUTER_ONLY","PAIRED_020_OUTER","PAIRED_040_OUTER","PAIRED_020_BLIND","PAIRED_040_BLIND"):
            points=fixture(name,42).runtime
            inner_strip=(np.abs(points[:,0]-4)<.03)&(np.abs(points[:,1])<1)
            self.assertFalse(inner_strip.any(),name)
        # Zero thickness is the same step geometry; visibility names must not
        # invent a second physical surface or move the scoring boundary.
        clouds=[fixture('PAIRED_000_'+view,42).runtime for view in ('INNER','OUTER','BOTH','BLIND')]
        def ordered(p):return p[np.lexsort((p[:,2],p[:,1],p[:,0]))]
        for p in clouds[1:]:np.testing.assert_array_equal(ordered(clouds[0]),ordered(p))

    def setUp(self):
        self.truth=fixture("NONRECTANGULAR",42);self.model=oracle_model(self.truth)

    def test_scorer_accepts_exact_control(self):
        validate_model(self.model)
        self.assertEqual(evaluate(self.model,self.truth,THRESHOLDS)["status"],"PASS")

    def test_forced_rectangle_fails(self):
        fake=oracle_model(fixture("SINGLE_HATCH",42))
        self.assertEqual(evaluate(fake,self.truth,THRESHOLDS)["status"],"FAIL")

    def test_all_partial_and_all_inferred_fail(self):
        self.model["hatches"][0]["status"]="PARTIAL"
        self.assertEqual(evaluate(self.model,self.truth,THRESHOLDS)["status"],"FAIL")
        self.model["hatches"][0]["status"]="COMPLETE_WITH_INFERENCE"
        for e in self.model["hatches"][0]["boundaries"]:e["evidence_flags"]=4
        self.assertEqual(evaluate(self.model,self.truth,THRESHOLDS)["status"],"FAIL")

    def test_duplicates_and_extra_complete_fail(self):
        clone=copy.deepcopy(self.model["hatches"][0]);clone["candidate_id"]="clone"
        self.model["hatches"].append(clone)
        self.assertEqual(evaluate(self.model,self.truth,THRESHOLDS)["status"],"FAIL")

    def test_duplicate_confirmed_primitives_cannot_hide_in_clean_hatch(self):
        edge=self.model["hatches"][0]["boundaries"][0]
        p=dict(kind="COAMING_OR_HOLD_WALL",segment=copy.deepcopy(edge),reason="TEST")
        self.model["structures"]=[p,copy.deepcopy(p)]
        self.assertIn("DUPLICATE_CONFIRMED_COAMING",evaluate(self.model,self.truth,THRESHOLDS)["failures"])

    def test_correct_beam_count_with_wrong_geometry_fails(self):
        truth=fixture("MULTI_HATCH_BEAM",42);m=oracle_model(truth)
        edge=copy.deepcopy(m["hatches"][0]["boundaries"][0]);edge['a']=[20,0,1];edge['b']=[24,0,1]
        m['structures']=[dict(kind="BEAM_OR_PARTITION",segment=edge,reason="TEST")]
        self.assertIn("BEAM_GEOMETRY",evaluate(m,truth,THRESHOLDS)["failures"])

    def test_short_accurate_segments_fail_coverage(self):
        for edge in self.model["hatches"][0]["boundaries"]:
            edge["support_intervals"]=[[0,.1]];edge["observed_support_length"]=.1
        result=evaluate(self.model,self.truth,THRESHOLDS)
        self.assertEqual(result["status"],"FAIL");self.assertIn("EDGE_COVERAGE",result["failures"])

    def test_negative_control_rejects_good_looking_false_hatch(self):
        self.assertEqual(evaluate(self.model,fixture("PURE_DECK",42),THRESHOLDS)["status"],"FAIL")

    def test_historical_and_fake_endpoint_fail_contract(self):
        e=self.model["hatches"][0]["boundaries"][0];e["evidence_flags"]=8
        with self.assertRaises(ValueError):validate_model(self.model)
        e["evidence_flags"]=1;e["start"]["endpoint_bounded"]=False
        with self.assertRaises(ValueError):validate_model(self.model)

    def test_invalid_se3_nan_and_extra_gt_field_fail_contract(self):
        self.model["T_B_input"][0][0]=1.01
        with self.assertRaises(ValueError):validate_model(self.model)
        self.model["T_B_input"][0][0]=float("nan")
        with self.assertRaises(ValueError):validate_model(self.model)
        self.model["T_B_input"][0][0]=1;self.model["GT"]=[]
        with self.assertRaises(ValueError):validate_model(self.model)

    def test_support_gap_cannot_be_hidden(self):
        e=self.model["hatches"][0]["boundaries"][0];e["support_intervals"]=[[0,.1]]
        with self.assertRaises(ValueError):validate_model(self.model)

    def test_debug_cloud_uses_runtime_points_and_never_colours_gap_observed(self):
        model=oracle_model(fixture('SINGLE_HATCH',42));edge=model['hatches'][0]['boundaries'][0]
        edge['support_intervals']=[[0,.1]]
        points=np.array([[0,-2,0],[-3.95,-2,0]],dtype='f4')
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);cache=folder/'input.xyzbin';ply=folder/'debug.ply';write_cache(points,cache)
            result=write_debug_cloud(ply,cache,model)
            def read():
                payload=ply.read_bytes().split(b'end_header\n',1)[1]
                return np.frombuffer(payload,dtype=[('xyz','<f8',(3,)),('rgb','u1',(3,))])
            output=read();np.testing.assert_array_equal(output['xyz'],points)
            self.assertEqual(result['point_count'],2)
            np.testing.assert_array_equal(output['rgb'][0],[140,140,140])
            edge['evidence_flags']=4;write_debug_cloud(ply,cache,model)
            np.testing.assert_array_equal(read()['rgb'][0],[240,160,30])


if __name__=="__main__":unittest.main()
