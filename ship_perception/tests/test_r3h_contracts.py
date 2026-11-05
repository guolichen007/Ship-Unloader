"""R3-H geometry, recall snapshot identity, and output contract regressions."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ship_perception.perception.r3h_geometry import dimensions
from ship_perception.perception.recall_snapshot import RecallSnapshot
from ship_perception.perception.context import VesselEvidence, freeze
from ship_perception.perception.local_rim_evidence import plane


class R3HContracts(unittest.TestCase):
    def test_rotated_review_dimensions_keep_xy_and_z_independent(self):
        theta=.51
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        polygon=(np.array([[0.,0.],[8.,0.],[8.,3.],[0.,3.]])@rotation.T+[90.,-200.]).tolist()
        result=dimensions(polygon)
        self.assertAlmostEqual(result["length_m"],8.)
        self.assertAlmostEqual(result["width_m"],3.)
        self.assertAlmostEqual(result["polygon_area_m2"],24.)
        self.assertLess(result["max_orthogonality_error_deg"],1e-10)

    def test_snapshot_roundtrip_and_config_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"recall.snapshot"
            identity=dict(input_sha="input",config_sha="one",source_sha="code")
            cache=RecallSnapshot(path,identity)
            value=VesselEvidence(freeze(dict(vessel_hypothesis_id="A",local_axes=[[1.,0.],[0.,1.]])),
                                 np.array([0,4,9],dtype="<i8"),"expected")
            cache.save(dict(owner=value,grid=np.array([[np.nan,2.],[3.,4.]]),keyed={0:[1,2]}))
            restored=RecallSnapshot(path,identity)
            self.assertEqual(restored.audit["status"],"HIT")
            np.testing.assert_array_equal(restored.value["owner"].point_indexes,[0,4,9])
            self.assertFalse(restored.value["grid"].flags.writeable)
            changed=RecallSnapshot(path,dict(identity,config_sha="two"))
            self.assertEqual(changed.audit["status"],"REBUILD")
            self.assertIsNone(changed.value)
            path.write_bytes(b"interrupted")
            self.assertEqual(RecallSnapshot(path,identity).audit["status"],"REBUILD")

    def test_missing_machine_output_is_a_failure_and_empty_cells_are_explicit(self):
        script=Path(__file__).resolve().parents[2]/"tools/r3h_oneclick.py"
        spec=importlib.util.spec_from_file_location("r3h_oneclick_contract",script)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            with self.assertRaisesRegex(ValueError,"MISSING_OUTPUT"):
                module.verify_output(path,"sha")
            for name in ("opening_cells.json","05_hatch_corners.json","03_results.json"):
                (path/name).write_text("[]" if name=="opening_cells.json" else "{}")
            (path/"01_geometry_only.png").write_bytes(b"image")
            (path/"船舱四角长宽_简表.csv").write_text("header")
            (path/"hatches_compact.json").write_text(json.dumps(dict(scene_source_sha256="sha",vessels=[])))
            self.assertEqual(module.verify_output(path,"sha")["vessels"],[])
            with self.assertRaisesRegex(ValueError,"OUTPUT_INPUT_SHA_MISMATCH"):
                module.verify_output(path,"other")

    def test_vertical_wall_and_cargo_slope_have_different_plane_roles(self):
        from ship_perception.r1_static.run import resolve_config
        cfg,_=resolve_config()
        x,z=np.meshgrid(np.linspace(0,3,20),np.linspace(-2,0,15))
        wall=np.column_stack((x.ravel(),np.zeros(x.size),z.ravel()))
        self.assertTrue(plane(wall,cfg,False)["qualified"])
        x,y=np.meshgrid(np.linspace(0,3,20),np.linspace(-1,1,15))
        cargo=np.column_stack((x.ravel(),y.ravel(),(.5*y).ravel()))
        self.assertFalse(plane(cargo,cfg,False)["qualified"])


if __name__=="__main__":
    unittest.main()
