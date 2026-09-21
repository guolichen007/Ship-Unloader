"""验收阈值不可借参数化被放宽，几何除法域必须有效。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'tools'))
from configure_v15 import generate


class Configuration(unittest.TestCase):
    def attempt(self,section=None,key=None,value=None):
        data=json.loads((SOURCE/'config/v15.json').read_text(encoding='utf-8'))
        if section:data[section][key]=value
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.json';path.write_text(json.dumps(data),encoding='utf-8')
            generate(path,Path(d)/'generated.hpp')

    def test_frozen_contract_cannot_be_weakened(self):
        for key,value in [('boundary_p95_m',.16),('corner_p95_m',.16),('real_frame_min',6),('real_joint_min',6),
                          ('real_deck_min',6),('real_association_min',5),('coverage_total_min',.89),
                          ('coverage_edge_min',.79),('deck_distance_p95_m',.11),('full_frames',199),('quick_frames',49)]:
            with self.subTest(key=key),self.assertRaises(ValueError):self.attempt('evaluation',key,value)

    def test_defaults_and_stronger_geometry_gate(self):
        self.attempt();self.attempt('evaluation','boundary_p95_m',.10)

    def test_invalid_geometry_domain_and_multiple_inferred_edges(self):
        for section,key,value in [('geometry','normal_radius_m',0),('boundary','profile_step_m',0),
                                  ('roi','support_search_m',0),('boundary','max_inferred_edges',2),
                                  ('geometry','planarity_min',1.1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):self.attempt(section,key,value)


if __name__=='__main__':unittest.main()
