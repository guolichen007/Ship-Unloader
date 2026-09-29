"""R2H-B2 provider independence, partial state and vessel isolation."""

import json
import unittest

import numpy as np

from ship_perception.r1_static.hatch_proposal import (
    _bev_provider, _raw3d_provider, discover_scene_hatches, propose_for_vessel,
)
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG
from ship_perception.tests.test_r1_boundary_topology import CONFIG
from ship_perception.tests.test_r1_rectangle_fusion import raw_scene
from ship_perception.tests.test_r2h_a_counterexamples import vessel_cloud


RESEARCH = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))


def synthetic_vessel():
    return dict(vessel_hypothesis_id="V_TEST", source_support_ids=["S_TEST"],
                local_axes=[[1, 0], [0, 1]], classification_status="VESSEL_HYPOTHESIS")


class IndependentProviders(unittest.TestCase):
    def test_height_provider_miss_keeps_bev_and_raw3d_proposal(self):
        result = propose_for_vessel(raw_scene(), synthetic_vessel(), CONFIG, RESEARCH,
                                    height_override=[])
        self.assertEqual(result["provider_summary"]["height"]["region_count"], 0)
        self.assertGreater(result["provider_summary"]["bev_count"], 0)
        self.assertGreater(result["provider_summary"]["raw3d_count"], 0)
        self.assertTrue(any(row["state"] == "HATCH_PROPOSAL"
                            for row in result["hatch_hypotheses"]))
        self.assertTrue(all(row["polygon_xy"] is None
                            for row in result["hatch_hypotheses"]))

    def test_three_strong_one_weak_is_partial_without_polygon(self):
        points = raw_scene()
        points[(points[:, 0] <= .3) &
               ((points[:, 1] < 8) | (points[:, 1] > 12)), 2] = 0
        result = propose_for_vessel(points, synthetic_vessel(), CONFIG, RESEARCH,
                                    height_override=[])
        partial = [row for row in result["hatch_hypotheses"]
                   if row["state"] == "PARTIAL_HATCH"]
        self.assertTrue(partial)
        self.assertEqual(partial[0]["measured_strong_side_count"], 3)
        self.assertIn(partial[0]["side_evidence"]["U0"]["state"],
                      ("MEASURED_WEAK", "MISSING"))
        self.assertIsNone(partial[0]["polygon_xy"])

    def test_raised_crossbeam_strip_does_not_become_hatch(self):
        points = raw_scene()
        points[(points[:, 1] > 9) & (points[:, 1] < 11) &
               (points[:, 0] > 0) & (points[:, 0] < 40), 2] = 1
        result = propose_for_vessel(points, synthetic_vessel(), CONFIG, RESEARCH,
                                    height_override=[])
        strips = [row for row in result["hatch_hypotheses"]
                  if row["bounds_axial"] is not None and
                  abs(row["bounds_axial"][1] - 9) < CONFIG["roi"]["support_search_m"] / 2
                  and abs(row["bounds_axial"][3] - 11) < CONFIG["roi"]["support_search_m"] / 2]
        self.assertTrue(strips)
        self.assertTrue(all(row["state"] != "PROVISIONAL_HATCH" for row in strips))
        self.assertTrue(any("RAISED_INTERIOR_ROLE_AMBIGUOUS" in row["role_conflicts"]
                            for row in strips))

    def test_two_observed_sides_never_form_complete_hatch(self):
        points = raw_scene()
        points[(points[:, 0] < 1) | (points[:, 0] > 39), 2] = 0
        result = propose_for_vessel(points, synthetic_vessel(), CONFIG, RESEARCH,
                                    height_override=[])
        self.assertFalse(any(row["state"] == "PROVISIONAL_HATCH"
                             for row in result["hatch_hypotheses"]))

    def test_four_raster_sides_need_independent_steel_role(self):
        result = propose_for_vessel(raw_scene(), synthetic_vessel(), CONFIG, RESEARCH,
                                    height_override=[])
        self.assertTrue(any(row["measured_strong_side_count"] == 4
                            for row in result["hatch_hypotheses"]))
        self.assertFalse(any(row["state"] == "PROVISIONAL_HATCH"
                             for row in result["hatch_hypotheses"]))

    def test_independent_provider_calls_keep_observation_provenance(self):
        points = raw_scene()
        bev, _ = _bev_provider(points, synthetic_vessel(), CONFIG)
        raw3d = _raw3d_provider(points, synthetic_vessel(), CONFIG)
        self.assertTrue(bev)
        self.assertTrue(raw3d)
        all_ids = [row["observation_id"] for row in bev + raw3d]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        for row in bev + raw3d:
            self.assertEqual(row["frame_id"], "STATIC_SINGLE_FRAME")
            self.assertEqual(row["sensor_id"], "UNKNOWN")
            self.assertEqual(row["role"], "UNKNOWN")
            self.assertTrue(row["support_interval"])

    def test_verified_rectangle_cannot_cross_vessel_or_bypass_four_sides(self):
        verified = dict(rectangle_id="R0", vessel_hypothesis_id="OTHER",
                        source_stage="R2F", status="PROVISIONAL_RECTANGLE",
                        bounds_axial=[0, 0, 40, 20])
        untrusted = propose_for_vessel(raw_scene(), synthetic_vessel(), CONFIG,
                                       RESEARCH, height_override=[],
                                       verified_rectangles=[verified])
        self.assertFalse(any(row["state"] == "PROVISIONAL_HATCH"
                             for row in untrusted["hatch_hypotheses"]))
        verified["vessel_hypothesis_id"] = "V_TEST"
        trusted = propose_for_vessel(raw_scene(), synthetic_vessel(), CONFIG,
                                     RESEARCH, height_override=[],
                                     verified_rectangles=[verified])
        self.assertTrue(any(row["state"] == "PROVISIONAL_HATCH"
                            for row in trusted["hatch_hypotheses"]))
        weak = raw_scene()
        weak[(weak[:, 0] <= .3) & ((weak[:, 1] < 8) | (weak[:, 1] > 12)), 2] = 0
        partial = propose_for_vessel(weak, synthetic_vessel(), CONFIG, RESEARCH,
                                     height_override=[], verified_rectangles=[verified])
        self.assertFalse(any(row["state"] == "PROVISIONAL_HATCH" and
                             row["measured_strong_side_count"] < 4
                             for row in partial["hatch_hypotheses"]))

    def test_unresolved_vessel_evidence_stays_unresolved(self):
        vessel = synthetic_vessel()
        vessel["classification_status"] = "UNRESOLVED"
        verified = dict(rectangle_id="R0", vessel_hypothesis_id="V_TEST",
                        source_stage="R2F", status="PROVISIONAL_RECTANGLE",
                        bounds_axial=[0, 0, 40, 20])
        result = propose_for_vessel(raw_scene(), vessel, CONFIG, RESEARCH,
                                    height_override=[], verified_rectangles=[verified])
        self.assertTrue(result["provider_evidence"])
        self.assertTrue(result["hatch_hypotheses"])
        self.assertTrue(all(row["state"] == "UNRESOLVED" and row["polygon_xy"] is None
                            for row in result["hatch_hypotheses"]))


class VesselIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        weaker = vessel_cloud(0, 23)
        weaker[weaker[:, 2] == 0, 2] = .3
        points = np.vstack((vessel_cloud(0, 0), weaker))
        cls.result = discover_scene_hatches(points, CONFIG, RESEARCH)

    def test_no_cross_vessel_height_score_suppression(self):
        self.assertEqual(len(self.result["per_vessel"]), 2)
        self.assertEqual([row["provider_summary"]["height"]["region_count"]
                          for row in self.result["per_vessel"]], [1, 1])
        self.assertTrue(all(all(item["vessel_hypothesis_id"] == row["vessel_hypothesis_id"]
                                for item in row["provider_evidence"])
                            for row in self.result["per_vessel"]))

    def test_no_cross_vessel_bev_evidence(self):
        self.assertTrue(all(row["provider_summary"]["bev_count"] > 0
                            for row in self.result["per_vessel"]))
        self._assert_provider_isolation("BEV_RECTILINEAR_PROVIDER")

    def test_no_cross_vessel_raw3d_evidence(self):
        self.assertTrue(all(row["provider_summary"]["raw3d_count"] > 0
                            for row in self.result["per_vessel"]))
        self._assert_provider_isolation("RAW3D_STRUCTURAL_PROVIDER")

    def _assert_provider_isolation(self, kind):
        sets = []
        for row in self.result["per_vessel"]:
            records = [item for item in row["provider_evidence"]
                       if item["provider_type"] == kind]
            self.assertTrue(records)
            self.assertTrue(all(item["vessel_hypothesis_id"] == row["vessel_hypothesis_id"]
                                for item in records))
            sets.append({item["provider_id"] for item in records})
        self.assertTrue(sets[0].isdisjoint(sets[1]))

    def test_no_cross_vessel_proposal_association(self):
        all_ids = {row["vessel_hypothesis_id"]:
                   {item["provider_id"] for item in row["provider_evidence"]}
                   for row in self.result["per_vessel"]}
        for row in self.result["per_vessel"]:
            vessel_id = row["vessel_hypothesis_id"]
            for candidate in row["hatch_hypotheses"]:
                self.assertEqual(candidate["vessel_hypothesis_id"], vessel_id)
                self.assertTrue(set(candidate["supporting_provider_ids"]) <= all_ids[vessel_id])
        self.assertEqual(self.result["target_vessel_status"], "NOT_SELECTED")


if __name__ == "__main__":
    unittest.main()
