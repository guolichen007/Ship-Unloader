"""Bridge diagnostics keep real mask topology and do not change selected regions."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from ship_perception.r1_static.height_structure_evaluator import (
    _decision,
    evaluate_scene,
)
from ship_perception.r1_static.height_structure_forensics import (
    _node_seeds,
    audit_scene,
    fragment_compatibility,
)
from ship_perception.r1_static.heightmap_topology import detect_regions
from ship_perception.tests.test_r1_heightmap_topology import rectangular_basins


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/v15.json").read_text(encoding="utf-8"))
RESEARCH = json.loads(
    (ROOT / "config/r1_heightmap_research.json").read_text(encoding="utf-8")
)


class BridgeForensics(unittest.TestCase):
    def test_real_mask_concavity_is_not_replaced_by_bbox(self):
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:5, 1:3] = True
        mask[3:5, 3:7] = True
        node = SimpleNamespace(node_id="l00-c0001", mask=mask, bbox_xy=(1, 1, 7, 5))
        grid = SimpleNamespace(x0=0.0, y0=0.0, cell_m=1.0)
        fov = SimpleNamespace(touches=lambda contour: False)
        seeds = _node_seeds(node, grid, fov)
        self.assertEqual(len(seeds), 1)
        self.assertGreater(len(seeds[0].contour_xy), 4)
        self.assertIn((3.0, 3.0), seeds[0].contour_xy)
        self.assertEqual(len(seeds[0].component.cells_rc), int(mask.sum()))

    def test_selected_baseline_signature_is_unchanged_by_full_audit(self):
        points = rectangular_basins(1).astype(np.float32)
        baseline, _ = detect_regions(points, CONFIG, RESEARCH)
        before = copy.deepcopy(baseline)
        rebuilt, records, _, nodes = audit_scene(points, CONFIG, RESEARCH, baseline)
        self.assertEqual(baseline, before)
        self.assertEqual(
            [row["source_node"] for row in rebuilt["regions"]],
            [row["source_node"] for row in baseline["regions"]],
        )
        self.assertEqual(
            [row["bbox_xy"] for row in rebuilt["regions"]],
            [row["bbox_xy"] for row in baseline["regions"]],
        )
        self.assertEqual(
            [row["topology_score"] for row in rebuilt["regions"]],
            [row["topology_score"] for row in baseline["regions"]],
        )
        self.assertEqual(len(records), len(nodes))
        self.assertTrue(all("observed_edges" in row for row in records))

    def test_missing_structural_lines_cannot_make_fragment_compatible(self):
        mask_a = np.zeros((13, 21), dtype=bool)
        mask_b = np.zeros_like(mask_a)
        mask_a[0:4, 0:20] = True
        mask_b[8:12, 0:20] = True
        a = SimpleNamespace(
            node_id="a",
            level_index=0,
            bbox_xy=(0, 0, 20, 4),
            parent=None,
            mask=mask_a,
        )
        b = SimpleNamespace(
            node_id="b",
            level_index=1,
            bbox_xy=(0, 8, 20, 12),
            parent=None,
            mask=mask_b,
        )
        rows = [dict(node_id=name, observed_edges=[]) for name in ("a", "b")]
        pair = fragment_compatibility([a, b], rows, CONFIG)["pairs"][0]
        self.assertEqual(pair["association_status"], "AMBIGUOUS")
        self.assertIsNone(pair["boundary_recoverability"])
        self.assertNotIn("merged_bbox_xy", pair)

    def test_weak_label_evaluator_only_compares_frozen_boxes(self):
        frozen = dict(
            selected_region_signature=[["selected", [0, 0, 10, 10], 1]],
            nodes=[dict(node_id="child", bbox_xy=[1, 1, 9, 9])],
        )
        original = copy.deepcopy(frozen)
        label = dict(annotations=[dict(corners=[[1, 1], [9, 1], [9, 9], [1, 9]])])
        metrics = evaluate_scene(frozen, label)
        self.assertEqual(metrics["metric_semantics"], "WEAK_LABEL_DIAGNOSTIC_ONLY")
        self.assertEqual(metrics["comparisons"][0]["best_all_node_id"], "child")
        self.assertEqual(metrics["comparisons"][0]["best_bbox_iou"], 0.64)
        self.assertEqual(frozen, original)

    def test_selection_discrepancy_is_reported_before_profile_failure(self):
        audit = dict(
            nodes=[
                dict(
                    observed_edge_count=1,
                    boundary_status="PARTIAL",
                    baseline_selected=False,
                )
            ]
        )
        weak = dict(comparisons=[dict(best_all_node_bbox_iou=0.72, best_bbox_iou=0.29)])
        decision = _decision("synthetic", audit, dict(pairs=[]), weak)
        self.assertEqual(decision["route"], "ROUTE_C")
        self.assertEqual(
            decision["first_bad_stage_hypothesis"], "HEIGHTMAP_CANDIDATE_SELECTION"
        )


if __name__ == "__main__":
    unittest.main()
