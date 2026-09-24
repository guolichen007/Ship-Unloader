"""S4-R2 structural line fusion invariants (occlusion gaps, parallel edges)."""

import json
from pathlib import Path
import unittest

import numpy as np

from ship_perception.r1_static.structural_line_fusion import fuse_lines


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/v15.json").read_text(encoding="utf-8"))


def _observation(segment_id, points, primitive_type="OBSERVED_PROFILE_BREAK", node_id="n0"):
    return dict(node_id=node_id, segment_id=segment_id, reference_family_id=None,
                primitive_type=primitive_type, positions=np.asarray(points, dtype=float))


class StructuralLineFusion(unittest.TestCase):
    def test_occlusion_gap_keeps_two_intervals(self):
        # Two collinear fragments of one physical edge separated by a large gap.
        frag_a = [[i * 0.1, 0.0] for i in range(30)]          # x in [0, 2.9]
        frag_b = [[i * 0.1, 0.0] for i in range(60, 90)]      # x in [6.0, 8.9]
        lines, _ = fuse_lines([
            _observation("s0", frag_a),
            _observation("s1", frag_b),
        ], CONFIG)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["observed_interval_count"], 2)
        self.assertEqual(len(lines[0]["source_segment_ids"]), 2)
        self.assertEqual(lines[0]["status"], "OBSERVED_SINGLE_MODAL_LINE")

    def test_parallel_separate_edges_do_not_merge(self):
        # Two long parallel edges with different normal offsets must stay apart.
        edge_a = [[i * 0.1, 0.0] for i in range(100)]         # y = 0
        edge_b = [[i * 0.1, 1.0] for i in range(100)]         # y = 1
        lines, _ = fuse_lines([
            _observation("s0", edge_a),
            _observation("s1", edge_b),
        ], CONFIG)
        self.assertEqual(len(lines), 2)

    def test_same_line_two_modal_fuses_multimodal(self):
        # A PROFILE break and a 3D face observation of the same edge fuse.
        profile = [[i * 0.1, 0.0] for i in range(40)]
        face = [[i * 0.1, 0.02] for i in range(40)]
        lines, _ = fuse_lines([
            _observation("s0", profile, "OBSERVED_PROFILE_BREAK"),
            _observation("s0", face, "OBSERVED_3D_FACE"),
        ], CONFIG)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["status"], "OBSERVED_MULTI_MODAL_LINE")


if __name__ == "__main__":
    unittest.main()
