"""Observed-only rectangle topology and adversarial corner coverage gates."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ship_perception.r1_static.boundary_topology import analyze, read_xyzrgb_ply


CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "config/v15.json").read_text(
        encoding="utf-8"
    )
)


def rectangle_case(angle_deg=0.0, gap=False, cargo=False):
    angle = math.radians(angle_deg)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    center = np.array([20.0, 10.0])

    def to_world(points):
        return (np.asarray(points) - center) @ rotation.T + center

    sides = [
        ((0, 0), (40, 0)),
        ((40, 0), (40, 20)),
        ((40, 20), (0, 20)),
        ((0, 20), (0, 0)),
    ]
    segments, edges = [], []
    for side_index, (a, b) in enumerate(sides):
        points = np.linspace(a, b, 201)
        if gap and side_index == 0:
            points = points[(points[:, 0] > 8) & (points[:, 0] < 32)]
        points = to_world(points)
        segment_id = "s%d" % side_index
        segments.append(
            dict(
                node_id="n0",
                seed_id="seed",
                segment_id=segment_id,
                profile_executed_after=True,
                profile_break_positions=points.tolist(),
                face_positions=[],
            )
        )
        ends = to_world([a, b])
        edges.append(
            dict(
                node_id="n0",
                a_raw=[*ends[0], 0],
                b_raw=[*ends[1], 0],
                observed_support_length=float(np.linalg.norm(ends[1] - ends[0])),
            )
        )
    if cargo:
        # A long interior cargo ridge is parallel to a side; it cannot supply
        # the missing boundary near the Height opening perimeter.
        points = to_world(np.linspace((0, 10), (40, 10), 101))
        segments.append(
            dict(
                node_id="n0",
                seed_id="seed",
                segment_id="cargo",
                profile_executed_after=True,
                profile_break_positions=points.tolist(),
                face_positions=[],
            )
        )
    corners = to_world([[0, 0], [0, 20], [40, 0], [40, 20]])
    bbox = [
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].max()),
    ]
    return segments, edges, [dict(node_id="n0", bbox_xy=bbox)]


class BoundaryTopology(unittest.TestCase):
    def test_complete_observed_rectangle_and_accounting(self):
        result = analyze(*rectangle_case(), CONFIG)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["status"], "COMPLETE_OBSERVED_RECTANGLE")
        self.assertEqual(candidate["observed_side_count"], 4)
        self.assertTrue(result["observation_accounting_pass"])
        self.assertEqual(
            result["observation_count"], result["used_count"] + result["rejected_count"]
        )

    def test_missing_corner_stays_partial(self):
        result = analyze(*rectangle_case(gap=True), CONFIG)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["status"], "PARTIAL_RECTANGLE")
        self.assertIsNone(candidate["polygon_xy"])

    def test_subline_points_recover_long_side(self):
        segments, edges, regions = rectangle_case()
        segments = segments[1:]
        edges = edges[1:]
        for index in range(40):
            points = np.column_stack((np.linspace(index, index + 1, 5), np.zeros(5)))
            segments.append(
                dict(
                    node_id="n0",
                    seed_id="seed",
                    segment_id="short_%02d" % index,
                    profile_executed_after=True,
                    profile_break_positions=points.tolist(),
                    face_positions=[],
                )
            )
        self.assertTrue(
            all(
                len(row["profile_break_positions"])
                < CONFIG["boundary"]["line_min_points"]
                for row in segments[3:]
            )
        )
        result = analyze(segments, edges, regions, CONFIG)
        self.assertEqual(
            result["candidates"][0]["status"], "COMPLETE_OBSERVED_RECTANGLE"
        )

    def test_long_interior_cargo_cannot_replace_missing_side(self):
        segments, edges, regions = rectangle_case(cargo=True)
        result = analyze(segments[1:], edges[1:], regions, CONFIG)
        self.assertEqual(result["candidates"][0]["status"], "PARTIAL_RECTANGLE")

    def test_overflow_role_cannot_supply_steel_side(self):
        segments, edges, regions = rectangle_case()
        segments[0]["reference_role"] = "ROLE_AMBIGUOUS_OVERFLOW"
        result = analyze(segments, edges, regions, CONFIG)
        self.assertEqual(result["candidates"][0]["status"], "PARTIAL_RECTANGLE")
        self.assertTrue(
            any(
                row["reject_reason"] == "REJECT_INTERIOR_OR_OVERFLOW_ROLE"
                for row in result["observations"]
            )
        )

    def test_rotation_and_long_cargo_do_not_change_boundary(self):
        result = analyze(*rectangle_case(angle_deg=7, cargo=True), CONFIG)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["status"], "COMPLETE_OBSERVED_RECTANGLE")
        self.assertEqual(candidate["observed_side_count"], 4)

    def test_ply_requires_complete_binary_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.ply"
            path.write_bytes(
                b"ply\nformat binary_little_endian 1.0\n"
                b"element vertex 1\nproperty float x\nproperty float y\n"
                b"property float z\nproperty uchar red\nproperty uchar green\n"
                b"property uchar blue\nend_header\n"
            )
            with self.assertRaisesRegex(ValueError, "PLY_LENGTH_MISMATCH"):
                read_xyzrgb_ply(path)


if __name__ == "__main__":
    unittest.main()
