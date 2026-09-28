"""Synthetic contracts for measured raster and structural rectangle fusion."""

import math
import unittest

import numpy as np

from ship_perception.r1_static.raw_topview_evidence import build_topview
from ship_perception.r1_static.rectangle_fusion import solve
from ship_perception.tests.test_r1_boundary_topology import CONFIG, rectangle_case


def raw_scene(rectangles=((0, 0, 40, 20),), *, interior_z=0.0, deck_z=1.0):
    x0 = min(row[0] for row in rectangles) - 3
    x1 = max(row[2] for row in rectangles) + 3
    y0 = min(row[1] for row in rectangles) - 3
    y1 = max(row[3] for row in rectangles) + 3
    x, y = np.meshgrid(np.arange(x0, x1, 0.15), np.arange(y0, y1, 0.15))
    inside = np.zeros(x.shape, dtype=bool)
    for left, bottom, right, top in rectangles:
        inside |= (x > left) & (x < right) & (y > bottom) & (y < top)
    z = np.where(inside, interior_z, deck_z)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def single_case(*, gap=False, cargo=False):
    segments, edges, regions = rectangle_case(gap=gap)
    return raw_scene(interior_z=1.5 if cargo else 0.0), segments, edges, regions


class RawTopviewEvidence(unittest.TestCase):
    def test_multiscale_raw_channels_and_valid_mask(self):
        points = np.asarray(
            ((0, 0, 1), (0, 0, 2), (0.02, 0.02, 3), (0.5, 0, 4)), dtype=float
        )
        grids = build_topview(points, ((1, 0), (0, 1)), CONFIG)
        self.assertEqual(set(grids), {"fine", "coarse"})
        self.assertEqual(grids["fine"].cell_m, CONFIG["geometry"]["candidate_voxel_m"])
        self.assertEqual(grids["coarse"].cell_m, CONFIG["geometry"]["coarse_voxel_m"])
        self.assertEqual(int(grids["fine"].count[0, 0]), 3)
        self.assertEqual(float(grids["fine"].median[0, 0]), 2.0)
        self.assertEqual(
            float(grids["fine"].maximum[0, 0] - grids["fine"].minimum[0, 0]), 2.0
        )
        self.assertFalse(grids["fine"].occupancy[0, 1])

    def test_nonorthogonal_axes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "INVALID_SHIP_AXES"):
            build_topview(np.asarray(((0, 0, 0),)), ((1, 0), (1, 0)), CONFIG)


class RectangleFusion(unittest.TestCase):
    def test_missing_corner_uses_line_intersection(self):
        result = solve(*single_case(gap=True), CONFIG)
        self.assertTrue(result["rectangles"])
        candidate = result["rectangles"][0]
        self.assertEqual(len(candidate["polygon_xy"]), 4)
        self.assertTrue(
            all(
                row["corner_source"] == "GEOMETRIC_LINE_INTERSECTION"
                for row in candidate["corners"]
            )
        )
        self.assertTrue(result["observation_accounting_pass"])

    def test_inner_rim_beats_outer_parallel_line(self):
        points, segments, edges, regions = single_case()
        exterior = np.column_stack((np.full(201, -1.0), np.linspace(0, 20, 201)))
        segments.append(
            dict(
                node_id="n0",
                seed_id="seed",
                segment_id="outer",
                profile_executed_after=True,
                profile_break_positions=exterior.tolist(),
                face_positions=[],
            )
        )
        edges.append(
            dict(
                node_id="n0",
                a_raw=[-1, 0, 0],
                b_raw=[-1, 20, 0],
                observed_support_length=20.0,
            )
        )
        result = solve(points, segments, edges, regions, CONFIG)
        self.assertTrue(result["rectangles"])
        self.assertLess(abs(result["rectangles"][0]["polygon_xy"][0][0]), 0.2)

    def test_one_weak_side_keeps_evidence_provenance(self):
        points, segments, edges, regions = single_case()
        result = solve(points, segments[1:], edges[1:], regions, CONFIG)
        self.assertTrue(result["rectangles"])
        self.assertLessEqual(
            result["rectangles"][0]["geometry_constrained_side_count"], 1
        )
        self.assertTrue(
            any(
                side["evidence_level"] == "OBSERVED_RASTER_BOUNDARY"
                for side in result["rectangles"][0]["sides"]
            )
        )

    def test_missing_3d_endcaps_use_measured_raw_raster_in_review(self):
        points, segments, edges, regions = single_case()
        # A grazing scan may resolve both long steel runs while neither
        # transverse face is returned by the 3D profile extractor.
        result = solve(points, [segments[0], segments[2]], edges, regions, CONFIG)
        self.assertTrue(result["rectangles"])
        candidate = result["rectangles"][0]
        self.assertEqual(candidate["status"], "REVIEW_REQUIRED_RECTANGLE")
        self.assertEqual(
            sum(
                side["evidence_level"] == "OBSERVED_RASTER_BOUNDARY"
                for side in candidate["sides"]
            ),
            2,
        )
        actual = np.asarray(candidate["polygon_xy"])
        self.assertLess(
            np.max(np.abs(actual - ((0, 0), (40, 0), (40, 20), (0, 20)))), 0.6
        )

    def test_cargo_ridge_does_not_split_opening_into_strips(self):
        points, segments, edges, regions = single_case()
        cargo = (
            (points[:, 0] > 2)
            & (points[:, 0] < 38)
            & (points[:, 1] > 8)
            & (points[:, 1] < 12)
        )
        points[cargo, 2] = 1.8
        ridge = np.column_stack((np.linspace(2, 38, 181), np.full(181, 10.0)))
        segments.append(
            dict(
                node_id="n0",
                seed_id="seed",
                segment_id="cargo_ridge",
                profile_executed_after=True,
                profile_break_positions=ridge.tolist(),
                face_positions=[],
            )
        )
        result = solve(points, segments, edges, regions, CONFIG)
        self.assertTrue(result["rectangles"])
        actual = np.asarray(result["rectangles"][0]["polygon_xy"])
        self.assertLess(
            np.max(np.abs(actual - ((0, 0), (40, 0), (40, 20), (0, 20)))), 0.6
        )

    def test_one_roi_contains_two_independent_hatches(self):
        points = raw_scene(((0, 0, 40, 20), (50, 0, 90, 20)))
        segments, edges, _ = rectangle_case()
        second_segments, second_edges, _ = rectangle_case()
        for row in second_segments:
            row["segment_id"] = "second_" + row["segment_id"]
            row["profile_break_positions"] = [
                [x + 50, y] for x, y in row["profile_break_positions"]
            ]
        for row in second_edges:
            row["a_raw"][0] += 50
            row["b_raw"][0] += 50
        regions = [dict(node_id="n0", bbox_xy=[-3, -3, 93, 23])]
        result = solve(
            points, segments + second_segments, edges + second_edges, regions, CONFIG
        )
        self.assertEqual(len(result["rectangles"]), 2)
        centers = sorted(
            float(np.mean(row["polygon_xy"], axis=0)[0]) for row in result["rectangles"]
        )
        self.assertTrue(np.allclose(centers, (20, 70), atol=0.2))

    def test_three_crossbeam_separated_hatches_are_independent(self):
        offsets = (0, 42, 84)
        points = raw_scene(tuple((x, 0, x + 40, 20) for x in offsets))
        segments = []
        edges = []
        for offset in offsets:
            part_segments, part_edges, _ = rectangle_case()
            for row in part_segments:
                row["segment_id"] = "%d_%s" % (offset, row["segment_id"])
                row["profile_break_positions"] = [
                    [x + offset, y] for x, y in row["profile_break_positions"]
                ]
            for row in part_edges:
                row["a_raw"][0] += offset
                row["b_raw"][0] += offset
            segments.extend(part_segments)
            edges.extend(part_edges)
        regions = [dict(node_id="n0", bbox_xy=[-3, -3, 127, 23])]
        result = solve(points, segments, edges, regions, CONFIG)
        centers = sorted(
            float(np.mean(row["polygon_xy"], axis=0)[0]) for row in result["rectangles"]
        )
        self.assertEqual(len(centers), 3)
        self.assertTrue(np.allclose(centers, (20, 62, 104), atol=0.6))

    def test_partial_3d_crossbeam_and_raster_keep_two_hatches_separate(self):
        points = raw_scene(((0, 0, 40, 20), (42, 0, 82, 20)))
        first_segments, first_edges, _ = rectangle_case()
        second_segments, second_edges, _ = rectangle_case()
        for row in second_segments:
            row["segment_id"] = "second_" + row["segment_id"]
            row["profile_break_positions"] = [
                [x + 42, y] for x, y in row["profile_break_positions"]
            ]
        for row in second_edges:
            row["a_raw"][0] += 42
            row["b_raw"][0] += 42
        crossbeam = np.column_stack((np.full(41, 40.1), np.linspace(8, 12, 41)))
        bridge_segment = dict(
            node_id="n0",
            seed_id="seed",
            segment_id="crossbeam",
            profile_executed_after=True,
            profile_break_positions=crossbeam.tolist(),
            face_positions=[],
        )
        regions = [dict(node_id="n0", bbox_xy=[-3, -3, 85, 23])]
        result = solve(
            points,
            [
                first_segments[0],
                first_segments[2],
                second_segments[0],
                second_segments[2],
                bridge_segment,
            ],
            first_edges + second_edges,
            regions,
            CONFIG,
        )
        centers = sorted(
            np.mean(row["polygon_xy"], axis=0)[0] for row in result["rectangles"]
        )
        self.assertEqual(len(centers), 2)
        self.assertTrue(np.allclose(centers, (20, 62), atol=0.6))

    def test_density_stripe_inside_one_hatch_is_not_a_crossbeam(self):
        points, segments, edges, regions = single_case()
        stripe = points[(points[:, 0] > 19.8) & (points[:, 0] < 20.2)]
        points = np.vstack((points, stripe, stripe, stripe))
        result = solve(points, segments, edges, regions, CONFIG)
        self.assertEqual(len(result["rectangles"]), 1)

    def test_smooth_high_cargo_does_not_become_confirmed_steel(self):
        result = solve(*single_case(cargo=True), CONFIG)
        self.assertEqual(result["provisional_rectangle_count"], 0)
        self.assertEqual(
            result["static_calibration_status"],
            "UNVERIFIED_NO_INDEPENDENT_INNER_EDGE_GOLDEN",
        )

    def test_overflow_reference_cannot_confirm_side(self):
        points, segments, edges, regions = single_case()
        segments[0]["reference_role"] = "ROLE_AMBIGUOUS_OVERFLOW"
        result = solve(points, segments, edges, regions, CONFIG)
        self.assertEqual(result["provisional_rectangle_count"], 0)
        self.assertFalse(result["rectangles"])

    def test_flat_hull_and_raised_deckhouse_are_not_provisional(self):
        for interior_z in (1.0, 2.0):
            with self.subTest(interior_z=interior_z):
                points = raw_scene(interior_z=interior_z)
                segments, edges, regions = rectangle_case()
                result = solve(points, segments, edges, regions, CONFIG)
                self.assertEqual(result["provisional_rectangle_count"], 0)

    def test_raised_deckhouse_beside_hatch_is_not_output_as_hatch(self):
        points = raw_scene(((0, 0, 40, 20), (45, 0, 60, 20)))
        house = (
            (points[:, 0] > 45)
            & (points[:, 0] < 60)
            & (points[:, 1] > 0)
            & (points[:, 1] < 20)
        )
        points[house, 2] = 2.0
        segments, edges, _ = rectangle_case()
        house_segments, house_edges, _ = rectangle_case()
        for row in house_segments:
            row["segment_id"] = "house_" + row["segment_id"]
            row["profile_break_positions"] = [
                [45 + x * 15 / 40, y] for x, y in row["profile_break_positions"]
            ]
        for row in house_edges:
            for end in ("a_raw", "b_raw"):
                row[end][0] = 45 + row[end][0] * 15 / 40
        result = solve(
            points,
            segments + house_segments,
            edges + house_edges,
            [dict(node_id="n0", bbox_xy=[-3, -3, 63, 23])],
            CONFIG,
        )
        self.assertEqual(len(result["rectangles"]), 1)
        center = np.mean(result["rectangles"][0]["polygon_xy"], axis=0)
        self.assertTrue(np.allclose(center, (20, 10), atol=0.6))

    def test_rotation_keeps_ship_axes_and_rectangle(self):
        angle = math.radians(7)
        rotation = np.asarray(
            ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
        )
        center = np.asarray((20.0, 10.0))
        points = raw_scene()
        points[:, :2] = (points[:, :2] - center) @ rotation.T + center
        segments, edges, regions = rectangle_case(angle_deg=7)
        result = solve(points, segments, edges, regions, CONFIG)
        self.assertTrue(result["rectangles"])
        candidate = np.asarray(result["rectangles"][0]["polygon_xy"])
        expected = (
            np.asarray(((0, 0), (40, 0), (40, 20), (0, 20))) - center
        ) @ rotation.T + center
        self.assertLess(
            float(np.max(np.linalg.norm(candidate - expected, axis=1))), 0.5
        )


if __name__ == "__main__":
    unittest.main()
