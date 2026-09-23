"""Coarse height regions remain provisional and retain raw-point ownership."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ship_perception.r1_static.heightmap_render import GRAY, render_regions
from ship_perception.r1_static.heightmap_topology import detect_regions


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/v15.json").read_text(encoding="utf-8"))
RESEARCH = json.loads((ROOT / "config/r1_heightmap_research.json").read_text(encoding="utf-8"))


def rectangular_basins(number=3):
    x, y = np.meshgrid(np.arange(0, 12 + number * 25, .1), np.arange(0, 22, .1))
    height = np.zeros_like(x)
    for index in range(number):
        left = 5 + index * 25
        height[(x > left) & (x < left + 20) & (y > 4) & (y < 16)] = -2
        # Cargo bumps and fine sampling gaps may alter the interior surface;
        # they do not define additional coarse opening topology.
        height[(x > left + 9) & (x < left + 12) &
               (y > 7) & (y < 12)] = -1.4
    return np.column_stack((x.ravel(), y.ravel(), height.ravel()))


class HeightmapTopology(unittest.TestCase):
    def test_coarse_basin_count_survives_cargo_bumps_and_translation(self):
        points = rectangular_basins()
        first, _ = detect_regions(points, CONFIG, RESEARCH)
        moved, _ = detect_regions(points + [100, 37.3, 4.7], CONFIG, RESEARCH)
        self.assertEqual(first["region_count"], 3)
        self.assertEqual(moved["region_count"], 3)
        self.assertTrue(all(row["steel_boundary_status"] == "UNKNOWN"
                            for row in first["regions"]))
        self.assertTrue(all(row["height_rise_side_count"] == 4 for row in first["regions"]))
        np.testing.assert_allclose(
            np.asarray([row["bbox_xy"] for row in first["regions"]]) + [100, 37.3, 100, 37.3],
            [row["bbox_xy"] for row in moved["regions"]], atol=.001)
        reordered, _ = detect_regions(points[::-1], CONFIG, RESEARCH)
        self.assertEqual([row["bbox_xy"] for row in first["regions"]],
                         [row["bbox_xy"] for row in reordered["regions"]])

    def test_flat_surface_is_not_called_hatch(self):
        points = rectangular_basins(0)
        result, _ = detect_regions(points, CONFIG, RESEARCH)
        self.assertEqual(result["region_count"], 0)
        self.assertEqual(result["status"], "NO_REGION_CANDIDATES")
        sparse, _ = detect_regions(np.asarray(((0, 0, 0), (1, 1, 0)), dtype=float),
                                   CONFIG, RESEARCH)
        self.assertEqual(sparse["status"], "NO_VESSEL_SUPPORT")
        self.assertEqual(sparse["region_count"], 0)

    def test_colored_ply_matches_coarse_region_ownership_and_box(self):
        points = rectangular_basins(2)
        result, auxiliary = detect_regions(points, CONFIG, RESEARCH)
        with tempfile.TemporaryDirectory() as directory:
            metadata = render_regions(directory, points, result, auxiliary, CONFIG)
            self.assertEqual(metadata["raw_vertex_count"], len(points))
            for name in ("height_surface.png", "height_levels.png", "hatch_regions.png",
                         "hatch_regions.ply", "hatch_boxes.ply", "region_point_ownership.npz"):
                self.assertTrue((Path(directory) / name).exists(), name)
            payload = (Path(directory) / "hatch_regions.ply").read_bytes().split(
                b"end_header\n", 1)[1]
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
            vertices = np.frombuffer(payload, dtype=dtype, count=len(points))
            actual = np.column_stack([vertices[name] for name in ("red", "green", "blue")])
            grid = auxiliary["coarse"]
            rows, cols = grid.cell_indices(points[:, :2])
            expected = np.tile(np.asarray(GRAY, dtype=np.uint8), (len(points), 1))
            for record, mask in zip(result["regions"], auxiliary["masks"]):
                expected[mask[rows, cols]] = record["color_rgb"]
                self.assertEqual(len(record["calibration_box_raw"]), 8)
                self.assertTrue((Path(directory) / "regions" /
                                 (record["region_id"] + "_points.ply")).exists())
            with np.load(Path(directory) / "region_point_ownership.npz") as archive:
                for record, mask in zip(result["regions"], auxiliary["masks"]):
                    np.testing.assert_array_equal(archive[record["region_id"]],
                                                  np.flatnonzero(mask[rows, cols]))
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(all(row["touches_true_crop_boundary"] is False
                                for row in result["regions"]))


if __name__ == "__main__":
    unittest.main()
