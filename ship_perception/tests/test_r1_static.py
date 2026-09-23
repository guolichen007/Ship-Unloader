"""Synthetic contracts for R1-S0 raw-frame geometry and fail-closed output."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ship_perception.r1_static.boundary_refinement import refine_boundaries
from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.hypothesis_solver import solve
from ship_perception.r1_static.local_deck import _physical_components, _ring_points, estimate_local_deck
from ship_perception.r1_static.local_opening import find_openings
from ship_perception.r1_static.model import Opening, Proposal, SeedComponent
from ship_perception.r1_static.run import analyze_points, resolve_config
from ship_perception.r1_static.structural_proposal import propose
from ship_perception.r1_static.visualization import write_colored_ply


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/v15.json").read_text(encoding="utf-8"))


def scene(number=1, *, tilt=False):
    x, y = np.meshgrid(np.arange(0, 12 + number * 10, 0.2), np.arange(0, 20, 0.2))
    z = 0.02 * x + 0.01 * y if tilt else np.zeros_like(x)
    for index in range(number):
        z[(x > 4 + 10 * index) & (x < 10 + 10 * index) & (y > 5) & (y < 15)] -= 2
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def envelope(number):
    return Proposal("test", (3, 4, 11 + (number - 1) * 10, 16), 100, 2, (0.5,))


class StaticGeometry(unittest.TestCase):
    def test_height_grid_quantiles_and_translation(self):
        points = np.array(((1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 2, 3)), dtype=float)
        a = R1HeightGrid.from_points(points, 0.5)
        b = R1HeightGrid.from_points(points + [100, 17.3, -5], 0.5)
        self.assertEqual(a.count[0, 0], 3)
        self.assertEqual(a.median[0, 0], 1)
        np.testing.assert_array_equal(a.count, b.count)
        np.testing.assert_allclose(a.q10 - 5, b.q10, equal_nan=True)

    def test_proposal_translation_invariance_and_no_absolute_y(self):
        points = scene()
        old, _ = propose(points, CONFIG)
        repeated, _ = propose(points, CONFIG)
        shifted, _ = propose(points + [100, 37.3, 4.7], CONFIG)
        self.assertEqual([row.record() for row in old], [row.record() for row in repeated])
        self.assertEqual(len(old), len(shifted))
        self.assertTrue(old)
        np.testing.assert_allclose(np.array(old[0].bbox_xy) + [100, 37.3, 100, 37.3],
                                   shifted[0].bbox_xy, atol=1e-4)

    def test_tilted_local_deck_and_one_to_three_openings(self):
        for number in (1, 2, 3):
            with self.subTest(number=number):
                points = scene(number, tilt=True)
                deck, plane, support = estimate_local_deck(points, envelope(number), CONFIG)
                self.assertEqual(deck["status"], "RESOLVED", deck)
                self.assertGreater(deck["support_sector_count"], 2)
                self.assertLess(deck["residual_p95_m"], CONFIG["geometry"]["plane_inlier_m"])
                openings, _ = find_openings(points, envelope(number), plane, CONFIG)
                self.assertEqual(len(openings), number)

    def test_one_sided_deck_fails_closed(self):
        points = scene()
        points = points[points[:, 1] < 5]
        deck, plane, _ = estimate_local_deck(points, envelope(1), CONFIG)
        self.assertIsNone(plane)
        self.assertNotEqual(deck["status"], "RESOLVED")

    def test_competing_local_planes_are_ambiguous(self):
        points = scene()
        ring = ((points[:, 0] < 3) | (points[:, 0] > 11) |
                (points[:, 1] < 4) | (points[:, 1] > 16))
        points[ring & (points[:, 1] > 10), 2] += 1
        deck, plane, _ = estimate_local_deck(points, envelope(1), CONFIG)
        self.assertEqual(deck["status"], "LOCAL_DECK_AMBIGUOUS")
        self.assertGreater(deck["competing_plane_count"], 0)
        self.assertIsNone(plane)
        translated = Proposal("test", (1003, 1004, 1011, 1016), 100, 2, (0.5,))
        moved, moved_plane, _ = estimate_local_deck(points + (1000, 1000, 1000),
                                                     translated, CONFIG)
        self.assertEqual(moved["status"], deck["status"], moved)
        self.assertIsNone(moved_plane)

    def test_local_deck_translation_and_ransac_duplicates(self):
        points = scene(tilt=True)
        points[:, 2] += np.random.default_rng(7).normal(0, 0.005, len(points))
        first, _, _ = estimate_local_deck(points, envelope(1), CONFIG)
        self.assertEqual(first["status"], "RESOLVED", first)
        self.assertEqual(first["competing_plane_count"], 0)
        for displacement in ((100, 37.3, 4.7), (1000, 1000, 1000)):
            shifted = Proposal("test", tuple(np.asarray(envelope(1).bbox_xy) +
                                             [displacement[0], displacement[1]] * 2),
                               100, 2, (0.5,))
            result, _, _ = estimate_local_deck(points + displacement, shifted, CONFIG)
            self.assertEqual(result["status"], first["status"], result)
            self.assertEqual(result["competing_plane_count"], 0)
            self.assertEqual(result["support_sector_count"], first["support_sector_count"])
            self.assertAlmostEqual(result["normal_raw"][0], first["normal_raw"][0], places=4)
            self.assertAlmostEqual(result["normal_raw"][1], first["normal_raw"][1], places=4)

    def test_physical_gap_and_multi_patch_union(self):
        line = np.array(((0, 0, 0), (0.8, 0, 0), (1.6, 0, 0),
                         (3.0, 0, 0)), dtype=float)
        self.assertEqual([len(group) for group in _physical_components(line, 1.0)], [3, 1])
        self.assertEqual([len(group) for group in _physical_components(line + 1000, 1.0)], [3, 1])
        parts = []
        for x0, x1, y0, y1 in ((0, 2, 3, 17), (10, 12, 3, 17),
                               (3, 9, 0, 2), (3, 9, 18, 20)):
            x, y = np.meshgrid(np.arange(x0, x1, 0.2), np.arange(y0, y1, 0.2))
            parts.append(np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))))
        deck, plane, support = estimate_local_deck(np.vstack(parts),
                                                   Proposal("patches", (3, 3, 9, 17), 100, 2, (0.5,)),
                                                   CONFIG)
        self.assertEqual(deck["status"], "RESOLVED", deck)
        self.assertGreaterEqual(deck["connectivity_component_count"], 4)
        self.assertGreaterEqual(len(deck["accepted_component_ids"]), 4)
        self.assertEqual(len(support), deck["support_count"])
        self.assertIsNotNone(plane)

    def test_seed_geometry_keeps_deck_inside_large_envelope(self):
        points = scene(2)
        components = []
        for col0, col1 in ((8, 20), (28, 40)):
            cells = tuple((row, col) for row in range(10, 30) for col in range(col0, col1))
            components.append(SeedComponent((0, 0), 0.5, cells,
                                            (col0 * 0.5, 5, col1 * 0.5, 15)))
        proposal = Proposal("multi", (4, 5, 20, 15), 480, 2, (0.5,), tuple(components))
        self.assertEqual(len(proposal.record()["seed_components"]), 2)
        self.assertEqual(proposal.record()["seed_components"][0]["cell_count"], 240)
        ring = _ring_points(points, proposal, CONFIG["roi"]["support_search_m"])
        self.assertTrue(np.any((ring[:, 0] > 11) & (ring[:, 0] < 13) & (ring[:, 2] == 0)))
        deck, plane, _ = estimate_local_deck(points, proposal, CONFIG)
        self.assertEqual(deck["status"], "RESOLVED", deck)
        self.assertIsNotNone(plane)

    def test_proposal_preserves_child_seed_components(self):
        proposals, _ = propose(scene(2), CONFIG)
        self.assertTrue(proposals)
        self.assertTrue(all(proposal.seed_components for proposal in proposals))
        for proposal in proposals:
            self.assertEqual(sum(len(child.cells_rc) for child in proposal.seed_components),
                             proposal.evidence_cells)
            self.assertEqual({child.cell_m for child in proposal.seed_components},
                             set(proposal.scales_m))

    def test_sparse_opening_drop_uses_only_observed_cells(self):
        openings, _ = find_openings(scene(), envelope(1),
                                    (np.array([0., 0., 1.]), 0.), CONFIG)
        self.assertTrue(openings)
        self.assertTrue(all(np.isfinite(opening.mean_drop_m) for opening in openings))
        json.dumps([opening.record() for opening in openings], allow_nan=False)

    def test_profile_break_complete_and_partial_no_inferred_side(self):
        points = scene()
        opening = Opening("o", "p", (4, 5, 10, 15), 100, 60, 2)
        plane = (np.array([0., 0., 1.]), 0.)
        complete = refine_boundaries(points, opening, plane, CONFIG)
        self.assertEqual(complete["status"], "COMPLETE_OBSERVED", complete)
        self.assertEqual(len(complete["boundaries"]), 4)
        self.assertTrue(all(e["evidence_type"] == "OBSERVED_PROFILE_BREAK" for e in complete["boundaries"]))
        partial_points = points[(points[:, 0] < 3.2) | (points[:, 0] > 4.8)]
        partial = refine_boundaries(partial_points, opening, plane, CONFIG)
        self.assertEqual(partial["status"], "PARTIAL", partial)
        self.assertIsNone(partial["polygon_raw"])
        self.assertLess(len(partial["boundaries"]), 4)

    def test_vertical_face_is_reported_as_3d_evidence(self):
        points = scene()
        wall = np.array([[4.0, y, z] for y in np.arange(5.2, 14.9, 0.1)
                         for z in np.linspace(-2, 0, 8)])
        result = refine_boundaries(np.vstack((points, wall)),
                                   Opening("o", "p", (4, 5, 10, 15), 100, 60, 2),
                                   (np.array([0., 0., 1.]), 0.), CONFIG)
        self.assertTrue(any("OBSERVED_3D_FACE" in edge["evidence_type"]
                            for edge in result["boundaries"]),
                        [edge["evidence_type"] for edge in result["boundaries"]])

    def test_negative_and_json_finite(self):
        x, y = np.meshgrid(np.arange(0, 10, 0.2), np.arange(0, 10, 0.2))
        flat = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        result, _ = analyze_points(flat, CONFIG)
        self.assertEqual(result["scene_status"], "NO_CONFIRMED_HATCH")
        self.assertEqual(result["confirmed_hatch_count"], 0)
        json.dumps(result, allow_nan=False)
        cargo = flat.copy()
        cargo[(cargo[:, 0] > 3) & (cargo[:, 0] < 7) & (cargo[:, 1] > 3) & (cargo[:, 1] < 7), 2] = 2
        cargo_result, _ = analyze_points(cargo, CONFIG)
        self.assertEqual(cargo_result["confirmed_hatch_count"], 0)

    def test_output_remains_in_translated_raw_frame(self):
        first, _ = analyze_points(scene(), CONFIG)
        shifted, _ = analyze_points(scene() + [100, 37.3, 4.7], CONFIG)
        self.assertEqual(first["confirmed_hatch_count"], 1)
        self.assertEqual(shifted["confirmed_hatch_count"], 1)
        np.testing.assert_allclose(np.asarray(first["hatches"][0]["center_raw"]) + [100, 37.3, 4.7],
                                   shifted["hatches"][0]["center_raw"], atol=0.1)

    def test_end_to_end_one_two_three_hatches_without_known_count(self):
        for number in (1, 2, 3):
            with self.subTest(number=number):
                result, _ = analyze_points(scene(number), CONFIG)
                self.assertEqual(result["confirmed_hatch_count"], number)
                self.assertEqual(result["scene_status"], "RESOLVED")
                self.assertTrue(all(h["status"] == "COMPLETE_OBSERVED" for h in result["hatches"]))

    def test_hypothesis_duplicate_conflict_and_ambiguity(self):
        edge = dict(evidence_type="OBSERVED_PROFILE_BREAK", coverage=0.9, fit_residual_p95_m=0.01)
        deck = dict(status="RESOLVED", residual_p95_m=0.01)
        rows = [dict(opening_id="a", bbox_xy=[0, 0, 5, 5], status="PARTIAL",
                     local_deck=deck, boundaries=[edge, edge]),
                dict(opening_id="b", bbox_xy=[0.1, 0.1, 5.1, 5.1], status="PARTIAL",
                     local_deck=deck, boundaries=[edge, edge])]
        decision = solve(rows, CONFIG)
        self.assertEqual(decision["scene_status"], "HATCH_HYPOTHESIS_AMBIGUOUS")
        self.assertEqual(decision["selected"], [])
        self.assertTrue(decision["candidate_conflicts"])

    def test_config_override_identity_and_frozen_evaluation(self):
        baseline, identity = resolve_config()
        changed, altered = resolve_config(overrides=["boundary.profile_step_m=0.04"])
        self.assertNotEqual(identity, altered)
        self.assertEqual(changed["boundary"]["profile_step_m"], 0.04)
        with self.assertRaisesRegex(ValueError, "R1_OVERRIDE_FORBIDDEN"):
            resolve_config(overrides=["evaluation.boundary_p95_m=0.2"])
        with tempfile.TemporaryDirectory() as directory:
            alternative = Path(directory) / "weakened.json"
            alternative.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "R1_CONFIG_PATH_FORBIDDEN"):
                resolve_config(alternative)
            weaker = json.loads(alternative.read_text(encoding="utf-8"))
            weaker["evaluation"]["boundary_p95_m"] = 0.2
            alternative.write_text(json.dumps(weaker), encoding="utf-8")
            with patch("ship_perception.r1_static.run.DEFAULT_CONFIG", alternative):
                with self.assertRaisesRegex(ValueError, "R1_CONFIG_CONTRACT_REJECTED"):
                    resolve_config(alternative)

    def test_binary_rgb_ply_parses(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "colored.ply"
            write_colored_ply(target, np.array(((1, 2, 3), (4, 5, 6))),
                              np.array(((255, 0, 0), (0, 255, 0)), dtype=np.uint8))
            raw = target.read_bytes()
            header, payload = raw.split(b"end_header\n", 1)
            self.assertIn(b"element vertex 2", header)
            data = np.frombuffer(payload, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                                          ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
            self.assertEqual(data[0]["x"], 1)
            self.assertEqual(data[1]["green"], 255)

    def test_product_import_isolation(self):
        source = (ROOT / "r1_static")
        texts = "\n".join(path.read_text(encoding="utf-8") for path in source.glob("*.py"))
        self.assertNotIn("import legacy", texts)
        self.assertNotIn("cnn_model.bin", texts)
        self.assertNotIn("import ship_perception.measurement", texts)
        self.assertIn("from ship_perception.tools.v15_pcd import decode", texts)


if __name__ == "__main__":
    unittest.main()
