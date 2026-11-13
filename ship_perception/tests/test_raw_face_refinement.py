"""Physical raw-face counterexamples; no customer points or scene-name priors."""

import unittest
from unittest import mock
import numpy as np

from ship_perception.io.frame import PointCloudFrame
from ship_perception.perception.context import VesselEvidence, freeze
from ship_perception.perception.local_evidence import VesselLocalEvidence
from ship_perception.perception.final_boundary import FinalBoundaryRefiner
from ship_perception.perception.raw_face_refinement import raw_face_section
from ship_perception.perception.raw_mode_consensus import resolve_section_modes
from ship_perception.r1_static.hatch_proposal import _opening_evidence
from ship_perception.r1_static.run import resolve_config


def terrace_scene(cargo=False):
    u, v = np.meshgrid(np.arange(0, 36, 0.1), np.arange(0, 24, 0.1))
    inside = (u > 4) & (u < 28) & (v > 4) & (v < 20)
    z = np.where(inside, -12.0, -5.0)
    z[(u >= 30) & (v > 4) & (v < 20)] = 0.0
    if cargo:
        z[(u > 22) & (u < 26) & (v > 4) & (v < 20)] = -5.0
    xyz = np.column_stack((u.ravel(), v.ravel(), z.ravel()))
    walls = []
    for x, low, high in ((28.0, -12.0, -5.0), (30.0, -5.0, 0.0)):
        along, height = np.meshgrid(
            np.arange(4.1, 20, 0.1), np.arange(low + 0.1, high, 0.1)
        )
        walls.append(
            np.column_stack((np.full(along.size, x), along.ravel(), height.ravel()))
        )
    if cargo:
        along, height = np.meshgrid(np.arange(4.1, 20, 0.1), np.arange(-11.9, -5, 0.1))
        walls.append(
            np.column_stack((np.full(along.size, 22.0), along.ravel(), height.ravel()))
        )
    # Mixed higher/lower returns near the real face corrupt wide-strip IQR,
    # while its independently measured 3D normal plane remains observable.
    band = xyz[(abs(xyz[:, 0] - 28) < 0.4) & (xyz[:, 1] > 4) & (xyz[:, 1] < 20)]
    ghosts = band[np.random.default_rng(71).random(len(band)) < 0.5].copy()
    ghosts[:, 2] = np.where(ghosts[:, 0] < 28, -5.0, -12.0)
    return np.vstack((xyz, *walls, ghosts))


def context(points):
    cfg, _ = resolve_config()
    frame = PointCloudFrame(
        points, None, "synthetic", 0, 0, "", ("x", "y", "z"), len(points)
    )
    vessel = VesselEvidence(
        freeze(
            dict(
                vessel_hypothesis_id="V_SYNTHETIC",
                local_axes=np.eye(2).tolist(),
                source_support_ids=["synthetic_raw"],
            )
        ),
        np.arange(len(points)),
        "synthetic",
    )
    cache = VesselLocalEvidence.build(frame, vessel, cfg)
    box = [4.0, 4.0, 31.0, 20.0]
    proposal = dict(
        bounds_axial=box,
        vessel_hypothesis_id=vessel.vessel_id,
        opening_evidence=_opening_evidence(cache.grid("fine"), box, cfg),
        measured_side_candidates=cache.side_candidates(box),
    )
    return cfg, cache, proposal


class RawFaceRefinement(unittest.TestCase):
    def raw_mode(self, rho):
        return dict(
            status="UNIQUE_SIGNED_RAW_STEP",
            rho=rho,
            modes=[rho],
            continuous=[rho, 0.05, 25, True],
            height=dict(transition=True),
        )

    def test_unique_raw_sections_resolve_only_one_continuous_measured_mode(self):
        cfg, _ = resolve_config()
        spans = [(0, 3), (3, 6), (6, 9), (9, 12)]
        records = [
            self.raw_mode(10),
            dict(
                status="MULTIMODE_BOUNDARY_CONFLICT",
                modes=[10.06, 11.2],
                mode_candidates=[self.raw_mode(10.06), self.raw_mode(11.2)],
            ),
            dict(
                status="MULTIMODE_BOUNDARY_CONFLICT",
                modes=[10.12, 11.8],
                mode_candidates=[self.raw_mode(10.12), self.raw_mode(11.8)],
            ),
            self.raw_mode(10.18),
        ]
        result = resolve_section_modes(records, spans, cfg)
        self.assertEqual(
            [r["continuous"][0] for r in result], [10, 10.06, 10.12, 10.18]
        )
        self.assertEqual(records[1]["status"], "MULTIMODE_BOUNDARY_CONFLICT")

    def test_two_spatially_supported_raw_line_families_stay_unresolved(self):
        cfg, _ = resolve_config()
        spans = [(0, 3), (3, 6), (6, 9), (9, 12)]
        records = [
            dict(
                status="MULTIMODE_BOUNDARY_CONFLICT",
                modes=[10, 11],
                mode_candidates=[self.raw_mode(10), self.raw_mode(11)],
            )
            for _ in spans
        ]
        result = resolve_section_modes(records, spans, cfg)
        self.assertTrue(
            all(r["status"] == "MULTIMODE_BOUNDARY_CONFLICT" for r in result)
        )

    def test_second_stable_family_is_not_hidden_by_two_unique_end_sections(self):
        cfg, _ = resolve_config()
        spans = [(i * 3, (i + 1) * 3) for i in range(6)]
        ambiguous = dict(
            status="MULTIMODE_BOUNDARY_CONFLICT",
            modes=[10, 11],
            mode_candidates=[self.raw_mode(10), self.raw_mode(11)],
        )
        records = [
            self.raw_mode(10),
            ambiguous,
            ambiguous,
            ambiguous,
            ambiguous,
            self.raw_mode(10),
        ]
        self.assertEqual(resolve_section_modes(records, spans, cfg), records)

    def test_two_equally_consistent_alternatives_are_never_averaged(self):
        cfg, _ = resolve_config()
        spans = [(0, 3), (3, 6), (6, 9), (9, 12)]
        ambiguous = dict(
            status="MULTIMODE_BOUNDARY_CONFLICT",
            modes=[9.7, 10.3],
            mode_candidates=[self.raw_mode(9.7), self.raw_mode(10.3)],
        )
        records = [self.raw_mode(10), ambiguous, ambiguous, self.raw_mode(10)]
        self.assertEqual(resolve_section_modes(records, spans, cfg), records)

    def test_raw_line_consensus_cannot_fill_a_missing_physical_section(self):
        cfg, _ = resolve_config()
        spans = [(0, 3), (3, 6), (6, 9), (9, 12)]
        records = [
            self.raw_mode(10),
            dict(
                status="MULTIMODE_BOUNDARY_CONFLICT",
                modes=[10, 11],
                mode_candidates=[self.raw_mode(10), self.raw_mode(11)],
            ),
            dict(status="INSUFFICIENT_RAW_SUPPORT", modes=[]),
            self.raw_mode(10),
        ]
        self.assertEqual(resolve_section_modes(records, spans, cfg), records)

    def test_measured_inner_face_excludes_higher_terrace_with_mixed_returns(self):
        cfg, cache, proposal = context(terrace_scene())
        with mock.patch.object(
            FinalBoundaryRefiner, "_normal_face_reanchor", return_value=None
        ), mock.patch.object(
            FinalBoundaryRefiner,
            "_search_raw_section",
            FinalBoundaryRefiner._search_raw_section_v1,
        ):
            previous = FinalBoundaryRefiner(cache.xyz, [], cfg).refine(
                proposal, np.eye(2)
            )
        current = FinalBoundaryRefiner(cache.xyz, [], cfg).refine(proposal, np.eye(2))
        self.assertGreater(previous["sides"]["U1"]["rho_after_m"], 29.0)
        self.assertAlmostEqual(current["sides"]["U1"]["rho_after_m"], 28.0, delta=0.15)
        self.assertGreaterEqual(
            current["sides"]["U1"]["support_section_count"],
            cfg["boundary"]["profile_min_sections"],
        )
        side = current["sides"]["U1"]
        face = side["opening_role_reanchor"].get("raw_normal_face")
        if face is not None:
            self.assertEqual(face["physical_role"], "UNKNOWN")
        else:
            # A unique paired raw profile now avoids an unnecessary normal
            # fallback. The same measured inner-face geometry is still required.
            self.assertGreaterEqual(
                side["positive_height_section_count"],
                cfg["boundary"]["profile_min_sections"],
            )
            self.assertGreaterEqual(
                side["raw_subvoxel_section_count"],
                cfg["boundary"]["profile_min_sections"],
            )
        self.assertFalse(current["sides"]["U1"]["steel_edge_certified"])

    def test_independent_vertical_cargo_face_cannot_cut_away_the_opening(self):
        cfg, cache, proposal = context(terrace_scene(cargo=True))
        current = FinalBoundaryRefiner(cache.xyz, [], cfg).refine(proposal, np.eye(2))
        self.assertGreaterEqual(current["sides"]["U1"]["rho_after_m"], 27.85)
        self.assertFalse(current["sides"]["U1"]["steel_edge_certified"])

    def test_reversing_raw_point_order_preserves_face_geometry(self):
        baseline = None
        points = terrace_scene()
        for variant in (points, points[::-1]):
            cfg, cache, proposal = context(variant)
            result = FinalBoundaryRefiner(cache.xyz, [], cfg).refine(
                proposal, np.eye(2)
            )
            if baseline is None:
                baseline = result["boundary_xy"]
            else:
                np.testing.assert_allclose(
                    result["boundary_xy"], baseline, atol=1e-9, rtol=0
                )

    def test_raw_normal_points_do_not_claim_nonexistent_surface_pairs(self):
        cfg, _ = resolve_config()
        u, v = np.meshgrid(
            np.r_[np.arange(7.0, 9.6, 0.1), np.arange(10.4, 13.0, 0.1)],
            np.arange(0, 4, 0.1),
        )
        base = np.column_stack(
            (u.ravel(), v.ravel(), np.where(u < 10, -4.0, 0.0).ravel())
        )
        along, z = np.meshgrid(np.arange(0, 4, 0.1), np.arange(-2.0, 0.0, 0.1))
        wall = np.column_stack((np.full(along.size, 10.0), along.ravel(), z.ravel()))
        face = dict(
            normal_axis=0,
            a_xy=[10.0, 0.0],
            b_xy=[10.0, 4.0],
            points=wall,
            residual_p95_m=0.01,
            point_index_sha256="synthetic_face",
        )
        result = raw_face_section(np.vstack((base, wall)), face, 0, (0.0, 4.0), -1, cfg)
        self.assertIsNotNone(result)
        self.assertTrue(result["raw_measurement_present"])
        self.assertFalse(result["surface_pair_present"])
        self.assertFalse(result["continuous"][3])
        self.assertEqual(result["measurement_source"], "RAW_NORMAL_FACE")


if __name__ == "__main__":
    unittest.main()
