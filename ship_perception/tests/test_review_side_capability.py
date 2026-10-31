"""Mechanism contracts for existing-side replay, without real scene coordinates."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from ship_perception.perception import review_side_capability as cap
from ship_perception.perception.review_measurement import (
    qualified_raw_pair_gain,
    raised_opposed_end_gain,
)
from ship_perception.r1_static.run import resolve_config
from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.vessel_height_topology import persistent_height_modes


class ReviewSideCapabilityContracts(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()

    def test_local_height_side_capability_matches_existing_without_corner_products(
        self,
    ):
        u, v = np.meshgrid(np.arange(-5, 25, 0.15), np.arange(-5, 15, 0.15))
        z = np.where((u > 0) & (u < 20) & (v > 0) & (v < 10), -4.0, 0.0)
        grid = R1HeightGrid.from_points(np.c_[u.ravel(), v.ravel(), z.ravel()], 0.15)
        old = persistent_height_modes(grid, [0, 0, 20, 10], self.config)
        with patch(
            "ship_perception.r1_static.vessel_height_topology.persistent_height_corners",
            side_effect=AssertionError("corner product forbidden"),
        ):
            new = cap.local_height_modes(grid, [0, 0, 20, 10], self.config)
        for key in ["levels_m", "topology", "modes", "footprint_edges"]:
            self.assertEqual(new[key], old[key])
        self.assertEqual(new["persistent_corners"], [])

    def test_only_self_qualified_pairs_authorize_raw_review_gain(self):
        row = dict(
            status="STRONG_UNIQUE_EDGE",
            measurement_source="RAW_PAIRED_PROFILE",
            sections=[
                dict(
                    section_record=dict(
                        status="UNIQUE_SIGNED_RAW_STEP",
                        height=dict(transition=True),
                        continuous=[0, 0.02, 20, True],
                    )
                )
                for _ in range(self.config["boundary"]["profile_min_sections"])
            ],
        )
        self.assertTrue(qualified_raw_pair_gain(row, self.config))
        row["sections"][0]["section_record"]["continuous"][3] = False
        self.assertFalse(qualified_raw_pair_gain(row, self.config))
        row["physical_role_improved"] = True
        self.assertFalse(qualified_raw_pair_gain(row, self.config))

    def test_opposed_end_faces_do_not_authorize_low_basin_or_lateral_crop(self):
        r = SimpleNamespace(
            identity_evidence=dict(
                nomination=dict(polarity="RAISED"),
                sides={
                    "U0": dict(verified_face_provider_ids=["A"]),
                    "U1": dict(verified_face_provider_ids=["B"]),
                },
            )
        )
        self.assertTrue(raised_opposed_end_gain(r, "U0", ["A"]))
        self.assertFalse(raised_opposed_end_gain(r, "V0", ["A"]))
        self.assertFalse(raised_opposed_end_gain(r, "U0", ["B"]))
        r.identity_evidence["nomination"]["polarity"] = "LOW"
        self.assertFalse(raised_opposed_end_gain(r, "U0", ["A"]))
        r.identity_evidence["nomination"]["polarity"] = "RAISED"
        r.identity_evidence["sides"]["U1"]["verified_face_provider_ids"] = ["A"]
        self.assertFalse(raised_opposed_end_gain(r, "U0", ["A"]))

    def fixture(self, family_span):
        r = SimpleNamespace(
            region_id="R",
            identity_role="CONFIRMED_INDEPENDENT_OPENING",
            conservative_bounds=[0.0, 0.0, 20.0, 10.0],
            identity_evidence={},
            side_topology_support={
                n: SimpleNamespace(
                    source_type="STRUCTURAL_LINE_FAMILY",
                    state="UNOBSERVED",
                    provider_ids=["F"] if n == "U0" else [],
                )
                for n, *_ in cap.SIDES
            },
        )
        grid = SimpleNamespace(
            cell_m=0.15,
            count=np.ones((2, 2)),
            median=np.zeros((2, 2)),
            axis_centers=lambda: (np.arange(2), np.arange(2)),
        )
        cache = SimpleNamespace(
            context_grid=lambda *a: grid,
            side_candidates=lambda *a: {n: [] for n, *_ in cap.SIDES},
            s4_topology_bridge=dict(
                side_families=[
                    dict(
                        family_id="F",
                        normal_uv=[1.0, 0.0],
                        rho=2.0,
                        rho_uncertainty=0.1,
                        support_span=family_span,
                    )
                ],
                node_partition_audit=[],
            ),
        )
        return r, cache

    def replay(self, family_span):
        r, cache = self.fixture(family_span)

        def profile(grid, span, anchor, name, config, **kw):
            if tuple(span) == (0.0, 10.0) and anchor == 0.0 and name == "V0":
                return dict(
                    accepted=True,
                    slope_deg=0.0,
                    dispersion_m=0.02,
                    median_contrast_m=2.0,
                    samples=[[y, 2.0, 2.0] for y in [2.0, 4.0, 6.0, 8.0]],
                )
            return dict(accepted=False, reason="UNOBSERVED", samples=[])

        with patch.object(
            cap,
            "local_height_modes",
            return_value=dict(modes={0: [], 1: []}, topology="LOW_BASIN"),
        ), patch.object(cap, "_enrich_side", return_value=[]), patch.object(
            cap,
            "_section_consensus",
            return_value=dict(transition_count=0, sections=[]),
        ), patch.object(
            cap, "_section_transition", side_effect=profile
        ):
            return cap.compile_constraints(
                cache, [r], self.config, {"R": {n: [] for n, *_ in cap.SIDES}}
            )["R"]["U0"][0]

    def test_s4_corroboration_requires_actual_observed_interval_overlap(self):
        near = self.replay([3.0, 7.0])
        far = self.replay([100.0, 101.0])
        self.assertTrue(near["independent_opening_role_gain"])
        self.assertEqual(near["rho"], 2.0)
        self.assertFalse(far["independent_opening_role_gain"])
        self.assertEqual(far["rho"], 0.0)
        self.assertEqual(far["state"], "CONSERVATIVE_PRIOR")

    def test_measured_intervals_follow_fitted_tangent_and_exclude_missing_sections(
        self,
    ):
        from ship_perception.perception.review_measurement import raw_profile_line

        row = dict(
            rho=2.0,
            fit=dict(slope=0.1, reference_along_m=10.0, outlier_section_indexes=[1]),
            sections=[
                dict(along=4.0, section_span_m=[3.0, 5.0]),
                dict(along=10.0, section_span_m=[9.0, 11.0]),
                dict(along=16.0, section_span_m=[15.0, 17.0]),
            ],
        )
        for axis in [0, 1]:
            line = raw_profile_line(row, axis, [0.0, 20.0])
            n = np.array(line["normal_uv"])
            t = np.array([-n[1], n[0]])
            self.assertEqual(len(line["measured_intervals"]), 2)
            recovered = []
            for interval in line["measured_intervals"]:
                points = line["rho"] * n + np.array(interval)[:, None] * t
                np.testing.assert_allclose(
                    points[:, axis],
                    2.0 + 0.1 * (points[:, 1 - axis] - 10.0),
                    atol=1e-12,
                )
                recovered.extend(points[:, 1 - axis])
            np.testing.assert_allclose(
                sorted(recovered), [3.0, 5.0, 15.0, 17.0], atol=1e-12
            )
            self.assertLess(max(recovered), 20.0)

    def test_partial_measurement_retains_internal_gaps_and_conservative_extensions(
        self,
    ):
        from ship_perception.tests.test_review_geometry_v29 import (
            ReviewGeometryContracts,
        )
        from ship_perception.perception.review_geometry import ReviewGeometryAssembler

        fixture = ReviewGeometryContracts()
        fixture.setUp()
        r = fixture.region()
        a = ReviewGeometryAssembler(self.config)
        result, _ = a.assemble(r, np.eye(2))
        mode = fixture.mode(0.0)
        mode.update(
            support_span=[2.0, 6.0], measured_intervals=[[2.0, 3.0], [5.0, 6.0]]
        )
        after = a.update(r, result, {"U0": [mode]}, np.eye(2))
        side = after["sides"]["U0"]
        self.assertEqual(side["measured_interval"], [[2.0, 3.0], [5.0, 6.0]])
        self.assertEqual(
            side["conservative_extension_interval"],
            [[0.0, 2.0], [3.0, 5.0], [6.0, 8.0]],
        )
        self.assertEqual(after["geometry_status"], "COMPLETE_REVIEW_GEOMETRY")
        self.assertFalse(after["steel_edge_certified"])
        self.assertEqual(side["final_state"], "GEOMETRY_CONSTRAINED")

    def test_runtime_has_no_baseline_dependency(self):
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "perception"
        forbidden = [
            "REVIEW_BASELINE_0930",
            "SEALED_INNER_EDGE_GOLDEN",
            "REVIEW_CAPABILITY_BASELINE_V1",
            "review/船舱标定图",
        ]
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    text = node.value.replace("\\", "/")
                    for token in forbidden:
                        self.assertNotIn(token, text, str(path))
        module = ast.parse(
            (root / "review_side_capability.py").read_text(encoding="utf8")
        )
        imports = [n.module for n in ast.walk(module) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(
            any("corner_topology" in (m or "") or "r2f" in (m or "") for m in imports)
        )

    def test_multimode_restores_prior_provenance_and_keeps_raw_candidates(self):
        from ship_perception.tests.test_review_geometry_v29 import (
            ReviewGeometryContracts,
        )
        from ship_perception.perception.review_geometry import ReviewGeometryAssembler

        fixture = ReviewGeometryContracts()
        fixture.setUp()
        r = fixture.region()
        a = ReviewGeometryAssembler(self.config)
        prior = fixture.mode(0.0, source="TOPOLOGY_SUPPORTED")
        prior.update(
            raw_support_ids=["TOPO_QUERY"],
            height_support_ids=["HEIGHT_PRIOR"],
            topology_support_ids=["FAMILY_PRIOR"],
        )
        result, _ = a.assemble(r, np.eye(2), topology_sides={"U0": [prior]})
        after = a.update(
            r, result, {"U0": [fixture.mode(-0.1), fixture.mode(-0.9)]}, np.eye(2)
        )
        side = after["sides"]["U0"]
        self.assertEqual(side["final_state"], "MULTIMODE")
        self.assertEqual(side["raw_support_ids"], ["TOPO_QUERY"])
        self.assertEqual(side["height_support_ids"], ["HEIGHT_PRIOR"])
        self.assertEqual(side["topology_support_ids"], ["FAMILY_PRIOR"])
        self.assertEqual(
            sum(m["source"] == "MEASURED_RAW" for m in side["candidate_modes"]), 2
        )


if __name__ == "__main__":
    unittest.main()
