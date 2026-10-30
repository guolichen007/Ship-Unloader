"""Physical line decisions must commute with coordinate transformations."""

import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ship_perception.perception.review_frame import ReviewFrame, line_displacements
from ship_perception.perception.review_geometry import ReviewGeometryAssembler
from ship_perception.perception.region_model import OpeningRegion, TopologySupport
from ship_perception.r1_static.run import resolve_config


class ReviewFrameContracts(unittest.TestCase):
    def test_one_input_projection_roundtrips_without_changing_geometry(self):
        angle = math.radians(13.7)
        axes = ((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle)))
        frame = ReviewFrame("V", "map", axes)
        xy = np.array([[-75.0, 4.0], [20.0, 16.0], [100.0, -80.0]])
        np.testing.assert_allclose(frame.to_input(frame.to_review(xy)), xy, atol=1e-12)
        self.assertEqual(frame.audit()["transform_count"], 1)
        self.assertEqual(frame.audit()["coordinate_frame"], "VESSEL_REVIEW_UV")

    def test_displacements_are_rotation_and_translation_invariant(self):
        a = dict(normal_uv=[1.0, 0.0], rho=0.0)
        b = dict(normal_uv=[math.cos(0.02), math.sin(0.02)], rho=0.1)
        points = np.array([[0.0, 0.0], [0.0, 8.0]])
        expected = line_displacements(a, b, points)
        rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
        translation = np.array([1030.0, -950.0])
        transformed = []
        for line in (a, b):
            n = rotation @ line["normal_uv"]
            transformed.append(
                dict(normal_uv=n.tolist(), rho=line["rho"] + float(n @ translation))
            )
        actual = line_displacements(*transformed, points @ rotation.T + translation)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_competing_mode_verdict_does_not_depend_on_map_origin(self):
        config, _ = resolve_config()
        polygons = []
        states = []
        for translation in [
            np.zeros(2),
            np.array([-1000.0, 2000.0]),
            np.array([7000.0, -3500.0]),
        ]:
            lo = translation
            hi = translation + [20.0, 8.0]
            region = OpeningRegion(
                "R",
                "V",
                tuple(np.r_[lo, hi]),
                tuple(np.r_[lo - 4, hi + 4]),
                (TopologySupport("S4_OPENING_TOPOLOGY", "SUPPORTING"),),
                component_bounds=(tuple(np.r_[lo, hi]),),
            )
            region.identity_role = "CONFIRMED_CONNECTED_OPENING"
            region.side_topology_support["V0"] = TopologySupport(
                "STRUCTURAL_LINE_FAMILY", "SUPPORTING"
            )
            assembler = ReviewGeometryAssembler(config)
            prior, _ = assembler.assemble(region, np.eye(2))
            modes = []
            for i, angle in enumerate([0.0, math.radians(1.0)]):
                normal = np.array([math.cos(angle), math.sin(angle)])
                direction = np.array([-normal[1], normal[0]])
                modes.append(
                    dict(
                        mode_id=str(i),
                        rho=4 * normal[1] + float(normal @ translation),
                        normal_uv=normal.tolist(),
                        support_span=(
                            np.array([0.0, 8.0]) + float(direction @ translation)
                        ).tolist(),
                        source="MEASURED_RAW",
                        state="MEASURED_WEAK",
                        uncertainty_m=0.05,
                        independent_raw_sections=4,
                        physical_role_improved=True,
                    )
                )
            result = assembler.update(region, prior, {"U0": modes}, np.eye(2))
            polygons.append(np.asarray(result["boundary_xy"]) - translation)
            states.append(result["sides"]["U0"]["final_state"])
        self.assertEqual(states, ["MEASURED_WEAK"] * 3)
        for polygon in polygons[1:]:
            np.testing.assert_allclose(polygon, polygons[0], atol=1e-9)

    def test_global_axis_ambiguity_cannot_skip_valid_local_observation_compilation(
        self,
    ):
        from ship_perception.perception import s4_topology_bridge as module

        config, _ = resolve_config()
        grid = SimpleNamespace(
            centers=lambda: np.meshgrid(np.arange(3.0), np.arange(2.0))
        )
        cache = SimpleNamespace(
            config=config,
            vessel=SimpleNamespace(vessel_id="V", axes=np.eye(2)),
            xyz=np.zeros((10, 3)),
        )
        s4 = dict(
            edges=[],
            segment_resolutions=[
                dict(
                    segment_id="S",
                    reference_role="RIM",
                    profiles=[dict(status="OBSERVED")],
                )
            ],
        )
        with patch.object(
            module, "_hierarchy", return_value=({}, dict(coarse=grid), [])
        ), patch.object(module, "analyze_scene_s4r1", return_value=s4), patch.object(
            module,
            "structural_axis_estimate",
            return_value=dict(
                status="AXIS_AMBIGUOUS",
                axis_a=None,
                axis_b=None,
                axis_uncertainty_deg=3.0,
            ),
        ):
            with patch.object(
                module.S4TopologyBridge, "_families"
            ) as families, patch.object(
                module.S4TopologyBridge, "_partition_nodes"
            ), patch.object(
                module.S4TopologyBridge, "_vertices"
            ), patch.object(
                module.S4TopologyBridge, "_relations"
            ):
                bridge = module.S4TopologyBridge(cache, {})
        families.assert_called_once()
        self.assertEqual(bridge.axis["status"], "AXIS_AMBIGUOUS")
        np.testing.assert_array_equal(bridge.axes, np.eye(2))
        self.assertIs(bridge.s4, s4)


if __name__ == "__main__":
    unittest.main()
