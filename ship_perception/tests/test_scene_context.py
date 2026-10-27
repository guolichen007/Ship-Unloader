"""Frozen scene provenance and isolated geometry failures."""

import json
import unittest
from dataclasses import replace
from unittest import mock
import numpy as np
from ship_perception.io.frame import PointCloudFrame
from ship_perception.perception.context import (
    BoundaryMeasurementContext,
    SceneEvidenceContext,
    VesselEvidence,
    freeze,
)
from ship_perception.r1_static.run import resolve_config


class SceneContextContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()

    def test_mixed_unresolved_source_recovers_only_proven_single_support_subset(self):
        def source(name, indexes, status, supports):
            values = np.frombuffer(
                np.asarray(indexes, dtype="<i8").tobytes(), dtype="<i8"
            )
            return VesselEvidence(
                freeze(
                    dict(
                        vessel_hypothesis_id=name,
                        classification_status=status,
                        source_support_ids=supports,
                        local_axes=[[1, 0], [0, 1]],
                        bbox_xy=[0, 0, 20, 3],
                    )
                ),
                values,
                name + "-hash",
            )

        xyz = np.array(
            [[3, 1, 0], [4, 1, 0], [9, 1, 0], [12, 1, 0], [5, 1, 0]], dtype=float
        )
        core = source("core", [0], "VESSEL_HYPOTHESIS", ["S0"])
        mixed = source("mixed", [1, 2, 3], "UNRESOLVED", ["S0", "S1"])
        bounds = dict(S0=[0, 0, 10, 3], S1=[8, 0, 20, 3])
        context = BoundaryMeasurementContext.build(
            core, [core, mixed], xyz, bounds, 1.0
        )
        np.testing.assert_array_equal(context.point_indexes, [0, 1])
        np.testing.assert_array_equal(core.point_indexes, [0])
        np.testing.assert_array_equal(mixed.point_indexes, [1, 2, 3])
        self.assertFalse(context.identity()["context_ownership_confirmed"])
        part = context.sources[1]
        self.assertEqual(part.identity()["origin_point_index_sha256"], "mixed-hash")
        self.assertEqual(part.identity()["support_ids"], ["S0"])
        with self.assertRaises(ValueError):
            part.point_indexes.setflags(write=True)
        # Bounds overlap and the unrelated support stay unknown. Missing
        # support bounds and a second resolved core both forbid borrowing.
        missing = BoundaryMeasurementContext.build(
            core, [core, mixed], xyz, dict(S0=bounds["S0"]), 1.0
        )
        np.testing.assert_array_equal(missing.point_indexes, [0])
        second = source("second", [4], "VESSEL_HYPOTHESIS", ["S0"])
        separated = BoundaryMeasurementContext.build(
            core, [core, mixed, second], xyz, bounds, 1.0
        )
        np.testing.assert_array_equal(separated.point_indexes, [0])
        other_xyz = xyz.copy()
        other_xyz[4, 0] = 13.0
        other_core = source("other", [4], "VESSEL_HYPOTHESIS", ["S1"])
        isolated = BoundaryMeasurementContext.build(
            core, [core, mixed, other_core], other_xyz, bounds, 1.0
        )
        np.testing.assert_array_equal(isolated.point_indexes, [0, 1])
        self.assertEqual(
            isolated.identity(),
            BoundaryMeasurementContext.build(
                core, [other_core, mixed, core], other_xyz, bounds, 1.0
            ).identity(),
        )

    def test_frozen_context_rejects_same_xyz_different_wire_identity(self):
        from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG

        u, v = np.meshgrid(np.arange(0, 12, 0.2), np.arange(0, 8, 0.2))
        points = np.column_stack(
            (u.ravel(), v.ravel(), np.where(u < 5, -1.0, 0.0).ravel())
        )
        frame = PointCloudFrame(
            points,
            None,
            "map",
            1,
            2,
            "topic",
            ("x", "y", "z"),
            len(points),
            json.dumps(dict(wire_sha256="first")),
        )
        context = SceneEvidenceContext.build(
            frame,
            self.config,
            json.loads(RESEARCH_CONFIG.read_text()),
            "config",
            "software",
        )
        context.validate_frame(frame)
        for vessel in context.vessels:
            self.assertFalse(vessel.point_indexes.flags.writeable)
            with self.assertRaises(ValueError):
                vessel.point_indexes.setflags(write=True)
            with self.assertRaises(TypeError):
                vessel.public["local_axes"] = [[1, 0], [0, 1]]
        other = replace(frame, provenance_json=json.dumps(dict(wire_sha256="second")))
        self.assertEqual(frame.sha256, other.sha256)
        with self.assertRaisesRegex(ValueError, "WIRE_IDENTITY_MISMATCH"):
            context.validate_frame(other)

    def test_legacy_vessel_ambiguity_preserves_sibling_and_context(self):
        from types import SimpleNamespace
        from ship_perception.r1_static.corner_topology_review import review_points

        scene = dict(
            vessel_hypotheses=[
                dict(
                    vessel_hypothesis_id=ident,
                    point_count=1,
                    classification_status="VESSEL_HYPOTHESIS",
                )
                for ident in ("V0", "V1")
            ]
        )
        context = SimpleNamespace(
            scene=freeze(scene),
            vessels=tuple(
                SimpleNamespace(vessel_id=ident, point_indexes=np.array([index]))
                for index, ident in enumerate(("V0", "V1"))
            ),
        )

        def solve(*args, **kwargs):
            if args[4] == "V0":
                raise ValueError("OPENING_U_STATION_TOPOLOGY_AMBIGUOUS")
            return dict(vessel_hypothesis_id="V1", rectangles=[dict(hatch_id="kept")])

        with mock.patch(
            "ship_perception.r1_static.corner_topology_review.infer_scene_vessels",
            side_effect=AssertionError("B1 must not repeat"),
        ), mock.patch(
            "ship_perception.r1_static.corner_topology_review.solve_vessel_openings",
            side_effect=solve,
        ):
            report, _, _ = review_points(
                "test",
                np.zeros((2, 3)),
                "sha",
                dict(input_sha256="sha", scene=scene),
                dict(rectangles=[]),
                self.config,
                {},
                scene_context=context,
            )
        self.assertEqual(report["vessels"][0]["status"], "AMBIGUOUS_TOPOLOGY")
        self.assertEqual(report["vessels"][1]["rectangles"][0]["hatch_id"], "kept")

    def test_boundary_context_never_promotes_or_merges_point_ownership(self):
        def source(name, indexes, status, supports):
            values = np.frombuffer(
                np.asarray(indexes, dtype="<i8").tobytes(), dtype="<i8"
            )
            return VesselEvidence(
                freeze(
                    dict(
                        vessel_hypothesis_id=name,
                        classification_status=status,
                        source_support_ids=supports,
                        local_axes=[[1, 0], [0, 1]],
                        bbox_xy=[0, 0, 1, 1],
                    )
                ),
                values,
                name + "-hash",
            )

        core = source("V0", [0, 2], "VESSEL_HYPOTHESIS", ["S0"])
        sibling = source("V1", [1, 3], "UNRESOLVED", ["S0"])
        unrelated = source("V2", [4], "UNRESOLVED", ["S1"])
        mixed = source("V3", [5], "UNRESOLVED", ["S0", "S1"])
        sources = [core, sibling, unrelated, mixed]
        context = BoundaryMeasurementContext.build(core, sources)
        np.testing.assert_array_equal(core.point_indexes, [0, 2])
        self.assertEqual([v.vessel_id for v in context.sources], ["V0", "V1"])
        np.testing.assert_array_equal(context.point_indexes, [0, 1, 2, 3])
        np.testing.assert_array_equal(context.point_source_codes, [0, 1, 0, 1])
        self.assertEqual(sibling.public["classification_status"], "UNRESOLVED")
        self.assertFalse(context.identity()["context_ownership_confirmed"])
        for values in (context.point_indexes, context.point_source_codes):
            with self.assertRaises(ValueError):
                values.setflags(write=True)
        self.assertEqual(
            context.identity(),
            BoundaryMeasurementContext.build(core, sources[::-1]).identity(),
        )
        # A second resolved core sharing the physical support makes sibling
        # attribution ambiguous. Neither boat may borrow the other or its band.
        other_core = source("V4", [6], "VESSEL_HYPOTHESIS", ["S0"])
        separated = BoundaryMeasurementContext.build(core, sources + [other_core])
        self.assertEqual([v.vessel_id for v in separated.sources], ["V0"])
        self.assertEqual(
            [v.vessel_id for v in separated.unattributed_sources], ["V1", "V3"]
        )
        unresolved = BoundaryMeasurementContext.build(sibling, sources)
        self.assertEqual([v.vessel_id for v in unresolved.sources], ["V1"])
        batch = BoundaryMeasurementContext.build_all(sources)
        self.assertEqual(batch[0].identity(), context.identity())
        self.assertEqual(batch[1].identity(), unresolved.identity())


if __name__ == "__main__":
    unittest.main()
