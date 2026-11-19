"""R2 business state and independent Raw tests."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from ship_perception.manual_review.contract import (
    FLAGS,
    IDENTITY,
    ContractError,
    digest,
)
from ship_perception.manual_review.frame_session import FrameSession
from ship_perception.manual_review.guided import guided_match
from ship_perception.manual_review.workbench import Workbench
from ship_perception.manual_review.workbench_auto import AutomaticRun
from tools.mhr_synthetic import rect, save_pcd, synthesized


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.xy = rect(np.array([12.0, -22.0]), 0, 18.0, 9.0)
        self.xyz = synthesized(self.xy)
        self.path = self.root / "input.pcd"
        save_pcd(self.path, self.xyz)
        self.frame = FrameSession(self.path)
        self.auto = AutomaticRun()
        self.review = Workbench(self.frame, self.root, self.auto, "TEST")
        self.ident = {k: self.frame.manifest[k] for k in IDENTITY}

    def auto_target(self):
        geometry = dict(
            xy=self.xy.tolist(),
            xyz=[
                [x, y, z] for (x, y), z in zip(self.xy.tolist(), [None, 3.0, 3.2, None])
            ],
            corner_ids=["P1", "P2", "P3", "P4"],
            length_m=18.0,
            width_m=9.0,
            xyz_complete=False,
        )
        self.auto.publish(
            dict(
                status="READY",
                bundle_digest="bundle",
                provenance={"original": "review-layer"},
                targets={
                    "OC:h": dict(
                        vessel_id="v",
                        geometry=geometry,
                        geometry_digest=digest(geometry),
                        state="REVIEW_UNRESOLVED",
                        grade="UNRESOLVED",
                        adopted_for_review=True,
                        review_gates={},
                    )
                },
            )
        )
        return self.auto.snapshot["targets"]["OC:h"]

    def action(self, name, payload):
        return self.review.handle("/api/v2/" + name, {**self.ident, **payload})

    def mutation(self, annotation="OC:h", parent=0, mid="m1"):
        return dict(
            annotation_id=annotation,
            expected_parent_revision=parent,
            mutation_id=mid,
            bundle_digest=self.auto.snapshot["bundle_digest"],
        )

    def draft(self, target=None, parent=0, req="q", dx=0.12):
        rough = rect(np.array([12.0 + dx, -22.12]), 0.006, 18.08, 9.08)
        uv = self.frame.projection.uv(rough).tolist()
        return dict(
            annotation_id=target or "new",
            expected_parent_revision=parent,
            bundle_digest=self.auto.snapshot["bundle_digest"],
            schema="manual_selection.1",
            request_id=req,
            operation="REFINE" if target else "CREATE",
            target_opening_id=target,
            target_vessel_id="v" if target else None,
            vertices_uv=uv,
            client_xy=rough.tolist(),
        )

    def test_direct_confirmation_is_exact_and_never_invokes_matcher(self):
        target = self.auto_target()
        form = dict(self.mutation(), geometry_digest=target["geometry_digest"])
        with patch(
            "ship_perception.manual_review.guided.guided_match",
            side_effect=AssertionError,
        ):
            with self.assertRaisesRegex(ContractError, "STALE_AUTO_GEOMETRY"):
                self.action("auto/confirm", dict(form, geometry_digest="fake"))
            confirmed = self.action("auto/confirm", form)
            self.assertEqual(confirmed, self.action("auto/confirm", form))
            self.assertEqual(confirmed["geometry"], target["geometry"])
            self.assertEqual(confirmed["state"], "ACCEPTED_AUTO")
            self.assertFalse(confirmed["control_eligible"])
            view = self.review.view()
            self.action(
                "scene/confirm",
                dict(
                    mutation_id="scene",
                    scene_digest=view["scene_digest"],
                    confirm_empty=False,
                ),
            )
            self.assertEqual(self.review.view()["scene_state"], "REVIEW_COMPLETE")
            export = self.action("export", {"scene_digest": view["scene_digest"]})
            self.assertEqual(export["accepted"][0]["geometry"], target["geometry"])
            void = self.action(
                "auto/void", dict(form, mutation_id="void", expected_parent_revision=1)
            )
            self.assertIsNone(void["geometry"])
            self.assertEqual(self.review.view()["scene_state"], "REVIEW_REOPENED")
            self.assertEqual(
                self.action("undo", self.mutation(parent=2, mid="undo"))["geometry"],
                target["geometry"],
            )
        reopened = Workbench(FrameSession(self.path), self.root, self.auto, "TEST")
        self.assertEqual(reopened.view()["heads"]["OC:h"]["revision"], 3)

    def test_empty_scene_is_explicit_and_unreviewed_scene_is_blocked(self):
        def confirm(empty=False):
            return self.action(
                "scene/confirm",
                dict(
                    mutation_id="check",
                    scene_digest=self.review.view()["scene_digest"],
                    confirm_empty=empty,
                ),
            )

        with self.assertRaisesRegex(ContractError, "EXPLICIT_EMPTY_REQUIRED"):
            confirm()
        confirm(True)
        self.auto_target()
        self.assertEqual(self.review.view()["scene_state"], "REVIEW_REOPENED")
        with self.assertRaisesRegex(ContractError, "SCENE_CHANGED_OR_INCOMPLETE"):
            self.action(
                "scene/confirm",
                dict(
                    mutation_id="next",
                    scene_digest=self.review.view()["scene_digest"],
                    confirm_empty=True,
                ),
            )

    def test_guided_create_and_existing_refine_remeasure_same_raw(self):
        for target in (None, "OC:h"):
            if target:
                orig = self.auto_target()["geometry"]
            payload = self.draft(target, req="with-target" if target else "create")
            candidate = self.action("refine", payload)
            self.assertEqual(candidate["match_state"], "GUIDED_REFINED_PENDING")
            self.assertEqual(candidate["ownership"], "UNVERIFIED")
            self.assertNotIn("selected_sides", candidate)
            self.assertLess(
                np.max(np.abs(np.array(candidate["geometry"]["xy"]) - self.xy)), 0.05
            )
            if target:
                self.assertNotEqual(candidate["geometry"]["xyz"], orig["xyz"])
            fields = {
                k: candidate[k]
                for k in (
                    "request_id",
                    "candidate_id",
                    "draft_digest",
                    "geometry_digest",
                )
            }
            form = {
                **self.mutation(
                    payload["annotation_id"], mid="mut" + payload["request_id"]
                ),
                **fields,
            }
            with self.assertRaisesRegex(ContractError, "SUGGESTION_BINDING"):
                self.action("refine/confirm", dict(form, geometry_digest="bad"))
            saved = self.action("refine/confirm", form)
            self.assertEqual(saved["state"], "ACCEPTED_REFINED")
            self.assertEqual(saved["geometry"]["xyz"], candidate["geometry"]["xyz"])

    def test_new_draft_and_cancel_invalidate_older_candidate(self):
        form = self.draft()
        a = self.action("refine", form)
        b = self.action("refine", dict(form, request_id="next"))
        with self.assertRaisesRegex(ContractError, "STALE_DRAFT"):
            self.action(
                "refine/confirm",
                {
                    **self.mutation("new"),
                    **{
                        k: a[k]
                        for k in (
                            "request_id",
                            "candidate_id",
                            "draft_digest",
                            "geometry_digest",
                        )
                    },
                },
            )
        self.action("refine/cancel", {"request_id": "next", "annotation_id": "new"})
        with self.assertRaisesRegex(ContractError, "STALE_DRAFT"):
            self.action(
                "refine/confirm",
                {
                    **self.mutation("new"),
                    **{
                        k: b[k]
                        for k in (
                            "request_id",
                            "candidate_id",
                            "draft_digest",
                            "geometry_digest",
                        )
                    },
                },
            )
        self.assertFalse(self.review.view()["heads"])

    def test_manual_fallback_does_not_execute_guided_match(self):
        with patch(
            "ship_perception.manual_review.guided.guided_match",
            side_effect=AssertionError,
        ):
            result = self.action(
                "manual/commit", dict(self.draft(), mutation_id="manual")
            )
        self.assertEqual(result["state"], "MANUAL_ONLY")
        self.assertEqual([z for _, _, z in result["geometry"]["xyz"]], [None] * 4)

    def test_duplicate_and_noise_keep_physical_line_and_gaps_never_impute(self):
        rough = np.asarray(self.draft()["client_xy"])
        for label in ("duplicate", "noise", "missing-z"):
            with self.subTest(label=label):
                if label == "duplicate":
                    xyz = np.repeat(self.xyz, 4, axis=0)
                elif label == "noise":
                    xyz = self.xyz + np.random.default_rng(25).normal(
                        0, 0.0015, self.xyz.shape
                    )
                else:
                    distances = np.linalg.norm(self.xyz[:, None, :2] - self.xy, axis=2)
                    xyz = self.xyz[np.min(distances, axis=1) > 0.55]
                p = self.root / (label + ".pcd")
                save_pcd(p, xyz)
                r = guided_match(FrameSession(p), rough, {"operation": "CREATE"}, {})
                self.assertEqual(r["match_state"], "GUIDED_REFINED_PENDING")
                self.assertLess(
                    np.max(
                        np.linalg.norm(
                            np.asarray(r["geometry"]["xy"]) - self.xy, axis=1
                        )
                    ),
                    0.05,
                )
                if label == "missing-z":
                    self.assertIn(None, r["geometry"]["z"])

    def test_realistic_competing_or_unobserved_structures_do_not_confirm(self):
        rough = np.asarray(self.draft()["client_xy"])
        for asset in ("dual", "shore", "high_cargo", "occluded", "sparse", "deformed"):
            with self.subTest(asset=asset):
                p = self.root / (asset + ".pcd")
                save_pcd(p, synthesized(self.xy, asset, np.random.default_rng(22)))
                result = guided_match(
                    FrameSession(p), rough, {"operation": "CREATE"}, {}
                )
                self.assertNotEqual(result["match_state"], "GUIDED_REFINED_PENDING")
                self.assertIsNone(result["geometry"])

    def test_parent_and_frame_conflicts_leave_no_mutation(self):
        target = self.auto_target()
        form = {
            **self.ident,
            **self.mutation(parent=1),
            "geometry_digest": target["geometry_digest"],
        }
        with self.assertRaisesRegex(ContractError, "STALE_PARENT"):
            self.review.handle("/api/v2/auto/confirm", form)
        with self.assertRaisesRegex(ContractError, "STALE"):
            self.review.handle(
                "/api/v2/auto/confirm", dict(form, frame_instance_id="wrong")
            )
        self.assertFalse(self.review.view()["heads"])

    def test_cancel_before_arrival_prevents_persisted_candidate(self):
        form = self.draft()
        self.action(
            "refine/cancel", {"request_id": form["request_id"], "annotation_id": "new"}
        )
        with self.assertRaisesRegex(ContractError, "STALE_DRAFT"):
            self.action("refine", form)
        self.assertFalse(self.review.view()["heads"])

    def test_direction_search_prunes_conflicting_prefixes_before_budget(self):
        from ship_perception.manual_review.guided import feasible_directions
        from ship_perception.manual_review.side_candidates import CONFIG

        lower = {"slope": -0.04}
        upper = {"slope": 0.04}
        axes = [[lower, upper], [upper], [lower], [lower, upper]]
        self.assertEqual(list(feasible_directions(axes, [90.0] * 4, CONFIG)), [])
        axes[2].append(upper)
        self.assertEqual(
            list(feasible_directions(axes, [90.0] * 4, CONFIG)),
            [(upper, upper, upper, upper)],
        )

    def test_long_oblique_opening_stays_anchored(self):
        heading = 0.38
        truth = rect(np.array([1.1, -24.6]), heading, 52.0, 11.0)
        path = self.root / "oblique.pcd"
        save_pcd(path, synthesized(truth))
        rough = rect(np.array([1.31, -24.70]), heading + np.radians(0.48), 52.28, 10.8)
        result = guided_match(FrameSession(path), rough, {"operation": "CREATE"}, {})
        self.assertEqual(result["match_state"], "GUIDED_REFINED_PENDING")
        self.assertLess(
            np.linalg.norm(np.asarray(result["geometry"]["xy"]) - truth, axis=1).max(),
            0.05,
        )
        self.assertTrue(
            all(
                a <= b
                for a, b in zip(
                    result["endpoint_moves_m"], result["search_half_width_m"]
                )
            )
        )
