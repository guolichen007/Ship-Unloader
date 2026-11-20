"""R3 provenance, arbitration and server-owned choice regression tests."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np

from ship_perception.manual_review.contract import (
    ContractError,
    IDENTITY,
    canonical,
    digest,
)
from ship_perception.manual_review.frame_session import FrameSession
from ship_perception.manual_review.guided_v3 import guided_match, tracks
from ship_perception.manual_review.side_candidates import CONFIG
from ship_perception.manual_review.source_identity import REQUIRED, file_hash, identify
from ship_perception.manual_review.workbench_auto import AutomaticRun
from ship_perception.manual_review.workbench_v3 import WorkbenchV3, corner_order
from tools.mhr_synthetic import rect, save_pcd, synthesized


class R3Tests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.xy = rect(np.array([12.0, -22.0]), 0.16, 18.0, 9.0)
        self.pcd = self.root / "input.pcd"
        save_pcd(self.pcd, synthesized(self.xy))
        self.frame = FrameSession(self.pcd)
        self.review = WorkbenchV3(
            self.frame,
            self.root,
            AutomaticRun(),
            {"status": "VERIFIED", "build_sha": "1" * 40},
        )
        self.identity = {k: self.frame.manifest[k] for k in IDENTITY}

    def api(self, action, payload):
        return self.review.handle(
            "/api/v3/refine" + action, dict(self.identity, **payload)
        )

    def request(self, key="first", xy=None):
        rough = self.xy + [0.12, -0.06] if xy is None else xy
        return dict(
            annotation_id="local",
            expected_parent_revision=0,
            request_id=key,
            schema="manual_selection.1",
            operation="CREATE",
            target_opening_id=None,
            target_vessel_id=None,
            bundle_digest=None,
            client_xy=rough.tolist(),
            vertices_uv=self.frame.projection.uv(rough).tolist(),
        )

    def binding(self, result):
        return dict(
            annotation_id="local",
            expected_parent_revision=0,
            bundle_digest=None,
            **{k: result[k] for k in ("request_id", "list_digest", "draft_digest")}
        )

    def commit(self, result, mutation="one"):
        return dict(
            self.binding(result),
            mutation_id=mutation,
            **{
                k: result[k]
                for k in ("candidate_id", "geometry_digest", "selection_id")
            }
        )

    def test_synthetic_rotated_create_and_cas(self):
        result = self.api("", self.request())
        self.assertEqual(result["match_state"], "GUIDED_REFINED_PENDING")
        measured = result["geometry"]
        self.assertLessEqual(
            max(np.linalg.norm(np.array(measured["xy"]) - self.xy, axis=1)), 0.05
        )
        self.assertLessEqual(abs(measured["axis_deg"] - np.degrees(0.16)), 0.5)
        self.assertFalse(measured["corner_z"][0].get("raw_indexes"))
        form = self.commit(result)
        with self.assertRaisesRegex(ContractError, "CANDIDATE_BINDING"):
            self.api("/confirm", dict(form, geometry_digest="wrong"))
        with self.assertRaisesRegex(ContractError, "INVALID_SCHEMA"):
            self.api("/confirm", dict(form, geometry=measured))
        saved = self.api("/confirm", form)
        self.assertFalse(saved["control_eligible"])
        self.assertEqual(saved, self.api("/confirm", form))
        with self.assertRaisesRegex(ContractError, "STALE_PARENT"):
            self.api("/confirm", dict(form, mutation_id="different"))

    def test_ownership_and_direction_contract(self):
        target = {"geometry": {"xy": self.xy.tolist()}}
        for selection in (self.xy[[2, 3, 0, 1]], self.xy[[2, 1, 0, 3]]):
            xy, mapping = corner_order(selection, target)
            np.testing.assert_array_equal(xy, self.xy)
            np.testing.assert_array_equal(
                selection[mapping["request_vertices"]], self.xy
            )
        outside = self.xy + np.array([100, 0])
        result = guided_match(
            self.frame, outside, {"target_opening_id": "target"}, {"target": target}
        )
        self.assertEqual(result["reason"], "WRONG_TARGET")
        result = guided_match(self.frame, self.xy, {}, {"target": target})
        self.assertEqual(result["reason"], "COMPETING_AUTO_TARGET")

    def test_cancel_and_stale_frame_keep_saved_head(self):
        result = self.api("", self.request())
        with self.assertRaisesRegex(ContractError, "STALE"):
            self.review.handle(
                "/api/v3/refine/confirm",
                dict(self.identity, **self.commit(result), view_id="other"),
            )
        self.api("/cancel", {"annotation_id": "local", "request_id": "first"})
        with self.assertRaisesRegex(ContractError, "STALE_DRAFT"):
            self.api("/confirm", self.commit(result))
        self.assertEqual(self.review.view()["heads"], {})

    def test_candidate_switch_revokes_old_selection_and_concurrent_commit(self):
        candidate = guided_match(self.frame, self.xy, {}, {})
        self.assertEqual(candidate["match_state"], "GUIDED_REFINED_PENDING")
        old = candidate["candidates"][0]
        second = dict(old, candidate_id="other")
        candidate.update(match_state="GUIDED_CHOICES_PENDING", candidates=[old, second])
        with patch(
            "ship_perception.manual_review.workbench_v3.guided_match",
            return_value=candidate,
        ):
            result = self.api("", self.request())
        self.assertIsNone(result["selection_id"])
        with self.assertRaisesRegex(ContractError, "SELECTION_BINDING"):
            self.api(
                "/confirm",
                dict(
                    self.commit(result),
                    candidate_id=old["candidate_id"],
                    geometry_digest=old["geometry_digest"],
                ),
            )
        forms = [
            dict(
                self.binding(result),
                candidate_id=c["candidate_id"],
                geometry_digest=c["geometry_digest"],
            )
            for c in (old, second)
        ]
        first = self.api("/select", forms[0])
        later = self.api("/select", forms[1])
        with self.assertRaisesRegex(ContractError, "SELECTION_BINDING"):
            self.api("/confirm", self.commit(first))

        def submit(key):
            try:
                return self.api("/confirm", self.commit(later, key))["state"]
            except ContractError as exc:
                return str(exc)

        with ThreadPoolExecutor(2) as threads:
            self.assertCountEqual(
                threads.map(submit, ["a", "b"]),
                ["ACCEPTED_GUIDED_CHOICE", "STALE_PARENT"],
            )

    def test_budget_no_geometry_and_unverified_source(self):
        with patch("ship_perception.manual_review.guided_v3.SEARCH_LIMIT", 0):
            result = self.api("", self.request("short"))
        self.assertEqual(result["match_state"], "SEARCH_INCOMPLETE")
        self.assertIsNone(result["geometry"])
        self.assertEqual(self.review.view()["heads"], {})
        result = self.api("", self.request())
        self.review.source_identity = {"status": "UNVERIFIED", "build_sha": None}
        with self.assertRaisesRegex(ContractError, "UNVERIFIED_SOURCE"):
            self.api("/confirm", self.commit(result))

    def test_profile_binding_and_stale_window(self):
        result = self.api("", self.request())
        bound = dict(self.binding(result), side_id="V0")
        section = self.api("/profile", bound)
        self.assertTrue(0 < len(section["points"]) <= 800)
        self.assertEqual(len(section["points"][0]), 2)
        with self.assertRaisesRegex(ContractError, "DRAFT_BINDING"):
            self.api("/profile", dict(bound, list_digest="other"))
        with patch(
            "ship_perception.manual_review.workbench_v3.monotonic", return_value=1e12
        ):
            with self.assertRaisesRegex(ContractError, "EXPIRED_OR_MISSING"):
                self.api("/profile", bound)

    def test_alias_does_not_chain_or_merge_disjoint_height_layers(self):
        def fit(shift, height, base):
            rows = [
                dict(
                    slab=2 * i,
                    t_m=float(i + 1),
                    d_m=shift,
                    top_m=height,
                    face=list(range(base + i * 3, base + i * 3 + 3)),
                )
                for i in range(8)
            ]
            return (0.0, shift, rows, 0.0)

        fits = [fit(0, 3, 0), fit(0.04, 3, 0), fit(0.08, 3, 0), fit(0.04, 4, 40)]
        with patch(
            "ship_perception.manual_review.guided.training_candidates",
            return_value=(fits, 4),
        ):
            hypotheses, _ = tracks([], 10, 0.6, CONFIG, lambda: None)
        self.assertEqual(len(hypotheses), 3)
        self.assertEqual(len(hypotheses[0]["fit_instances"]), 2)
        self.assertLessEqual(hypotheses[0]["alias_max_displacement_m"], 0.05)


class PackageIdentityTests(unittest.TestCase):
    def test_release_identity_missing_wrong_parent_and_tampered(self):
        with TemporaryDirectory() as temp:
            parent = Path(temp)
            (parent / ".git").mkdir()
            root = parent / "arbitrary"
            root.mkdir()
            self.assertEqual(identify(root)["reason"], "MISSING_PROJECT_IDENTITY")
            files = {}
            for name in REQUIRED:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode("utf8"))
                files[name] = file_hash(path)
            manifest = dict(
                version="MHR-R3",
                protocols=[2, 3],
                clean=True,
                build_sha="1" * 40,
                production_source_sha256="2" * 64,
                files=files,
            )
            (root / "mhr-build.json").write_bytes(canonical(manifest))
            self.assertEqual(identify(root)["build_sha"], "1" * 40)
            (root / REQUIRED[0]).write_bytes(b"tampered")
            self.assertEqual(identify(root)["reason"], "INVALID_RELEASE_MANIFEST")
            (root / "mhr-build.json").write_bytes(canonical(dict(manifest, files={})))
            self.assertEqual(identify(root)["reason"], "INVALID_RELEASE_MANIFEST")
