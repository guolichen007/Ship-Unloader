"""MHR coordinate, identity, browser HTTP boundary, and revision contracts."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from http.client import HTTPConnection
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter, sleep
from unittest.mock import patch
import unittest

import numpy as np

from ship_perception.manual_review.contract import (
    ContractError,
    IDENTITY,
    Projection,
    canonical,
    digest,
    parse_json,
    selection,
)
from ship_perception.manual_review.frame_session import FrameSession, projection_for
from ship_perception.manual_review.revisions import Review
from ship_perception.manual_review.server import LocalServer


def example(frame):
    uv = [[0.05, 0.5], [17.05, 0.5], [17.05, 19.25], [0.05, 19.25]]
    return {
        **{k: frame.manifest[k] for k in IDENTITY},
        "schema": "manual_selection.1",
        "request_id": "req",
        "annotation_id": "anno",
        "expected_parent_revision": 0,
        "operation": "CREATE",
        "target_opening_id": None,
        "target_vessel_id": None,
        "vertices_uv": uv,
        "client_xy": frame.projection.xy(uv).tolist(),
    }


class ManualReviewContracts(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pcd = self.root / ("\u56fe \u5730.pcd")
        self.pcd.write_text(
            "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
            "WIDTH 5\nHEIGHT 1\nPOINTS 5\nDATA ascii\n"
            "-4 -18.25 0\n4 -14.125 1\n1 -4 2\nnan 0 1\n1 -4 2\n",
            encoding="ascii",
        )
        self.frame = FrameSession(self.pcd)
        self.request = example(self.frame)

    def test_projection(self):
        p = Projection(-2240134.531, 8030550.156, 0.1, 8425, 315)
        uv = np.array([[0, 0], [p.nx, p.ny], [0.5, 0.5], [17.235, 18.912]])
        xy = p.xy(uv)
        self.assertLessEqual(np.max(np.abs(p.xy(p.uv(xy)) - xy)), 1e-8)
        self.assertGreater(p.xy([0, 0.5])[1], p.xy([0, p.ny - 0.5])[1])
        # Power-of-two cell fixture assesses the half-open boundaries exactly.
        g = Projection(-4, -8, 0.125, 8, 16)
        cells, valid = g.cells(
            g.xy([[0, 16], [0.5, 15.5], [8, 16], [0, 0], [-0.01, 4], [0.5, -0.5]])
        )
        self.assertEqual(valid.tolist(), [True, True, False, False, False, False])
        self.assertEqual(cells.tolist(), [[0, 0], [0, 0]])
        with self.assertRaisesRegex(ContractError, "GRID_BUDGET"):
            projection_for(np.array([[0.0, 0.0, 0.0], [1e6, 1e6, 0.0]]), 0.1)
        with self.assertRaisesRegex(ContractError, "XY_PRECISION"):
            projection_for(np.array([[1e8, 0.0, 0.0]]), 0.1)

    def test_manifest_raw_rows_and_layers(self):
        m = dict(self.frame.manifest)
        self.assertEqual(m.pop("manifest_sha256"), digest(m))
        self.assertEqual(m["wire_sha256"], sha256(self.pcd.read_bytes()).hexdigest())
        self.assertEqual(m["source_count"], 5)
        self.assertEqual(m["finite_count"], 4)
        self.assertEqual(m["dtype"], "<f4")
        self.assertFalse(self.frame.frame.xyz.flags.writeable)
        for name, data in self.frame.layers.items():
            self.assertEqual(sha256(data).hexdigest(), m["layers"][name])
            self.assertEqual(int.from_bytes(data[16:20], "big"), m["nx"])
            self.assertEqual(int.from_bytes(data[20:24], "big"), m["ny"])
        self.assertEqual(
            self.frame.index.query_indexes([1, -4, 1, -4]).tolist(), [2, 3]
        )

    def test_overlay_is_bound_and_never_written(self):
        manifest = self.frame.manifest
        overlay = self.root / "automatic.json"
        data = dict(
            schema="manual_auto_overlay.1",
            wire_sha256=manifest["wire_sha256"],
            semantic_sha256=manifest["semantic_sha256"],
            frame_id=manifest["frame_id"],
            targets=[
                dict(opening_id="h1", vessel_id="v1", xy=self.request["client_xy"])
            ],
        )
        expected = canonical(data)
        overlay.write_bytes(expected)
        bound = FrameSession(self.pcd, overlay=overlay)
        request = dict(
            example(bound),
            operation="REFINE",
            target_opening_id="h1",
            target_vessel_id="v1",
            annotation_id="h1",
        )
        review = Review(bound, self.root)
        result = review.suggest(request)
        self.assertIsNone(result["geometry"])
        self.assertEqual(overlay.read_bytes(), expected)
        overlay.write_bytes(canonical(dict(data, wire_sha256="invalid")))
        with self.assertRaisesRegex(ContractError, "OVERLAY_CHANGED"):
            review.suggest(request)
        self.assertFalse(FrameSession(self.pcd, overlay=overlay).targets)

    def test_selection_rejects_invalid_shapes_and_identity(self):
        p, m = self.frame.projection, self.frame.manifest
        expected = selection(self.request, m, p, {})
        self.assertEqual(expected.tolist(), self.request["client_xy"])
        for key in IDENTITY:
            with self.subTest(key=key), self.assertRaisesRegex(ContractError, "STALE"):
                selection(dict(self.request, **{key: "wrong"}), m, p, {})
        for vertices in (
            [[0.0, 0.0]] * 4,
            [[1, 1], [21, 1], [15, 24], [1, 21]],
            [[1, 1], [21, 22], [1, 22], [21, 1]],
            [[1, 1], [21, 1], [21, 14], [-1, 14]],
        ):
            with self.subTest(vertices=vertices), self.assertRaises(ContractError):
                selection(
                    dict(
                        self.request,
                        vertices_uv=vertices,
                        client_xy=p.xy(vertices).tolist(),
                    ),
                    m,
                    p,
                    {},
                )
        with self.assertRaisesRegex(ContractError, "XY_MISMATCH"):
            selection(dict(self.request, client_xy=[[0, 0]] * 4), m, p, {})

    def test_loopback_route_host_origin_and_json(self):
        with LocalServer(self.frame) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:

                def call(path, body=None, **overrides):
                    data = canonical(body) if isinstance(body, dict) else body
                    headers = {
                        "X-MHR-Token": server.token,
                        "Origin": server.origin,
                        "Content-Type": "application/json",
                        **overrides,
                    }
                    c = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                    try:
                        c.request(
                            "GET" if data is None else "POST", path, data, headers
                        )
                        r = c.getresponse()
                        response = parse_json(r.read())
                        self.assertFalse(response["steel_certified"])
                        self.assertFalse(response["control_eligible"])
                        return r.status, response
                    finally:
                        c.close()

                self.assertEqual(call("/api/selection", self.request)[0], 200)
                for bad in (
                    {"Host": "localhost"},
                    {"Origin": "https://example.invalid"},
                    {"X-MHR-Token": "wrong"},
                ):
                    self.assertEqual(
                        call("/api/selection", self.request, **bad)[0], 403
                    )
                self.assertEqual(call("/../secret")[0], 404)
                self.assertEqual(call("/api/selection", b"x" * 16385)[0], 413)
                for payload in (b'{"x":0,"x":0}', b'{"x":NaN}', b'{"x":1e99999}'):
                    self.assertEqual(call("/api/selection", payload)[0], 400)
            finally:
                server.shutdown()

    def test_revisions_compare_swap_retry_undo_export_restart(self):
        service = Review(self.frame, self.root)
        result = service.suggest(self.request)
        self.assertIsNone(result["geometry"])
        self.assertEqual(service.suggest(self.request), result)
        with self.assertRaisesRegex(ContractError, "ID_PAYLOAD_CONFLICT"):
            service.suggest(dict(self.request, annotation_id="different"))
        identity = {k: self.request[k] for k in IDENTITY}
        proposal = dict(
            **identity,
            mutation_id="m0",
            annotation_id="anno",
            expected_parent_revision=0,
            request_id=result["request_id"],
            suggestion_id=result["suggestion_id"],
            candidate_id=result["candidate_id"],
            geometry_digest=result["rough_digest"],
        )
        with self.assertRaisesRegex(ContractError, "SUGGESTION_BINDING"):
            service.handle("/api/manual", dict(proposal, candidate_id="forged"))
        with self.assertRaisesRegex(ContractError, "GEOMETRY_DIGEST"):
            service.handle("/api/manual", dict(proposal, geometry_digest="forged"))
        with self.assertRaisesRegex(ContractError, "CANDIDATE_NOT_MATCHED"):
            service.handle(
                "/api/accept", dict(proposal, geometry_digest=result["geometry_digest"])
            )

        def write(mid):
            try:
                return service.handle("/api/manual", dict(proposal, mutation_id=mid))
            except ContractError as e:
                return str(e)

        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(write, ("m0", "m1")))
        self.assertEqual(sum(r == "STALE_PARENT" for r in results), 1)
        chosen = "m0" if isinstance(results[0], dict) else "m1"
        reset = Review(self.frame, self.root)
        self.assertEqual(
            reset.handle("/api/manual", dict(proposal, mutation_id=chosen))["revision"],
            1,
        )
        self.assertEqual(reset.view()["heads"], {"anno": 1})
        out = reset.handle(
            "/api/export",
            dict(
                **identity,
                mutation_id="m2",
                annotation_id="anno",
                expected_parent_revision=1,
            ),
        )
        self.assertIn("false,false", out["csv"])
        self.assertIn("MANUAL_ONLY", out["json"])
        undo = reset.handle(
            "/api/undo",
            dict(
                **identity,
                mutation_id="m3",
                annotation_id="anno",
                expected_parent_revision=1,
            ),
        )
        self.assertEqual(undo["revision"], 2)
        self.assertIsNone(undo["geometry"])
        replaced = FrameSession(self.pcd)
        with self.assertRaisesRegex(ContractError, "STALE"):
            Review(replaced, self.root).handle(
                "/api/manual", dict(proposal, mutation_id=chosen)
            )

    def test_sqlite_abort_leaves_old_state(self):
        service = Review(self.frame, self.root)
        with self.assertRaises(RuntimeError):
            with service.store.transaction() as db:
                db.execute(
                    "INSERT INTO revisions VALUES (?,?,?,?)",
                    (service.store.scope, "sample", 1, b"{}"),
                )
                raise RuntimeError("injected write interruption")
        self.assertEqual(service.store.heads(), {})
        with self.assertRaises(ContractError):
            with service.store.transaction() as db:
                db.execute("INSERT INTO table_that_does_not_exist VALUES (0)")
        self.assertEqual(service.store.heads(), {})

    def test_cancelled_early_and_during_match_cannot_save(self):
        service = Review(self.frame, self.root)
        cancel = {k: self.request[k] for k in IDENTITY}
        cancel["request_id"] = self.request["request_id"]
        service.handle("/api/cancel", cancel)
        self.assertEqual(service.suggest(self.request)["execution_state"], "CANCELLED")
        active = dict(self.request, request_id="running")
        entered = Event()

        def slow(frame, xy, req, cancelled):
            entered.set()
            deadline = perf_counter() + 3
            while not cancelled() and perf_counter() < deadline:
                sleep(0.005)
            return {
                "geometry": None,
                "execution_state": "CANCELLED",
                "reason": "CANCELLED",
            }

        with patch("ship_perception.manual_review.revisions.match", slow):
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(service.suggest, active)
                self.assertTrue(entered.wait(2))
                service.handle("/api/cancel", dict(cancel, request_id="running"))
                self.assertEqual(
                    future.result(timeout=3)["execution_state"], "CANCELLED"
                )
        with service.store.transaction() as db:
            self.assertEqual(
                db.execute("SELECT count(*) FROM suggestions").fetchone()[0], 0
            )

    def test_uncommitted_child_process_exit(self):
        service = Review(self.frame, self.root)
        script = """import sqlite3, sys, os
c = sqlite3.connect(sys.argv[1])
c.execute("INSERT INTO revisions VALUES ('scope','test',1,'tampered')")
os._exit(91)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(service.store.path)], timeout=10
        )
        self.assertEqual(result.returncode, 91)
        with service.store.transaction() as db:
            self.assertEqual(
                db.execute("SELECT count(*) FROM revisions").fetchone()[0], 0
            )

    def test_replay_uses_server_pixel_roundtrip_for_raw_measurement(self):
        from tools.mhr_replay import replay
        from tools.mhr_synthetic import rect, save_pcd, synthesized
        from ship_perception.manual_review.matcher import match

        pcd = self.root / "replay.pcd"
        truth = rect(np.array([12.0, -22.0]), 0, 18.0, 9.0)
        save_pcd(pcd, synthesized(truth))
        frame = FrameSession(pcd)
        rough = rect(np.array([12.11, -22.12]), 0.015, 17.8, 8.9)
        pixel = frame.projection.uv(rough)
        server_xy = frame.projection.xy(pixel)
        request = {
            **{k: frame.manifest[k] for k in IDENTITY},
            "operation": "CREATE",
            "vertices_uv": pixel.tolist(),
            "client_xy": np.nextafter(server_xy, np.inf).tolist(),
        }
        result = match(frame, server_xy, request)
        self.assertEqual(result["match_state"], "MATCHED_REVIEW")
        evidence = {"request": request, "result": result}
        self.assertGreater(replay(frame, evidence)["raw_rows"], 0)
