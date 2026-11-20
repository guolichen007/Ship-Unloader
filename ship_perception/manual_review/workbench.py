"""Versioned workbench actions, using the R1 transaction and coordinate contracts."""

import csv
from io import StringIO
from threading import BoundedSemaphore, Event, Lock
from time import monotonic, perf_counter
from uuid import uuid4

import numpy as np

from .contract import (
    FLAGS,
    IDENTITY,
    ContractError,
    canonical,
    check_id,
    check_identity,
    digest,
    parse_json,
    selection,
)
from .matcher import anchor_geometry
from .revisions import RevisionStore, now

PARENT = {"annotation_id", "expected_parent_revision"}
SELECT = PARENT | {
    "schema",
    "request_id",
    "operation",
    "target_opening_id",
    "target_vessel_id",
    "vertices_uv",
    "client_xy",
}
MUTATION = PARENT | {"mutation_id", "bundle_digest"}
VIEW = set(IDENTITY)
ROUTES = {
    "/api/v2/auto/confirm": ("AUTO_BASELINE_ACCEPTED", {"geometry_digest"}),
    "/api/v2/auto/void": ("VOID", {"geometry_digest"}),
    "/api/v2/refine/confirm": (
        "GUIDED_REFINED_ACCEPTED",
        {"request_id", "candidate_id", "geometry_digest", "draft_digest"},
    ),
    "/api/v2/manual/commit": ("MANUAL_ONLY", SELECT - PARENT),
    "/api/v2/undo": ("UNDO", set()),
}


def keys(payload, names):
    if set(payload) != VIEW | names:
        raise ContractError("INVALID_SCHEMA")


def append_revision(db, store, record):
    db.execute(
        "INSERT INTO revisions VALUES (?,?,?,?)",
        (store.scope, record["annotation_id"], record["revision"], canonical(record)),
    )


def compact_evidence(result):
    """Reference immutable per-side Raw observations once in the private audit."""
    sides = []
    selected = []
    for index, side in enumerate(result["sides"]):
        rows = {id(row): i for i, row in enumerate(side["observations"])}
        modes = []
        for mode in side["modes"]:
            modes.append(
                dict(
                    {k: v for k, v in mode.items() if k != "train"},
                    train_refs=[rows[id(row)] for row in mode["train"]],
                )
            )
        sides.append(dict(side, modes=modes))
        if result.get("selected_sides"):
            chosen = result["selected_sides"][index]
            selected.append(
                dict(
                    side_id=side["side_id"],
                    train_sha256=chosen["modes"][0]["train_sha256"],
                    heldout_refs=[
                        rows[id(row)] for row in chosen.get("joint_heldout", [])
                    ],
                    joint_reason=chosen.get("joint_reason"),
                    joint_heldout_sha256=chosen.get("joint_heldout_sha256"),
                )
            )
    return dict(
        result,
        sides=sides,
        selected_sides=selected,
        evidence_schema="mhr_raw_observation_refs.2",
    )


class Workbench:
    def __init__(self, frame, output, auto, source):
        self.frame, self.auto, self.source = frame, auto, source
        self.store = RevisionStore(output / "r2-state", frame)
        self.lock = Lock()
        self.slots = BoundedSemaphore(2)
        self.active = {}
        self.cancelled = {}
        with self.store.transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS r2_latest (view TEXT, annotation TEXT, request TEXT,"
                " PRIMARY KEY(view,annotation))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS r2_scene (scope TEXT PRIMARY KEY, marker BLOB)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS r2_scene_log (id TEXT PRIMARY KEY, scope TEXT, marker BLOB)"
            )

    def parent(self, db, payload):
        annotation = check_id(payload["annotation_id"])
        number = payload["expected_parent_revision"]
        current = self.store.head(db, annotation)
        if type(number) is not int or number != (current["revision"] if current else 0):
            raise ContractError("STALE_PARENT", 409)
        return current

    def view_db(self, db, auto):
        revisions = db.execute(
            "SELECT r.body FROM revisions r JOIN (SELECT annotation,max(seq) AS seq"
            " FROM revisions WHERE scope=? GROUP BY annotation) h"
            " ON r.annotation=h.annotation AND r.seq=h.seq WHERE r.scope=?",
            (self.store.scope, self.store.scope),
        ).fetchall()
        heads = {
            r["annotation_id"]: r for r in (parse_json(row[0]) for row in revisions)
        }
        marker = db.execute(
            "SELECT marker FROM r2_scene WHERE scope=?", (self.store.scope,)
        ).fetchone()
        eligible = {
            a: r
            for a, r in heads.items()
            if a not in auto["targets"]
            or r.get("baseline_digest") == auto["targets"][a]["geometry_digest"]
        }
        pending = sorted(
            a
            for a in auto["targets"]
            if a not in eligible
            or eligible[a]["state"]
            not in ("ACCEPTED_AUTO", "ACCEPTED_REFINED", "ACCEPTED_GUIDED_CHOICE", "MANUAL_ONLY", "VOID")
        )
        scene_digest = digest(
            dict(
                bundle=auto["bundle_digest"],
                status=auto["status"],
                heads={k: digest(r) for k, r in heads.items()},
            )
        )
        scene_state = (
            "REVIEW_COMPLETE"
            if marker and parse_json(marker[0])["scene_digest"] == scene_digest
            else "REVIEW_REOPENED" if marker else "REVIEW_PENDING"
        )
        return dict(
            heads=heads,
            pending=pending,
            scene_digest=scene_digest,
            scene_state=scene_state,
        )

    def view(self):
        auto = self.auto.snapshot
        with self.store.transaction() as db:
            state = self.view_db(db, auto)
        return dict(**FLAGS, manifest=self.frame.manifest, auto=auto, **state)

    def read(self, path):
        if path == "/api/v2/view":
            return self.view()
        if path == "/api/v2/revisions":
            with self.store.transaction() as db:
                rows = [
                    parse_json(r[0])
                    for r in db.execute(
                        "SELECT body FROM revisions WHERE scope=? ORDER BY annotation,seq",
                        (self.store.scope,),
                    )
                ]
            return dict(revisions=rows)
        raise ContractError("ROUTE_NOT_FOUND", 404)

    def selection(self, payload, auto):
        return selection(
            {k: payload[k] for k in VIEW | SELECT},
            self.frame.manifest,
            self.frame.projection,
            auto["targets"],
        )

    @staticmethod
    def bind_bundle(payload, auto):
        if payload["bundle_digest"] != auto["bundle_digest"]:
            raise ContractError("STALE_AUTO_BUNDLE", 409)

    def handle(self, path, payload):
        check_identity(payload, self.frame.manifest)
        auto = self.auto.snapshot
        if path == "/api/v2/refine":
            keys(payload, SELECT | {"bundle_digest"})
            self.bind_bundle(payload, auto)
            return self.refine(payload, auto)
        if path == "/api/v2/refine/cancel":
            keys(payload, {"request_id", "annotation_id"})
            return self.cancel(payload)
        if path == "/api/v2/scene/confirm":
            return self.scene_confirm(payload, auto)
        if path == "/api/v2/export":
            keys(payload, {"scene_digest"})
            return self.export(payload, auto)
        if path not in ROUTES:
            raise ContractError("ROUTE_NOT_FOUND", 404)
        action, fields = ROUTES[path]
        keys(payload, MUTATION | fields)
        self.bind_bundle(payload, auto)
        return self.mutate(action, payload, auto)

    def latest(self, db, annotation):
        result = db.execute(
            "SELECT request FROM r2_latest WHERE view=? AND annotation=?",
            (self.frame.manifest["view_id"], annotation),
        ).fetchone()
        return result[0] if result else None

    def cancel(self, payload):
        annotation, request = (
            check_id(payload[k]) for k in ("annotation_id", "request_id")
        )
        with self.lock:
            self.prune_cancellations()
            if (
                len(self.cancelled) >= 512
                and (annotation, request) not in self.cancelled
            ):
                raise ContractError("CANCEL_QUEUE_FULL", 429)
            self.cancelled[annotation, request] = monotonic() + 20
            with self.store.transaction() as db:
                if self.latest(db, annotation) == request:
                    db.execute(
                        "DELETE FROM r2_latest WHERE view=? AND annotation=?",
                        (payload["view_id"], annotation),
                    )
            event = self.active.get((annotation, request))
            if event:
                event.set()
        return dict(execution_state="CANCELLED")

    def prune_cancellations(self):
        current = monotonic()
        self.cancelled = {
            k: expiry for k, expiry in self.cancelled.items() if expiry > current
        }

    def refine(self, payload, auto):
        from .guided import guided_match

        xy = self.selection(payload, auto)
        annotation, request = (payload[k] for k in ("annotation_id", "request_id"))
        key = digest([payload["view_id"], request])
        if not self.slots.acquire(blocking=False):
            raise ContractError("MATCHER_BUSY", 429)
        stop = Event()
        try:
            with self.lock:
                self.prune_cancellations()
                if (annotation, request) in self.cancelled:
                    raise ContractError("STALE_DRAFT", 409)
                with self.store.transaction() as db:
                    self.parent(db, payload)
                    previous = self.store.replay(db, "suggestions", key, payload)
                    if previous:
                        if self.latest(db, annotation) != request:
                            raise ContractError("STALE_DRAFT", 409)
                        return self.preview(previous)
                    if (annotation, request) in self.active:
                        raise ContractError("REQUEST_IN_PROGRESS", 409)
                    db.execute(
                        "INSERT OR REPLACE INTO r2_latest VALUES (?,?,?)",
                        (payload["view_id"], annotation, request),
                    )
                for (current, _), old in self.active.items():
                    if annotation == current:
                        old.set()
                self.active[annotation, request] = stop
            started = perf_counter()
            result = guided_match(
                self.frame, xy, payload, auto["targets"], cancelled=stop.is_set
            )
            result.update(
                **FLAGS,
                view_id=payload["view_id"],
                request_id=request,
                annotation_id=annotation,
                draft_digest=digest(payload["vertices_uv"]),
                candidate_id=uuid4().hex,
                geometry_digest=digest(result.get("geometry")),
                request=payload,
                source=self.source,
            )
            result["latency_s"] = perf_counter() - started
            evidence = canonical(compact_evidence(result))
            if len(evidence) > 24 * 1024**2:
                result.update(
                    geometry=None,
                    execution_state="BUDGET_EXCEEDED",
                    reason="EVIDENCE_SIZE",
                    match_state=None,
                )
                return self.preview(result)
            with self.store.transaction() as db:
                self.parent(db, payload)
                if self.latest(db, annotation) != request or stop.is_set():
                    raise ContractError("STALE_DRAFT", 409)
                db.execute(
                    "INSERT INTO suggestions VALUES (?,?,?)",
                    (key, digest(payload), evidence),
                )
            return self.preview(result)
        finally:
            with self.lock:
                if self.active.get((annotation, request)) is stop:
                    self.active.pop((annotation, request))
            self.slots.release()

    @staticmethod
    def preview(result):
        public = {
            k: v
            for k, v in result.items()
            if k
            not in (
                "sides",
                "selected_sides",
                "request",
                "coverage_audit",
                "separator_evidence",
            )
        }
        public["side_summary"] = [
            {k: s[k] for k in ("side_id", "reason", "mode_count")}
            for s in result.get("sides", [])
        ]
        geometry = public.get("geometry")
        if geometry:
            public["geometry"] = {k: v for k, v in geometry.items() if k != "corner_z"}
        return public

    def mutate(self, action, payload, auto):
        mutation = check_id(payload["mutation_id"])
        annotation = check_id(payload["annotation_id"])
        target = auto["targets"].get(annotation)
        signed = dict(action=action, payload=payload)
        with self.store.transaction() as db:
            replay = self.store.replay(db, "mutations", mutation, signed)
            if replay:
                return replay
            current = self.parent(db, payload)
            number = current["revision"] if current else 0
            geometry = None
            previous = number or None
            provenance = {}
            if action in ("AUTO_BASELINE_ACCEPTED", "VOID"):
                if (
                    not target
                    or payload["geometry_digest"] != target["geometry_digest"]
                ):
                    raise ContractError("STALE_AUTO_GEOMETRY", 409)
                geometry = (
                    target["geometry"] if action == "AUTO_BASELINE_ACCEPTED" else None
                )
                provenance = {
                    k: target[k]
                    for k in ("state", "grade", "adopted_for_review", "review_gates")
                }
                provenance["native"] = auto["provenance"]
                state = "ACCEPTED_AUTO" if geometry else "VOID"
            elif action == "UNDO":
                if not current:
                    raise ContractError("NO_REVISION", 409)
                old = (
                    self.store.at(db, annotation, current["restore_pointer"])
                    if current["restore_pointer"]
                    else None
                )
                geometry = old["geometry"] if old else None
                state = old["state"] if old else "AUTO_PENDING"
                provenance = old["provenance"] if old else {}
                previous = old["restore_pointer"] if old else None
            elif action == "MANUAL_ONLY":
                xy = self.selection(payload, auto)
                geometry = anchor_geometry(xy)
                geometry.update(
                    xyz=[[x, y, None] for x, y in geometry["xy"]],
                    xy_complete=True,
                    xyz_complete=False,
                )
                state = "MANUAL_ONLY"
                provenance = dict(draft_digest=digest(payload["vertices_uv"]))
            else:
                request = check_id(payload["request_id"])
                if self.latest(db, annotation) != request:
                    raise ContractError("STALE_DRAFT", 409)
                row = db.execute(
                    "SELECT body FROM suggestions WHERE id=?",
                    (digest([payload["view_id"], request]),),
                ).fetchone()
                candidate = parse_json(row[0]) if row else None
                if (
                    not candidate
                    or candidate["match_state"] != "GUIDED_REFINED_PENDING"
                ):
                    raise ContractError("CANDIDATE_NOT_MATCHED", 409)
                if (
                    candidate["annotation_id"] != annotation
                    or candidate["request"]["expected_parent_revision"] != number
                    or candidate["request"]["bundle_digest"] != auto["bundle_digest"]
                    or any(
                        payload[k] != candidate[k]
                        for k in ("candidate_id", "draft_digest", "geometry_digest")
                    )
                ):
                    raise ContractError("SUGGESTION_BINDING", 409)
                geometry = candidate["geometry"]
                state = "ACCEPTED_REFINED"
                provenance = dict(
                    candidate_id=candidate["candidate_id"],
                    evidence_sha256=digest(candidate),
                    ownership=candidate["ownership"],
                )
            record = dict(
                **FLAGS,
                schema="mhr_workbench_revision.2",
                source=self.source,
                action=action,
                annotation_id=annotation,
                parent=number,
                revision=number + 1,
                restore_pointer=previous,
                state=state,
                geometry=geometry,
                provenance=provenance,
                time=now(),
                baseline_digest=target["geometry_digest"] if target else None,
                human_review="OPERATOR_ACTION_RECORDED",
                frame={k: self.frame.manifest[k] for k in IDENTITY},
            )
            if action == "UNDO" and old:
                record["baseline_digest"] = old["baseline_digest"]
            append_revision(db, self.store, record)
            db.execute(
                "INSERT INTO mutations VALUES (?,?,?)",
                (mutation, digest(signed), canonical(record)),
            )
            db.execute(
                "DELETE FROM r2_latest WHERE view=? AND annotation=?",
                (payload["view_id"], annotation),
            )
        return record

    def scene_confirm(self, payload, auto):
        keys(payload, {"mutation_id", "scene_digest", "confirm_empty"})
        if type(payload["confirm_empty"]) is not bool:
            raise ContractError("INVALID_CONFIRM_EMPTY")
        mutation = check_id(payload["mutation_id"])
        signed = dict(action="SCENE_CONFIRM", payload=payload)
        with self.store.transaction() as db:
            replay = self.store.replay(db, "mutations", mutation, signed)
            if replay:
                return replay
            state = self.view_db(db, auto)
            if state["scene_digest"] != payload["scene_digest"] or state["pending"]:
                raise ContractError("SCENE_CHANGED_OR_INCOMPLETE", 409)
            if auto["status"] in ("RUNNING", "LOADING"):
                raise ContractError("AUTO_RUNNING", 409)
            if (
                not any(r["geometry"] for r in state["heads"].values())
                and payload["confirm_empty"] is not True
            ):
                raise ContractError("EXPLICIT_EMPTY_REQUIRED", 409)
            marker = dict(
                **FLAGS,
                scene_digest=state["scene_digest"],
                time=now(),
                scene_state="REVIEW_COMPLETE",
            )
            body = canonical(marker)
            db.execute(
                "INSERT OR REPLACE INTO r2_scene VALUES (?,?)", (self.store.scope, body)
            )
            db.execute(
                "INSERT INTO r2_scene_log VALUES (?,?,?)",
                (mutation, self.store.scope, body),
            )
            db.execute(
                "INSERT INTO mutations VALUES (?,?,?)", (mutation, digest(signed), body)
            )
        return marker

    def export(self, payload, auto):
        if auto is not self.auto.snapshot:
            raise ContractError("STALE_AUTO_BUNDLE", 409)
        with self.store.transaction() as db:
            state = self.view_db(db, auto)
        if payload["scene_digest"] != state["scene_digest"]:
            raise ContractError("STALE_SCENE", 409)
        accepted = [
            r
            for r in state["heads"].values()
            if r["geometry"] is not None and r["annotation_id"] not in state["pending"]
        ]
        void = [
            r
            for r in state["heads"].values()
            if r["state"] == "VOID" and r["annotation_id"] not in state["pending"]
        ]
        stream = StringIO()
        writer = csv.writer(stream)
        writer.writerow(
            (
                "target_id",
                "source",
                "corner",
                "x_m",
                "y_m",
                "z_m",
                "length_m",
                "width_m",
                "steel_certified",
                "control_eligible",
            )
        )
        for r in accepted:
            g = r["geometry"]
            for corner, (x, y, z) in zip(g["corner_ids"], g["xyz"]):
                writer.writerow(
                    (
                        r["annotation_id"],
                        r["state"],
                        corner,
                        x,
                        y,
                        z,
                        g["length_m"],
                        g["width_m"],
                        "false",
                        "false",
                    )
                )
        return dict(
            **FLAGS,
            schema="mhr_workbench_export.2",
            scene_state=state["scene_state"],
            scene_digest=state["scene_digest"],
            wire_sha256=self.frame.manifest["wire_sha256"],
            accepted=accepted,
            rejected=void,
            pending=state["pending"],
            csv=stream.getvalue(),
        )
