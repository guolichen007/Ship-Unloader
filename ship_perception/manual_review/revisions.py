"""Transactional private evidence and revision log using the standard SQLite library."""

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from threading import Event, Lock
from time import monotonic
from uuid import uuid4
import hashlib
import platform
import sqlite3
import scipy

import numpy as np

from .contract import (
    FLAGS,
    IDENTITY,
    VERSION,
    ContractError,
    canonical,
    check_id,
    check_identity,
    digest,
    parse_json,
    selection,
)
from .matcher import anchor_geometry, match


def now():
    return datetime.now(timezone.utc).isoformat()


class RevisionStore:
    def __init__(self, root, frame):
        root = Path(root).resolve() / "mhr_private"
        if root.is_symlink():
            raise ContractError("INVALID_STORE_PATH")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = root / "reviews.sqlite3"
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ContractError("INVALID_STORE_PATH")
        self.path = path
        self.scope = digest(
            {
                k: frame.manifest[k]
                for k in (
                    "wire_sha256",
                    "semantic_sha256",
                    "frame_id",
                    "overlay_status",
                    "config",
                    "matching_config",
                    "version",
                )
            }
            | {"overlay_sha256": frame.manifest.get("overlay_sha256")}
        )
        with self.transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS suggestions (id TEXT PRIMARY KEY, digest TEXT, body BLOB)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS revisions (scope TEXT, annotation TEXT, seq INTEGER, body BLOB,"
                " PRIMARY KEY(scope,annotation,seq))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS mutations (id TEXT PRIMARY KEY, digest TEXT, body BLOB)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS scopes (id TEXT PRIMARY KEY, binding BLOB)"
            )
            db.execute(
                "INSERT OR IGNORE INTO scopes VALUES (?,?)",
                (self.scope, canonical(frame.manifest)),
            )

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(str(self.path), timeout=2.0)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA max_page_count=131072")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except sqlite3.Error as exc:
            db.rollback()
            raise ContractError("LOCAL_STATE_IO_ERROR", 500) from exc
        finally:
            db.close()

    def heads(self):
        with self.transaction() as db:
            return dict(
                db.execute(
                    "SELECT annotation,max(seq) FROM revisions WHERE scope=? GROUP BY annotation",
                    (self.scope,),
                ).fetchall()
            )

    def head(self, db, annotation):
        row = db.execute(
            "SELECT body FROM revisions WHERE scope=? AND annotation=? ORDER BY seq DESC LIMIT 1",
            (self.scope, annotation),
        ).fetchone()
        return parse_json(row[0]) if row else None

    def at(self, db, annotation, revision):
        row = db.execute(
            "SELECT body FROM revisions WHERE scope=? AND annotation=? AND seq=?",
            (self.scope, annotation, revision),
        ).fetchone()
        return parse_json(row[0]) if row else None

    @staticmethod
    def replay(db, table, identifier, payload):
        query = (
            "SELECT digest,body FROM suggestions WHERE id=?"
            if table == "suggestions"
            else "SELECT digest,body FROM mutations WHERE id=?"
        )
        row = db.execute(query, (identifier,)).fetchone()
        if row:
            if digest(payload) != row[0]:
                raise ContractError("ID_PAYLOAD_CONFLICT", 409)
            return parse_json(row[1])
        return None


class Review:
    def __init__(self, frame, output, source=None):
        self.frame = frame
        self.source = source or VERSION
        self.store = RevisionStore(output, frame)
        self.lock = Lock()
        self.active = {}
        self.cancelled = {}

    def view(self):
        heads = self.store.heads()
        with self.store.transaction() as db:
            active = {k: self.store.at(db, k, v)["geometry"] for k, v in heads.items()}
        return {"review_enabled": True, "heads": heads, "saved_geometry": active}

    def verify_overlay(self):
        if self.frame.overlay_path is not None:
            with self.frame.overlay_path.open("rb") as stream:
                data = stream.read(1024**2 + 1)
            if (
                hashlib.sha256(data).hexdigest()
                != self.frame.manifest["overlay_sha256"]
            ):
                raise ContractError("OVERLAY_CHANGED", 409)

    def handle(self, route, payload):
        check_identity(payload, self.frame.manifest)
        if route == "/api/match":
            return self.suggest(payload)
        if route == "/api/cancel":
            if set(payload) != set(IDENTITY) | {"request_id"}:
                raise ContractError("INVALID_CANCEL")
            request = check_id(payload.get("request_id"))
            with self.lock:
                self.prune_cancelled()
                pending = self.active.get(request)
                if len(self.cancelled) >= 512 and request not in self.cancelled:
                    raise ContractError("CANCEL_QUEUE_FULL", 429)
                self.cancelled[request] = monotonic() + 20
                if pending is not None:
                    pending.set()
            return {
                "execution_state": "CANCEL_PENDING" if pending else "NO_ACTIVE_REQUEST"
            }
        if route in ("/api/accept", "/api/manual", "/api/undo", "/api/export"):
            return self.mutate(route[5:], payload)
        raise ContractError("ROUTE_NOT_FOUND", 404)

    def prune_cancelled(self):
        instant = monotonic()
        self.cancelled = {k: v for k, v in self.cancelled.items() if v > instant}

    def suggest(self, payload):
        xy = selection(
            payload, self.frame.manifest, self.frame.projection, self.frame.targets
        )
        request = payload["request_id"]
        annotation = payload["annotation_id"]
        self.verify_overlay()
        with self.lock:
            self.prune_cancelled()
            if request in self.cancelled:
                return {
                    **FLAGS,
                    "execution_state": "CANCELLED",
                    "match_state": None,
                    "request_id": request,
                    "view_id": payload["view_id"],
                    "rough_xy": xy.tolist(),
                    "geometry": None,
                    "reason": "CANCELLED",
                }
            if self.active:
                raise ContractError("MATCHER_BUSY", 429)
            with self.store.transaction() as db:
                saved = self.store.replay(db, "suggestions", request, payload)
                if saved is not None:
                    return public_result(saved)
                head = self.store.head(db, annotation)
                current = head["revision"] if head else 0
                if current != payload["expected_parent_revision"]:
                    raise ContractError("STALE_PARENT", 409)
            stop = Event()
            self.active[request] = stop
        try:
            result = match(self.frame, xy, payload, cancelled=stop.is_set)
            rough = anchor_geometry(xy)
            result.update(
                request_id=request,
                view_id=self.frame.manifest["view_id"],
                request=payload,
                source_version=self.source,
                created_at=now(),
                environment=dict(
                    python=platform.python_version(),
                    numpy=np.__version__,
                    scipy=scipy.__version__,
                    system=platform.system(),
                ),
                rough_digest=digest(rough),
                geometry_digest=digest(result["geometry"]),
                candidate_id=uuid4().hex,
                suggestion_id=uuid4().hex,
            )
            with self.lock:
                if stop.is_set() or result["execution_state"] != "COMPLETED":
                    return {
                        **FLAGS,
                        "request_id": request,
                        "view_id": payload["view_id"],
                        "rough_xy": xy.tolist(),
                        "execution_state": (
                            "CANCELLED" if stop.is_set() else result["execution_state"]
                        ),
                        "match_state": None,
                        "geometry": None,
                        "reason": result["reason"],
                    }
                evidence = canonical(result)
                if len(evidence) > 24 * 1024**2:
                    return {
                        **FLAGS,
                        "request_id": request,
                        "view_id": payload["view_id"],
                        "rough_xy": xy.tolist(),
                        "execution_state": "BUDGET_EXCEEDED",
                        "match_state": None,
                        "geometry": None,
                        "reason": "EVIDENCE_SIZE",
                    }
                with self.store.transaction() as db:
                    head = self.store.head(db, annotation)
                    if (head["revision"] if head else 0) != current:
                        raise ContractError("STALE_PARENT", 409)
                    db.execute(
                        "INSERT INTO suggestions VALUES (?,?,?)",
                        (request, digest(payload), evidence),
                    )
            return public_result(result)
        finally:
            with self.lock:
                self.active.pop(request, None)

    def mutate(self, action, payload):
        keys = set(IDENTITY) | {
            "mutation_id",
            "annotation_id",
            "expected_parent_revision",
        }
        if action in ("accept", "manual"):
            keys |= {"request_id", "suggestion_id", "candidate_id", "geometry_digest"}
        if set(payload) != keys:
            raise ContractError("INVALID_REVISION_SCHEMA")
        mid = check_id(payload["mutation_id"])
        annotation = check_id(payload["annotation_id"])
        parent = payload["expected_parent_revision"]
        if type(parent) is not int or parent < 0:
            raise ContractError("INVALID_PARENT")
        self.verify_overlay()
        signed = {"action": action, "payload": payload}
        with self.store.transaction() as db:
            previously = self.store.replay(db, "mutations", mid, signed)
            if previously is not None:
                return previously
            head = self.store.head(db, annotation)
            current = head["revision"] if head else 0
            if current != parent:
                raise ContractError("STALE_PARENT", 409)
            if action == "export":
                if not head:
                    raise ContractError("NO_REVISION", 409)
                return export(head)
            if action == "undo":
                if not head:
                    raise ContractError("NO_REVISION", 409)
                revert = head["restore_pointer"]
                old = self.store.at(db, annotation, revert) if revert else None
                content = old["content"] if old else None
                restore_pointer = old["restore_pointer"] if old else None
            else:
                body = db.execute(
                    "SELECT body FROM suggestions WHERE id=?",
                    (check_id(payload["request_id"]),),
                ).fetchone()
                if not body:
                    raise ContractError("UNKNOWN_SUGGESTION", 409)
                candidate = parse_json(body[0])
                check_identity(candidate["request"], self.frame.manifest)
                if (
                    candidate["request"]["annotation_id"] != annotation
                    or candidate["request"]["expected_parent_revision"] != parent
                    or any(
                        payload[k] != candidate[k]
                        for k in ("suggestion_id", "candidate_id")
                    )
                ):
                    raise ContractError("SUGGESTION_BINDING", 409)
                stored_digest = (
                    "geometry_digest" if action == "accept" else "rough_digest"
                )
                if payload["geometry_digest"] != candidate[stored_digest]:
                    raise ContractError("GEOMETRY_DIGEST_MISMATCH", 409)
                if action == "accept" and candidate["match_state"] != "MATCHED_REVIEW":
                    raise ContractError("CANDIDATE_NOT_MATCHED", 409)
                geometry = (
                    candidate["geometry"]
                    if action == "accept"
                    else anchor_geometry(np.array(candidate["rough_xy"]))
                )
                content = dict(
                    **FLAGS,
                    geometry=geometry,
                    geometry_complete=True,
                    source_kind="RAW_REFINED" if action == "accept" else "MANUAL_ONLY",
                    ownership=candidate["ownership"],
                    match_state=candidate["match_state"],
                    human_review="ACCEPTED",
                    suggestion_id=candidate["suggestion_id"],
                    request_id=candidate["request_id"],
                    candidate_id=candidate["candidate_id"],
                    evidence_sha256=digest(candidate),
                )
                restore_pointer = current or None
            revision = dict(
                **FLAGS,
                schema="manual_revision.1",
                revision=current + 1,
                parent=current,
                restore_pointer=restore_pointer,
                annotation_id=annotation,
                action=action.upper(),
                content=content,
                geometry=None if content is None else content["geometry"],
                frame={k: self.frame.manifest[k] for k in IDENTITY},
                source_version=self.source,
                operator="LOCAL_SESSION_OPERATOR_UNVERIFIED",
                created_at=now(),
            )
            db.execute(
                "INSERT INTO revisions VALUES (?,?,?,?)",
                (self.store.scope, annotation, current + 1, canonical(revision)),
            )
            db.execute(
                "INSERT INTO mutations VALUES (?,?,?)",
                (mid, digest(signed), canonical(revision)),
            )
            return revision


def public_result(result):
    visible = {
        k: v for k, v in result.items() if k not in ("separators", "sides", "request")
    }
    visible["sides"] = []
    for side in result["sides"]:
        brief = {
            k: v
            for k, v in side.items()
            if k not in ("modes", "observations", "joint_heldout")
        }
        brief["modes"] = [
            {
                **{k: v for k, v in mode.items() if k not in ("train", "heldout")},
                **{
                    k: [
                        {
                            f: r[f]
                            for f in (
                                "t_m",
                                "d_m",
                                "top_m",
                                "relief_m",
                                "unique_face_count",
                            )
                        }
                        for r in mode[k]
                    ]
                    for k in ("train", "heldout")
                },
            }
            for mode in side["modes"]
        ]
        visible["sides"].append(brief)
    visible["separator_count"] = len(result["separators"])
    return visible


def export(revision):
    stream = StringIO(newline="")
    out = csv.writer(stream)
    out.writerow(
        (
            "角点",
            "X(米)",
            "Y(米)",
            "Z(米)",
            "水平长(米)",
            "水平宽(米)",
            "帧",
            "语义SHA256",
            "钢边认证",
            "设备控制资格",
        )
    )
    g = revision["geometry"]
    if g is not None:
        frame_name = revision["frame"]["frame_id"]
        if frame_name.lstrip().startswith(("=", "+", "-", "@")):
            frame_name = "'" + frame_name
        for identifier, (x, y), z in zip(g["corner_ids"], g["xy"], g["z"]):
            out.writerow(
                (
                    identifier,
                    x,
                    y,
                    z,
                    g["length_m"],
                    g["width_m"],
                    frame_name,
                    revision["frame"]["semantic_sha256"],
                    "false",
                    "false",
                )
            )
    return {"json": canonical(revision).decode("utf-8"), "csv": stream.getvalue()}
