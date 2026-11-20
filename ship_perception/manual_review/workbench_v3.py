"""R3 guided review actions. All saved coordinates come from server candidates."""

from itertools import product
from time import monotonic, perf_counter
from threading import Event
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
    rectangle_axes,
)
from .guided_v3 import SIDES, guided_match
from .revisions import now
from .workbench import MUTATION, PARENT, SELECT, Workbench, append_revision, keys

REFINE = "/api/v3/refine"
BIND = PARENT | {"bundle_digest", "request_id", "draft_digest", "list_digest"}
CHOICE = BIND | {"candidate_id", "geometry_digest"}
TTL_S = 600
MAX_EVIDENCE = 24 * 1024**2


def corner_order(xy, target):
    if target is None:
        return xy, dict(
            request_vertices=[0, 1, 2, 3], definition="REQUEST_VERTEX_ORDER"
        )
    reference = np.asarray(target["geometry"]["xy"], dtype=np.float64)
    a, b = xy - xy.mean(axis=0), reference - reference.mean(axis=0)
    a, b = (
        a / np.linalg.norm(a, axis=1)[:, None],
        b / np.linalg.norm(b, axis=1)[:, None],
    )
    permutations = [
        [(start + i * sign) % 4 for i in range(4)]
        for start, sign in product(range(4), (1, -1))
    ]
    order = sorted((float(np.sum((a[p] - b) ** 2)), p) for p in permutations)
    if order[1][0] - order[0][0] < 0.05:
        raise ContractError("CORNER_ANCHOR_AMBIGUOUS", 409)
    p = order[0][1]
    return xy[p], dict(request_vertices=p, definition="AUTO_P1_TO_P4")


def public_geometry(geometry):
    if geometry is None:
        return None
    return dict(
        geometry,
        corner_z=[
            {k: v for k, v in corner.items() if k != "raw_indexes"}
            for corner in geometry["corner_z"]
        ],
    )


class WorkbenchV3(Workbench):
    def __init__(self, frame, output, auto, identity):
        super().__init__(frame, output, auto, identity.get("build_sha") or "UNVERIFIED")
        self.source_identity = identity
        with self.store.transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS r3_drafts (id TEXT PRIMARY KEY, "
                "payload_digest TEXT, body BLOB, expires REAL, selected BLOB)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS r3_expiry ON r3_drafts(expires)")

    def read(self, path):
        if path == "/api/v3/view":
            return dict(
                self.view(), source_identity=self.source_identity, protocol="mhr.3"
            )
        return super().read(path)

    def handle(self, path, payload):
        if (
            path == "/api/v2/scene/confirm"
            and self.source_identity["status"] != "VERIFIED"
        ):
            check_identity(payload, self.frame.manifest)
            raise ContractError("UNVERIFIED_SOURCE", 409)
        if path != REFINE and not path.startswith(REFINE + "/"):
            return super().handle(path, payload)
        check_identity(payload, self.frame.manifest)
        auto = self.auto.snapshot
        if path == REFINE + "/cancel":
            keys(payload, {"request_id", "annotation_id"})
            return super().cancel(dict(payload, request_id=self.slot(payload)))
        if path == REFINE:
            extra = {"edge_hint"} if "edge_hint" in payload else set()
            keys(payload, SELECT | {"bundle_digest"} | extra)
            self.bind_bundle(payload, auto)
            return self.solve(payload, auto)
        if path == REFINE + "/profile":
            keys(payload, BIND | {"side_id"})
            self.bind_bundle(payload, auto)
            return self.profile(payload)
        if path == REFINE + "/select":
            keys(payload, CHOICE)
            self.bind_bundle(payload, auto)
            return self.select_candidate(payload)
        if path == REFINE + "/confirm":
            keys(payload, CHOICE | MUTATION | {"selection_id"})
            self.bind_bundle(payload, auto)
            return self.confirm(payload, auto)
        raise ContractError("ROUTE_NOT_FOUND", 404)

    @staticmethod
    def slot(payload):
        return "r3:" + digest(check_id(payload["request_id"]))[:60]

    @staticmethod
    def record_id(payload):
        return digest([payload["view_id"], check_id(payload["request_id"]), "mhr.3"])

    def stored(self, db, payload, *, binding=True):
        self.parent(db, payload)
        if self.latest(db, payload["annotation_id"]) != self.slot(payload):
            raise ContractError("STALE_DRAFT", 409)
        row = db.execute(
            "SELECT body,expires,selected FROM r3_drafts WHERE id=?",
            (self.record_id(payload),),
        ).fetchone()
        if not row or row[1] <= monotonic():
            raise ContractError("EXPIRED_OR_MISSING_DRAFT", 409)
        result = parse_json(row[0])
        if not result:
            raise ContractError("REQUEST_IN_PROGRESS", 409)
        signed = result["request"]
        if (
            any(
                payload[k] != signed[k]
                for k in IDENTITY
                + ("annotation_id", "bundle_digest", "expected_parent_revision")
            )
            or binding
            and any(payload[k] != result[k] for k in ("draft_digest", "list_digest"))
        ):
            raise ContractError("DRAFT_BINDING", 409)
        return result, parse_json(row[2]) if row[2] else None

    def hint(self, db, payload, xy):
        hint = payload.get("edge_hint")
        if hint is None and "edge_hint" not in payload:
            return None
        if not isinstance(hint, dict) or set(hint) != {"request_id", "side_id", "uv"}:
            raise ContractError("INVALID_EDGE_HINT")
        old, _ = self.stored(
            db, dict(payload, request_id=hint["request_id"]), binding=False
        )
        if (
            old["match_state"]
            not in (
                "PARTIAL",
                "AMBIGUOUS",
                "GUIDED_CHOICES_PENDING",
                "SEARCH_INCOMPLETE",
            )
            or "edge_hint" in old["request"]
            or any(payload[k] != old["request"][k] for k in SELECT - {"request_id"})
        ):
            raise ContractError("EDGE_HINT_NOT_AVAILABLE", 409)
        uv = hint["uv"]
        if (
            hint["side_id"] not in SIDES
            or not isinstance(uv, list)
            or len(uv) != 2
            or any(type(n) not in (float, int) for n in uv)
            or any(
                not 0 <= v <= limit
                for v, limit in zip(
                    uv, (self.frame.projection.nx, self.frame.projection.ny)
                )
            )
        ):
            raise ContractError("INVALID_EDGE_HINT")
        i = SIDES.index(hint["side_id"])
        axes, lengths, orientation = rectangle_axes(xy)
        t = axes[i]
        n = orientation * np.array([-t[1], t[0]])
        relative = self.frame.projection.xy(uv) - xy[i]
        station, offset = float(relative @ t), float(relative @ n)
        widths = old.get("search_half_width_m", [])
        if (
            len(widths) != 4
            or not 0.25 * lengths[i] <= station <= 0.75 * lengths[i]
            or abs(offset) > widths[i]
        ):
            raise ContractError("EDGE_HINT_OUTSIDE_WINDOW")
        return (hint["side_id"], station, offset)

    def solve(self, payload, auto):
        if auto["status"] in ("LOADING", "RUNNING"):
            raise ContractError("AUTO_RUNNING", 409)
        xy = self.selection(payload, auto)
        xy, mapping = corner_order(
            xy, auto["targets"].get(payload["target_opening_id"])
        )
        annotation = payload["annotation_id"]
        key, slot = self.record_id(payload), self.slot(payload)
        if not self.slots.acquire(blocking=False):
            raise ContractError("MATCHER_BUSY", 429)
        stop = Event()
        active = (annotation, slot)
        try:
            with self.lock:
                self.prune_cancellations()
                if active in self.cancelled:
                    raise ContractError("STALE_DRAFT", 409)
                with self.store.transaction() as db:
                    self.parent(db, payload)
                    row = db.execute(
                        "SELECT payload_digest FROM r3_drafts WHERE id=?", (key,)
                    ).fetchone()
                    if row:
                        if row[0] != digest(payload):
                            raise ContractError("REQUEST_ID_REUSED", 409)
                        result, selected = self.stored(db, payload, binding=False)
                        return self.preview_v3(result, selected)
                    if active in self.active:
                        raise ContractError("REQUEST_IN_PROGRESS", 409)
                    hint = self.hint(db, payload, xy)
                    db.execute(
                        "INSERT OR REPLACE INTO r2_latest VALUES (?,?,?)",
                        (payload["view_id"], annotation, slot),
                    )
                    # Cap storage growth. Keep spent IDs until expiry, including cancelled requests.
                    db.execute(
                        "DELETE FROM r3_drafts WHERE expires<?", (monotonic() - TTL_S,)
                    )
                    if (
                        db.execute("SELECT count(*) FROM r3_drafts").fetchone()[0]
                        >= 256
                    ):
                        raise ContractError("DRAFT_STORAGE_LIMIT", 429)
                    db.execute(
                        "INSERT INTO r3_drafts VALUES(?,?,?,?,NULL)",
                        (key, digest(payload), b"{}", monotonic() + TTL_S),
                    )
                for (current, _), event in self.active.items():
                    if current == annotation:
                        event.set()
                self.active[active] = stop
            result = guided_match(
                self.frame, xy, payload, auto["targets"], stop.is_set, edge_hint=hint
            )
            for candidate in result["candidates"]:
                geometry = candidate["geometry"]
                geometry["corner_mapping"] = mapping
                candidate["geometry_digest"] = digest(geometry)
            result.update(
                request=payload,
                request_id=payload["request_id"],
                view_id=payload["view_id"],
                annotation_id=annotation,
                source=self.source_identity,
                corner_mapping=mapping,
                draft_digest=digest(payload["vertices_uv"]),
                list_digest=digest(
                    [
                        (c["candidate_id"], c["geometry_digest"], c["confirmable"])
                        for c in result["candidates"]
                    ]
                ),
            )
            selected = None
            if result["match_state"] == "GUIDED_REFINED_PENDING":
                selected = dict(
                    id=result["candidates"][0]["candidate_id"], selection_id=uuid4().hex
                )
            evidence = canonical(result)
            if len(evidence) > MAX_EVIDENCE:
                result.update(
                    geometry=None,
                    candidates=[],
                    attempted=[],
                    sides=[],
                    recommendation_id=None,
                    match_state="SEARCH_INCOMPLETE",
                    search_complete=False,
                    execution_state="BUDGET_EXCEEDED",
                    reason="EVIDENCE_SIZE",
                    list_digest=digest([]),
                )
                evidence, selected = canonical(result), None
            with self.store.transaction() as db:
                self.parent(db, payload)
                if (
                    self.latest(db, annotation) != slot
                    or stop.is_set()
                    or self.auto.snapshot is not auto
                ):
                    raise ContractError("STALE_DRAFT", 409)
                db.execute(
                    "UPDATE r3_drafts SET body=?,selected=? WHERE id=?",
                    (evidence, canonical(selected) if selected else None, key),
                )
            return self.preview_v3(result, selected)
        finally:
            with self.lock:
                if self.active.get(active) is stop:
                    self.active.pop(active)
            self.slots.release()

    @staticmethod
    def preview_v3(r, selected=None):
        fields = (
            "match_state",
            "execution_state",
            "reason",
            "ownership",
            "view_id",
            "request_id",
            "annotation_id",
            "draft_digest",
            "list_digest",
            "recommendation_id",
            "search_complete",
            "elapsed_s",
            "corner_mapping",
        )
        candidates = [
            dict(c, geometry=public_geometry(c["geometry"])) for c in r["candidates"]
        ]
        choice = next(
            (c for c in candidates if selected and c["candidate_id"] == selected["id"]),
            None,
        )
        best = choice or next(
            (c for c in candidates if c["candidate_id"] == r["recommendation_id"]), None
        )
        return dict(
            **FLAGS,
            **{k: r[k] for k in fields},
            candidates=candidates,
            selection_id=selected["selection_id"] if choice else None,
            candidate_id=choice["candidate_id"] if choice else None,
            geometry_digest=choice["geometry_digest"] if choice else None,
            geometry=best["geometry"] if best else None,
            side_summary=[
                {k: s[k] for k in ("side_id", "mode_count", "reason")}
                for s in r["sides"]
            ]
        )

    @staticmethod
    def bound_candidate(result, payload):
        candidate = next(
            (
                c
                for c in result["candidates"]
                if c["candidate_id"] == check_id(payload["candidate_id"])
            ),
            None,
        )
        if not (
            result["search_complete"]
            and candidate
            and candidate["confirmable"]
            and candidate["geometry_digest"] == payload["geometry_digest"]
            and digest(candidate["geometry"]) == candidate["geometry_digest"]
        ):
            raise ContractError("CANDIDATE_BINDING", 409)
        return candidate

    def select_candidate(self, payload):
        with self.store.transaction() as db:
            result, _ = self.stored(db, payload)
            if result["match_state"] != "GUIDED_CHOICES_PENDING":
                raise ContractError("CHOICES_NOT_AVAILABLE", 409)
            candidate = self.bound_candidate(result, payload)
            selected = dict(id=candidate["candidate_id"], selection_id=uuid4().hex)
            db.execute(
                "UPDATE r3_drafts SET selected=? WHERE id=?",
                (canonical(selected), self.record_id(payload)),
            )
            return self.preview_v3(result, selected)

    def profile(self, payload):
        side = payload["side_id"]
        if side not in SIDES:
            raise ContractError("INVALID_SIDE")
        with self.store.transaction() as db:
            result, _ = self.stored(db, payload)
        if len(result["sides"]) != 4:
            raise ContractError("SIDE_PROFILE_UNAVAILABLE", 409)
        i = SIDES.index(side)
        xy = np.asarray(result["rough_xy"])
        axes, lengths, orientation = rectangle_axes(xy)
        t = axes[i]
        n = orientation * np.array([-t[1], t[0]])
        sample_a, sample_b = (
            xy[i] + 0.25 * lengths[i] * t,
            xy[i] + 0.75 * lengths[i] * t,
        )
        slab_data = result["sides"][i]
        middle = [
            o
            for o in slab_data["observations"]
            if o["slab"] % 2 == 0 and abs(o["t_m"] - lengths[i] / 2) < lengths[i] * 0.25
        ]
        indices = sorted(
            {
                j
                for o in middle
                for field in ("face", "cap", "interior", "competing_interior")
                for j in o[field]
            }
        )
        step = max(1, (len(indices) + 799) // 800)
        points = self.frame.frame.xyz[indices[::step]].astype(np.float64)
        return dict(
            **FLAGS,
            side_id=side,
            frame_view=payload["view_id"],
            request_id=payload["request_id"],
            list_digest=payload["list_digest"],
            sample_kind="TRAIN_OBSERVATIONS",
            offset_window_m=result["search_half_width_m"][i],
            station_line_uv=self.frame.projection.uv([sample_a, sample_b]).tolist(),
            midpoint_uv=self.frame.projection.uv([(sample_a + sample_b) / 2])[
                0
            ].tolist(),
            normal_uv=[
                float(n[0] / self.frame.projection.s),
                float(-n[1] / self.frame.projection.s),
            ],
            points=np.column_stack(((points[:, :2] - xy[i]) @ n, points[:, 2])).tolist()
        )

    def confirm(self, payload, auto):
        mutation = check_id(payload["mutation_id"])
        signed = dict(action="CONFIRM_GUIDED_V3", payload=payload)
        annotation = payload["annotation_id"]
        with self.store.transaction() as db:
            replay = self.store.replay(db, "mutations", mutation, signed)
            if replay:
                return replay
            if self.source_identity["status"] != "VERIFIED":
                raise ContractError("UNVERIFIED_SOURCE", 409)
            current = self.parent(db, payload)
            result, selected = self.stored(db, payload)
            candidate = self.bound_candidate(result, payload)
            if (
                not selected
                or selected["id"] != candidate["candidate_id"]
                or selected["selection_id"] != payload["selection_id"]
            ):
                raise ContractError("SELECTION_BINDING", 409)
            if self.auto.snapshot is not auto:
                raise ContractError("STALE_AUTO_BUNDLE", 409)
            number = payload["expected_parent_revision"]
            target = auto["targets"].get(annotation)
            choice = result["match_state"] == "GUIDED_CHOICES_PENDING"
            record = dict(
                **FLAGS,
                schema="mhr_workbench_revision.3",
                source=self.source,
                action="OPERATOR_SELECTED_RAW" if choice else "GUIDED_REFINED_ACCEPTED",
                annotation_id=annotation,
                parent=number,
                revision=number + 1,
                restore_pointer=number if current else None,
                state="ACCEPTED_GUIDED_CHOICE" if choice else "ACCEPTED_REFINED",
                geometry=candidate["geometry"],
                provenance=dict(
                    candidate_id=candidate["candidate_id"],
                    ownership=result["ownership"],
                    evidence_sha256=digest(result),
                    source_identity=self.source_identity,
                ),
                time=now(),
                baseline_digest=target["geometry_digest"] if target else None,
                human_review="OPERATOR_ACTION_RECORDED",
                frame={k: self.frame.manifest[k] for k in IDENTITY}
            )
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
