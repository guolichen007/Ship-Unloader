"""Canonical JSON, frame identity and selection validation."""

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

FLAGS = {"steel_certified": False, "control_eligible": False}
RAW_ROWS = "decoded_finite_xyz_row_v1"
VERSION = "mhr-r1.1"
IDENTITY = (
    "session_id",
    "frame_instance_id",
    "view_id",
    "manifest_sha256",
    "wire_sha256",
    "semantic_sha256",
    "frame_id",
)
ID = re.compile(r"[A-Za-z0-9_:-]{1,64}")
CLIENT_XY_TOLERANCE_M = 2e-7
MIN_EDGE_M = 1.0
MAX_EDGE_M = 250.0


class ContractError(ValueError):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.status = status


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def parse_json(data):
    def pairs(items):
        obj = dict(items)
        if len(obj) != len(items):
            raise ContractError("DUPLICATE_JSON_KEY")
        return obj

    def invalid(_):
        raise ContractError("NONFINITE_JSON")

    def finite_float(value):
        parsed = float(value)
        if not np.isfinite(parsed):
            return invalid(value)
        return parsed

    return json.loads(
        data.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid,
        parse_float=finite_float,
    )


def check_id(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ContractError("INVALID_ID")
    return value


def check_identity(payload, manifest):
    if not isinstance(payload, dict):
        raise ContractError("INVALID_OBJECT")
    if any(payload.get(key) != manifest[key] for key in IDENTITY):
        raise ContractError("STALE", 409)


def points4(value):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or any(type(x) not in (float, int) or abs(x) > 1e12 for x in row)
            for row in value
        )
    ):
        raise ContractError("INVALID_VERTICES")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ContractError("INVALID_VERTICES")
    return result


def rectangle_axes(xy):
    edges = np.roll(xy, -1, axis=0) - xy
    sizes = np.linalg.norm(edges, axis=1)
    if np.any(sizes < MIN_EDGE_M) or np.any(sizes > MAX_EDGE_M):
        raise ContractError("RECTANGLE_SIZE")
    axes = edges / sizes[:, None]
    adjacent = np.roll(axes, -1, axis=0)
    cross = axes[:, 0] * adjacent[:, 1] - axes[:, 1] * adjacent[:, 0]
    if (
        not (np.all(cross > 0) or np.all(cross < 0))
        or np.max(np.abs(np.sum(axes * adjacent, axis=1))) > 2e-6
        or np.max(np.abs(sizes[:2] - sizes[2:])) > 2e-6
    ):
        raise ContractError("NOT_ORDERED_RECTANGLE")
    return axes, sizes, float(np.sign(cross[0]))


@dataclass(frozen=True)
class Projection:
    x0: float
    y0: float
    s: float
    nx: int
    ny: int

    def xy(self, uv):
        u, v = np.asarray(uv, dtype=np.float64).T
        return np.stack(
            (self.x0 + u * self.s, self.y0 + (self.ny - v) * self.s), axis=-1
        )

    def uv(self, xy):
        x, y = np.asarray(xy, dtype=np.float64).T
        return np.stack(
            ((x - self.x0) / self.s, self.ny - (y - self.y0) / self.s), axis=-1
        )

    def cells(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        colrow = np.floor((xy - (self.x0, self.y0)) / self.s)
        origin = np.array([self.x0, self.y0])
        counts = np.array([self.nx, self.ny])
        valid = ((xy >= origin) & (xy < origin + self.s * counts)).all(axis=1)
        return np.minimum(colrow[valid], counts - 1)[:, ::-1].astype(np.int64), valid


def selection(payload, manifest, projection, targets):
    check_identity(payload, manifest)
    expected = set(IDENTITY) | {
        "schema",
        "request_id",
        "annotation_id",
        "expected_parent_revision",
        "operation",
        "target_opening_id",
        "target_vessel_id",
        "vertices_uv",
        "client_xy",
    }
    if set(payload) != expected or payload["schema"] != "manual_selection.1":
        raise ContractError("INVALID_SELECTION_SCHEMA")
    check_id(payload["request_id"])
    annotation = check_id(payload["annotation_id"])
    parent = payload["expected_parent_revision"]
    if type(parent) is not int or parent < 0:
        raise ContractError("INVALID_PARENT")
    target = payload["target_opening_id"]
    vessel = payload["target_vessel_id"]
    if payload["operation"] == "REFINE":
        check_id(target)
        check_id(vessel)
        if (
            target not in targets
            or targets[target]["vessel_id"] != vessel
            or annotation != target
        ):
            raise ContractError("INVALID_TARGET")
    elif (
        payload["operation"] != "CREATE"
        or target is not None
        or vessel is not None
        or annotation in targets
    ):
        raise ContractError("INVALID_OPERATION")
    uv = points4(payload["vertices_uv"])
    if np.any(uv < 0) or np.any(uv > (projection.nx, projection.ny)):
        raise ContractError("OUTSIDE_MAP")
    xy = projection.xy(uv)
    if np.max(np.abs(xy - points4(payload["client_xy"]))) > CLIENT_XY_TOLERANCE_M:
        raise ContractError("XY_MISMATCH")
    rectangle_axes(xy)
    return xy
