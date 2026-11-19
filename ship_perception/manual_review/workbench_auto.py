"""Bind frozen native review outputs to the current decoded frame."""

import hashlib
from pathlib import Path
import subprocess
import sys
from threading import Lock
from time import perf_counter
from uuid import uuid4

import numpy as np

from ship_perception.perception.pipeline import resolve_config, source_identity
from ship_perception.perception.r3h_geometry import dimensions

from .contract import FLAGS, ContractError, check_id, digest, parse_json, points4

FILES = {
    "03_results.json": 256 << 20,
    "05_hatch_corners.json": 48 << 20,
    "review_hatches_compact.json": 4 << 20,
}
ORDER = ["U0/V0", "U1/V0", "U1/V1", "U0/V1"]


def require(condition):
    if not condition:
        raise ContractError("AUTO_OUTPUT_BINDING")


def unique(records, key):
    by_id = {check_id(r[key]): r for r in records}
    require(len(by_id) == len(records))
    return by_id


def load_auto(folder, frame):
    folder = Path(folder).resolve()
    checksums, records = {}, {}
    for name, limit in FILES.items():
        with (folder / name).open("rb") as stream:
            data = stream.read(limit + 1)
        require(len(data) <= limit)
        records[name] = parse_json(data)
        checksums[name] = hashlib.sha256(data).hexdigest()
    full, corners, compact = (records[name] for name in FILES)
    manifest = frame.manifest
    for document in (full, corners):
        native = document["input"]
        require(
            document["input_sha256"]
            == native["provenance"]["wire_sha256"]
            == manifest["wire_sha256"]
        )
        require(
            native["semantic_sha256"] == manifest["semantic_sha256"]
            and native["frame_id"] == manifest["frame_id"]
            and native["point_count"] == manifest["finite_count"]
        )
        require(document["review_mode"] == "review-first")
    require(
        full["schema"]
        == corners["source_result_schema"]
        == "ship_perception.hatch_once.3"
    )
    require(
        corners["schema"] == "ship_perception.hatch_corners.3"
        and corners["units"] == "m"
        and corners["coordinate_frame"] == manifest["frame_id"]
        and corners["corner_order"] == ORDER
    )
    require(corners["source_result_sha256"] == checksums["03_results.json"])
    require(corners["producer_git_sha"] == full["producer_git_sha"])
    require(full["runtime_source_sha256"] == source_identity()["runtime_source_sha256"])
    require(
        full["config_sha256"] == resolve_config()[1]
        and full["config_hash"] == corners["config_hash"]
    )
    native_geometry = full["geometry_only"]
    require(native_geometry["opening_review_level"] == "PRECISION_RECTANGLE_REVIEW")
    native_hatches = unique(native_geometry["hatches"], "hatch_id")
    strict = unique(corners["hatches"], "hatch_id")
    require(native_hatches.keys() == strict.keys())
    for key, row in strict.items():
        src = native_hatches[key]
        for prop in (
            "opening_identity",
            "vessel_hypothesis_id",
            "topology_status",
            "geometry_status",
        ):
            require(row[prop] == src[prop])
        require(row["boundary_xy_m"] == src["boundary_xy"])
    cells = unique(
        [
            c
            for c in native_geometry["selected_opening_cells"]
            if c.get("geometry") is not None
        ],
        "cell_id",
    )
    displayed = unique(compact, "cell_id")
    require(cells.keys() == displayed.keys() and len(displayed) <= 128)
    targets = {}
    for key, row in displayed.items():
        cell = cells[key]
        layer = cell.get("review_geometry") or {}
        adopted = layer.get("adopted_for_review", False)
        primary = layer if adopted else cell["geometry"]
        require(
            row["adopted_for_review"] == adopted
            and row["vessel_id"] == cell["vessel_id"]
        )
        require(
            row["boundary_xy"] == primary["boundary_xy"]
            and row["corners_xyz"] == primary["corners_xyz"]
        )
        require(
            row["state"] == layer.get("state", "RETAINED_BASELINE")
            and row["grade"] == layer.get("grade", "UNRESOLVED")
        )
        points = points4(row["boundary_xy"])
        edges = np.roll(points, -1, axis=0) - points
        turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(
            edges[:, 0], -1
        )
        require(np.all(turns > 0) or np.all(turns < 0))
        xyz = row["corners_xyz"]
        require(len(xyz) == 4)
        for p, q in zip(row["boundary_xy"], xyz):
            require(
                len(q) == 3
                and q[:2] == p
                and (q[2] is None or type(q[2]) in (int, float) and abs(q[2]) < 1e7)
            )
        size = dimensions(row["boundary_xy"])
        require(
            size["length_m"] == row["length_m"] and size["width_m"] == row["width_m"]
        )
        require(row["steel_certified"] is False and row["control_eligible"] is False)
        # Keep the adopted display layer and its original dimensions and state.
        geometry = dict(
            xy=row["boundary_xy"],
            z=[p[2] for p in xyz],
            xyz=xyz,
            length_m=row["length_m"],
            width_m=row["width_m"],
            dimension_definition="frozen_review_layer_horizontal_opposing_edges",
            corner_ids=["P1", "P2", "P3", "P4"],
            corner_order=ORDER,
            xy_complete=True,
            xyz_complete=all(p[2] is not None for p in xyz),
        )
        target = dict(
            **FLAGS,
            opening_id=key,
            vessel_id=check_id(cell["vessel_id"]),
            geometry=geometry,
            geometry_digest=digest(geometry),
            state=row["state"],
            grade=row["grade"],
            adopted_for_review=adopted,
            review_gates=row.get("review_gates", {}),
            source_kind="AUTO_BASELINE",
        )
        targets[key] = target
    provenance = dict(
        files=checksums,
        wire_sha256=manifest["wire_sha256"],
        semantic_sha256=manifest["semantic_sha256"],
        frame_id=manifest["frame_id"],
        producer=full["producer_git_sha"],
        runtime_source_sha256=full["runtime_source_sha256"],
        config_hash=full["config_hash"],
        review_mode=full["review_mode"],
        native_pipeline_status=full["pipeline_health"]["status"],
    )
    require(full["pipeline_health"]["status"] not in ("FAILED", "ERROR"))
    return dict(
        status="READY",
        bundle_digest=digest(provenance),
        provenance=provenance,
        targets=targets,
    )


class AutomaticRun:
    def __init__(self, status="DISABLED"):
        self.lock = Lock()
        self.snapshot = dict(status=status, bundle_digest=None, targets={})
        self.child = None
        self.stopped = False

    def publish(self, result):
        with self.lock:
            if not self.stopped:
                self.snapshot = result

    def import_result(self, folder, frame):
        started = perf_counter()
        try:
            bundle = load_auto(folder, frame)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
            RecursionError,
        ):
            self.publish(
                dict(
                    status="FAILED",
                    reason="AUTO_OUTPUT_BINDING",
                    bundle_digest=None,
                    targets={},
                )
            )
        else:
            self.publish(dict(**bundle, import_s=perf_counter() - started))

    def run(self, pcd, root, frame):
        started = perf_counter()
        directory = Path(root).resolve() / "r2-auto" / uuid4().hex
        try:
            directory.mkdir(parents=True)
            script = Path(__file__).resolve().parents[2] / "tools" / "r3h_oneclick.py"
            with (directory / "worker.log").open("wb") as log:
                with self.lock:
                    if self.stopped:
                        return
                    self.child = subprocess.Popen(
                        [
                            sys.executable,
                            "-s",
                            str(script),
                            str(pcd),
                            str(directory),
                            "--worker",
                            "--mode",
                            "full",
                            "--frame-id",
                            frame.manifest["frame_id"],
                            "--expected-sha",
                            frame.manifest["wire_sha256"],
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    child = self.child
                try:
                    code = child.wait(timeout=310)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    raise ContractError("AUTO_TIMEOUT")
            results = list(directory.glob("full/*/03_results.json"))
            if code or len(results) != 1:
                raise ContractError("AUTO_WORKER_FAILED")
            bundle = load_auto(results[0].parent, frame)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
            RecursionError,
        ) as exc:
            reason = (
                str(exc) if isinstance(exc, ContractError) else "AUTO_OUTPUT_BINDING"
            )
            self.publish(
                dict(
                    status="FAILED",
                    reason=reason,
                    bundle_digest=None,
                    targets={},
                    auto_s=perf_counter() - started,
                )
            )
        else:
            self.publish(dict(**bundle, auto_s=perf_counter() - started))

    def close(self):
        with self.lock:
            self.stopped = True
            child = self.child
        if child and child.poll() is None:
            child.kill()
            child.wait()
