"""Private acceptance runner. Historical geometry enters only as explicit test ROI."""

import argparse
from collections import Counter
from hashlib import sha256
from http.client import HTTPConnection
import json
from math import ceil
from pathlib import Path
from threading import Thread
from time import perf_counter
from uuid import uuid4
from unittest.mock import patch

import numpy as np

from ship_perception.manual_review.contract import IDENTITY, canonical, rectangle_axes
from ship_perception.manual_review.frame_session import FrameSession
from ship_perception.manual_review.server import LocalServer, source_sha
from ship_perception.manual_review.workbench import Workbench
from ship_perception.manual_review.workbench_auto import AutomaticRun, FILES
from ship_perception.manual_review.workbench_server import STATIC


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, obj):
    path.write_bytes(canonical(obj))


def api(server, path, payload=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=12)
    start = perf_counter()
    try:
        request = dict(headers={"X-MHR-Token": server.token})
        if payload is not None:
            request["headers"].update(
                Origin=server.origin, **{"Content-Type": "application/json"}
            )
            request["body"] = canonical(payload)
        connection.request("GET" if payload is None else "POST", path, **request)
        response = connection.getresponse()
        data = json.loads(response.read())
        assert data["steel_certified"] is False and data["control_eligible"] is False
        return response.status, data, perf_counter() - start
    finally:
        connection.close()


def fetch(server):
    code, view, _ = api(server, "/api/v2/view")
    assert code == 200
    return view, {k: view["manifest"][k] for k in IDENTITY}


def read_suggestion(review, request):
    from ship_perception.manual_review.contract import digest

    with review.store.transaction() as db:
        row = db.execute(
            "SELECT body FROM suggestions WHERE id=?",
            (digest([request["view_id"], request["request_id"]]),),
        ).fetchone()
    return json.loads(row[0]) if row else None


def render_svg(path, projection, baseline, rough, geometry):
    dims = f'width="{projection.nx}" height="{projection.ny}"'
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" {dims}>',
        f'<rect {dims} fill="#152330"/><image href="height.png" {dims}/>',
    ]
    for poly, color, dash in (
        (baseline, "#88a9c0", "5,4"),
        (rough, "#ffc96c", "9,5"),
        (geometry and geometry["xy"], "#70edb4", ""),
    ):
        if poly is not None:
            points = " ".join(",".join(map(str, row)) for row in projection.uv(poly))
            parts.append(
                f'<polygon points="{points}" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}" fill="none"/>'
            )
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf8")


def run_frame(out, spec, folder, r1, auto_rows, results, branch):
    from tools.mhr_synthetic import rect

    started = perf_counter()
    frame = FrameSession(spec["pcd"])
    cold = perf_counter() - started
    auto = AutomaticRun("LOADING")
    auto.import_result(folder, frame)
    assert auto.snapshot["status"] == "READY", auto.snapshot
    review = Workbench(frame, out, auto, branch)
    manifest = frame.manifest
    write(out / "manifest.json", manifest)
    (out / "height.png").write_bytes(frame.layers["height"])
    original = {
        name: sha256((folder / name).read_bytes()).hexdigest() for name in FILES
    }
    with LocalServer(frame, review=review, static_assets=STATIC) as server:
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            view, identity = fetch(server)
            targets = view["auto"]["targets"]
            assert set(targets) == {row["overlay_id"] for row in spec["targets"]}
            baseline_targets = [
                dict(id=k, geometry=t["geometry"], digest=t["geometry_digest"])
                for k, t in targets.items()
            ]
            with patch(
                "ship_perception.manual_review.guided.guided_match",
                side_effect=AssertionError("AUTO_REACHED_MATCHER"),
            ):
                for key, target in targets.items():
                    req = {
                        **identity,
                        "mutation_id": uuid4().hex,
                        "annotation_id": key,
                        "expected_parent_revision": 0,
                        "bundle_digest": view["auto"]["bundle_digest"],
                        "geometry_digest": target["geometry_digest"],
                    }
                    code, saved, elapsed = api(server, "/api/v2/auto/confirm", req)
                    assert (
                        code == 200
                        and saved["state"] == "ACCEPTED_AUTO"
                        and saved["geometry"] == target["geometry"]
                    )
                    auto_rows.append(
                        dict(
                            scene=spec["id"],
                            target=key,
                            wire=manifest["wire_sha256"],
                            cold_seconds=cold,
                            import_seconds=auto.snapshot["import_s"],
                            confirm_seconds=elapsed,
                            exact=True,
                        )
                    )
                view, _ = fetch(server)
                code, marker, _ = api(
                    server,
                    "/api/v2/scene/confirm",
                    {
                        **identity,
                        "mutation_id": uuid4().hex,
                        "scene_digest": view["scene_digest"],
                        "confirm_empty": False,
                    },
                )
                assert code == 200 and marker["scene_state"] == "REVIEW_COMPLETE"
                code, exported, _ = api(
                    server,
                    "/api/v2/export",
                    {**identity, "scene_digest": view["scene_digest"]},
                )
                assert code == 200
                assert {
                    x["annotation_id"]: x["geometry"] for x in exported["accepted"]
                } == {k: t["geometry"] for k, t in targets.items()}
            write(out / "auto-targets.json", baseline_targets)
            write(out / "auto-export.json", exported)
            (out / "auto-export.csv").write_text(exported["csv"], encoding="utf8")

            def probe(label, ordinal, xy, key, cohort, baseline_state=None):
                baseline = targets[key]["geometry"]["xy"] if key else None
                vessel = targets[key]["vessel_id"] if key else None
                ann = key or uuid4().hex
                request = {
                    **identity,
                    "annotation_id": ann,
                    "request_id": uuid4().hex,
                    "schema": "manual_selection.1",
                    "operation": "REFINE" if key else "CREATE",
                    "target_opening_id": key,
                    "target_vessel_id": vessel,
                    "expected_parent_revision": 1 if key else 0,
                    "bundle_digest": auto.snapshot["bundle_digest"],
                    "vertices_uv": frame.projection.uv(xy).tolist(),
                    "client_xy": xy.tolist(),
                }
                code, answer, elapsed = api(server, "/api/v2/refine", request)
                record = read_suggestion(review, request)
                write(
                    out / (label + ".json"),
                    dict(
                        request=request,
                        result=record or answer,
                        baseline_state=baseline_state,
                    ),
                )
                geometry = answer.get("geometry")
                render_svg(
                    out / (label + ".svg"),
                    frame.projection,
                    baseline,
                    xy.tolist(),
                    geometry,
                )
                row = dict(
                    scene=spec["id"],
                    ordinal=ordinal,
                    case=label,
                    cohort=cohort,
                    operation=request["operation"],
                    state=answer.get("match_state")
                    or answer.get("execution_state")
                    or "INVALID",
                    http=code,
                    reason=answer.get("reason", answer.get("error")),
                    seconds=elapsed,
                    r1_state=baseline_state,
                    xyz_complete=geometry and geometry["xyz_complete"],
                    z_null=sum(x is None for x in geometry["z"]) if geometry else None,
                )
                results.append(row)
                return row

            for i, target in enumerate(spec["targets"]):
                original_xy = np.asarray(target["xy"])
                axes, sizes, orientation = rectangle_axes(original_xy)
                if orientation < 0:
                    original_xy = original_xy[[1, 0, 3, 2]]
                    axes, sizes, _ = rectangle_axes(original_xy)
                angle = float(np.arctan2(axes[0, 1], axes[0, 0]))
                raw = rect(
                    original_xy.mean(axis=0) + np.array([0.17, -0.12]),
                    angle + np.radians(0.50),
                    sizes[0] + 0.18,
                    sizes[1] - 0.19,
                )
                probe(f"{i}-first", i, raw, target["overlay_id"], "SCRIPTED_FIRST")
                if r1:
                    for kind in ("translation", "rotation", "independent_size"):
                        saved = read(
                            r1
                            / "04_m2_m3"
                            / "real-strip"
                            / spec["id"]
                            / f"{i}-{kind}.json"
                        )
                        orig = saved["request"]
                        r = saved["result"]
                        probe(
                            f"{i}-r1-{kind}",
                            i,
                            np.asarray(orig["client_xy"]),
                            orig["target_opening_id"],
                            "R1_57_AB",
                            r["match_state"] or r["execution_state"],
                        )
                        if kind == "translation":
                            probe(
                                f"{i}-r1-create-as-refine",
                                i,
                                np.asarray(orig["client_xy"]),
                                target["overlay_id"],
                                "R1_CREATE_INTENT_AS_REFINE",
                            )
            # Use the same decoded frame and a complete absence of automatic context.
            with auto.lock:
                cached = auto.snapshot
                auto.snapshot = dict(status="DISABLED", bundle_digest=None, targets={})
            try:
                for i, target in enumerate(spec["targets"]):
                    saved = read(
                        r1
                        / "04_m2_m3"
                        / "local-only-no-overlay"
                        / spec["id"]
                        / f"{i}-translation.json"
                    )
                    probe(
                        f"{i}-local-only",
                        i,
                        np.asarray(saved["request"]["client_xy"]),
                        None,
                        "LOCAL_ONLY19",
                        saved["result"]["match_state"]
                        or saved["result"]["execution_state"],
                    )
            finally:
                with auto.lock:
                    auto.snapshot = cached
            for i, target in enumerate(spec["targets"]):
                wrong = next(
                    (v["overlay_id"] for v in spec["targets"] if v != target), None
                )
                if wrong:
                    result = probe(
                        f"{i}-wrong-target",
                        i,
                        np.asarray(target["xy"]),
                        wrong,
                        "WRONG_TARGET",
                    )
                    assert result["state"] != "GUIDED_REFINED_PENDING"
        finally:
            server.shutdown()
    assert original == {
        name: sha256((folder / name).read_bytes()).hexdigest() for name in FILES
    }


def quantiles(values):
    values = sorted(values)
    if not values:
        return dict(n=0, p50=None, p95=None, maximum=None)
    return dict(
        n=len(values),
        p50=values[ceil(len(values) * 0.5) - 1],
        p95=values[ceil(len(values) * 0.95) - 1],
        maximum=values[-1],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("native_runs", type=Path)
    parser.add_argument("r1_evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    imports = {}
    for p in args.native_runs.rglob("review_hatches_compact.json"):
        native = read(p.with_name("05_hatch_corners.json"))
        assert native["input_sha256"] not in imports
        imports[native["input_sha256"]] = p.parent
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    scenes = read(args.spec)
    r1_identity = read(args.r1_evidence / "02_inputs/verified.json")
    wire = {str(Path(r["pcd"]).resolve()): r["wire"] for r in r1_identity}
    auto_rows, results = [], []
    branch = source_sha()
    for spec in scenes:
        folder = output / spec["id"]
        folder.mkdir()
        run_frame(
            folder,
            spec,
            imports[wire[str(Path(spec["pcd"]).resolve())]],
            args.r1_evidence,
            auto_rows,
            results,
            branch,
        )
        write(output / "auto-accept.json", auto_rows)
        write(output / "refinement.json", results)
        print(spec["id"], len(auto_rows), len(results), flush=True)
    cohorts = {}
    for name in sorted({r["cohort"] for r in results}):
        rows = [r for r in results if r["cohort"] == name]
        success = [r for r in rows if r["state"] == "GUIDED_REFINED_PENDING"]
        cohorts[name] = dict(
            n=len(rows),
            counts=dict(Counter(r["state"] for r in rows)),
            by_reason=dict(Counter(r["reason"] for r in rows)),
            by_operation={
                k: dict(Counter(r["state"] for r in rows if r["operation"] == k))
                for k in ("CREATE", "REFINE")
            },
            success=quantiles([r["seconds"] for r in success]),
            reject=quantiles([r["seconds"] for r in rows if r not in success]),
        )
    from ship_perception.perception.runtime_metrics import peak_rss_bytes

    report = dict(
        source_sha=branch,
        frames=len(scenes),
        automatic_targets=len(auto_rows),
        confirm=quantiles([r["confirm_seconds"] for r in auto_rows]),
        cold=quantiles([r["cold_seconds"] for r in auto_rows]),
        import_time=quantiles([r["import_seconds"] for r in auto_rows]),
        cohorts=cohorts,
        rss_peak_bytes=peak_rss_bytes(),
        human_review="PENDING",
    )
    write(output / "summary.json", report)
    print(report)


if __name__ == "__main__":
    main()
