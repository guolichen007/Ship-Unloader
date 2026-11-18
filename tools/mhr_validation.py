"""Preregistered synthetic matrix and private PCD runner. Never imported by runtime."""

import argparse
from collections import Counter
import csv
import json
from math import ceil
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from ship_perception.manual_review.contract import IDENTITY, canonical
from ship_perception.manual_review.frame_session import FrameSession
from ship_perception.manual_review.matcher import match
from ship_perception.manual_review.revisions import Review
from ship_perception.manual_review.side_candidates import CONFIG
from tools.mhr_synthetic import frame_for, rect, synthesized

EVAL_SEED = 802331
REAL_SEED = 19412
POSITIVE = 20
DISTURBANCES = 5
VARIANTS = {
    "noisy": "POS",
    "duplicate": "POS",
    "reordered": "POS",
    "near_square": "POS",
    "partial_sampling": "POS",
    "corner_z_gap": "POS",
    "sparse": "UNRESOLVED",
    "cropped": "UNRESOLVED",
    "dual": "UNRESOLVED",
    "occluded": "UNRESOLVED",
    "all_occluded": "UNRESOLVED",
    "high_cargo": "UNRESOLVED",
    "cluster": "NEG",
    "shore": "NEG",
    "deformed": "NEG",
    "water": "NEG",
    "two_holds": "NEG",
    "unequal_holds": "NEG",
    "wrong_box": "NEG",
    "two_vessels": "NEG",
}


def plan():
    return dict(
        config=CONFIG.audit(),
        development_seed=49072,
        evaluation_seed=EVAL_SEED,
        real_seed=REAL_SEED,
        positives=POSITIVE,
        disturbances=DISTURBANCES,
        variants=VARIANTS,
        replicates=5,
        corner_gate_m=0.05,
        axis_gate_deg=0.5,
        real_perturbations=["translation", "rotation", "independent_size"],
        statistics="nearest_rank_include_all",
        further_real_warm_repeats=43,
    )


def score(r, truth, expected):
    g = r["geometry"]
    matched = r["match_state"] == "MATCHED_REVIEW" and g is not None
    row = dict(
        match_state=r["match_state"],
        execution_state=r["execution_state"],
        reason=r["reason"],
        matched=matched,
        expected=expected,
        corner_error_m=None,
        axis_error_deg=None,
        z_null_count=sum(z is None for z in g["z"]) if g else None,
        side_reasons=[s["reason"] for s in r.get("sides", [])],
    )
    if matched and truth is not None:
        measured = np.array(g["xy"])
        row["corner_error_m"] = float(np.linalg.norm(measured - truth, axis=1).max())
        edges = np.roll(truth, -1, axis=0) - truth
        true_axis = edges[np.argmax(np.linalg.norm(edges[:2], axis=1))]
        deg = np.degrees(np.arctan2(true_axis[1], true_axis[0]))
        row["axis_error_deg"] = float(abs((g["axis_deg"] - deg + 90) % 180 - 90))
    if expected == "POS":
        row["passed"] = (
            matched
            and row["corner_error_m"] <= 0.05
            and (g["near_square_axis_ambiguous"] or row["axis_error_deg"] <= 0.5)
        )
    elif expected in ("NEG", "UNRESOLVED"):
        row["passed"] = not matched and r["execution_state"] == "COMPLETED"
    return row


def evaluate_synthetic(root):
    rng = np.random.default_rng(EVAL_SEED)
    rows = []
    for i in range(POSITIVE):
        c, angle = rng.uniform(-450, 450, 2), rng.uniform(-np.pi, np.pi)
        length, width = rng.uniform(12, 27), rng.uniform(8, 11)
        truth = rect(c, angle, length, width)
        points = synthesized(truth)
        frame = frame_for(points)
        for j in range(DISTURBANCES):
            offset, shape = rng.uniform(-0.70, 0.70, 2), rng.uniform(-0.65, 0.65, 2)
            turn = rng.uniform(-2.15, 2.15)
            rough = rect(
                c + offset,
                angle + np.radians(turn),
                length + shape[0],
                width + shape[1],
            )
            run_case(
                root, rows, f"clean-{i:02d}-{j}", frame, rough, truth, "POS", "clean"
            )
    for variant, expected in VARIANTS.items():
        for j in range(5):
            c, angle = rng.uniform(-180, 180, 2), rng.uniform(-np.pi, np.pi)
            length, width = rng.uniform(15, 19), rng.uniform(9, 11)
            if variant == "near_square":
                length = width * 1.015
            truth = rect(c, angle, length, width)
            rough = rect(
                c + rng.uniform(-0.3, 0.3, 2),
                angle + rng.uniform(-0.025, 0.025),
                length + 0.24,
                width - 0.17,
            )
            base_variants = {
                "near_square",
                "duplicate",
                "reordered",
                "partial_sampling",
                "corner_z_gap",
                "water",
                "two_holds",
                "unequal_holds",
                "wrong_box",
                "two_vessels",
            }
            p = synthesized(
                truth, "clean" if variant in base_variants else variant, rng
            )
            if variant == "duplicate":
                p = np.repeat(p, 4, axis=0)
            if variant == "reordered":
                p = p[rng.permutation(len(p))]
            if variant == "partial_sampling":
                p = p[rng.random(len(p)) > 0.26]
            if variant == "corner_z_gap":
                p = p[np.linalg.norm(p[:, None, :2] - truth, axis=2).min(axis=1) > 0.49]
            if variant == "wrong_box":
                rough += (length + 10) * np.array([np.cos(angle), np.sin(angle)])
            if variant == "water":
                points = rng.uniform(-length, length, (2500, 2)) + c
                p = np.column_stack((points, np.zeros(len(points))))
            if variant in ("two_holds", "unequal_holds", "two_vessels"):
                scale = 0.77 if variant == "unequal_holds" else 1
                gap = 4 if variant == "two_vessels" else 1.1
                offset = (length * (1 + scale) / 2 + gap) * np.array(
                    [np.cos(angle), np.sin(angle)]
                )
                p = np.vstack(
                    [p, synthesized(rect(c + offset, angle, length * scale, width))]
                )
                rough = rect(c + offset / 2, angle, length * (1 + scale) + gap, width)
            run_case(
                root,
                rows,
                f"{variant}-{j}",
                frame_for(p),
                rough,
                truth,
                expected,
                variant,
            )
    return rows


def run_case(root, rows, case, frame, rough, truth, expected, variant):
    started = perf_counter()
    r = match(frame, rough, {"operation": "CREATE"})
    evidence = dict(
        case_id=case, result=r, offline_truth_xy=truth.tolist(), expected=expected
    )
    (root / (case + ".json")).write_bytes(canonical(evidence))
    rows.append(
        dict(
            case=case,
            variant=variant,
            wall_s=perf_counter() - started,
            **score(r, truth, expected),
        )
    )


def real_request(frame, xy, identifier, annotation, target):
    uv = frame.projection.uv(xy)
    return {
        **{k: frame.manifest[k] for k in IDENTITY},
        "schema": "manual_selection.1",
        "request_id": identifier,
        "annotation_id": annotation,
        "expected_parent_revision": 0,
        "operation": "REFINE" if target else "CREATE",
        "target_opening_id": target,
        "target_vessel_id": frame.targets[target]["vessel_id"] if target else None,
        "vertices_uv": uv.tolist(),
        "client_xy": xy.tolist(),
    }


def evaluate_real(root, spec):
    from ship_perception.manual_review.contract import ContractError, rectangle_axes

    rng = np.random.default_rng(REAL_SEED)
    rows = []
    for scene in spec:
        tag = scene["id"]
        folder = root / tag
        folder.mkdir(exist_ok=True)
        start = perf_counter()
        frame = FrameSession(scene["pcd"], overlay=scene.get("overlay"))
        cold = perf_counter() - start
        review = Review(frame, folder)
        (folder / "manifest.json").write_bytes(canonical(frame.manifest))
        for name, image in frame.layers.items():
            (folder / (name + ".png")).write_bytes(image)
        cases = []
        for number, target in enumerate(scene["targets"]):
            ref = np.array(target["xy"])
            axes, sizes, orient = rectangle_axes(ref)
            long_angle = np.arctan2(axes[0, 1], axes[0, 0])
            if orient < 0:
                ref = ref[[1, 0, 3, 2]]
                axes, sizes, _ = rectangle_axes(ref)
                long_angle = np.arctan2(axes[0, 1], axes[0, 0])
            c = ref.mean(axis=0)
            perturbations = [
                ("translation", rng.uniform(-0.75, 0.75, 2), 0, 0, 0),
                ("rotation", np.zeros(2), rng.choice((-1, 1)) * np.radians(1.85), 0, 0),
                ("independent_size", rng.uniform(-0.22, 0.22, 2), 0, 0.7, -0.58),
            ]
            for kind, shift, turn, dl, dw in perturbations:
                rough = rect(c + shift, long_angle + turn, sizes[0] + dl, sizes[1] + dw)
                target_id = target.get("overlay_id") if kind != "translation" else None
                if target_id not in frame.targets:
                    target_id = None
                cases.append((f"{number}-{kind}", rough, ref, target_id, "REAL"))
        if len(scene["targets"]) > 1:
            polygons = [np.array(t["xy"]) for t in scene["targets"]]
            world = np.vstack(polygons)
            left, bottom = world.min(axis=0)
            right, top = world.max(axis=0)
            combined = rect(
                np.array([(left + right) / 2, (bottom + top) / 2]),
                0,
                right - left,
                top - bottom,
            )
            cases.append(("multiple-holds", combined, None, None, "NEG"))
        p = frame.projection
        cases.append(
            (
                "outside",
                p.xy([[-10, 80], [45, 80], [45, 120], [-10, 120]]),
                None,
                None,
                "INVALID",
            )
        )
        cases.append(("wrong-frame", cases[0][1], None, None, "INVALID"))
        cases.append(
            ("blank", p.xy([[1, 1], [51, 1], [51, 51], [1, 51]]), None, None, "NEG")
        )
        for index in range(4):
            candidate = cases[index % max(1, len(scene["targets"]) * 3)]
            cases.append((f"warm-{index}", *candidate[1:]))
        for identifier, rough, reference, target, expected in cases:
            req = real_request(frame, rough, identifier, target or identifier, target)
            if identifier == "wrong-frame":
                req["frame_instance_id"] = "wrong"
            started = perf_counter()
            try:
                r = review.handle("/api/match", req)
                with review.store.transaction() as db:
                    body = db.execute(
                        "SELECT body FROM suggestions WHERE id=?", (identifier,)
                    ).fetchone()
                if body:
                    r = json.loads(body[0])
            except ContractError as e:
                r = dict(
                    match_state="STALE" if str(e) == "STALE" else "INVALID",
                    execution_state="COMPLETED",
                    reason=str(e),
                    geometry=None,
                    sides=[],
                )
            wall = perf_counter() - started
            support = score(r, reference, expected)
            rows.append(
                dict(
                    scene=tag,
                    case=identifier,
                    local_only=target is None,
                    wall_s=wall,
                    cold_s=cold,
                    **support,
                )
            )
            (folder / (identifier + ".json")).write_bytes(
                canonical(
                    dict(
                        request=req,
                        result=r,
                        offline_reference=(
                            None if reference is None else reference.tolist()
                        ),
                        offline_only=True,
                    )
                )
            )
            render_overlay(
                folder / (identifier + ".svg"), frame, rough, reference, r["geometry"]
            )
        (folder / "cold.json").write_bytes(
            canonical(dict(seconds=cold, **frame.timings))
        )
    return rows


def render_overlay(path, frame, rough, reference, geometry):
    p = frame.projection
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{p.nx}" height="{p.ny}" viewBox="0 0 {p.nx} {p.ny}">',
        f'<rect width="{p.nx}" height="{p.ny}" fill="#17212b"/>',
        f'<image href="height.png" width="{p.nx}" height="{p.ny}"/>',
    ]
    for polygon, color, dash in (
        (reference, "#85a9ff", "3,6"),
        (rough, "#ffce73", "8,5"),
        (None if not geometry else geometry["xy"], "#75ffbd", ""),
    ):
        if polygon is not None:
            vertices = " ".join(",".join(map(str, pt)) for pt in p.uv(polygon))
            svg.append(
                f'<polygon points="{vertices}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>'
            )
    svg.append("</svg>")
    path.write_text("".join(svg), encoding="utf-8")


def summary(rows):
    times = sorted(row["wall_s"] for row in rows)
    return dict(
        n=len(rows),
        counts=Counter(row["match_state"] or row["execution_state"] for row in rows),
        p50_s=times[ceil(0.50 * len(times)) - 1],
        p95_s=times[ceil(0.95 * len(times)) - 1],
        max_s=times[-1],
        wrong_unique=sum(
            r["expected"] in ("NEG", "UNRESOLVED") and r["matched"] for r in rows
        ),
        expected_pos=sum(r["expected"] == "POS" for r in rows),
        pos_pass=sum(r["expected"] == "POS" and r["passed"] for r in rows),
        z_null=sum(r["z_null_count"] or 0 for r in rows),
        local_only_matches=sum(
            r.get("local_only", False) and r["matched"] for r in rows
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--real-manifest", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_bytes(canonical(plan()))
    rows = (
        evaluate_real(args.output, json.loads(args.real_manifest.read_bytes()))
        if args.real_manifest
        else evaluate_synthetic(args.output)
    )
    (args.output / "rows.json").write_bytes(canonical(rows))
    with (args.output / "rows.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        out = csv.DictWriter(stream, fieldnames=rows[0].keys(), extrasaction="ignore")
        out.writeheader()
        out.writerows(rows)
    report = summary(rows)
    try:
        import psutil

        rss = psutil.Process().memory_info()
        report["rss_bytes"] = getattr(rss, "peak_wset", rss.rss)
        report["rss_source"] = "peak_wset" if hasattr(rss, "peak_wset") else "final_rss"
    except ImportError:
        report["rss_source"] = "NOT_RUN"
    report["platform"] = platform.platform()
    (args.output / "summary.json").write_bytes(canonical(report))
    print(json.dumps(report))
    if not args.real_manifest and any(not r["passed"] for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
