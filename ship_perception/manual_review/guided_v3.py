"""Bounded R3 arbitration. R2 measurement remains separately available."""

from dataclasses import replace
from heapq import heappop, heappush
from math import prod
from time import perf_counter

import numpy as np

from . import guided as r2
from .contract import FLAGS, RAW_ROWS, digest, rectangle_axes
from .matcher import anchor_geometry, corner_heights, fit_joint, query_strip, revalidate
from .side_candidates import CONFIG, ExecutionStop, checkpoint, profile, raw_set

SIDES = ("V0", "U1", "V1", "U0")
SEARCH_LIMIT = 96
REVIEW_LIMIT = 6
ALIAS_ENVELOPE_M = 0.05


def line_distance(a, b, length):
    slope, offset = a[0] - b[0], a[1] - b[1]
    return max(abs(offset), abs(offset + slope * length))


def tracks(rows, length, width, cfg, check):
    fits, _ = r2.training_candidates(rows, length, width, cfg, check, cluster=False)
    groups = []
    for mode in fits:
        check()
        for group in groups:
            if all(
                line_distance(mode, member, length) <= ALIAS_ENVELOPE_M
                and r2.equivalent(mode, member, length)
                for member in group
            ):
                group.append(mode)
                break
        else:
            groups.append([mode])
    output = []
    for group in groups:
        slope, intercept, train, rms = group[0]
        stations = [row["t_m"] for row in train]
        heights = [row["top_m"] for row in train]
        face = sorted({i for row in train for i in row["face"]})
        score = dict(
            train_rms_m=rms,
            unsupported_fraction=1 - (max(stations) - min(stations)) / length,
            intent_penalty=min(1.0, abs(intercept + slope * length / 2) / width),
        )
        output.append(
            dict(
                slope=slope,
                intercept_m=intercept,
                train=train,
                train_sha256=digest(train),
                track_id=digest([face, sorted(row["slab"] for row in train)]),
                rms_m=rms,
                station_extent_m=[min(stations), max(stations)],
                visible_height_range_m=[min(heights), max(heights)],
                spatial_scatter_m=rms,
                alias_max_displacement_m=max(
                    line_distance(a, b, length)
                    for i, a in enumerate(group)
                    for b in group[i:]
                ),
                fit_instances=[
                    [m, b, error, digest(support)] for m, b, support, error in group
                ],
                representative_rule="MOST_STATIONS_THEN_TRAIN_RMS",
                score_parts=score,
                score=rms / cfg.residual_m
                + score["unsupported_fraction"]
                + 0.15 * score["intent_penalty"],
            )
        )
    return output, len(fits)


def combinations(groups, lengths, cfg, check, report):
    """Explore ordered prefixes with a nonnegative additive lower bound."""
    queue = [(0.0, (), -float("inf"), float("inf"))]
    while queue:
        check()
        score, path, lo, hi = heappop(queue)
        i = len(path)
        if i == len(groups):
            yield tuple(groups[j][k] for j, k in enumerate(path))
            continue
        # A fitted side is allowed to shorten. The endpoint residual bounds
        # constrain slope over its minimum feasible projected span.
        minimum_span = max(1, lengths[i] - 2 * cfg.move_m) * np.cos(
            np.radians(cfg.angle_deg)
        )
        band = 4 * cfg.residual_m / minimum_span
        for j, mode in enumerate(groups[i]):
            # Include the complete alias envelope in the direction bound.
            bound = band + 2 * mode["alias_max_displacement_m"] / lengths[i]
            left = max(lo, mode["slope"] - bound)
            right = min(hi, mode["slope"] + bound)
            if left <= right:
                heappush(queue, (score + mode["score"], (*path, j), left, right))
            else:
                report["pruned_by_direction"] += prod(map(len, groups[i + 1 :]))
        if len(queue) > 8192:
            report["queued_at_limit"] = len(queue)
            raise ExecutionStop("PREFIX_QUEUE_LIMIT")


def independent_sides(sides):
    train = set()
    audit = set()
    face_by_side = set()
    for side in sides:
        rows = side["modes"][0]["train"]
        heldout = side.get("joint_heldout", [])
        face = {i for row in rows + heldout for i in row["face"]}
        if face_by_side & face:
            return False
        face_by_side.update(face)
        train.update(raw_set(rows))
        audit.update(raw_set(heldout))
    return not (train & audit)


def final_geometry(frame, corrected, sides, cfg, check):
    results = []
    for p in corrected:
        check()
        ids = frame.index.query_indexes(
            [p[0] - 0.8, p[1] - 0.8, p[0] + 0.8, p[1] + 0.8]
        )
        if len(ids) > cfg.slab_points:
            raise ExecutionStop("CORNER_POINT_BUDGET")
        results.append(ids)
    ids = np.unique(np.concatenate(results))
    z, evidence = corner_heights(
        frame.frame.xyz[ids].astype(np.float64), ids, corrected, sides, cfg
    )
    geometry = anchor_geometry(corrected, z, evidence)
    geometry.update(
        xyz=[[x, y, h] for (x, y), h in zip(geometry["xy"], z)],
        xy_complete=True,
        xyz_complete=all(h is not None for h in z),
    )
    return geometry


def guided_match(
    frame,
    xy,
    request,
    targets,
    cancelled=lambda: False,
    primitive=profile,
    *,
    edge_hint=None
):
    started = perf_counter()
    check = lambda: checkpoint(started + CONFIG.seconds, cancelled)
    report = dict(
        **FLAGS,
        algorithm="guided.3",
        ownership="UNVERIFIED",
        raw_rows=RAW_ROWS,
        execution_state="COMPLETED",
        match_state="NO_MATCH",
        reason="RAW_SUPPORT_INSUFFICIENT",
        geometry=None,
        sides=[],
        attempted=[],
        candidates=[],
        search_complete=False,
        pruned_by_direction=0,
        examined=0,
        joint_combinations=0,
        recommendation_id=None,
        phases_s={},
        rough_xy=xy.tolist()
    )

    def finish(state, reason):
        report.update(
            match_state=state, reason=reason, elapsed_s=perf_counter() - started
        )
        return report

    try:
        guard = r2.ownership_guard(xy, request, targets)
        if guard:
            return finish("AMBIGUOUS", guard)
        axes, lengths, orientation = rectangle_axes(xy)
        normals = orientation * np.column_stack((-axes[:, 1], axes[:, 0]))
        cfg = replace(CONFIG, angle_deg=r2.ANGLE_DEG)
        widths = [
            min(
                r2.MAX_WIDTH,
                max(
                    r2.MIN_WIDTH,
                    frame.projection.s * 3,
                    0.30 + length * np.tan(np.radians(1.5)) / 2,
                ),
            )
            for length in lengths
        ]
        report["search_half_width_m"] = widths
        for name, a, t, n, length, width in zip(
            SIDES, xy, axes, normals, lengths, widths
        ):
            check()
            pts, ids, box = query_strip(frame, a, t, n, length, width + 0.68, cfg)
            sections, observed = primitive(pts, ids, a, t, n, length, width, cfg, check)
            modes, count = tracks(observed, length, width, cfg, check)
            if edge_hint and name == edge_hint[0]:
                _, station, offset = edge_hint
                # Refit the entire side within the original band and select only
                # lines consistent with the operator's local position window.
                modes = [
                    m
                    for m in modes
                    if abs(m["intercept_m"] + station * m["slope"] - offset)
                    <= 2 * cfg.residual_m
                ]
            report["sides"].append(
                dict(
                    side_id=name,
                    reason="TRAINING_SUPPORTED" if modes else "TRAINING_INSUFFICIENT",
                    sections=sections,
                    observations=observed,
                    modes=modes,
                    aabb=box.tolist(),
                    mode_count=len(modes),
                    pre_alias_count=count,
                )
            )
        report["phases_s"]["nomination"] = perf_counter() - started
        groups = [side["modes"] for side in report["sides"]]
        report["joint_combinations"] = prod(map(len, groups))
        if any(not group for group in groups):
            report["search_complete"] = True
            return finish(
                "PARTIAL" if any(groups) else "NO_MATCH",
                "INCOMPLETE_PHYSICAL_TRAINING_SUPPORT",
            )
        training = []
        joint_started = perf_counter()
        for combination in combinations(
            groups, lengths, replace(cfg, move_m=max(widths)), check, report
        ):
            if report["examined"] >= SEARCH_LIMIT:
                raise ExecutionStop("JOINT_COMBINATION_LIMIT")
            report["examined"] += 1
            attempt = [
                dict(side, modes=[m]) for side, m in zip(report["sides"], combination)
            ]
            corrected, joint = fit_joint(
                xy, attempt, replace(cfg, move_m=max(widths)), check
            )
            audit = dict(
                track_ids=[m["track_id"] for m in combination],
                joint=joint,
                score=[m["score_parts"] for m in combination],
            )
            report["attempted"].append(audit)
            if corrected is None:
                audit["reason"] = "JOINT_GEOMETRY_CONFLICT"
                continue
            moves = [
                float(
                    max(abs((corrected[[i, (i + 1) % 4]] - xy[[i, (i + 1) % 4]]) @ n))
                )
                for i, n in enumerate(normals)
            ]
            guard = r2.ownership_guard(corrected, request, targets)
            audit.update(
                xy=corrected.tolist(),
                endpoint_moves_m=moves,
                rank_score=sum(m["score"] for m in combination),
            )
            if guard or any(m > w for m, w in zip(moves, widths)):
                audit["reason"] = guard or "OUTSIDE_INTENT_BAND"
                continue
            audit["reason"] = "PENDING_INDEPENDENT_VETO"
            training.append((corrected, attempt, audit))
        report["phases_s"]["joint"] = perf_counter() - joint_started
        report["search_complete"] = True
        report["training_feasible"] = len(training)
        if not training:
            return finish("PARTIAL", "JOINT_GEOMETRY_CONFLICT")
        audit_started = perf_counter()
        qualified = []
        for corrected, attempt, audit in training:
            check()
            passed = revalidate(xy, corrected, attempt, cfg)
            audit["heldout"] = [
                dict(
                    reason=s["joint_reason"],
                    digest=s["joint_heldout_sha256"],
                    refs=s["joint_heldout"],
                )
                for s in attempt
            ]
            if not passed or not independent_sides(attempt):
                audit["reason"] = (
                    "INDEPENDENT_HELDOUT_HAS_GAPS"
                    if not passed
                    else "CROSS_SIDE_RAW_REUSE"
                )
                continue
            qualified.append((corrected, attempt, audit))
        report["phases_s"]["heldout"] = perf_counter() - audit_started
        report["heldout_qualified"] = len(qualified)
        if len(qualified) > REVIEW_LIMIT:
            raise ExecutionStop("FINAL_VERIFICATION_LIMIT")
        for corrected, attempt, audit in qualified:
            check()
            stage = perf_counter()
            coverage, separator = r2.scan_separators(frame, corrected, check, cfg)
            report["phases_s"]["separator"] = (
                report["phases_s"].get("separator", 0) + perf_counter() - stage
            )
            audit["coverage"] = coverage
            if separator:
                audit.update(reason="INTERIOR_SEPARATOR", separator=separator)
                continue
            stage = perf_counter()
            geometry = final_geometry(frame, corrected, attempt, cfg, check)
            report["phases_s"]["corners"] = (
                report["phases_s"].get("corners", 0) + perf_counter() - stage
            )
            audit["reason"] = "INDEPENDENT_RAW_SUPPORT"
            candidate_id = digest([audit["track_ids"], geometry["xy"]])
            report["candidates"].append(
                dict(
                    candidate_id=candidate_id,
                    geometry=geometry,
                    geometry_digest=digest(geometry),
                    track_ids=audit["track_ids"],
                    source_kind="GUIDED_RAW",
                    score_parts=audit["score"],
                    score=audit["rank_score"],
                    uncertainty=dict(
                        train_rms_m=[s["modes"][0]["rms_m"] for s in attempt],
                        kind="TRAIN_RESIDUAL_NOT_CONFIDENCE_BOUND",
                    ),
                    endpoint_moves_m=audit["endpoint_moves_m"],
                    confirmable=True,
                )
            )
        check()
        if not report["candidates"]:
            return finish("PARTIAL", "NO_INDEPENDENTLY_VERIFIED_RECTANGLE")
        if len(report["candidates"]) == 1 and len(training) != 1:
            # With two splits the heldout may veto, but must never elect a new winner.
            report["candidates"][0]["confirmable"] = False
            return finish("AMBIGUOUS", "VETO_CANNOT_ADJUDICATE_COMPETITION")
        report["candidates"].sort(key=lambda c: (c["score"], c["candidate_id"]))
        best = report["candidates"][0]
        report["recommendation_id"] = best["candidate_id"]
        report["geometry"] = best["geometry"]
        if len(report["candidates"]) > 1:
            report["score_gap"] = report["candidates"][1]["score"] - best["score"]
            return finish("GUIDED_CHOICES_PENDING", "VERIFIED_PHYSICAL_COMPETITION")
        return finish("GUIDED_REFINED_PENDING", "INDEPENDENT_RAW_SUPPORT")
    except ExecutionStop as exc:
        reason = str(exc)
        report.update(
            search_complete=False,
            geometry=None,
            execution_state=(
                reason if reason in ("CANCELLED", "TIMEOUT") else "BUDGET_EXCEEDED"
            ),
        )
        for candidate in report["candidates"]:
            candidate["confirmable"] = False
        return finish("SEARCH_INCOMPLETE", reason)
