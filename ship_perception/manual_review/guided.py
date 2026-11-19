"""Intent-constrained Raw candidate search, separate from the strict R1 matcher."""

from dataclasses import replace
from math import prod
from time import perf_counter

import numpy as np

from .contract import FLAGS, RAW_ROWS, digest, rectangle_axes
from .matcher import anchor_geometry, corner_heights, fit_joint, query_strip, revalidate
from .side_candidates import (
    CONFIG,
    ExecutionStop,
    checkpoint,
    line_fit,
    profile,
    select_support,
    separation,
)

ANGLE_DEG = 3.0
MAX_BRANCHES = 96
MIN_WIDTH = 0.60
MAX_WIDTH = 1.25


def polygon_overlap(a, b):
    """Convex polygon intersection as a fraction of the smaller polygon."""

    def cross(x, y):
        return float(x[0] * y[1] - x[1] * y[0])

    def area(rows):
        if len(rows) < 3:
            return 0.0
        u = np.array(rows) - rows[0]
        return abs(sum(cross(v, w) for v, w in zip(u, np.roll(u, -1, axis=0)))) / 2

    orientation = np.sign(cross(b[1] - b[0], b[2] - b[1]))
    out = list(a)
    for p, q in zip(b, np.roll(b, -1, axis=0)):
        edge = q - p
        path, out = out, []
        if not path:
            return 0.0
        for x, y in zip(path, path[1:] + path[:1]):
            dx, dy = (orientation * cross(edge, v - p) for v in (x, y))
            if dx >= 0:
                out.append(x)
            if (dx >= 0) != (dy >= 0):
                out.append(x + (y - x) * dx / (dx - dy))
    return area(out) / max(min(area(a), area(b)), 1e-8)


def ownership_guard(xy, request, targets):
    target = request.get("target_opening_id")
    if (
        target
        and polygon_overlap(xy, np.asarray(targets[target]["geometry"]["xy"])) < 0.20
    ):
        return "WRONG_TARGET"
    if any(
        key != target and polygon_overlap(xy, np.asarray(row["geometry"]["xy"])) > 0.20
        for key, row in targets.items()
    ):
        return "COMPETING_AUTO_TARGET"
    return None


def equivalent(a, b, length):
    a_slope, a_offset, a_train, _ = a
    b_slope, b_offset, b_train, _ = b
    delta = a_offset - b_offset
    if max(abs(delta), abs(delta + (a_slope - b_slope) * length)) > 0.085:
        return False
    left = {r["slab"]: r for r in a_train}
    right = {r["slab"]: r for r in b_train}
    common = left.keys() & right.keys()
    if len(common) < 0.65 * min(len(left), len(right)):
        return False
    shared = 0
    for slab in common:
        p, q = left[slab], right[slab]
        intersect = len(set(p["face"]) & set(q["face"]))
        if abs(p["top_m"] - q["top_m"]) <= 0.10 and intersect >= 0.65 * min(
            len(p["face"]), len(q["face"])
        ):
            shared += 1
    return shared >= 0.65 * max(len(left), len(right))


def training_candidates(rows, length, width, cfg, check):
    train = [
        r for r in rows if r["role"] == "INNER_RIM_SUPPORTED" and r["slab"] % 2 == 0
    ]
    if len(train) < cfg.sections_per_split:
        return [], 0
    t = np.array([r["t_m"] for r in train])
    d = np.array([r["d_m"] for r in train])
    found = {}
    limit = np.tan(np.radians(ANGLE_DEG))
    for angle in [
        0,
        *np.arange(0.25, ANGLE_DEG + 0.001, 0.25),
        *np.arange(-0.25, -ANGLE_DEG - 0.001, -0.25),
    ]:
        check()
        slope = float(np.tan(np.radians(angle)))
        residual = d - slope * t
        for phase in (0.0, cfg.residual_m):
            bins = np.floor((residual + phase) / (2 * cfg.residual_m))
            for group, count in zip(*np.unique(bins, return_counts=True)):
                if count < cfg.sections_per_split:
                    continue
                offset = float(np.median(residual[bins == group]))
                support = select_support(train, slope, offset, cfg)
                if len(support) < cfg.sections_per_split:
                    continue
                slope_fit, offset_fit = line_fit(support)
                if (
                    abs(slope_fit) > limit
                    or max(abs(offset_fit), abs(offset_fit + slope_fit * length))
                    > width
                ):
                    continue
                support = select_support(train, slope_fit, offset_fit, cfg)
                if (
                    len(support) < cfg.sections_per_split
                    or support[-1]["t_m"] - support[0]["t_m"]
                    < length * cfg.coverage * 0.8
                ):
                    continue
                key = tuple((r["slab"], r["d_m"], r["top_m"]) for r in support)
                rms = float(
                    np.sqrt(
                        np.mean(
                            [
                                (r["d_m"] - offset_fit - slope_fit * r["t_m"]) ** 2
                                for r in support
                            ]
                        )
                    )
                )
                if key not in found or rms < found[key][3]:
                    found[key] = (slope_fit, offset_fit, support, rms)
    groups = []
    for mode in sorted(found.values(), key=lambda m: (-len(m[2]), m[3], m[1], m[0])):
        group = next(
            (g for g in groups if all(equivalent(m, mode, length) for m in g)), None
        )
        if group is None:
            groups.append([mode])
        else:
            group.append(mode)
    return [g[0] for g in groups], len(found)


def scan_separators(frame, xy, check, cfg):
    axes, lengths, orientation = rectangle_axes(xy)
    normals = orientation * np.column_stack((-axes[:, 1], axes[:, 0]))
    checked = 0
    for a, t, n, length, width in zip(xy, axes, normals, lengths, np.roll(lengths, -1)):
        lo, hi = 0.9, width - 0.9
        if hi <= lo:
            continue
        # Lanes overlap by the maximum end-to-center drift of the allowed line.
        step = 0.7
        drift = length * np.tan(np.radians(ANGLE_DEG)) / 2
        centers = np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / step)) + 1))
        for center in centers:
            check()
            half = drift + step / 2
            points, ids, _ = query_strip(
                frame, a + n * center, t, n, length, half + 0.68, cfg
            )
            # Isolate one directional lane and run the original physical primitive.
            # Context outside the interior is retained, but a selected side's rim
            # cannot qualify as an interior separator.
            _, modes = profile(
                points, ids, a + n * center, t, n, length, half, cfg, check
            )
            qualified = [r for r in modes if lo <= center + r["d_m"] <= hi]
            fitted, _ = training_candidates(qualified, length, half, cfg, check)
            for slope, intercept, train, _ in fitted:
                if not all(
                    lo <= center + v <= hi
                    for v in (intercept, intercept + slope * length)
                ):
                    continue
                options = [
                    r
                    for r in qualified
                    if r["role"] == "INNER_RIM_SUPPORTED" and r["slab"] % 2 == 1
                ]
                heldout = select_support(options, slope, intercept, cfg)
                if separation(train, heldout, length, cfg) == "QUALIFIED":
                    return dict(coverage="SEPARATOR_OBSERVED", lanes=checked + 1), dict(
                        origin=(a + n * center).tolist(),
                        slope=slope,
                        intercept=intercept,
                        train=train,
                        heldout=heldout,
                    )
            checked += 1
    return dict(coverage="ALL_PLANNED_LANES_SCANNED", lanes=checked), None


def feasible_directions(groups, lengths, cfg):
    """Prune only prefixes whose shared slope interval is already empty."""

    def walk(i, selected, lower, upper):
        if i == len(groups):
            yield selected
            return
        band = 4 * cfg.residual_m / lengths[i]
        for mode in groups[i]:
            lo = max(lower, mode["slope"] - band)
            hi = min(upper, mode["slope"] + band)
            if lo <= hi:
                yield from walk(i + 1, (*selected, mode), lo, hi)

    return walk(0, (), -float("inf"), float("inf"))


def guided_match(frame, xy, request, targets, cancelled=lambda: False):
    started = perf_counter()
    check = lambda: checkpoint(started + CONFIG.seconds, cancelled)
    r = dict(
        **FLAGS,
        match_state="NO_MATCH",
        execution_state="COMPLETED",
        reason="RAW_SUPPORT_INSUFFICIENT",
        geometry=None,
        sides=[],
        ownership="UNVERIFIED",
        source_kind="GUIDED_RAW",
        raw_rows=RAW_ROWS,
        algorithm="guided.2",
        rough_xy=xy.tolist(),
    )

    def reject(state, reason):
        r.update(match_state=state, reason=reason)
        return r

    try:
        guard = ownership_guard(xy, request, targets)
        if guard:
            return reject("AMBIGUOUS", guard)
        axes, lengths, orientation = rectangle_axes(xy)
        normals = orientation * np.column_stack((-axes[:, 1], axes[:, 0]))
        cfg = replace(CONFIG, angle_deg=ANGLE_DEG)
        widths = [
            min(
                MAX_WIDTH,
                max(
                    MIN_WIDTH,
                    frame.projection.s * 3,
                    0.30 + length * np.tan(np.radians(1.5)) / 2,
                ),
            )
            for length in lengths
        ]
        r["search_half_width_m"] = widths
        for name, a, t, n, length, width in zip(
            ("V0", "U1", "V1", "U0"), xy, axes, normals, lengths, widths
        ):
            check()
            pts, ids, box = query_strip(frame, a, t, n, length, width + 0.68, cfg)
            sections, rows = profile(pts, ids, a, t, n, length, width, cfg, check)
            fits, raw_count = training_candidates(rows, length, width, cfg, check)
            modes = [
                dict(
                    slope=m,
                    intercept_m=b,
                    train=train,
                    rms_m=rms,
                    train_sha256=digest(train),
                )
                for m, b, train, rms in fits
            ]
            r["sides"].append(
                dict(
                    side_id=name,
                    reason="TRAINING_SUPPORTED" if modes else "TRAINING_INSUFFICIENT",
                    mode_count=len(modes),
                    pre_alias_count=raw_count,
                    modes=modes,
                    observations=rows,
                    sections=sections,
                    aabb=box.tolist(),
                )
            )
        groups = [side["modes"] for side in r["sides"]]
        if any(not group for group in groups):
            return reject(
                "PARTIAL" if any(groups) else "NO_MATCH",
                "INCOMPLETE_PHYSICAL_TRAINING_SUPPORT",
            )
        count = prod(map(len, groups))
        r["joint_combinations"] = count
        feasible = []
        examined = 0
        for combination in feasible_directions(groups, lengths, cfg):
            check()
            examined += 1
            r["joint_direction_feasible_examined"] = examined
            if examined > MAX_BRANCHES:
                raise ExecutionStop("JOINT_COMBINATION_BUDGET")
            attempt = [
                dict(side, modes=[m]) for side, m in zip(r["sides"], combination)
            ]
            corrected, joint = fit_joint(
                xy, attempt, replace(cfg, move_m=max(widths)), check
            )
            if corrected is None:
                continue
            moves = [
                float(
                    max(abs((corrected[[i, (i + 1) % 4]] - xy[[i, (i + 1) % 4]]) @ n))
                )
                for i, n in enumerate(normals)
            ]
            if any(m > w for m, w in zip(moves, widths)):
                continue
            if ownership_guard(corrected, request, targets):
                continue
            feasible.append((corrected, attempt, joint, moves))
            if len(feasible) > 1:
                return reject("AMBIGUOUS", "MULTIPLE_PHYSICAL_RECTANGLES")
        if not feasible:
            return reject("PARTIAL", "JOINT_GEOMETRY_CONFLICT")
        corrected, sides, joint, moves = feasible[0]
        r["joint"] = joint
        r["endpoint_moves_m"] = moves
        if not revalidate(xy, corrected, sides, cfg):
            r["selected_sides"] = sides
            return reject("PARTIAL", "INDEPENDENT_HELDOUT_HAS_GAPS")
        r["selected_sides"] = sides
        coverage, separator = scan_separators(frame, corrected, check, cfg)
        r["coverage_audit"] = coverage
        if separator:
            r["separator_evidence"] = separator
            return reject("AMBIGUOUS", "INTERIOR_SEPARATOR")
        results = []
        for p in corrected:
            ids = frame.index.query_indexes(
                [p[0] - 0.8, p[1] - 0.8, p[0] + 0.8, p[1] + 0.8]
            )
            if len(ids) > cfg.slab_points:
                raise ExecutionStop("CORNER_POINT_BUDGET")
            results.append(ids)
        ids = np.unique(np.concatenate(results))
        z, z_evidence = corner_heights(
            frame.frame.xyz[ids].astype(np.float64), ids, corrected, sides, cfg
        )
        geometry = anchor_geometry(corrected, z, z_evidence)
        geometry.update(
            xyz=[[x, y, h] for (x, y), h in zip(geometry["xy"], z)],
            xy_complete=True,
            xyz_complete=all(h is not None for h in z),
        )
        check()
        r.update(
            match_state="GUIDED_REFINED_PENDING",
            reason="INDEPENDENT_RAW_SUPPORT",
            geometry=geometry,
        )
        return r
    except ExecutionStop as exc:
        reason = str(exc)
        r.update(
            match_state=None,
            reason=reason,
            execution_state=(
                reason if reason in ("CANCELLED", "TIMEOUT") else "BUDGET_EXCEEDED"
            ),
        )
        return r
