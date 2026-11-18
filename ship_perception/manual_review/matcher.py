"""Local rectangle inference. Automatic overlays supply no measurement values."""

from dataclasses import replace
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from .contract import FLAGS, RAW_ROWS, VERSION, digest, rectangle_axes
from .side_candidates import (
    CONFIG,
    ExecutionStop,
    analyze_side,
    checkpoint,
    fit_modes,
    profile,
    raw_set,
    select_support,
    separation,
)


def corners(q, center, e, f):
    cx, cy, theta, length, width = q
    a = np.cos(theta) * e + np.sin(theta) * f
    b = -np.sin(theta) * e + np.cos(theta) * f
    c = center + cx * e + cy * f
    u, v = length * a / 2, width * b / 2
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


def anchor_geometry(xy, z=None, height=None):
    axes, sizes, _ = rectangle_axes(xy)
    long_first = sizes[0] >= sizes[1]
    axis = axes[0 if long_first else 1]
    return dict(
        xy=xy.tolist(),
        z=z if z is not None else [None] * 4,
        corner_z=(
            height
            if height is not None
            else [
                {"reason": "MANUAL_NO_Z_EVIDENCE", "raw_indexes": []} for _ in range(4)
            ]
        ),
        corner_ids=["P1", "P2", "P3", "P4"],
        side_ids=["V0", "U1", "V1", "U0"],
        side_vertices={
            "V0": ["P1", "P2"],
            "U1": ["P2", "P3"],
            "V1": ["P3", "P4"],
            "U0": ["P4", "P1"],
        },
        units="m",
        dimension_definition="horizontal_input_xy",
        length_m=float(max(sizes[:2])),
        width_m=float(min(sizes[:2])),
        axis_deg=float(np.degrees(np.arctan2(axis[1], axis[0])) % 180),
        near_square_axis_ambiguous=bool(
            abs(sizes[0] - sizes[1]) <= 0.05 * max(sizes[:2])
        ),
        corner_anchor="request_vertex_order",
    )


def fit_joint(xy, sides, config, check):
    axes, sizes, orientation = rectangle_axes(xy)
    inward = orientation * np.column_stack((-axes[:, 1], axes[:, 0]))
    center, e, f = xy.mean(axis=0), axes[0], axes[1]
    groups = []
    for a, t, n, side in zip(xy, axes, inward, sides):
        train = side["modes"][0]["train"]
        groups.append(
            np.asarray([a + r["t_m"] * t + r["d_m"] * n - center for r in train])
        )

    def residual(q):
        check()
        frame_xy = corners(q, np.zeros(2), e, f)
        edge = np.roll(frame_xy, -1, axis=0) - frame_xy
        edge /= np.linalg.norm(edge, axis=1)[:, None]
        norm = np.column_stack((-edge[:, 1], edge[:, 0]))
        return np.concatenate(
            [
                (g - frame_xy[i]) @ norm[i] / np.sqrt(len(g)) / config.residual_m
                for i, g in enumerate(groups)
            ]
        )

    initial = np.array([0.0, 0.0, 0.0, sizes[0], sizes[1]])
    move, tilt = config.move_m, np.radians(config.angle_deg)
    low = np.array(
        [-move, -move, -tilt, max(1, sizes[0] - 2 * move), max(1, sizes[1] - 2 * move)]
    )
    high = np.array(
        [move, move, tilt, min(250, sizes[0] + 2 * move), min(250, sizes[1] + 2 * move)]
    )
    started = perf_counter()
    fixed_ids = [0, 1, 3, 4]

    def fixed_residual(v):
        q = initial.copy()
        q[fixed_ids] = v
        return residual(q)

    fixed_fit = least_squares(
        fixed_residual,
        initial[fixed_ids],
        bounds=(low[fixed_ids], high[fixed_ids]),
        loss="soft_l1",
        max_nfev=25,
    )
    fixed = initial.copy()
    fixed[fixed_ids] = fixed_fit.x
    refined_fit = least_squares(
        residual, fixed, bounds=(low, high), loss="soft_l1", max_nfev=35
    )
    refined = refined_fit.x

    def verify_basin(candidate):
        points = corners(candidate, center, e, f)
        for i, (a, t, n, side) in enumerate(zip(xy, axes, inward, sides)):
            mode = side["modes"][0]
            end = points[[i, (i + 1) % 4]]
            displacement = (end - xy[[i, (i + 1) % 4]]) @ n
            along = (end - a) @ t
            if (
                np.max(np.abs(displacement)) > move
                or np.max(
                    np.abs(displacement - (mode["slope"] * along + mode["intercept_m"]))
                )
                > 2 * config.residual_m
            ):
                return False
        return True

    fixed_ok, refined_ok = verify_basin(fixed), verify_basin(refined)
    use_refined = (
        refined_ok
        and refined_fit.success
        and (not fixed_ok or refined_fit.cost < fixed_fit.cost - 1e-10)
    )
    chosen = refined if use_refined else fixed
    report = dict(
        fixed_q=fixed.tolist(),
        refined_q=refined.tolist(),
        fixed_cost=float(fixed_fit.cost),
        refined_cost=float(refined_fit.cost),
        fixed_basin=fixed_ok,
        refined_basin=refined_ok,
        selected="REFINED" if use_refined else "FIXED",
        seconds=perf_counter() - started,
    )
    if not use_refined and not (fixed_ok and fixed_fit.success):
        return None, report
    return corners(chosen, center, e, f), report


def revalidate(xy, corrected, sides, cfg):
    axes, sizes, orientation = rectangle_axes(xy)
    normals = orientation * np.column_stack((-axes[:, 1], axes[:, 0]))
    for i, (a, t, n, length, side) in enumerate(zip(xy, axes, normals, sizes, sides)):
        final = corrected[[i, (i + 1) % 4]] - a
        t1, t2 = final @ t
        d1, d2 = final @ n
        slope = (d2 - d1) / (t2 - t1)
        intercept = d1 - slope * t1
        mode = side["modes"][0]
        options = [
            o
            for o in side["observations"]
            if o["slab"] % 2 == 1 and o["role"] == "INNER_RIM_SUPPORTED"
        ]
        support = select_support(options, slope, intercept, cfg)
        reason = separation(mode["train"], support, length, cfg)
        side["joint_heldout"] = support
        side["joint_heldout_sha256"] = digest(support)
        side["joint_reason"] = reason
    return all(s["joint_reason"] == "QUALIFIED" for s in sides)


def separators(p, ids, corrected, cfg, check):
    """Veto a distributed physical face through the selected interior."""
    axes, lengths, orientation = rectangle_axes(corrected)
    normals = orientation * np.column_stack((-axes[:, 1], axes[:, 0]))
    found = []
    for a, t, n, length, width in zip(
        corrected, axes, normals, lengths, np.roll(lengths, -1)
    ):
        half = width / 2 - 0.9
        if half <= 0.12:
            continue
        interior = a + n * width / 2
        _, rows = profile(p, ids, interior, t, n, length, half, cfg, check)
        rows = [
            r
            for r in rows
            if abs(r["d_m"]) < half and r["role"] == "INNER_RIM_SUPPORTED"
        ]
        config = replace(cfg, move_m=half)
        for slope, intercept, train, _ in fit_modes(rows, length, config, check):
            test = select_support(
                [r for r in rows if r["slab"] % 2 == 1], slope, intercept, config
            )
            if separation(train, test, length, config) == "QUALIFIED":
                found.append(
                    dict(
                        edge_origin=interior.tolist(),
                        tangent=t.tolist(),
                        normal=n.tolist(),
                        slope=slope,
                        intercept_m=intercept,
                        train=train,
                        heldout=test,
                    )
                )
    return found


def corner_heights(p, ids, xy, sides, cfg):
    axes, lengths, orientation = rectangle_axes(xy)
    normals = orientation * np.column_stack((-axes[:, 1], axes[:, 0]))
    estimates, records = [], []
    for i, corner in enumerate(xy):
        delta = p[:, :2] - corner
        planes = []
        for side, sign in ((i, 1), ((i - 1) % 4, -1)):
            station = delta @ axes[side] * sign
            offset = delta @ normals[side]
            mode = sides[side]["modes"][0]
            samples = sorted(
                mode["train"] + sides[side]["joint_heldout"], key=lambda r: r["t_m"]
            )
            reference = samples[0 if sign == 1 else -1]["top_m"]
            close = (
                (station > 0) & (station <= 0.40) & (offset < -0.09) & (offset >= -0.46)
            )
            close_ids = np.flatnonzero(close)
            near = close_ids[np.abs(p[close_ids, 2] - reference) <= cfg.cap_band_m]
            distinct = np.unique(p[near], axis=0)
            conflict = (
                np.count_nonzero(np.abs(p[close_ids, 2] - reference) > cfg.cap_band_m)
                >= 4
            )
            if len(distinct) >= 5 and not conflict and np.ptp(distinct[:, 2]) <= 0.08:
                planes.append((float(np.median(distinct[:, 2])), ids[near].tolist()))
        if len(planes) == 2 and abs(planes[0][0] - planes[1][0]) <= cfg.cap_band_m:
            estimates.append(float(np.median([planes[0][0], planes[1][0]])))
            records.append(
                dict(
                    reason="ADJACENT_RIM_CAPS_AT_CORNER",
                    raw_indexes=sorted(set(planes[0][1] + planes[1][1])),
                    side_z_m=[p[0] for p in planes],
                )
            )
        else:
            estimates.append(None)
            records.append(
                dict(
                    reason=(
                        "RIM_CAPS_COMPETING"
                        if len(planes) == 2
                        else "CORNER_CAP_SUPPORT_INSUFFICIENT"
                    ),
                    raw_indexes=sorted({row for _, rs in planes for row in rs}),
                )
            )
    return estimates, records


def query_strip(frame, a, t, n, length, width, cfg):
    bound = n * (width + 0.001)
    end = a + t * length
    vertices = np.array([a - bound, a + bound, end - bound, end + bound])
    box = np.r_[vertices.min(axis=0), vertices.max(axis=0)]
    ids = frame.index.query_indexes(box)
    xy = frame.frame.xyz[ids, :2].astype(np.float64)
    shift = xy - a
    u, v = shift @ t, shift @ n
    keep = (u >= -0.001) & (u <= length + 0.001) & (v >= -width) & (v <= width)
    ids = ids[keep]
    if len(ids) > cfg.roi_points:
        raise ExecutionStop("STRIP_POINT_BUDGET")
    return frame.frame.xyz[ids].astype(np.float64), ids, box


def match(frame, xy, req, config=CONFIG, cancelled=lambda: False):
    started = perf_counter()
    check = lambda: checkpoint(started + config.seconds, cancelled)
    result = dict(
        **FLAGS,
        schema="manual_suggestion.1",
        execution_state="COMPLETED",
        match_state="NO_MATCH",
        reason="NO_LOCAL_PHYSICAL_EVIDENCE",
        source_kind="USER_SELECTION",
        human_review="PENDING",
        ownership="LOCAL_ONLY_UNVERIFIED",
        rough_xy=xy.tolist(),
        geometry=None,
        geometry_complete=False,
        sides=[],
        raw_rows=RAW_ROWS,
        config=config.audit(),
        version=VERSION,
        separators=[],
    )
    try:
        axes, sizes, orientation = rectangle_axes(xy)
        normals = orientation * np.column_stack((-axes[:, 1], axes[:, 0]))
        for name, a, t, n, length in zip(
            ("V0", "U1", "V1", "U0"), xy, axes, normals, sizes
        ):
            check()
            points, ids, box = query_strip(
                frame, a, t, n, length, config.move_m + 0.68, config
            )
            report = analyze_side(points, ids, a, t, n, length, config, check)
            result["sides"].append(
                dict(
                    side_id=name,
                    rough_origin_xy=a.tolist(),
                    tangent=t.tolist(),
                    inward_normal=n.tolist(),
                    roi_box_xy=box.tolist(),
                    roi_count=len(ids),
                    **report,
                )
            )
        sides = result["sides"]
        supports = [
            (
                raw_set(r["modes"][0]["train"] + r["modes"][0]["heldout"])
                if len(r["modes"]) == 1
                else set()
            )
            for r in sides
        ]
        result["cross_side_shared_counts"] = [
            [len(a & b) for b in supports] for a in supports
        ]
        if any(s["role"] == "CONFLICTING" for s in sides):
            result.update(match_state="AMBIGUOUS", reason="SIDE_COMPETITION")
        elif not all(side["qualified"] for side in sides):
            if any(side["modes"] for side in sides):
                result.update(match_state="PARTIAL", reason="SIDE_EVIDENCE_INCOMPLETE")
        else:
            corrected, joint = fit_joint(xy, sides, config, check)
            result["joint"] = joint
            if corrected is None or not revalidate(xy, corrected, sides, config):
                result.update(match_state="PARTIAL", reason="JOINT_REVALIDATION_FAILED")
            else:
                pad = config.move_m + 1.0
                box = np.r_[corrected.min(axis=0) - pad, corrected.max(axis=0) + pad]
                ids = frame.index.query_indexes(box)
                if len(ids) > config.roi_points:
                    raise ExecutionStop("INTERIOR_POINT_BUDGET")
                points = frame.frame.xyz[ids].astype(np.float64)
                result["separators"] = separators(points, ids, corrected, config, check)
                if result["separators"]:
                    result.update(match_state="AMBIGUOUS", reason="INTERIOR_SEPARATOR")
                elif req["operation"] == "REFINE":
                    result.update(
                        match_state="PARTIAL", reason="TARGET_MEMBERSHIP_UNVERIFIED"
                    )
                else:
                    z, height = corner_heights(points, ids, corrected, sides, config)
                    result.update(
                        match_state="MATCHED_REVIEW",
                        reason="LOCAL_PHYSICAL_EVIDENCE",
                        geometry=anchor_geometry(corrected, z, height),
                        geometry_complete=True,
                        source_kind="RAW_REFINED",
                    )
        check()
    except ExecutionStop as stop:
        result.update(
            execution_state=(
                str(stop)
                if str(stop) in ("TIMEOUT", "CANCELLED")
                else "BUDGET_EXCEEDED"
            ),
            match_state=None,
            reason=str(stop),
            geometry=None,
            geometry_complete=False,
            source_kind="USER_SELECTION",
        )
    result["processing_s"] = perf_counter() - started
    return result
