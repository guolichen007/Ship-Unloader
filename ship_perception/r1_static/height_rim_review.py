"""Offline multi-section Raw XYZ height-rim geometry review.

Height transitions localize an opening candidate. They never certify steel.
The B5 polygon and all acceptance statuses remain available separately.
"""

import warnings

import numpy as np
from scipy.stats import theilslopes

from .raw_topview_evidence import build_topview


def _section_transition(grid, u_bounds, rho_anchor, side_id, config):
    """Return robust U-section transition samples near an existing side."""
    u, v = grid.centers()
    u, v = u[0], v[:, 0]
    z, count = grid.median, grid.count
    half = config["boundary"]["profile_half_length_m"]
    step_u = 2 * half
    step_v = max(1, int(round(config["boundary"]["side_probe_m"] / grid.cell_m)))
    search = config["roi"]["boundary_search_m"]
    direction = 1 if side_id == "V1" else -1
    samples = []
    for center_u in np.arange(u_bounds[0] + half, u_bounds[1] - half, step_u):
        columns = (u >= center_u - half) & (u <= center_u + half)
        if columns.sum() < 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            section_z = np.nanmedian(np.where(count[:, columns] > 0,
                                              z[:, columns], np.nan), axis=1)
        candidates = []
        for index in np.flatnonzero(abs(v - rho_anchor) <= search):
            inside = index - direction * step_v
            outside = index + direction * step_v
            if min(inside, outside) < 0 or max(inside, outside) >= len(v):
                continue
            if not np.isfinite(section_z[[inside, outside]]).all():
                continue
            inside_fraction = np.mean(count[inside, columns] > 0)
            outside_fraction = np.mean(count[outside, columns] > 0)
            if min(inside_fraction, outside_fraction) < config["roi"]["min_enclosure_ratio"]:
                continue
            contrast = float(section_z[outside] - section_z[inside])
            if contrast < config["roi"]["opening_drop_m"]:
                continue
            candidates.append((float(v[index]), contrast))
        if candidates:
            peak = max(score for _, score in candidates)
            near_peak = [(rho, score) for rho, score in candidates if
                         score >= peak - config["roi"]["opening_drop_m"]]
            groups = []
            for item in near_peak:
                if groups and item[0] - groups[-1][-1][0] <= grid.cell_m * 1.5:
                    groups[-1].append(item)
                else:
                    groups.append([item])
            group = max(groups, key=lambda rows: (
                np.mean([score for _, score in rows]),
                -abs(np.median([rho for rho, _ in rows]) - rho_anchor)))
            samples.append([float(center_u),
                            float(np.median([rho for rho, _ in group])),
                            float(np.median([score for _, score in group]))])
    if len(samples) < config["boundary"]["profile_min_sections"]:
        return dict(accepted=False, reason="TOO_FEW_OCCUPIED_SECTIONS", samples=samples)
    xy = np.asarray(samples)
    fit = theilslopes(xy[:, 1], xy[:, 0])
    residual = xy[:, 1] - (fit.intercept + fit.slope * xy[:, 0])
    dispersion = float(np.median(np.abs(residual)))
    angle_deg = float(np.degrees(np.arctan(fit.slope)))
    accepted = (dispersion <= config["boundary"]["corner_join_m"] and
                abs(angle_deg) <= config["boundary"]["merge_angle_deg"] and
                xy[-1, 0] - xy[0, 0] >= config["roi"]["min_deck_span_m"] *
                config["boundary"]["profile_min_sections"])
    return dict(accepted=bool(accepted),
                reason="PERSISTENT_OPENING_TRANSITION" if accepted else
                "DISPERSED_OR_SHORT_TRANSITION",
                slope_deg=angle_deg, dispersion_m=dispersion,
                median_contrast_m=float(np.median(xy[:, 2])), samples=samples)


def _guard_review_displacement(detail, current, side_id, bootstrap, config):
    """Keep a B4 edge from jumping to a remote cargo or quay transition."""
    if not detail["accepted"] or bootstrap:
        return detail
    move = float(np.median(np.asarray(detail["samples"])[:, 1]) - current)
    inward = move if side_id == "V0" else -move
    if inward > config["boundary"]["corner_join_m"] or abs(move) > (
            config["roi"]["boundary_search_m"] / 2):
        detail["accepted"] = False
        detail["reason"] = "LARGE_MOVE_REQUIRES_STRUCTURAL_CORROBORATION"
    detail["candidate_delta_m"] = move
    return detail


def refine_height_review(points, vessel, b2, b4, config):
    """Share one refined axis across a vessel's review rectangles."""
    rectangles = vessel.get("rectangles", [])
    if not rectangles:
        return dict(status="NO_RECTANGLE", axis_delta_deg=0.0, rectangles={})
    ident = vessel["vessel_hypothesis_id"]
    initial_axes = (next((np.asarray(row["axes"]) for row in b4["rectangles"] if
                          row["vessel_hypothesis_id"] == ident), None)
                    if b4 is not None else None)
    if initial_axes is None:
        initial_axes = next((np.asarray(row["local_axes"]) for row in
                             b2["scene"]["vessel_hypotheses"] if
                             row["vessel_hypothesis_id"] == ident), None)
    if initial_axes is None:
        return dict(status="NO_RECTANGLE_AXES", axis_delta_deg=0.0, rectangles={})
    grid = build_topview(points, initial_axes, config)["fine"]
    evidence = {}
    slopes = []
    for row in rectangles:
        box = np.asarray(row["polygon_after"]) @ initial_axes.T
        u_bounds = (float(box[:, 0].min()), float(box[:, 0].max()))
        details = {}
        for side_id in ("V0", "V1"):
            # B5's prior stage can be a better search anchor than a later
            # constrained side, especially for a composite B2 seed.
            current = float(box[:, 1].min() if side_id == "V0" else box[:, 1].max())
            prior = row.get("side_stage_arbitration", {}).get(side_id, {}).get("r2g_rho")
            anchor = float(prior) if row["hatch_id"].startswith("I") and prior is not None else current
            detail = _section_transition(grid, u_bounds, anchor, side_id, config)
            detail = _guard_review_displacement(
                detail, current, side_id, row["hatch_id"].startswith("I"), config)
            detail["rho_before_m"] = current
            detail["rho_search_anchor_m"] = anchor
            details[side_id] = detail
            if detail["accepted"] and side_id == "V1":
                slopes.append(detail["slope_deg"])
        evidence[row["hatch_id"]] = details
    axis_delta = float(np.median(slopes)) if slopes else 0.0
    initial_angle = float(np.arctan2(initial_axes[0, 1], initial_axes[0, 0]))
    final_angle = initial_angle + np.radians(axis_delta)
    axes = np.asarray([[np.cos(final_angle), np.sin(final_angle)],
                       [-np.sin(final_angle), np.cos(final_angle)]])
    output = {}
    for row in rectangles:
        projected = np.asarray(row["polygon_after"]) @ axes.T
        u0, u1 = float(projected[:, 0].min()), float(projected[:, 0].max())
        v0, v1 = float(projected[:, 1].min()), float(projected[:, 1].max())
        details = evidence[row["hatch_id"]]
        for side_id in ("V0", "V1"):
            detail = details[side_id]
            if not detail["accepted"]:
                continue
            samples = np.asarray(detail["samples"])
            world = samples[:, :2] @ initial_axes
            rho = float(np.median((world @ axes.T)[:, 1]))
            if side_id == "V0":
                v0 = rho
            else:
                v1 = rho
        if v1 <= v0 + config["boundary"]["corner_join_m"]:
            output[row["hatch_id"]] = dict(
                status="REJECTED_DEGENERATE_GEOMETRY", polygon=row["polygon_after"],
                axes=initial_axes.tolist(), sides=details)
            continue
        polygon = np.asarray([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]) @ axes
        output[row["hatch_id"]] = dict(
            status="HEIGHT_TRANSITION_REVIEW_NOT_STEEL",
            polygon=polygon.tolist(), axes=axes.tolist(), sides=details,
            axis_delta_deg=axis_delta)
    return dict(status="HEIGHT_TRANSITION_REVIEW_NOT_STEEL",
                axis_initial_deg=float(np.degrees(initial_angle)),
                axis_delta_deg=axis_delta,
                axis_review_deg=float(np.degrees(final_angle)),
                rectangles=output)
