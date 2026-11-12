"""Held-out spatial sections for non-certified review geometry only."""

import numpy as np
from .interior_rim_search import raw_transition_nominations, transition_families
from .local_rim_evidence import digest
from .sections import physical_sections
from .review_numerics import roundoff_m

SIDES = {
    "V0": (1, 0, 1, [0, 1]),
    "V1": (1, 0, -1, [3, 2]),
    "U0": (0, 1, 1, [0, 3]),
    "U1": (0, 1, -1, [1, 2]),
}


def evaluate_section(section, slope, intercept, config, position_noise_m=None):
    b = config["boundary"]
    target = slope * section["station_m"] + intercept
    proposals = [m for m in section["transitions"] if m["signed_height_change_m"] < 0]
    if not proposals:
        return dict(
            passed=False,
            first_failure="NO_INWARD_HEIGHT_TRANSITION",
            station_m=section["station_m"],
        )
    match = min(proposals, key=lambda m: abs(m["offset_m"] - target))
    residual = abs(match["offset_m"] - target) / np.sqrt(1 + slope * slope)
    sigma = max(
        match["uncertainty_m"],
        (
            config["geometry"]["candidate_voxel_m"]
            if position_noise_m is None
            else position_noise_m
        ),
    )
    profile = section["coarse_profile"]
    outside = [
        r
        for r in profile
        if target - b["side_probe_m"]
        <= r["t_m"]
        < target - config["geometry"]["plane_inlier_m"]
        and r["raw_count"] >= b["profile_min_points"]
    ]
    span = np.ptp([r["t_m"] for r in outside]) if len(outside) > 1 else 0.0
    # This is a 2-D plateau witness, never the 3-D deck/steel qualification.
    if len(outside) >= 2:
        x = np.array([r["t_m"] for r in outside])
        z = np.array([r["median_z_m"] for r in outside])
        s = np.linalg.lstsq(
            np.column_stack((x - x.mean(), np.ones(len(x)))), z, rcond=None
        )[0][0]
        rough = float(np.median(abs(z - np.median(z))))
    else:
        s = np.inf
        rough = np.inf
    plateau = bool(
        span
        >= b["profile_deck_support_span_m"] - config["geometry"]["candidate_voxel_m"]
        and abs(s)
        <= np.tan(np.deg2rad(b["profile_deck_slope_max_deg"]))
        + rough / max(span, config["geometry"]["candidate_voxel_m"])
    )
    opposites = [
        m
        for m in section["transitions"]
        if m["signed_height_change_m"] > 0
        and 0 < target - m["offset_m"] <= b["beam_max_width_m"]
    ]
    ridge = any(
        abs(m["outside_height_m"] - match["inside_height_m"])
        <= config["geometry"]["plane_inlier_m"]
        for m in opposites
    )
    numerical = roundoff_m(target, match["offset_m"])
    passed = bool(residual <= sigma + numerical and not ridge)
    return dict(
        passed=passed,
        first_failure=(
            "PASS"
            if passed
            else (
                "HOLDOUT_POSITION"
                if residual > sigma + numerical
                else "CARGO_RIDGE_OPPOSITE_POLARITY"
            )
        ),
        station_m=section["station_m"],
        position_m=match["offset_m"],
        residual_m=residual,
        uncertainty_m=sigma,
        polarity=match["signed_height_change_m"],
        outside_plateau=plateau,
        outside_slope=float(s) if np.isfinite(s) else None,
        raw_indexes=match["profile_raw_indexes"],
        raw_digest=match["raw_index_digest"],
    )


def heldout_side(points, indexes, core, member, side, polygon, envelope, config):
    axis, along, sign, pair = SIDES[side]
    b = config["boundary"]
    g = config["geometry"]
    owned = points[member]
    lo, hi = np.quantile(owned[:, along], [0, 1])
    length = hi - lo
    if length <= b["line_min_length_m"]:
        return dict(
            state="REVIEW_UNRESOLVED",
            first_failure="INSUFFICIENT_OBSERVED_SPAN",
            candidates=[],
        )
    anchor = float(np.median(points[core, axis]) * sign)
    outer = float(envelope[axis] if sign == 1 else -envelope[axis + 2])
    q = np.column_stack((points[:, along] - lo, sign * points[:, axis], points[:, 2]))
    edges = physical_sections((0.0, length), g["candidate_voxel_m"], config)
    sections = []
    for i, (a, z) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (
            (q[:, 0] >= a)
            & ((q[:, 0] <= z) if i == len(edges) - 2 else (q[:, 0] < z))
            & (q[:, 1] >= outer)
            & (q[:, 1] <= anchor)
        )
        modes, profile = raw_transition_nominations(
            q[mask], indexes[mask], (a, z), (outer, anchor), config
        )
        profile.update(
            section_id=i,
            station_m=(a + z) / 2,
            transitions=modes,
            source_raw_indexes=indexes[mask].tolist(),
        )
        sections.append(profile)
    # Alternating non-overlapping physical slabs, fixed before nominations.
    train = [s for i, s in enumerate(sections) if i % 2 == 0]
    validation = [s for i, s in enumerate(sections) if i % 2]
    training = [
        dict(
            s,
            transitions=[
                m for m in s["transitions"] if m["signed_height_change_m"] < 0
            ],
        )
        for s in train
    ]
    families = transition_families(
        training, length, config, position_tolerance_m=g["candidate_voxel_m"]
    )
    t_ids = (
        np.concatenate([np.asarray(s["source_raw_indexes"], np.int64) for s in train])
        if train
        else np.array([], np.int64)
    )
    v_ids = (
        np.concatenate(
            [np.asarray(s["source_raw_indexes"], np.int64) for s in validation]
        )
        if validation
        else np.array([], np.int64)
    )
    assert not np.intersect1d(t_ids, v_ids).size
    p = np.asarray(polygon)[pair]
    s = p[:, along] - lo
    t = sign * p[:, axis]
    old_slope = (t[1] - t[0]) / (s[1] - s[0])
    old_intercept = t[0] - old_slope * s[0]
    old_rows = [
        evaluate_section(row, old_slope, old_intercept, config) for row in validation
    ]
    old_coverage = sum(r["passed"] for r in old_rows) / max(1, len(validation))
    models = []
    for f in families:
        validations = [
            evaluate_section(
                row,
                f["slope"],
                f["intercept_m"],
                config,
                max(g["candidate_voxel_m"], f["profile_residual_m"]),
            )
            for row in validation
        ]
        n = sum(r["passed"] for r in validations)
        train_n = f["independent_sections"]
        coverage = (n + train_n) / max(1, len(sections))
        good = [r for r in validations if r["passed"]]
        sigma = max(
            g["plane_inlier_m"],
            f["profile_residual_m"],
            max((r["residual_m"] for r in good), default=b["profile_max_residual_m"]),
        )
        position = sign * (f["intercept_m"] + f["slope"] * length / 2)
        attachment_roundoff = roundoff_m(points[core, :2], f["intercept_m"])
        inside = (
            sign * points[core, axis]
            >= f["slope"] * (points[core, along] - lo)
            + f["intercept_m"]
            - sigma
            - attachment_roundoff
        )
        attachment = bool(
            len(points[core]) >= b["profile_min_points"] and np.all(inside)
        )
        total = train_n + n
        failures = []
        if train_n < 2:
            failures.append("INSUFFICIENT_TRAIN_SECTIONS")
        if n < 2:
            failures.append("INSUFFICIENT_HOLDOUT_SECTIONS")
        if total < b["profile_min_sections"]:
            failures.append("INSUFFICIENT_DISJOINT_SPATIAL_SECTIONS")
        if not attachment:
            failures.append("SPATIAL_CORE_WOULD_BE_CLIPPED")
        if not good:
            failures.append("POLARITY_OR_OUTSIDE_PLATEAU")
        qualified = not failures
        width_sigma = sigma + abs(f["slope"]) * length / 2
        model = dict(
            mode_id="REVIEW_TRAIN:"
            + digest(
                np.unique([i for s in f["sections"] for i in s["profile_raw_indexes"]])
            ),
            side=side,
            axis=axis,
            sign=sign,
            position_m=float(position),
            slope=float(f["slope"]),
            intercept_m=float(f["intercept_m"]),
            along_origin_m=float(lo),
            along_span_m=[float(lo), float(hi)],
            source_family="HELDOUT_HEIGHT_TRANSITION_AND_SPATIAL_OWNERSHIP",
            unobserved_or_estimated=True,
            steel_certified=False,
            review_state="REVIEW_ESTIMATED" if qualified else "REVIEW_UNRESOLVED",
            qualified_for_review=qualified,
            train_section_ids=[
                train[s["section_id"]]["section_id"] for s in f["sections"]
            ],
            holdout_section_ids=[
                s["section_id"] for s, r in zip(validation, validations) if r["passed"]
            ],
            train_raw_digest=digest(t_ids),
            holdout_raw_digest=digest(v_ids),
            train_holdout_overlap=0,
            nomination_raw_indexes=sorted(
                {i for s in f["sections"] for i in s["profile_raw_indexes"]}
            ),
            holdout_evidence=validations,
            train_coverage=f["observed_section_coverage"],
            holdout_coverage=n / max(1, len(validation)),
            coverage=coverage,
            position_uncertainty_m=float(width_sigma),
            position_interval_m=[
                float(position - width_sigma),
                float(position + width_sigma),
            ],
            interval_semantic="OBSERVATION_AND_RECTANGLE_APPROXIMATION_NOT_SENSOR_CALIBRATED",
            opening_side_spatial_attachment=attachment,
            attachment_numerical_roundoff_m=attachment_roundoff,
            steel_face_contact=False,
            first_failure=failures[0] if failures else "PASS",
            failure_gates=failures,
            grade=(
                ("B" if coverage >= b["min_edge_coverage"] else "C")
                if qualified
                else "UNRESOLVED"
            ),
            score=float(
                n
                / max(1, len(validation))
                * coverage
                / (1 + sigma / b["profile_max_residual_m"])
            ),
            old_holdout_coverage=old_coverage,
        )
        n = np.zeros(2)
        n[along] = -f["slope"]
        n[axis] = sign
        n /= np.linalg.norm(n)
        uv = np.zeros((2, 2))
        uv[:, along] = [lo, hi]
        uv[:, axis] = sign * (f["intercept_m"] + f["slope"] * np.array([0.0, length]))
        model.update(
            normal_uv=n.tolist(),
            rho=float(uv[0] @ n),
            predicted_endpoints_uv=uv.tolist(),
            source_point_indexes=[],
            profile_source_point_indexes=model["nomination_raw_indexes"],
        )
        models.append(model)
    source_supports = {}

    def intern(row, field):
        if field in row:
            ids = row.pop(field)
            ref = digest(np.asarray(ids, np.int64))
            source_supports.setdefault(ref, ids)
            row["raw_support_ref"] = ref

    for slab in sections:
        slab.pop("source_raw_indexes", None)
        for transition in slab["transitions"]:
            intern(transition, "profile_raw_indexes")
    for row in old_rows:
        intern(row, "raw_indexes")
    for model in models:
        for row in model["holdout_evidence"]:
            intern(row, "raw_indexes")
    return dict(
        state="CANDIDATES_EVALUATED",
        side=side,
        training_section_ids=[s["section_id"] for s in train],
        holdout_section_ids=[s["section_id"] for s in validation],
        training_digest=digest(t_ids),
        holdout_digest=digest(v_ids),
        train_holdout_overlap=0,
        old_holdout_evidence=old_rows,
        old_holdout_coverage=old_coverage,
        candidates=models,
        section_profiles=sections,
        source_supports=source_supports,
    )
