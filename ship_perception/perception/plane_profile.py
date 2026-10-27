"""Exact existing deck-profile recipe with batched per-bin quantiles.

Only the calculation changes: Raw returns, bins, quantile interpolation,
bracket tests and competing-break rejection match boundary_refinement._profile.
"""

import numpy as np


def observed_profile(points, tree, center, tangent, inward, plane, config):
    row = profile_record(
        points, tree, center, tangent, inward, plane, config, "LOCAL_RAW"
    )
    return row if row["valid"] else None


def profile_record(points, tree, center, tangent, inward, plane, config, profile_id):
    row = dict(
        profile_id=profile_id,
        sample_count=0,
        deck_support_count=0,
        lower_support_count=0,
        break_position=None,
        break_strength=None,
        residual=None,
        valid=False,
        reason="INSUFFICIENT_PROFILE_POINTS",
        face_position=None,
    )
    b = config["boundary"]
    half, width, step = (
        b["profile_half_length_m"],
        b["profile_half_width_m"],
        b["profile_step_m"],
    )
    ids = tree.query_ball_point(center, half + width)
    if not ids:
        return row
    local = points[ids].astype(float)
    delta = local[:, :2] - center
    along = delta @ inward
    use = (abs(delta @ tangent) <= width) & (abs(along) <= half)
    if int(use.sum()) < b["profile_min_points"]:
        return row
    local, along = local[use], along[use]
    residual = local @ plane[0] + plane[1]
    precision = (
        np.finfo(points.dtype).eps * max(1.0, float(abs(local[:, :2]).max())) * 2
        if np.issubdtype(points.dtype, np.floating)
        else 0.0
    )
    scaled = along / step
    nearest = np.rint(scaled)
    bins = np.floor(
        np.where(abs(scaled - nearest) <= precision / step, nearest, scaled)
    ).astype(int)
    # Sorting within each bin provides the same NumPy linear quantile without
    # thousands of independent np.quantile calls and Python allocations.
    order = np.lexsort((residual, bins))
    keys, starts, counts = np.unique(bins[order], return_index=True, return_counts=True)
    rank = (counts - 1) * b["profile_lower_quantile"]
    low = np.floor(rank).astype(int)
    high = np.ceil(rank).astype(int)
    values = residual[order]
    fraction = rank - low
    delta = values[starts + high] - values[starts + low]
    quantiles = np.where(
        fraction >= 0.5,
        values[starts + high] - delta * (1 - fraction),
        values[starts + low] + delta * fraction,
    )
    positions = (keys + 0.5) * step
    face = (
        (counts >= max(3, config["geometry"]["normal_min_points"] // 2))
        & (
            values[starts + counts - 1] - values[starts]
            >= b["face_vertical_span_min_m"]
        )
        & (values[starts + counts - 1] >= -config["roi"]["support_band_m"])
        & (values[starts] <= -b["profile_min_drop_m"])
    )
    if face.any():
        choices = positions[face]
        row["face_position"] = (
            center + choices[np.argmin(abs(choices))] * inward
        ).tolist()
    deck = abs(quantiles) <= config["roi"]["support_band_m"]
    lower = quantiles < -b["profile_min_drop_m"]
    row.update(
        sample_count=len(keys),
        deck_support_count=int(deck.sum()),
        lower_support_count=int(lower.sum()),
    )
    if len(keys) < b["profile_min_points"] or deck.sum() < 3 or lower.sum() < 3:
        row["reason"] = "DECK_OR_LOWER_NOT_OBSERVED"
        return row
    options = []
    for index in np.flatnonzero(lower):
        prior = np.flatnonzero(
            deck[:index]
            & (positions[index] - positions[:index] > 0)
            & (
                positions[index] - positions[:index]
                <= b["profile_max_bracket_m"] + precision
            )
        )
        if not len(prior):
            continue
        j = prior[-1]
        preceding = np.flatnonzero(deck[: j + 1])
        if len(preceding) < 3 or lower[index:].sum() < 3:
            continue
        if (
            positions[preceding[-1]] - positions[preceding[0]]
            < b["profile_deck_support_span_m"]
        ):
            continue
        if lower[j + 1 : index].any():
            continue
        strength = (quantiles[j] - quantiles[index]) / (positions[index] - positions[j])
        if strength < b["profile_step_slope_min"]:
            continue
        options.append(
            (
                (positions[j] + positions[index]) / 2,
                strength,
                abs(positions[index] - positions[j]),
            )
        )
    if not options:
        row["reason"] = "NO_OBSERVED_DECK_TO_LOWER_BREAK"
        return row
    options.sort(key=lambda item: (abs(item[0]), -item[1]))
    best = options[0]
    if any(
        abs(other[0] - best[0]) > b["profile_breakpoint_stability_m"]
        for other in options
    ):
        row["reason"] = "COMPETING_BREAKS"
        return row
    row.update(
        break_position=(center + best[0] * inward).tolist(),
        break_strength=float(best[1]),
        residual=float(best[2] / 2),
        valid=True,
        reason=None,
    )
    return row
