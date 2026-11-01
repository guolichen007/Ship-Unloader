"""Trace existing Raw surface pairs without supplying a new boundary estimate.

The paired midpoint is a geometric step observation. Neither endpoint's Z is
a rim-top elevation, and a contextual return never acquires vessel ownership.
"""

import hashlib
import numpy as np


def trace_section_pairs(measurement, section, axis, span, sign, config):
    """Missing trace inputs must not veto an existing Raw observation."""
    anchor = section.get("section_record", {}).get("rho")
    primary = getattr(getattr(measurement, "local", None), "primary_cache", None)
    if primary is None:
        primary = getattr(measurement, "local", None)
    if (
        anchor is None
        or not all(
            hasattr(measurement, k) for k in ("spatial", "raw_uvz", "raw_point_indexes")
        )
        or not hasattr(getattr(primary, "vessel", None), "point_indexes")
    ):
        return dict(
            status="UNAVAILABLE_ORIGINAL_RAW_PAIR_TRACE_INPUT",
            decision_input=False,
            corner_z_source=False,
            steel_certified=False,
            ownership_promoted=False,
            raw_midpoint_rho_m=None,
            source_pair_point_indexes=[],
            pairs=[],
        )
    from .rim_top_height import owned_measurement_mask

    width = config["geometry"]["candidate_voxel_m"]
    box = [0.0] * 4
    box[axis] = np.nextafter(anchor - width, -np.inf)
    box[axis + 2] = np.nextafter(anchor + width, np.inf)
    box[1 - axis], box[3 - axis] = span
    ids = measurement.spatial.query_indexes(box)
    trace = trace_surface_pairs(
        measurement.raw_uvz[ids],
        measurement.raw_point_indexes[ids],
        owned_measurement_mask(measurement)[ids],
        axis,
        anchor,
        span,
        sign,
        config,
    )
    value = trace["raw_midpoint_rho_m"]
    trace["continuous_rho_difference_m"] = (
        None if value is None else float(value - section["rho"])
    )
    return trace


def trace_surface_pairs(points, point_indexes, owned, axis, rho, span, sign, config):
    """Retain the original witnesses used by the existing midpoint primitive."""
    points = np.asarray(points)
    point_indexes = np.asarray(point_indexes, dtype=np.int64)
    owned = np.asarray(owned, dtype=bool)
    if points.shape != (len(point_indexes), 3) or owned.shape != point_indexes.shape:
        raise ValueError("RAW_PAIR_PROVENANCE_INDEX_ALIGNMENT")
    width = config["geometry"]["candidate_voxel_m"]
    keep = (
        (abs(points[:, axis] - rho) <= width)
        & (points[:, 1 - axis] >= span[0])
        & (points[:, 1 - axis] <= span[1])
    )
    points, point_indexes, owned = points[keep], point_indexes[keep], owned[keep]
    result = dict(
        status="INSUFFICIENT_RAW_SUPPORT",
        decision_input=False,
        used_for_detection=False,
        corner_z_source=False,
        steel_certified=False,
        anchor_rho_m=float(rho),
        normal_axis=axis,
        inward_sign=sign,
        section_span_m=list(span),
        query_point_count=len(points),
        query_owned_point_count=int(owned.sum()),
        query_context_point_count=int((~owned).sum()),
        raw_midpoint_rho_m=None,
        source_pair_point_indexes=[],
        pairs=[],
        uncertainty_semantic="RAW_BRACKETS_NOT_CALIBRATED_STEEL_OR_CORNER_UNCERTAINTY",
        ownership_promoted=False,
    )
    minimum = config["boundary"]["line_min_points"]
    if len(points) < minimum:
        return result
    signed = sign * (points[:, axis] - rho)
    outside, inside = points[signed <= 0, 2], points[signed > 0, 2]
    if min(len(outside), len(inside)) < minimum:
        return result
    outside_z, inside_z = float(np.median(outside)), float(np.median(inside))
    result.update(
        outside_surface_median_z=outside_z,
        inside_surface_median_z=inside_z,
        surface_z_semantic="PROFILE_RETURN_LEVELS_NOT_CORNER_RIM_TOP",
    )
    if abs(outside_z - inside_z) < config["boundary"]["profile_min_drop_m"]:
        result["status"] = "NO_DISTINCT_SURFACE_LEVELS"
        return result
    bins = np.floor(
        (points[:, 1 - axis] - span[0]) / config["boundary"]["max_support_gap_m"]
    ).astype(int)
    witnesses = []
    for index in np.unique(bins):
        in_bin = bins == index
        high = in_bin & (abs(points[:, 2] - outside_z) < abs(points[:, 2] - inside_z))
        low = in_bin & ~high
        outer, inner = high & (signed <= 0), low & (signed >= 0)
        if not outer.any() or not inner.any():
            continue
        outer_coordinate, inner_coordinate = signed[outer].max(), signed[inner].min()
        outer_ids = outer & (signed == outer_coordinate)
        inner_ids = inner & (signed == inner_coordinate)
        witnesses.extend(point_indexes[outer_ids | inner_ids].tolist())
        result["pairs"].append(
            dict(
                along_bin=int(index),
                outside_point_indexes=sorted(point_indexes[outer_ids].tolist()),
                inside_point_indexes=sorted(point_indexes[inner_ids].tolist()),
                outside_owned_point_indexes=sorted(
                    point_indexes[outer_ids & owned].tolist()
                ),
                inside_owned_point_indexes=sorted(
                    point_indexes[inner_ids & owned].tolist()
                ),
                midpoint_rho_m=float(
                    rho + sign * (outer_coordinate + inner_coordinate) / 2
                ),
                bracket_width_m=float(inner_coordinate - outer_coordinate),
            )
        )
    result["source_pair_point_indexes"] = sorted(set(witnesses))
    result["source_pair_point_index_sha256"] = hashlib.sha256(
        np.asarray(result["source_pair_point_indexes"], dtype="<i8").tobytes()
    ).hexdigest()
    if len(result["pairs"]) < min(minimum, config["boundary"]["profile_min_sections"]):
        result["status"] = "INSUFFICIENT_PAIRED_INTERVALS"
        return result
    midpoints = np.array([p["midpoint_rho_m"] for p in result["pairs"]])
    estimate = float(np.median(midpoints))
    result.update(
        status="TRACED_EXISTING_RAW_SURFACE_PAIRS",
        raw_midpoint_rho_m=estimate,
        midpoint_dispersion_m=float(np.median(abs(midpoints - estimate))),
        bracket_width_quantiles_m=np.quantile(
            [p["bracket_width_m"] for p in result["pairs"]], [0.5, 0.95, 1.0]
        ).tolist(),
        paired_interval_count=len(result["pairs"]),
        independent_section_vote_count=1,
    )
    return result
