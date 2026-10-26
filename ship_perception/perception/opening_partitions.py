"""An ambiguous envelope cannot erase independently validated opening cells."""

import numpy as np


def _bounds(row, axes):
    uv = np.asarray(row["boundary_xy"]) @ axes.T
    return np.r_[uv.min(0), uv.max(0)]


def _independent_cell(row, parent, config):
    if (
        row is parent
        or row.get("topology_status") != "INDEPENDENT_OPENING_GROUP"
        or row.get("vessel_hypothesis_id") != parent.get("vessel_hypothesis_id")
        or row.get("containment", {}).get("status") != "CONTAINED"
        or row.get("exterior", {}).get("status") != "ENCLOSED_OBSERVED_LOW_COMPONENT"
    ):
        return False
    required = config["boundary"]["profile_min_sections"]
    full = sum(
        min(s["raw_subvoxel_section_count"], s["positive_height_section_count"])
        >= required
        for s in row["sides"].values()
    )
    return full >= 4 - config["boundary"]["max_inferred_edges"]


def _facing_support(row, name, station, config):
    # The fourth side may remain conservative in a three-side partial cell.
    # Its independently nominated Raw3D+BEV station must still agree with the
    # parent's two measured modes. No new side or sub-threshold fit is created.
    side = row.get("deck_ring", {}).get("sides", {}).get(name, {})
    rise = side.get("signed_height_rise_m")
    return (
        side.get("raw3d_support", False)
        and side.get("bev_support", False)
        and side.get("coverage", 0) >= config["roi"]["min_enclosure_ratio"]
        and rise is not None
        and rise >= config["roi"]["opening_drop_m"]
        and abs(row["sides"][name]["rho_after_m"] - station)
        <= config["boundary"]["corner_join_m"]
    )


def explained_envelopes(rows, axes, config):
    """Defer only envelopes already explained by separate, supported cells.

    A numeric separator contains two measured Raw3D+BEV modes. A broad
    transverse-band conflict does not: it must retain its unresolved envelope.
    This does not widen the beam-splitting gate or create child proposals.
    """
    boxes = {id(row): _bounds(row, axes) for row in rows}
    deferred = []
    for parent in rows:
        stations = parent.get("topology_ambiguity")
        if (
            parent.get("topology_status") != "AMBIGUOUS_TOPOLOGY"
            or not isinstance(stations, list)
            or len(stations) != 2
        ):
            continue
        first, second = map(float, stations)
        if not np.isfinite([first, second]).all() or first >= second:
            continue
        outer = boxes[id(parent)]
        tolerance = config["boundary"]["corner_join_m"]
        sides = [[], []]
        for child in rows:
            if not _independent_cell(child, parent, config):
                continue
            box = boxes[id(child)]
            if np.any(box[:2] < outer[:2] - tolerance) or np.any(
                box[2:] > outer[2:] + tolerance
            ):
                continue
            overlap = max(0.0, min(box[3], outer[3]) - max(box[1], outer[1]))
            if (
                overlap / max(np.finfo(float).eps, outer[3] - outer[1])
                < config["boundary"]["min_edge_coverage"]
            ):
                continue
            if abs(box[0] - outer[0]) <= tolerance and _facing_support(
                child, "U1", first, config
            ):
                sides[0].append(child)
            if abs(box[2] - outer[2]) <= tolerance and _facing_support(
                child, "U0", second, config
            ):
                sides[1].append(child)
        pair = min(
            (
                (left, right)
                for left in sides[0]
                for right in sides[1]
                if boxes[id(left)][2] < boxes[id(right)][0]
            ),
            key=lambda pair: tuple(r["proposal_id"] for r in pair),
            default=None,
        )
        if pair is None:
            continue
        left, right = pair
        deferred.append(
            dict(
                proposal_id=parent["proposal_id"],
                kept=left["proposal_id"],
                explained_by_proposal_ids=[left["proposal_id"], right["proposal_id"]],
                reason="AMBIGUOUS_ENVELOPE_EXPLAINED_BY_INDEPENDENT_OPENINGS",
                envelope_boundary_xy=parent["boundary_xy"],
                measured_separator_modes_m=stations,
                independent_opening_coordinates_created=False,
                beam_width_gate_changed=False,
            )
        )
    identifiers = {row["proposal_id"] for row in deferred}
    return [row for row in rows if row["proposal_id"] not in identifiers], deferred
