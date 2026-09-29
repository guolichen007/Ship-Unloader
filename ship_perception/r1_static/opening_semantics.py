"""Static opening-interior review and B2-to-calibration evidence bridge.

Height is local context, not a steel label. A proposed side is moved only to a
measured B2 Raw3D/BEV mode on the same scene support and with an opening-side
transition. The output distinguishes the suggested geometry from confirmed
opening-side boundaries.
"""

import numpy as np
from scipy import ndimage

from .raw_topview_evidence import build_topview
from .rectangle_fusion import _opening_relation


SIDE_IDS = ("U0", "U1", "V0", "V1")


def _box(polygon, axes):
    projected = np.asarray(polygon, dtype=float) @ np.asarray(axes).T
    return [
        float(projected[:, 0].min()),
        float(projected[:, 1].min()),
        float(projected[:, 0].max()),
        float(projected[:, 1].max()),
    ]


def _iou(first, second):
    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )
    a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return float(overlap / max(a + b - overlap, 1e-9))


def _interior_evidence(grid, box, config):
    """Relative local median evidence; loaded cargo may invert the sign."""
    u, v = grid.centers()
    u0, v0, u1, v1 = box
    inset = config["boundary"]["side_probe_m"]
    width = config["roi"]["support_search_m"] / 2
    inside = (
        (u >= u0 + inset)
        & (u <= u1 - inset)
        & (v >= v0 + inset)
        & (v <= v1 - inset)
        & grid.occupancy
    )
    outer_box = (
        (u >= u0 - width) & (u <= u1 + width) & (v >= v0 - width) & (v <= v1 + width)
    )
    box_mask = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    ring = outer_box & ~box_mask & grid.occupancy
    minimum = config["roi"]["min_deck_support_cells"]
    if min(int(inside.sum()), int(ring.sum())) < minimum:
        return dict(
            status="INSUFFICIENT_INTERIOR_OR_RING_SUPPORT",
            interior_cells=int(inside.sum()),
            ring_cells=int(ring.sum()),
        )
    interior = grid.median[inside]
    surrounding = grid.median[ring]
    interior_z = float(np.median(interior))
    surrounding_z = float(np.median(surrounding))
    depth = surrounding_z - interior_z
    threshold = config["roi"]["opening_drop_m"]
    lower = inside & (grid.median < surrounding_z - threshold)
    raised = inside & (grid.median > surrounding_z + threshold)
    active = lower if depth > threshold else raised if depth < -threshold else inside
    labels, number = ndimage.label(active)
    counts = np.bincount(labels[active], minlength=number + 1)[1:]
    significant = counts[counts >= minimum]
    corner_evidence = []
    for cu, cv, su, sv in (
        (u0, v0, 1, 1),
        (u1, v0, -1, 1),
        (u1, v1, -1, -1),
        (u0, v1, 1, -1),
    ):
        near = (
            (su * (u - cu) >= 0)
            & (su * (u - cu) < width)
            & (sv * (v - cv) >= 0)
            & (sv * (v - cv) < width)
            & grid.occupancy
        )
        far = (
            (su * (u - cu) < 0)
            & (su * (u - cu) >= -width)
            & (sv * (v - cv) < 0)
            & (sv * (v - cv) >= -width)
            & grid.occupancy
        )
        if min(int(near.sum()), int(far.sum())) < max(1, minimum // 4):
            corner_evidence.append(dict(status="UNRESOLVED", local_contrast_m=None))
        else:
            contrast = float(np.median(grid.median[far]) - np.median(grid.median[near]))
            corner_evidence.append(
                dict(
                    status=(
                        "RELATIVE_HEIGHT_CORNER"
                        if abs(contrast) >= threshold
                        else "HEIGHT_AMBIGUOUS"
                    ),
                    local_contrast_m=contrast,
                )
            )
    return dict(
        status=(
            "LOWER_INTERIOR"
            if depth > threshold
            else (
                "RAISED_OR_CARGO_INTERIOR" if depth < -threshold else "HEIGHT_AMBIGUOUS"
            )
        ),
        interior_median_z_m=interior_z,
        interior_z_mad_m=float(np.median(np.abs(interior - interior_z))),
        surrounding_ring_median_z_m=surrounding_z,
        surrounding_ring_z_mad_m=float(np.median(np.abs(surrounding - surrounding_z))),
        local_basin_depth_m=float(depth),
        interior_low_fraction=float(np.mean(interior < surrounding_z - threshold)),
        interior_high_fraction=float(np.mean(interior > surrounding_z + threshold)),
        interior_component_count=int(len(significant)),
        interior_largest_component_fraction=float(
            max(significant, default=0) / max(1, int(active.sum()))
        ),
        corners=corner_evidence,
        corner_support_count=sum(
            row["status"] == "RELATIVE_HEIGHT_CORNER" for row in corner_evidence
        ),
        interior_cells=int(inside.sum()),
        ring_cells=int(ring.sum()),
    )


def _provider_lines(
    per_vessel, axes, normal_axis, span, before, config, confirmed_vessel_id
):
    """Read raw line support, including adjacent unresolved scene context.

    A different confirmed vessel is never allowed to supply a Hatch side.
    The line must itself overlap the current Hatch side; shared separators may
    legitimately support both adjacent openings.
    """
    grouped = []
    tolerance = config["boundary"]["corner_join_m"]
    search = config["roi"]["support_search_m"]
    for context in per_vessel:
        status = context["vessel_classification_status"]
        source_id = context["vessel_hypothesis_id"]
        if status == "VESSEL_HYPOTHESIS" and source_id != confirmed_vessel_id:
            continue
        for row in context["provider_evidence"]:
            if row["geometry_type"] != "AXIAL_LINE_INTERVAL":
                continue
            if row["rho_or_region"]["normal_axis"] != normal_axis:
                continue
            if row["provider_type"] not in (
                "RAW3D_STRUCTURAL_PROVIDER",
                "BEV_RECTILINEAR_PROVIDER",
            ):
                continue
            endpoints = np.asarray(row["source_geometry"]["endpoints_xy"], dtype=float)
            if endpoints.shape != (2, 2):
                continue
            local = endpoints @ axes.T
            rho = float(np.mean(local[:, normal_axis]))
            along = sorted(float(value) for value in local[:, 1 - normal_axis])
            overlap = max(0.0, min(span[1], along[1]) - max(span[0], along[0]))
            if (
                abs(rho - before) > search
                or overlap < config["boundary"]["line_min_length_m"]
                or abs(local[1, normal_axis] - local[0, normal_axis]) > search
            ):
                continue
            coverage = (
                min(1.0, overlap / max(span[1] - span[0], 1e-9)) * row["coverage"]
            )
            grouped.append(
                dict(
                    rho=rho,
                    provider_type=row["provider_type"],
                    coverage=float(coverage),
                    provider_id=row["provider_id"],
                    source_vessel_id=source_id,
                    source_classification=status,
                    tilt_m=float(abs(local[1, normal_axis] - local[0, normal_axis])),
                )
            )
    # A physical rim can have multiple nearby grid interfaces. Keep separate
    # groups beyond the existing corner-join resolution, never average all.
    groups = []
    for row in sorted(grouped, key=lambda item: item["rho"]):
        if not groups or row["rho"] - groups[-1][0]["rho"] > tolerance:
            groups.append([])
        groups[-1].append(row)
    return groups


def _side_review(side_id, box, old_side, per_vessel, vessel_id, axes, fine, config):
    normal_axis = 0 if side_id[0] == "U" else 1
    lower = side_id[1] == "0"
    before = box[normal_axis] if lower else box[normal_axis + 2]
    span = (box[1], box[3]) if normal_axis == 0 else (box[0], box[2])
    sign = 1 if lower else -1
    minimum = config["roi"]["min_enclosure_ratio"]
    candidates = []
    for group in _provider_lines(
        per_vessel, axes, normal_axis, span, before, config, vessel_id
    ):
        rho = float(np.median([item["rho"] for item in group]))
        raw3d = max(
            (
                item["coverage"]
                for item in group
                if item["provider_type"] == "RAW3D_STRUCTURAL_PROVIDER"
            ),
            default=0.0,
        )
        bev = max(
            (
                item["coverage"]
                for item in group
                if item["provider_type"] == "BEV_RECTILINEAR_PROVIDER"
            ),
            default=0.0,
        )
        relation = _opening_relation(fine, normal_axis, rho, span, sign, config)
        candidates.append(
            dict(
                rho=rho,
                raw3d_coverage=raw3d,
                bev_coverage=bev,
                opening_relation=relation,
                provider_ids=[item["provider_id"] for item in group],
                source_vessel_ids=sorted({item["source_vessel_id"] for item in group}),
                source_classifications=sorted(
                    {item["source_classification"] for item in group}
                ),
            )
        )
    # Only a measured line with independent providers and a supported local
    # transition can replace an R2G side. Proximity and width are never enough.
    qualified = [
        row
        for row in candidates
        if row["raw3d_coverage"] >= minimum
        and row["bev_coverage"] >= minimum
        and row["opening_relation"]["status"] == "OPENING_LOWER"
        and row["opening_relation"]["support_fraction"] >= minimum
        and row["opening_relation"]["far_support_fraction"] >= minimum
    ]
    # First physical transition outward from the interior wins. Later strong
    # lines can be outer coaming, vessel side or a deckhouse.
    qualified.sort(key=lambda row: (-row["rho"] if lower else row["rho"]))
    selected = qualified[0] if qualified else None
    # A cargo-filled opening can reverse local height contrast. In that case
    # a long Raw3D line is a reviewable side hypothesis, not confirmed steel.
    raw3d_only = [
        row
        for row in candidates
        if row["raw3d_coverage"] >= minimum
        and row["opening_relation"]["support_fraction"] >= minimum
        and row["opening_relation"]["far_support_fraction"] >= minimum
    ]
    raw3d_only.sort(key=lambda row: (-row["rho"] if lower else row["rho"]))
    suggested = raw3d_only[0] if raw3d_only else None
    # All candidate modes are expressed in ``axes``. When the B2 vessel axis
    # replaces R2G's axis, the old rho must first be reprojected into that
    # same frame; comparing axial values from different frames can move an
    # otherwise unchanged side by metres.
    previous_rho = float(before)
    weak_outer_fallback = (
        old_side["role"] != "INNER_EDGE"
        and max(old_side["structural_coverage"], old_side["raster_coverage"]) < minimum
    )
    if (
        old_side.get("mode_conflict")
        and not weak_outer_fallback
        and abs((selected or suggested or {"rho": previous_rho})["rho"] - previous_rho)
        > config["boundary"]["corner_join_m"]
    ):
        role = "UNKNOWN"
        after = previous_rho
        evidence_status = "MULTIMODE_CONFLICT_KEEP_R2G"
    elif selected is None:
        previous_role = (
            "OUTER_HULL_EDGE"
            if weak_outer_fallback
            else (
                "UNKNOWN"
                if old_side["role"] == "AMBIGUOUS"
                else "HATCH_INNER_EDGE_CANDIDATE"
            )
        )
        role = "UNKNOWN" if suggested and weak_outer_fallback else previous_role
        after = suggested["rho"] if suggested and weak_outer_fallback else previous_rho
        evidence_status = (
            "RAW3D_ONLY_REVIEW_GEOMETRY"
            if suggested and weak_outer_fallback
            else "UNRESOLVED_KEEP_R2G_GEOMETRY"
        )
    else:
        role = "HATCH_INNER_EDGE_CANDIDATE"
        after = selected["rho"]
        evidence_status = "B2_RAW3D_BEV_TRANSITION"
    return dict(
        side_id=side_id,
        rho_before=previous_rho,
        rho_after=float(after),
        delta_m=float(after - previous_rho),
        role=role,
        r2g_rho_before=float(old_side["rho_after"]),
        previous_role=old_side["role"],
        evidence_status=evidence_status,
        selected_provider_ids=(
            (selected or suggested)["provider_ids"] if (selected or suggested) else []
        ),
        selected_source_vessel_ids=(
            (selected or suggested)["source_vessel_ids"]
            if (selected or suggested)
            else []
        ),
        candidate_modes=candidates,
    )


def _reconcile_measured_shared_width(rows, config):
    """Use equal width only to resolve a shared *measured* parallel rim.

    This never constructs a side at the width predicted by its neighbor. Each
    hatch must independently observe the same Raw3D + BEV opening transition.
    """
    tolerance = config["boundary"]["corner_join_m"]
    minimum = config["roi"]["min_enclosure_ratio"]
    for index, first in enumerate(rows):
        if first.get("sides") is None:
            continue
        for second in rows[index + 1 :]:
            if (
                second.get("sides") is None
                or first["vessel_hypothesis_id"] != second["vessel_hypothesis_id"]
                or first["axis_source"] != second["axis_source"]
            ):
                continue
            a, b = first["sides"], second["sides"]
            shared_separator = min(
                abs(a["U0"]["rho_after"] - b["U1"]["rho_after"]),
                abs(a["U1"]["rho_after"] - b["U0"]["rho_after"]),
            )
            if shared_separator > tolerance:
                continue
            for side_id, opposite in (("V0", "V1"), ("V1", "V0")):
                prior_width_mismatch = abs(
                    (a["V1"]["rho_after"] - a["V0"]["rho_after"])
                    - (b["V1"]["rho_after"] - b["V0"]["rho_after"])
                )
                if (
                    prior_width_mismatch <= tolerance
                    or abs(a[opposite]["rho_after"] - b[opposite]["rho_after"])
                    > tolerance
                    or not any(
                        side[side_id]["evidence_status"]
                        == "MULTIMODE_CONFLICT_KEEP_R2G"
                        for side in (a, b)
                    )
                ):
                    continue
                common = []
                for left in a[side_id]["candidate_modes"]:
                    for right in b[side_id]["candidate_modes"]:
                        if abs(left["rho"] - right["rho"]) > tolerance:
                            continue
                        pair = (left, right)
                        if not all(
                            mode["raw3d_coverage"] >= minimum
                            and mode["bev_coverage"] >= minimum
                            and mode["opening_relation"]["status"] == "OPENING_LOWER"
                            and mode["opening_relation"]["support_fraction"] >= minimum
                            for mode in pair
                        ):
                            continue
                        # The far probe of one hatch may be hidden by cargo or
                        # wharf returns. The independent second hatch still
                        # has to see the far side of this same measured line.
                        if (
                            max(
                                mode["opening_relation"]["far_support_fraction"]
                                for mode in pair
                            )
                            < minimum
                        ):
                            continue
                        rho = float(np.median((left["rho"], right["rho"])))
                        inward = (
                            rho
                            > max(a[side_id]["rho_before"], b[side_id]["rho_before"])
                            if side_id == "V0"
                            else rho
                            < min(a[side_id]["rho_before"], b[side_id]["rho_before"])
                        )
                        if not inward:
                            continue
                        after_mismatch = abs(
                            abs(a[opposite]["rho_after"] - rho)
                            - abs(b[opposite]["rho_after"] - rho)
                        )
                        if prior_width_mismatch - after_mismatch <= tolerance:
                            continue
                        strength = min(
                            mode[key]
                            for mode in pair
                            for key in ("raw3d_coverage", "bev_coverage")
                        )
                        displacement = sum(
                            abs(rho - side[side_id]["rho_before"]) for side in (a, b)
                        )
                        common.append((strength, -displacement, rho, pair))
                if not common:
                    continue
                _, _, rho, pair = max(common, key=lambda item: item[:2])
                for sides, mode in zip((a, b), pair):
                    side = sides[side_id]
                    side["rho_after"] = rho
                    side["delta_m"] = rho - side["rho_before"]
                    side["role"] = "HATCH_INNER_EDGE_CANDIDATE"
                    side["evidence_status"] = "MEASURED_SHARED_WIDTH_TIE_BREAK"
                    side["selected_provider_ids"] = mode["provider_ids"]
                    side["selected_source_vessel_ids"] = mode["source_vessel_ids"]
                first["status"] = second["status"] = "PARTIAL_HATCH"


def bridge_final_rectangles(points, r2g, b2, config):
    """Consume B2 observations to review each existing R2G opening per vessel."""
    all_vessels = b2["scene"]["vessel_hypotheses"]
    status_by_id = {
        row["vessel_hypothesis_id"]: row["classification_status"] for row in all_vessels
    }
    # Earlier B2 map reviews did not repeat classification in per_vessel.
    # Recover it from the authoritative Scene object, never infer a target.
    per_vessel = [
        dict(
            row,
            vessel_classification_status=status_by_id.get(
                row["vessel_hypothesis_id"], "UNRESOLVED"
            ),
        )
        for row in b2["per_vessel"]
    ]
    vessels = [
        row
        for row in all_vessels
        if row["classification_status"] == "VESSEL_HYPOTHESIS"
    ]
    result = []
    grid_cache = {}
    for hatch in r2g["rectangles"]:
        polygon = np.asarray(hatch["polygon_after"], dtype=float)
        center = polygon.mean(axis=0)
        # A vessel owns the rectangle only when its measured spatial support
        # covers the center. Ambiguous ownership never creates a target vessel.
        possible = [
            row
            for row in vessels
            if row["bbox_xy"][0] <= center[0] <= row["bbox_xy"][2]
            and row["bbox_xy"][1] <= center[1] <= row["bbox_xy"][3]
        ]
        if len(possible) != 1:
            result.append(
                dict(
                    hatch_id=hatch["hatch_id"],
                    vessel_hypothesis_id=None,
                    status="UNRESOLVED_VESSEL_ASSOCIATION",
                    polygon_before=hatch["polygon_after"],
                    polygon_after=None,
                )
            )
            continue
        vessel = possible[0]
        vessel_id = vessel["vessel_hypothesis_id"]
        # Preserve an already supported R2G ship axis. A weak outer-frame
        # fallback is the case where the independent B2 vessel axis may expose
        # a long inner rim hidden by angular drift in the old search frame.
        weak_outer_side = any(
            side["role"] != "INNER_EDGE"
            and max(side["structural_coverage"], side["raster_coverage"])
            < config["roi"]["min_enclosure_ratio"]
            for side in hatch["sides"].values()
        )
        axis_source = "B2_LOCAL_VESSEL" if weak_outer_side else "R2G_SUPPORTED"
        axes = np.asarray(
            vessel["local_axes"] if weak_outer_side else r2g["axes"], dtype=float
        )
        cache_key = (vessel_id, axis_source)
        if cache_key not in grid_cache:
            grid_cache[cache_key] = build_topview(points, axes, config)
        fine = grid_cache[cache_key]["fine"]
        coarse = grid_cache[cache_key]["coarse"]
        box = _box(polygon, axes)
        lineage = []
        for context in per_vessel:
            if context["vessel_hypothesis_id"] != vessel_id:
                continue
            for candidate in context["hatch_hypotheses"]:
                bounds = candidate.get("bounds_axial")
                if bounds is not None:
                    vessel_axes = np.asarray(vessel["local_axes"])
                    raw_corners = (
                        np.asarray(
                            [
                                [bounds[0], bounds[1]],
                                [bounds[2], bounds[1]],
                                [bounds[2], bounds[3]],
                                [bounds[0], bounds[3]],
                            ]
                        )
                        @ vessel_axes
                    )
                    candidate_box = _box(raw_corners, axes)
                    lineage.append((_iou(box, candidate_box), candidate))
        match = max(lineage, key=lambda item: item[0]) if lineage else None
        sides = {
            side_id: _side_review(
                side_id,
                box,
                hatch["sides"][side_id],
                per_vessel,
                vessel_id,
                axes,
                fine,
                config,
            )
            for side_id in SIDE_IDS
        }
        # An uncertain shared transverse beam cannot split into two different
        # edges just because independent Hatch side rankings prefer them.
        for previous in result:
            if previous.get("vessel_hypothesis_id") != vessel_id:
                continue
            for current_key, previous_key in (("U0", "U1"), ("U1", "U0")):
                current_side = sides[current_key]
                prior_side = previous["sides"][previous_key]
                if (
                    abs(current_side["rho_before"] - prior_side["rho_before"])
                    <= config["boundary"]["corner_join_m"]
                ):
                    if (
                        abs(current_side["rho_after"] - prior_side["rho_after"])
                        > config["boundary"]["corner_join_m"]
                    ):
                        for side in (current_side, prior_side):
                            side["rho_after"] = side["rho_before"]
                            side["delta_m"] = 0.0
                            side["role"] = "SEPARATOR_STRUCTURE"
                            side["evidence_status"] = "SHARED_SEPARATOR_MODE_CONFLICT"
                        previous["status"] = "PARTIAL_HATCH"
        after = [
            sides["U0"]["rho_after"],
            sides["V0"]["rho_after"],
            sides["U1"]["rho_after"],
            sides["V1"]["rho_after"],
        ]
        if after[0] >= after[2] or after[1] >= after[3]:
            after = box
            for side_id, target in zip(("U0", "V0", "U1", "V1"), box):
                side = sides[side_id]
                side["rho_after"] = float(target)
                side["delta_m"] = float(target - side["rho_before"])
                side["role"] = "UNKNOWN"
                side["evidence_status"] = "GEOMETRY_REJECTED_KEEP_REPROJECTED_R2G"
            status = "UNRESOLVED_GEOMETRY"
        else:
            supported = sum(
                side["evidence_status"] == "B2_RAW3D_BEV_TRANSITION"
                for side in sides.values()
            )
            same_vessel_support = all(
                side["evidence_status"] == "B2_RAW3D_BEV_TRANSITION"
                and side["selected_source_vessel_ids"] == [vessel_id]
                for side in sides.values()
            )
            status = (
                "PROVISIONAL_HATCH"
                if same_vessel_support
                else "PARTIAL_HATCH" if supported >= 2 else "UNRESOLVED"
            )
        corners = (
            np.asarray(
                [
                    [after[0], after[1]],
                    [after[2], after[1]],
                    [after[2], after[3]],
                    [after[0], after[3]],
                ]
            )
            @ axes
        )
        result.append(
            dict(
                hatch_id=hatch["hatch_id"],
                vessel_hypothesis_id=vessel_id,
                hatch_hypothesis_id=(
                    match[1]["hatch_hypothesis_id"]
                    if match and match[0] >= config["roi"]["min_enclosure_ratio"]
                    else None
                ),
                hatch_hypothesis_iou=match[0] if match else 0.0,
                nearest_b2_proposal_id=(
                    match[1]["hatch_hypothesis_id"] if match else None
                ),
                status=status,
                axes=axes.tolist(),
                axis_source=axis_source,
                polygon_before=hatch["polygon_after"],
                polygon_after=corners.tolist(),
                opening_interior=_interior_evidence(coarse, after, config),
                sides=sides,
            )
        )
    _reconcile_measured_shared_width(result, config)
    # A later Hatch can reveal a shared-separator conflict. Rebuild both
    # polygons after all such pairwise checks have completed.
    for row in result:
        if row.get("sides") is None:
            continue
        sides = row["sides"]
        a = [
            sides["U0"]["rho_after"],
            sides["V0"]["rho_after"],
            sides["U1"]["rho_after"],
            sides["V1"]["rho_after"],
        ]
        axes = np.asarray(row["axes"])
        row["polygon_after"] = (
            np.asarray([[a[0], a[1]], [a[2], a[1]], [a[2], a[3]], [a[0], a[3]]]) @ axes
        ).tolist()
        row["opening_interior"] = _interior_evidence(
            grid_cache[(row["vessel_hypothesis_id"], row["axis_source"])]["coarse"],
            a,
            config,
        )
    return dict(
        schema="ship_perception.v15r.opening_semantics.1",
        target_vessel_status="NOT_SELECTED",
        static_calibration_status="UNVERIFIED_NO_INDEPENDENT_INNER_EDGE_GOLDEN",
        rectangles=result,
    )
