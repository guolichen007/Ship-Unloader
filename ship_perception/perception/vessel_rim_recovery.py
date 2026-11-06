"""Own-Raw A/B for erroneous outer sides, separate from local refinement."""

import copy
from dataclasses import replace
import numpy as np

from .constrained_rectangle import constrain_review_geometry, RectangleInfeasible
from .interior_rim_search import InteriorRimSearch
from .local_rim_evidence import digest
from .opening_cells import coarse_review
from .review_geometry import SIDES
from .width_consistency import width_consistency, side_line


def inside_polygon(points, polygon, tolerance=0.0):
    p = np.asarray(polygon)
    points = np.asarray(points)
    edges = np.roll(p, -1, axis=0) - p
    sign = 1.0 if np.cross(p - p[0], np.roll(p, -1, axis=0) - p[0]).sum() >= 0 else -1.0
    distances = np.stack(
        [
            sign * np.cross(edge, points - point) / np.linalg.norm(edge)
            for point, edge in zip(p, edges)
        ],
        axis=1,
    )
    return np.all(distances >= -tolerance, axis=1)


def unique_raw_modes(entry, config):
    groups = []
    for mode in sorted(
        (m for m in entry["candidates"] if m["qualified_for_recovery"]),
        key=lambda m: (
            -m["observed_section_coverage"],
            m["uncertainty_m"],
            m["mode_id"],
        ),
    ):
        matched = None
        ids = set(mode["source_point_indexes"])
        for group in groups:
            old = group[0]
            other = set(old["source_point_indexes"])
            overlap = len(ids & other) / max(1, min(len(ids), len(other)))
            normal = np.asarray(mode["normal_uv"])
            oldnormal = np.asarray(old["normal_uv"])
            if overlap >= config["frame"]["same_plane_overlap_min"] and abs(
                normal @ oldnormal
            ) >= np.cos(np.deg2rad(config["boundary"]["merge_angle_deg"])):
                matched = group
                break
        if matched is None:
            groups.append([mode])
        else:
            matched.append(mode)
    return groups


def _member_gate(
    search, cell, polygon_before, polygon_after, entries, selected, uncertainty
):
    part = search.partitions[cell["cell_id"]]
    raw = search.local.xyz
    source = search.local.ids
    credible = part["indexes"][part["trusted"]]
    keep = inside_polygon(raw[credible, :2], polygon_after, uncertainty)
    audit = dict(
        trusted_member_count=len(credible),
        lost_trusted_count=int((~keep).sum()),
        lost_trusted_digest=digest(source[credible[~keep]]),
        original_mask_modified=False,
    )
    if not np.all(keep):
        return False, dict(audit, reason="TRUSTED_INTERIOR_MEMBER_WOULD_BE_LOST")
    ids = np.flatnonzero(part["original"])
    old = inside_polygon(raw[ids, :2], polygon_before, uncertainty)
    new = inside_polygon(raw[ids, :2], polygon_after, uncertainty)
    clipped = ids[old & ~new]
    clipped_sources = source[clipped]
    explained = set()
    witnesses = []
    for side, mode in selected.items():
        entry = entries[side]
        for row in mode["section_evidence"]:
            if not row["inner_rim_positive"]:
                continue
            plane = row["deck_plane"]
            surface = entry["surface_sources"][plane["context_surface_ref"]]
            explained.update(surface["source_point_indexes"])
            explained.update(row["vertical_face"]["point_indexes"])
            witnesses.append(
                dict(
                    side=side,
                    section=row["section_id"],
                    deck_source=plane["context_surface_ref"],
                    face_source_digest=row["vertical_face"]["raw_index_digest"],
                )
            )
    reclassified = np.isin(clipped_sources, np.fromiter(explained, dtype=np.int64))
    unknown = clipped_sources[~reclassified]
    # Neighbor membership remains a separate veto, even in overlapping old masks.
    neighbor = part["indexes"][part["other"]]
    conflict = neighbor[
        inside_polygon(raw[neighbor, :2], polygon_after)
        & ~inside_polygon(raw[neighbor, :2], polygon_before)
    ]
    audit.update(
        clipped_original_count=len(clipped),
        clipped_original_digest=digest(clipped_sources),
        reclassified_actual_deck_or_face_count=int(reclassified.sum()),
        reclassified_raw_indexes=clipped_sources[reclassified].tolist(),
        unexplained_clipped_count=len(unknown),
        unexplained_clipped_raw_indexes=unknown.tolist(),
        new_neighbor_capture_count=len(conflict),
        new_neighbor_capture_digest=digest(source[conflict]),
        reclassification_witnesses=witnesses,
    )
    allowed = not len(unknown) and not len(conflict)
    audit["reason"] = (
        "ALL_CLIPPED_MEMBERS_HAVE_CURRENT_RAW_ROLE_WITNESSES"
        if allowed
        else "UNEXPLAINED_MEMBER_LOSS_OR_NEIGHBOR_CAPTURE"
    )
    return allowed, audit


def recovery_ab(region, cell, anchor, seed, report, search, assembler, config, bridge):
    """Return an adopted strict rectangle or None, keeping every alternative."""
    audit = dict(
        status="CANDIDATE_NOT_ADOPTED",
        reason="NO_UNIQUE_COMPLETE_OWN_RAW_CHAIN",
        legacy_solver_state=report.get(
            "legacy_solver_state", cell.get("strict_rectangle_state", "UNKNOWN")
        ),
        old_boundary_uv=anchor["boundary_uv"],
        new_boundary_uv=None,
        sides={},
        steel_certified=False,
        common_width_is_measurement=False,
    )
    report["shadow_ab"] = audit
    selected = {}
    for side, entry in report["candidates"].items():
        groups = unique_raw_modes(entry, config)
        for mode in entry["candidates"]:
            mode["selection_reason"] = (
                "COMPLETE_CHAIN_AWAITS_AB"
                if mode["qualified_for_recovery"]
                else "PARTIAL_RAW_OR_MATERIAL_ROLE_NOT_CONFIRMED"
            )
        if len(groups) != 1:
            if groups:
                entry["selection_reason"] = "MULTIPLE_INDEPENDENT_RAW_LINE_FAMILIES"
            continue
        mode = groups[0][0]
        old_modes = seed.get("sides", {}).get(side, {}).get("candidate_modes", [])
        old_selected = seed.get("sides", {}).get(side, {}).get("selected_mode_id")
        old_strong = any(
            m["mode_id"] == old_selected
            and m.get("independent_opening_role_gain")
            and m.get("independent_raw_sections", 0)
            >= config["boundary"]["profile_min_sections"]
            for m in old_modes
        )
        if old_strong:
            entry["selection_reason"] = (
                "EXISTING_INDEPENDENT_INNER_ROLE_CONFLICT_REQUIRES_REVIEW"
            )
            continue
        mode["profile_mode_aliases"] = [m["mode_id"] for m in groups[0]]
        selected[side] = mode
    if not selected:
        return None
    axes = getattr(
        search.local.measurement.local, "primary_cache", search.local.measurement.local
    ).vessel.axes
    before = np.asarray(anchor["boundary_uv"])
    if seed:
        review_region = region
        sides = copy.deepcopy(seed["sides"])
    else:
        bounds = tuple(np.r_[before.min(0), before.max(0)])
        envelope = tuple(
            np.r_[
                np.minimum(region.search_envelope[:2], bounds[:2]),
                np.maximum(region.search_envelope[2:], bounds[2:]),
            ]
        )
        review_region = replace(
            region, conservative_bounds=bounds, search_envelope=envelope
        )
        sides = {
            name: assembler._prior(review_region, name, axis, pos, sign)
            for name, axis, pos, sign in SIDES
        }
    for side, mode in selected.items():
        layout = next(row for row in SIDES if row[0] == side)
        audit["sides"][side] = dict(
            legacy_side_evidence=copy.deepcopy(sides[side]),
            proposed_mode_id=mode["mode_id"],
            raw_source_digest=mode["raw_index_digest"],
            recovery_permit="COMPLETE_OWN_DECK_FACE_INTERIOR_CHAIN_SUBJECT_TO_MEMBER_AB",
            original_corner_join_m=config["boundary"]["corner_join_m"],
        )
        sides[side] = assembler._prior(review_region, *layout)
        assembler._apply_mode(
            sides[side],
            dict(
                mode,
                source="MEASURED_RAW",
                state="MEASURED_WEAK",
                independent_opening_role_gain=True,
                physical_role_improved=True,
            ),
        )
        if sides[side]["selected_mode_id"] != mode["mode_id"]:
            audit["reason"] = "EXISTING_SEPARATOR_OR_MODE_GUARD_REJECTED_RECOVERY"
            return None
    trial = assembler._result(review_region, axes, sides)
    try:
        alternate = constrain_review_geometry(review_region, trial, config, bridge)
    except RectangleInfeasible as error:
        audit.update(reason=str(error), solver_audit=error.audit)
        return None
    after = np.asarray(alternate["boundary_xy"]) @ axes.T
    audit.update(
        new_boundary_uv=after.tolist(),
        max_corner_move_m=float(np.max(np.linalg.norm(after - before, axis=1))),
        rectangle_raw_conflicts=alternate["rectangle_constraint_conflict_count"],
    )
    if alternate["rectangle_constraint_conflict_count"]:
        audit["reason"] = "RECOVERY_MODEL_CONFLICTS_WITH_CURRENT_RAW"
        return None
    uncertainty = max(mode["uncertainty_m"] for mode in selected.values())
    accepted, members = _member_gate(
        search, cell, before, after, report["candidates"], selected, uncertainty
    )
    audit["members"] = members
    for side, mode in selected.items():
        raw = search.local.xyz[
            search.local.positions(
                np.asarray(mode["source_point_indexes"], dtype=np.int64)
            ),
            :2,
        ]
        old = side_line(before, side)
        new = side_line(after, side)
        old_res = float(np.median(abs(raw @ np.asarray(old["normal_uv"]) - old["rho"])))
        new_res = float(np.median(abs(raw @ np.asarray(new["normal_uv"]) - new["rho"])))
        gain = old_res - new_res
        points = np.asarray(after)[[0, 1] if side == "V0" else [3, 2]]
        n = np.asarray(mode["normal_uv"])
        t = np.array([-n[1], n[0]])
        low, high = mode["support_span"]
        leverage = np.maximum(0.0, np.maximum(low - points @ t, points @ t - high))
        observable = mode["uncertainty_m"] * np.sqrt(
            1 + (leverage / max(high - low, mode["uncertainty_m"])) ** 2
        )
        backed = bool(
            np.all(
                abs(points @ n - mode["rho"])
                <= observable
                + 64
                * np.finfo(float).eps
                * np.maximum(1.0, np.linalg.norm(points, axis=1))
            )
        )
        audit["sides"][side].update(
            old_raw_residual_m=old_res,
            new_raw_residual_m=new_res,
            gain_m=gain,
            uncertainty_m=mode["uncertainty_m"],
            corners_within_observed_line_uncertainty=backed,
        )
        accepted = (
            accepted
            and gain > mode["uncertainty_m"]
            and new_res <= mode["uncertainty_m"]
            and backed
        )
    if not accepted:
        audit["reason"] = "RAW_GAIN_OR_MEMBER_OR_OBSERVABLE_CORNER_GUARD_FAILED"
        return None
    audit.update(
        status="COMMITTED_RAW_ERROR_RECOVERY",
        reason="OWN_RAW_GAIN_AND_EXPLICIT_MEMBER_RECLASSIFICATION_VERIFIED",
    )
    alternate["rim_error_recovery"] = audit
    alternate["steel_certified"] = False
    report["original_geometry_retained"] = False
    for side, mode in selected.items():
        report["candidates"][side]["adopted_mode_id"] = mode["mode_id"]
        mode["selection_reason"] = "COMMITTED_RAW_ERROR_RECOVERY"
    return alternate


def recover_vessel_sides(
    cells, strict_hatches, seeds, regions, local_rim, assembler, config, bridge
):
    """One optional recovery phase in the existing selected-vessel pipeline."""
    measurement = local_rim.measurement
    axes = getattr(measurement.local, "primary_cache", measurement.local).vessel.axes
    baseline = {h["opening_identity"]: h for h in strict_hatches}
    anchors = {}
    polygons = {}
    for cell in cells:
        key = cell["primary_region_id"]
        anchor = baseline.get(key) or coarse_review(
            cell, axes, measurement, "INFEASIBLE" if key in seeds else "NOT_RUN"
        )
        if anchor is None:
            continue
        anchors[cell["cell_id"]] = anchor
        polygons[cell["cell_id"]] = (
            np.asarray(anchor["boundary_xy"]) @ axes.T
        ).tolist()
    usable = [c for c in cells if c["cell_id"] in polygons]
    vessel_audits, requests = width_consistency(usable, polygons, config)
    search = InteriorRimSearch(local_rim)
    adopted = dict(baseline)
    for cell in usable:
        cell["same_vessel_width_consistency"] = vessel_audits[cell["vessel_id"]]
        cell["width_recovery_requested_sides"] = requests[cell["cell_id"]]
        if not requests[cell["cell_id"]]:
            cell["rim_error_recovery"] = dict(
                status="FAST_PATH_RETAINED", geometry_modified=False
            )
            continue
        result = search.search(cell, polygons[cell["cell_id"]], regions, usable)
        cell["rim_error_recovery"] = result
        key = cell["primary_region_id"]
        result["legacy_solver_state"] = (
            "FEASIBLE"
            if key in baseline
            else "INFEASIBLE" if key in seeds else "NOT_RUN"
        )
        if key not in regions:
            continue
        alternate = recovery_ab(
            regions[key],
            cell,
            anchors[cell["cell_id"]],
            seeds.get(key, {}),
            result,
            search,
            assembler,
            config,
            bridge,
        )
        if alternate is not None:
            adopted[key] = alternate
    return list(adopted.values()), vessel_audits
