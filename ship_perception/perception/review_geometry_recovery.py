"""Uncertified review proposals, separate from frozen identity and strict XY/Z."""

import copy
import numpy as np
from scipy.optimize import least_squares
from .local_rim_evidence import LocalRimEvidence, digest
from .interior_rim_search import cell_members
from .opening_interior_support import spatial_interior_core
from .review_side_evidence import heldout_side, SIDES
from .r3h_geometry import dimensions


def rectangle(bounds):
    u0, v0, u1, v1 = bounds
    return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], float)


def inside(xy, bounds, tolerance=0.0):
    return np.all(
        (xy >= np.asarray(bounds[:2]) - tolerance)
        & (xy <= np.asarray(bounds[2:]) + tolerance),
        axis=1,
    )


def select_supported(entry, config, neighbor_positions=()):
    qualified = [m for m in entry["candidates"] if m["qualified_for_review"]]
    ranked = sorted(
        qualified,
        key=lambda m: (
            -m["score"],
            -m["coverage"],
            m["position_uncertainty_m"],
            m["mode_id"],
        ),
    )
    if not ranked:
        return None, []
    best = ranked[0]
    near = [
        m
        for m in ranked
        # One held-out slab's relative support uncertainty. An absolute unit
        # margin would declare every weak mode tied when the best score is low.
        if m["score"]
        >= best["score"] * (1 - 1 / max(1, len(entry["holdout_section_ids"])))
    ]
    if neighbor_positions and len(near) > 1:
        # Only current own-Raw-qualified modes can be disambiguated by another cell.
        def distance(m):
            return min(
                abs(m["position_m"] - p)
                / max(
                    m["position_uncertainty_m"] + s,
                    config["geometry"]["candidate_voxel_m"],
                )
                for p, s in neighbor_positions
            )

        best = min(near, key=lambda m: (distance(m), -m["score"], m["mode_id"]))
    alternatives = [
        m
        for m in near
        if abs(m["position_m"] - best["position_m"])
        > m["position_uncertainty_m"] + best["position_uncertainty_m"]
    ]
    if alternatives:
        tied = [
            m
            for m in alternatives
            if abs(m["holdout_coverage"] - best["holdout_coverage"])
            <= 1 / max(1, len(entry["holdout_section_ids"]))
        ]
        if tied and (
            not neighbor_positions
            or any(abs(distance(m) - distance(best)) <= 1 for m in tied)
        ):
            return None, [best] + tied[:1]
    return best, alternatives[:2]


def joint_bounds(cells, plans, config):
    """Robust measured-side factors; width factors switch off for genuine variation."""
    rows = []
    initial = []
    sigma = []
    for c in cells:
        key = c["cell_id"]
        plan = plans[key]
        old = plan["old_bounds"]
        pos = list(old)
        unc = np.full(4, config["geometry"]["coarse_voxel_m"])
        for side, m in plan["selected"].items():
            i = {"U0": 0, "V0": 1, "U1": 2, "V1": 3}[side]
            pos[i] = m["position_m"]
            unc[i] = max(
                m["position_uncertainty_m"], config["geometry"]["candidate_voxel_m"]
            )
        rows.append(key)
        initial.append(pos)
        sigma.append(unc)
    x0 = np.asarray(initial)
    sig = np.asarray(sigma)
    links = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = plans[rows[i]], plans[rows[j]]
            if a["vessel_id"] != b["vessel_id"] or not all(
                s in p["selected"] for p in (a, b) for s in ("V0", "V1")
            ):
                continue
            w = x0[:, 3] - x0[:, 1]
            noise = float(
                np.hypot(np.hypot(*sig[i, [1, 3]]), np.hypot(*sig[j, [1, 3]]))
            )
            # A compatibility likelihood from OWN observed widths, never an equal-width rule.
            weight = np.exp(-0.5 * ((w[i] - w[j]) / noise) ** 2)
            links.append((i, j, noise, float(weight)))

    def residual(x):
        q = x.reshape(-1, 4)
        r = ((q - x0) / sig).ravel().tolist()
        for i, j, s, w in links:
            r.append(np.sqrt(w) * ((q[i, 3] - q[i, 1]) - (q[j, 3] - q[j, 1])) / s)
        return np.asarray(r)

    solved = least_squares(residual, x0.ravel(), loss="soft_l1").x.reshape(-1, 4)
    return {key: solved[i] for i, key in enumerate(rows)}, [
        dict(
            first=rows[i],
            second=rows[j],
            own_width_uncertainty_m=s,
            soft_weight=w,
            equal_width_required=False,
        )
        for i, j, s, w in links
    ]


def recover_review_geometries(cells, regions, measurement, config, local_rim=None):
    local = (
        local_rim if local_rim is not None else LocalRimEvidence(measurement, config)
    )
    axes = getattr(measurement.local, "primary_cache", measurement.local).vessel.axes
    masks = {
        c["cell_id"]: cell_members(c, regions, local.xyz, local.owned)[0] for c in cells
    }
    extents = {
        k: np.r_[local.xyz[m, :2].min(0), local.xyz[m, :2].max(0)]
        for k, m in masks.items()
        if m is not None and np.any(m)
    }
    plans = {}
    all_evidence = {}
    for cell in cells:
        if cell.get("geometry") is None:
            continue
        key = cell["cell_id"]
        old = cell["geometry"]
        p = np.asarray(old["boundary_xy"]) @ axes.T
        bounds = np.r_[p.min(0), p.max(0)]
        layer = dict(
            state="REVIEW_UNRESOLVED",
            boundary_xy=old["boundary_xy"],
            corners_xyz=old["corners_xyz"],
            baseline_boundary_xy=old["boundary_xy"],
            strict_rectangle_state=cell["strict_rectangle_state"],
            steel_certified=False,
            certification_status="NOT_CERTIFIED",
            control_eligible=False,
            human_review="NOT_VERIFIED",
            adopted_for_review=False,
            source_family="RETAINED_STRICT_BASELINE",
            unobserved_or_estimated=True,
        )
        cell["review_geometry"] = layer
        cell["steel_certified_geometry"] = dict(
            state="NOT_CERTIFIED", geometry=None, control_eligible=False
        )
        cell["review_gates"] = dict(
            OPENING_TOPOLOGY_GATE="NOT_VERIFIED",
            REVIEW_GEOMETRY_GATE="NOT_VERIFIED",
            STEEL_PRECISION_GATE="NOT_CERTIFIED",
        )
        vessel_cells = [c for c in cells if c["vessel_id"] == cell["vessel_id"]]
        if len(vessel_cells) < 2 or not any(
            c.get("width_recovery_requested_sides") for c in vessel_cells
        ):
            layer["reason"] = "FAST_PATH_IDENTITY_AND_EXISTING_LOCAL_REVIEW_RETAINED"
            continue
        original = masks[key]
        if original is None or key not in extents:
            layer["reason"] = "FROZEN_MEMBER_PROVENANCE_MISSING"
            continue
        low = np.min([regions[r].search_envelope[:2] for r in cell["candidate_ids"]], 0)
        high = np.max(
            [regions[r].search_envelope[2:] for r in cell["candidate_ids"]], 0
        )
        for n in cells:
            nk = n["cell_id"]
            if nk == key or nk not in extents or n["vessel_id"] != cell["vessel_id"]:
                continue
            if np.mean(extents[nk][[0, 2]]) < np.mean(extents[key][[0, 2]]):
                low[0] = max(low[0], extents[nk][2])
            else:
                high[0] = min(high[0], extents[nk][0])
        envelope = np.r_[low, high]
        ids = local.index.query_indexes(envelope)
        ids = ids[local.owned[ids]]
        raw = local.xyz[ids]
        raw_ids = local.ids[ids]
        member = original[ids].copy()
        other = np.zeros(len(ids), bool)
        for n in cells:
            if (
                n["cell_id"] != key
                and n["vessel_id"] == cell["vessel_id"]
                and masks[n["cell_id"]] is not None
            ):
                other |= masks[n["cell_id"]][ids]
        member &= ~other
        core, audit = spatial_interior_core(raw, member, raw_ids, p, config)
        layer["interior_support"] = audit
        layer["material_connected_core_count"] = (
            cell.get("rim_error_recovery", {})
            .get("member_partition", {})
            .get("trusted_member_count", 0)
        )
        if not core.any():
            layer["reason"] = "SPATIAL_CORE_UNOBSERVABLE_NOT_VACUOUS_PASS"
            continue
        evidence = {
            s: heldout_side(raw, raw_ids, core, member, s, p, envelope, config)
            for s in SIDES
        }
        all_evidence[key] = evidence
        plans[key] = dict(
            old_bounds=bounds,
            selected={},
            vessel_id=cell["vessel_id"],
            evidence=evidence,
            layer=layer,
            core=core,
            raw=raw,
            ids=ids,
            raw_ids=raw_ids,
            other=other,
            original=member,
            cell=cell,
        )
    for key, plan in plans.items():
        for side, e in plan["evidence"].items():
            neighbors = []
            if side.startswith("V"):
                for nk, other in plans.items():
                    if nk == key or other["vessel_id"] != plan["vessel_id"]:
                        continue
                    neighbor_best, _ = select_supported(other["evidence"][side], config)
                    if neighbor_best is not None:
                        neighbors.append(
                            (
                                neighbor_best["position_m"],
                                neighbor_best["position_uncertainty_m"],
                            )
                        )
            best, alternatives = select_supported(e, config, neighbors)
            e["mutually_exclusive_alternatives"] = [m["mode_id"] for m in alternatives]
            e["arbitration"] = (
                "RAW_SUPPORTED_VESSEL_TIE_BREAK"
                if best is not None and neighbors
                else (
                    "OWN_HELDOUT_EVIDENCE"
                    if best is not None
                    else "AMBIGUOUS_OR_UNOBSERVABLE"
                )
            )
            if best is None:
                continue
            i = {"U0": 0, "V0": 1, "U1": 2, "V1": 3}[side]
            # A well-explained old side stays; uncertainty is not permission to move it.
            movement = abs(best["position_m"] - plan["old_bounds"][i])
            gain = best["holdout_coverage"] - e["old_holdout_coverage"]
            if gain <= 0 or movement <= best["position_uncertainty_m"]:
                continue
            plan["selected"][side] = best
            e["review_selected_mode_id"] = best["mode_id"]
            e["mutually_exclusive_alternatives"] = [m["mode_id"] for m in alternatives]
    if not plans:
        return
    ordered = [c for c in cells if c["cell_id"] in plans]
    bounds_by_id, links = joint_bounds(ordered, plans, config)
    for key, plan in plans.items():
        layer = plan["layer"]
        b = bounds_by_id[key]
        old = plan["old_bounds"]
        cell = plan["cell"]
        raw = plan["raw"]
        layer.update(
            side_evidence=plan["evidence"],
            joint_soft_factors=links,
            selected_modes={s: m["mode_id"] for s, m in plan["selected"].items()},
        )
        if not plan["selected"]:
            layer["reason"] = "NO_OWN_HELDOUT_GAIN"
            continue
        if b[2] <= b[0] or b[3] <= b[1]:
            layer["reason"] = "NONPOSITIVE_RECTANGLE"
            continue
        # Keep all existing separator inequalities; never remove a conflicting guard.
        conflicts = []
        for guard in cell["separator_constraints"]:
            axis = guard.get("axis", 0)
            value = guard["value"]
            sense = guard.get("sense", guard.get("bound"))
            if sense in ("MIN", "LOWER", "lower_u") and b[axis] < value - 64 * np.finfo(
                float
            ).eps * max(1.0, abs(value)):
                conflicts.append(guard)
            if sense in ("MAX", "UPPER", "upper_u") and b[
                axis + 2
            ] > value + 64 * np.finfo(float).eps * max(1.0, abs(value)):
                conflicts.append(guard)
        lost = ~inside(raw[plan["core"], :2], b, config["geometry"]["plane_inlier_m"])
        captured = plan["other"] & inside(raw[:, :2], b) & ~inside(raw[:, :2], old)
        layer["guards"] = dict(
            spatial_core_loss=int(lost.sum()),
            neighbor_capture=int(captured.sum()),
            separator_crossing=len(conflicts),
            empty_core_pass_forbidden=True,
        )
        if np.any(lost) or np.any(captured) or conflicts:
            layer["reason"] = "SPATIAL_OR_NEIGHBOR_OR_SEPARATOR_GUARD"
            continue
        uv = rectangle(b)
        xy = uv @ axes
        reclassified = (
            plan["original"] & inside(raw[:, :2], old) & ~inside(raw[:, :2], b)
        )
        layer.update(
            state="REVIEW_ESTIMATED",
            adopted_for_review=True,
            boundary_uv=uv.tolist(),
            boundary_xy=xy.tolist(),
            corners_xyz=[[float(x), float(y), None] for x, y in xy],
            **dimensions(xy.tolist()),
            source_family="OWN_SPATIALLY_DISJOINT_RAW_SECTIONS_WITH_SOFT_VESSEL_CONTEXT",
            unobserved_or_estimated=True,
            reason="HELDOUT_RAW_GAIN_AND_SPATIAL_CORE_PROTECTED;NOT_STEEL_CERTIFIED",
            reclassified_boundary_raw_count=int(reclassified.sum()),
            reclassified_boundary_raw_digest=digest(plan["raw_ids"][reclassified]),
            reclassified_boundary_raw_indexes=plan["raw_ids"][reclassified].tolist(),
            original_frozen_members_modified=False,
            per_side_intervals={
                s: m["position_interval_m"] for s, m in plan["selected"].items()
            },
            unknown_or_retained_sides=[s for s in SIDES if s not in plan["selected"]],
            side_source_families={
                s: (
                    plan["selected"][s]["source_family"]
                    if s in plan["selected"]
                    else "OLD_SIDE_PRIOR_IN_COMMON_VESSEL_FRAME"
                )
                for s in SIDES
            },
            core_containment_tolerance_m=config["geometry"]["plane_inlier_m"],
            max_corner_move_m=float(
                np.max(
                    np.linalg.norm(
                        xy - np.asarray(cell["geometry"]["boundary_xy"]), axis=1
                    )
                )
            ),
            grade=(
                "B"
                if all(m["grade"] == "B" for m in plan["selected"].values())
                else "C"
            ),
            z_reason="REVIEW_XY_CHANGED;NO_OLD_Z_TRANSFERRED",
            opening_identity=cell["primary_region_id"],
            vessel_hypothesis_id=cell["vessel_id"],
        )
