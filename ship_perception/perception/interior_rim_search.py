"""Recover side observations from an established OpeningCell's interior.

This consumes the frozen vessel/Cell identities. It cannot recall or merge
openings, and a corridor prediction is never promoted to a Raw measurement.
"""

import time
import numpy as np
from scipy.spatial import cKDTree

from .local_rim_evidence import digest, gravity_surfaces, isotropic_face
from .opening_rim_source import member_mask
from .sections import physical_sections
from .recovery_normals import SpatialRawNormals


def metric_bins(values, step):
    scaled = np.asarray(values) / step
    return np.floor(
        scaled + 64 * np.finfo(float).eps * np.maximum(1.0, abs(scaled))
    ).astype(np.int64)


def cell_members(cell, regions, raw, owned):
    members = np.zeros(len(raw), bool)
    for key in cell["candidate_ids"]:
        region = regions.get(key)
        if region is None or region.vessel_id != cell["vessel_id"]:
            return None, "CELL_MEMBER_PROVENANCE_INCOMPLETE"
        mask = member_mask(raw, region)
        if mask is None:
            return None, "CELL_MEMBER_PROVENANCE_INCOMPLETE"
        members |= mask
    return members & owned, "FROZEN_SAME_CELL_ALIAS_MEMBER_UNION"


def interior_partition(points, members, indexes, polygon, config, normal_cache=None):
    """Keep frozen masks; isolate connected, full-3D supported interior material."""
    g, b = config["geometry"], config["boundary"]
    step = g["coarse_voxel_m"]
    origin = np.asarray(polygon).min(0)
    cells = metric_bins(points[:, :2] - origin, step)
    valid = np.flatnonzero(members)
    empty = dict(
        status="INTERIOR_ANCHOR_UNOBSERVABLE",
        original_member_count=len(valid),
        original_member_digest=digest(indexes[valid]),
        trusted_member_count=0,
        uncertain_member_count=len(valid),
        original_mask_modified=False,
    )
    if not len(valid):
        return np.zeros(len(points), bool), empty
    keys, inverse, counts = np.unique(
        cells[valid], axis=0, return_inverse=True, return_counts=True
    )
    grouped = np.argsort(inverse, kind="stable")
    ends = np.r_[0, np.cumsum(counts)]
    groups = [valid[grouped[ends[i] : ends[i + 1]]] for i in range(len(keys))]
    centers = origin + (keys + 0.5) * step
    samples = []
    for i, selected in enumerate(groups):
        target = np.r_[centers[i], np.median(points[selected, 2])]
        delta = points[selected] - target
        dist = np.einsum("ij,ij->i", delta, delta)
        order = np.lexsort(
            (points[selected, 2], points[selected, 1], points[selected, 0], dist)
        )
        samples.append(selected[order[0]])
    samples = np.asarray(samples)
    normal_cache = (
        normal_cache
        if normal_cache is not None
        else SpatialRawNormals(points, indexes, config)
    )
    valid_normals, normals = normal_cache.estimate(samples)
    nonvertical = valid_normals & (
        abs(normals[:, 2]) > np.sin(np.deg2rad(g["normal_angle_deg"]))
    )
    row = {}
    details = []
    for i, selected in enumerate(groups):
        n = normals[i]
        sample = points[samples[i]]
        errors = (points[selected] - sample) @ n
        residual = float(np.median(abs(errors - np.median(errors))))
        height = (
            float(sample[2] - (centers[i] - sample[:2]) @ n[:2] / n[2])
            if nonvertical[i]
            else float(sample[2])
        )
        details.append(
            dict(selected=selected, normal=n, seed=sample, height=height, mad=residual)
        )
        if counts[i] >= b["profile_min_points"] and nonvertical[i]:
            row[tuple(keys[i])] = i
    if not row:
        return np.zeros(len(points), bool), empty
    center = np.asarray(polygon).mean(0)
    seed = min(
        row,
        key=lambda key: float(
            np.linalg.norm(
                (origin + (np.asarray(key) + 0.5) * step - center)
                / np.maximum(np.ptp(polygon, axis=0), step)
            )
        ),
    )
    connected = {seed}
    pending = [seed]
    while pending:
        key = pending.pop()
        i = row[key]
        a = details[i]
        for offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            other = (key[0] + offset[0], key[1] + offset[1])
            if other in connected or other not in row:
                continue
            j = row[other]
            other_plane = details[j]
            shift = centers[j] - centers[i]
            forward = a["height"] - shift @ a["normal"][:2] / a["normal"][2]
            backward = (
                other_plane["height"]
                + shift @ other_plane["normal"][:2] / other_plane["normal"][2]
            )
            residual = max(
                abs(forward - other_plane["height"]), abs(backward - a["height"])
            )
            tolerance = (
                config["roi"]["opening_drop_m"]
                + a["mad"]
                + other_plane["mad"]
                + g["plane_inlier_m"]
            )
            if residual <= tolerance:
                connected.add(other)
                pending.append(other)
    if len(connected) < config["roi"]["min_opening_cells"]:
        return np.zeros(len(points), bool), dict(empty, connected_cells=len(connected))
    trusted = np.zeros(len(points), bool)
    lookup = {tuple(key): i for i, key in enumerate(keys)}
    for key in connected:
        a = details[row[key]]
        for offset in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            other = (key[0] + offset[0], key[1] + offset[1])
            if other not in lookup:
                continue
            selected = groups[lookup[other]]
            tolerance = max(g["plane_inlier_m"], a["mad"])
            if offset != (0, 0):
                tolerance += config["roi"]["opening_drop_m"]
            trusted[
                selected[abs((points[selected] - a["seed"]) @ a["normal"]) <= tolerance]
            ] = True
    anchor = np.median(points[trusted, :2], axis=0)
    return trusted, dict(
        empty,
        status="OWNED_CONNECTED_INTERIOR_HEIGHT_FAMILY",
        anchor_uv=anchor.tolist(),
        connected_cells=len(connected),
        trusted_member_count=int(trusted.sum()),
        trusted_member_digest=digest(indexes[trusted]),
        uncertain_member_count=int((members & ~trusted).sum()),
        uncertain_member_digest=digest(indexes[members & ~trusted]),
        material_normal_seed_raw_indexes=indexes[
            samples[[row[key] for key in sorted(connected)]]
        ].tolist(),
        uncertainty_basis="FULL_3D_NONSECTION_NORMALS_AND_CONNECTED_FROZEN_INTERIOR_HEIGHT_FAMILY",
    )


def profile_bins(points, indexes, step, origin):
    codes = metric_bins(points[:, 1] - origin, step)
    keys, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    ends = np.r_[0, np.cumsum(counts)]
    groups = [order[ends[i] : ends[i + 1]] for i in range(len(keys))]
    medians = np.array([np.median(points[g, 2]) for g in groups])
    return keys, counts, medians, groups


def raw_transition_nominations(points, indexes, span, search, config):
    """Coarse height transitions nominate; fine brackets retain original rows."""
    boundary = config["boundary"]
    coarse = config["geometry"]["candidate_voxel_m"]
    fine = config["geometry"]["refine_voxel_m"]
    minimum = boundary["profile_min_points"]
    if not len(points):
        return [], dict(span_m=list(span), raw_count=0, coarse_profile=[])
    keys, counts, z, groups = profile_bins(points, indexes, coarse, search[0])
    jump = np.flatnonzero(
        (counts[:-1] >= minimum)
        & (counts[1:] >= minimum)
        & ((np.diff(keys) - 1) * coarse <= boundary["max_support_gap_m"])
        & (abs(np.diff(z)) >= config["roi"]["opening_drop_m"])
    )
    proposed = []
    for i in jump:
        lo = search[0] + (keys[i] + 0.5) * coarse
        hi = search[0] + (keys[i + 1] + 0.5) * coarse
        local = (points[:, 1] >= lo - coarse) & (points[:, 1] <= hi + coarse)
        where = np.flatnonzero(local)
        raw = points[where]
        fk, fc, fz, fg = profile_bins(raw, indexes[where], fine, lo - coarse)
        valid = np.flatnonzero(
            (fc[:-1] >= minimum)
            & (fc[1:] >= minimum)
            & ((np.diff(fk) - 1) * fine <= boundary["profile_max_bracket_m"])
            & (abs(np.diff(fz)) >= boundary["profile_min_drop_m"])
        )
        if not len(valid):
            continue
        j = max(
            valid,
            key=lambda j: (
                abs(fz[j + 1] - fz[j]),
                -abs((lo + hi) / 2 - (lo - coarse + (fk[j] + 1) * fine)),
            ),
        )
        left, right = where[fg[j]], where[fg[j + 1]]
        bracket = [float(points[left, 1].max()), float(points[right, 1].min())]
        offset = float(np.mean(bracket))
        if not search[0] <= offset <= search[1]:
            continue
        ids = np.unique(np.r_[indexes[left], indexes[right]])
        proposed.append(
            dict(
                offset_m=offset,
                station_m=float(np.mean(span)),
                bracket_m=bracket,
                bracket_width_m=bracket[1] - bracket[0],
                signed_height_change_m=float(fz[j + 1] - fz[j]),
                outside_height_m=float(fz[j]),
                inside_height_m=float(fz[j + 1]),
                uncertainty_m=max(
                    config["geometry"]["plane_inlier_m"], (bracket[1] - bracket[0]) / 2
                ),
                profile_raw_indexes=ids.tolist(),
                raw_index_digest=digest(ids),
                observed=True,
                steel_certified=False,
            )
        )
    # Coarse neighbours can nominate the same fine bracket; no observation is
    # discarded from the IR, only identical bracket aliases are evaluated once.
    unique = {m["raw_index_digest"]: m for m in proposed}
    profile = [
        dict(
            t_m=float(search[0] + (key + 0.5) * coarse),
            raw_count=int(count),
            median_z_m=float(height),
        )
        for key, count, height in zip(keys, counts, z)
    ]
    return list(unique.values()), dict(
        span_m=list(span),
        raw_count=len(points),
        raw_index_digest=digest(indexes),
        coarse_resolution_m=coarse,
        fine_resolution_m=fine,
        coarse_profile=profile,
        coarse_transition_count=len(jump),
        fine_bracket_count=len(unique),
    )


def transition_families(sections, length, config):
    rows = []
    for i, section in enumerate(sections):
        distinct = []
        for mode in sorted(
            section["transitions"],
            key=lambda m: (
                -abs(m["signed_height_change_m"]),
                m["uncertainty_m"],
                m["offset_m"],
            ),
        ):
            if any(
                abs(mode["offset_m"] - old["offset_m"])
                <= config["boundary"]["max_support_gap_m"]
                for old in distinct
            ):
                continue
            distinct.append(mode)
            if len(distinct) >= config["geometry"]["max_plane_hypotheses"]:
                break
        section["family_nomination_digests"] = [
            mode["raw_index_digest"] for mode in distinct
        ]
        rows.extend(dict(mode, section_id=i) for mode in distinct)
    if not rows:
        return []
    s = np.array([r["station_m"] for r in rows])
    t = np.array([r["offset_m"] for r in rows])
    tolerance = np.maximum(
        [r["uncertainty_m"] for r in rows], config["boundary"]["profile_max_residual_m"]
    )
    seeds = []
    for i, a in enumerate(rows):
        seeds.append((0.0, t[i]))
        for j in range(i + 1, len(rows)):
            if rows[j]["section_id"] == a["section_id"]:
                continue
            slope = (t[j] - t[i]) / (s[j] - s[i])
            if abs(slope) <= np.tan(np.deg2rad(config["boundary"]["merge_angle_deg"])):
                seeds.append((slope, t[i] - slope * s[i]))
    candidates = []
    for slope, intercept in seeds:
        residual = abs(t - slope * s - intercept) / np.sqrt(1 + slope * slope)
        by_section = {}
        for i in np.flatnonzero(residual <= tolerance):
            key = rows[i]["section_id"]
            if key not in by_section or residual[i] < residual[by_section[key]]:
                by_section[key] = i
        ids = list(by_section.values())
        coverage = sum(np.ptp(sections[k]["span_m"]) for k in by_section) / length
        candidates.append(
            (coverage, len(ids), float(np.median(residual[ids])), slope, intercept, ids)
        )
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2], c[4], c[3]))
    accepted = []
    seen_evidence = set()
    for coverage, count, error, slope, intercept, ids in candidates:
        signature = tuple(sorted(rows[i]["raw_index_digest"] for i in ids))
        if signature in seen_evidence:
            continue
        seen_evidence.add(signature)
        if len(ids) >= 2:
            design = np.column_stack((s[ids] - length / 2, np.ones(len(ids))))
            base = 1 / tolerance[ids] ** 2
            weight = base.copy()
            for _ in range(config["geometry"]["max_plane_hypotheses"]):
                fitted = np.linalg.lstsq(
                    design * np.sqrt(weight[:, None]),
                    t[ids] * np.sqrt(weight),
                    rcond=None,
                )[0]
                residual = abs(t[ids] - design @ fitted)
                updated = base * np.minimum(
                    1.0, tolerance[ids] / np.maximum(residual, np.finfo(float).eps)
                )
                if np.allclose(updated, weight, rtol=0, atol=np.finfo(float).eps):
                    break
                weight = updated
            slope = float(fitted[0])
            intercept = float(fitted[1] - slope * length / 2)
            error = float(np.quantile(residual, 0.95))
        endpoints = np.array([intercept, intercept + slope * length])
        if any(
            np.max(
                abs(
                    endpoints
                    - np.array(
                        [f["intercept_m"], f["intercept_m"] + f["slope"] * length]
                    )
                )
            )
            <= config["boundary"]["merge_distance_m"]
            for f in accepted
        ):
            continue
        selected = [rows[i] for i in ids]
        accepted.append(
            dict(
                slope=float(slope),
                intercept_m=float(intercept),
                observed_section_coverage=float(coverage),
                independent_sections=count,
                profile_residual_m=error,
                sections=selected,
            )
        )
        if len(accepted) >= config["geometry"]["max_plane_hypotheses"]:
            break
    return accepted


def section_role(
    points,
    indexes,
    full,
    full_ids,
    tree,
    cut,
    member,
    other,
    config,
    surfaces,
    normal_cache=None,
    tangent_uv=None,
    inward_uv=None,
):
    b = config["boundary"]
    g = config["geometry"]
    probe = b["side_probe_m"]
    position_uncertainty = g["plane_inlier_m"]
    outer = (points[:, 1] >= cut - probe - position_uncertainty) & (
        points[:, 1] < cut - position_uncertainty
    )
    inner = (points[:, 1] >= cut + position_uncertainty) & (
        points[:, 1] < cut + probe + position_uncertainty
    )
    eligible = []
    for surface in surfaces:
        support = surface["_mask"] & outer
        raw = points[support]
        if (
            len(raw) < g["min_plane_points"]
            or np.ptp(raw[:, 1]) < b["profile_deck_support_span_m"]
            or np.ptp(raw[:, 0]) < b["line_min_length_m"]
        ):
            continue
        cells = metric_bins(raw[:, :2] - raw[:, :2].min(0), g["candidate_voxel_m"])
        fill = len(np.unique(cells, axis=0)) / np.prod(cells.max(0) + 1)
        if (
            fill < config["roi"]["deck_patch_fill_min"]
            or cut - raw[:, 1].max() > b["side_inner_gap_max_m"]
        ):
            continue
        eligible.append((surface, support, float(fill)))
    deck = None
    if eligible:
        plane, support, fill = min(eligible, key=lambda x: x[0]["residual_mad_m"])
        deck = {k: v for k, v in plane.items() if not k.startswith("_")}
        deck.update(
            raw_index_digest=digest(indexes[support]),
            source_point_indexes=indexes[support].tolist(),
            context_surface_ref=plane["surface_id"],
            fill=fill,
        )
    face = None
    if deck:
        face = (
            normal_cache.face(points, indexes, cut, tangent_uv, inward_uv)
            if normal_cache is not None
            else isotropic_face(points, indexes, full, full_ids, tree, cut, config)
        )
    attached = bool(
        np.count_nonzero(inner & member) >= b["profile_min_points"]
        and np.count_nonzero(outer & member) < b["profile_min_points"]
    )
    foreign = bool(np.any(other & (inner | outer)))
    cargo_ridge = False
    observed_interval = None
    if deck and face and face["qualified"]:
        face_points = full[np.isin(full_ids, face["point_indexes"])]
        coefficients = np.asarray(deck["coefficients"])
        deck_height = face_points[:, :2] @ coefficients[:2] + coefficients[2]
        cargo_ridge = bool(
            np.count_nonzero(
                deck_height - face_points[:, 2] >= b["face_vertical_span_min_m"]
            )
            < g["normal_min_points"]
        )
        observed_interval = [
            float(face_points[:, 0].min()),
            float(face_points[:, 0].max()),
        ]
    positive = bool(
        deck
        and face
        and face["qualified"]
        and attached
        and not foreign
        and not cargo_ridge
    )
    return dict(
        deck_plane=deck,
        vertical_face=face,
        inside_trusted_count=int(np.count_nonzero(inner & member)),
        outside_trusted_count=int(np.count_nonzero(outer & member)),
        other_cell_contamination=foreign,
        cargo_above_deck_face_only_risk=cargo_ridge,
        observed_face_s_interval_m=observed_interval,
        opening_attachment=attached,
        inner_rim_positive=positive,
        reason=(
            "OWNED_DECK_FACE_INTERIOR_CHAIN"
            if positive
            else "DECK_FACE_OR_TRUSTED_INTERIOR_ATTACHMENT_INCOMPLETE"
        ),
    )


class InteriorRimSearch:
    def __init__(self, local_rim):
        self.local = local_rim
        self.config = local_rim.config
        self._member_cache = {}
        self.partitions = {}

    def members(self, cell, regions):
        if cell["cell_id"] not in self._member_cache:
            self._member_cache[cell["cell_id"]] = cell_members(
                cell, regions, self.local.xyz, self.local.owned
            )
        return self._member_cache[cell["cell_id"]]

    def search(self, cell, polygon, regions, other_cells):
        started = time.perf_counter()
        cfg = self.config
        measurement = self.local.measurement
        original, reason = self.members(cell, regions)
        if original is None:
            return dict(status=reason, candidates={}, original_geometry_retained=True)
        p = np.asarray(polygon)
        margin = cfg["boundary"]["profile_half_length_m"]
        ids = self.local.index.query_indexes(
            np.r_[p.min(0) - margin, p.max(0) + margin]
        )
        contamination = int((~self.local.owned[ids]).sum())
        ids = ids[self.local.owned[ids]]
        raw = self.local.xyz[ids]
        raw_ids = self.local.ids[ids]
        other = np.zeros(len(ids), bool)
        for neighbor in other_cells:
            if (
                neighbor["cell_id"] == cell["cell_id"]
                or neighbor["vessel_id"] != cell["vessel_id"]
            ):
                continue
            mask, _ = self.members(neighbor, regions)
            if mask is not None:
                other |= mask[ids]
        owned = original[ids] & ~other
        normal_cache = SpatialRawNormals(raw, raw_ids, cfg) if len(raw) else None
        trusted, partition = interior_partition(
            raw, owned, raw_ids, p, cfg, normal_cache
        )
        self.partitions[cell["cell_id"]] = dict(
            indexes=ids,
            trusted=trusted,
            original=original,
            other=other,
            raw_source_indexes=raw_ids,
        )
        report = dict(
            status="INTERIOR_SEARCH_COMPLETED",
            cell_id=cell["cell_id"],
            member_partition=partition,
            contamination_raw_count=contamination,
            source_raw_index_digest=digest(raw_ids),
            original_geometry_retained=True,
            candidates={},
            steel_certified=False,
        )
        if partition["status"] != "OWNED_CONNECTED_INTERIOR_HEIGHT_FAMILY":
            possible = np.flatnonzero(owned)
            if not len(possible):
                report["status"] = "NO_OWNED_INTERIOR_SEARCH_ANCHOR"
                return report
            center = p.mean(0)
            distance = np.linalg.norm(
                (raw[possible, :2] - center)
                / np.maximum(np.ptp(p, axis=0), cfg["geometry"]["coarse_voxel_m"]),
                axis=1,
            )
            nearest = possible[int(np.argmin(distance))]
            partition.update(
                anchor_uv=raw[nearest, :2].tolist(),
                search_anchor_raw_index=int(raw_ids[nearest]),
                search_only=True,
                automatic_recovery_allowed=False,
            )
            report["status"] = "OWNED_INTERIOR_ANCHOR_REQUIRES_MATERIAL_CONFIRMATION"
        a = p[0]
        tangent = p[1] - a
        length = float(np.linalg.norm(tangent))
        tangent /= length
        inward = np.array([-tangent[1], tangent[0]])
        if (p.mean(0) - a) @ inward < 0:
            inward = -inward
        cross = float(np.mean((p[[3, 2]] - a) @ inward))
        base = np.column_stack(
            ((raw[:, :2] - a) @ tangent, (raw[:, :2] - a) @ inward, raw[:, 2])
        )
        live = (
            (base[:, 0] >= -margin)
            & (base[:, 0] <= length + margin)
            & (base[:, 1] >= -cfg["boundary"]["side_probe_m"])
            & (base[:, 1] <= cross + cfg["boundary"]["side_probe_m"])
        )
        base, raw_ids, trusted, other = (
            base[live],
            raw_ids[live],
            trusted[live],
            other[live],
        )
        for side in cell["width_recovery_requested_sides"]:
            sign = 1 if side == "V0" else -1
            origin = a if sign == 1 else a + cross * inward
            points = base.copy()
            points[:, 1] = base[:, 1] if sign == 1 else cross - base[:, 1]
            anchor = float(
                (np.asarray(partition["anchor_uv"]) - origin) @ (inward * sign)
            )
            search = (-cfg["boundary"]["side_probe_m"], anchor)
            tree = None
            limits = physical_sections(
                (0.0, length), cfg["geometry"]["candidate_voxel_m"], cfg
            )
            sections = []
            masks = []
            for section_index, (lo, hi) in enumerate(zip(limits[:-1], limits[1:])):
                live = (
                    (points[:, 0] >= lo)
                    & (
                        (points[:, 0] <= hi)
                        if section_index == len(limits) - 2
                        else (points[:, 0] < hi)
                    )
                    & (points[:, 1] >= search[0])
                    & (points[:, 1] <= search[1])
                )
                transitions, section = raw_transition_nominations(
                    points[live], raw_ids[live], (lo, hi), search, cfg
                )
                section["transitions"] = transitions
                sections.append(section)
                masks.append(live)
            families = transition_families(sections, length, cfg)
            surface_cache = {}
            candidates = []
            for family in families:
                roles = []
                face_ids = []
                if (
                    family["independent_sections"]
                    >= cfg["boundary"]["profile_min_sections"]
                ):
                    for transition in family["sections"]:
                        section_id = transition["section_id"]
                        live = masks[section_id]
                        if section_id not in surface_cache:
                            surface_cache[section_id] = gravity_surfaces(
                                points[live], raw_ids[live], cfg
                            )
                        role = section_role(
                            points[live],
                            raw_ids[live],
                            points,
                            raw_ids,
                            tree,
                            transition["offset_m"],
                            trusted[live],
                            other[live],
                            cfg,
                            surface_cache[section_id],
                            normal_cache,
                            tangent,
                            sign * inward,
                        )
                        roles.append(dict(section_id=section_id, **role))
                        if (
                            role["vertical_face"]
                            and role["vertical_face"]["qualified"]
                            and not role["other_cell_contamination"]
                        ):
                            face_ids.extend(role["vertical_face"]["point_indexes"])
                ids_face = np.unique(face_ids).astype(np.int64)
                n_st = np.array([-family["slope"], 1.0])
                n_st /= np.linalg.norm(n_st)
                n_uv = n_st[0] * tangent + n_st[1] * sign * inward
                rho = float(family["intercept_m"] * n_st[1] + origin @ n_uv)
                support_ids = np.unique(
                    [i for m in family["sections"] for i in m["profile_raw_indexes"]]
                ).astype(np.int64)
                uncertainty = max(
                    cfg["geometry"]["plane_inlier_m"],
                    family["profile_residual_m"],
                    max(m["uncertainty_m"] for m in family["sections"]),
                )
                profile_uncertainty = uncertainty
                if len(ids_face):
                    observed = self.local.xyz[self.local.positions(ids_face), :2]
                    weights = np.zeros(len(ids_face))
                    for row in roles:
                        if (
                            not row["vertical_face"]
                            or not row["vertical_face"]["qualified"]
                            or row["other_cell_contamination"]
                        ):
                            continue
                        belongs = np.isin(
                            ids_face, row["vertical_face"]["point_indexes"]
                        )
                        weights[belongs] = 1 / max(1, int(belongs.sum()))
                    base_weight = weights.copy()
                    for _ in range(cfg["geometry"]["max_plane_hypotheses"]):
                        center = np.average(observed, axis=0, weights=weights)
                        delta = observed - center
                        _, _, vectors = np.linalg.svd(
                            delta * np.sqrt(weights[:, None]), full_matrices=False
                        )
                        fitted = vectors[-1]
                        errors = abs(delta @ fitted)
                        updated = base_weight * np.minimum(
                            1.0,
                            cfg["boundary"]["line_inlier_m"]
                            / np.maximum(errors, np.finfo(float).eps),
                        )
                        if np.allclose(
                            updated, weights, rtol=0, atol=np.finfo(float).eps
                        ):
                            break
                        weights = updated
                    if fitted @ n_uv < 0:
                        fitted = -fitted
                    n_uv = fitted
                    rho = float(center @ n_uv)
                    uncertainty = max(
                        cfg["geometry"]["plane_inlier_m"],
                        float(np.quantile(abs(observed @ n_uv - rho), 0.95)),
                    )
                    for row in roles:
                        if (
                            row["vertical_face"]
                            and row["vertical_face"]["qualified"]
                            and not row["other_cell_contamination"]
                        ):
                            belongs = np.isin(
                                ids_face, row["vertical_face"]["point_indexes"]
                            )
                            uncertainty = max(
                                uncertainty,
                                float(np.median(abs(observed[belongs] @ n_uv - rho))),
                            )
                if n_uv[1] < 0:
                    n_uv = -n_uv
                    rho = -rho
                along_dir = np.array([-n_uv[1], n_uv[0]])
                endpoints = np.array(
                    [
                        origin + family["intercept_m"] * sign * inward,
                        origin
                        + length * tangent
                        + (family["intercept_m"] + family["slope"] * length)
                        * sign
                        * inward,
                    ]
                )
                endpoints += (rho - endpoints @ n_uv)[:, None] * n_uv
                actual_points = (
                    observed
                    if len(ids_face)
                    else self.local.xyz[self.local.positions(support_ids), :2]
                )
                actual_along = actual_points @ along_dir
                span = [float(actual_along.min()), float(actual_along.max())]
                positive = [
                    row["section_id"] for row in roles if row["inner_rim_positive"]
                ]
                observed_spans = [
                    row["observed_face_s_interval_m"]
                    for row in roles
                    if row["inner_rim_positive"]
                ]
                coverage = sum(hi - lo for lo, hi in observed_spans) / length
                qualified = bool(
                    len(positive) >= cfg["boundary"]["profile_min_sections"]
                    and coverage >= cfg["boundary"]["min_edge_coverage"]
                    and uncertainty <= cfg["boundary"]["profile_max_residual_m"]
                )
                candidates.append(
                    dict(
                        mode_id="INTERIOR_RAW:" + digest(support_ids),
                        side=side,
                        normal_uv=n_uv.tolist(),
                        rho=rho,
                        support_span=span,
                        source="MEASURED_RAW" if qualified else "INFERRED_GEOMETRY",
                        state="MEASURED_WEAK" if qualified else "CONSERVATIVE_PRIOR",
                        uncertainty_m=uncertainty,
                        profile_localization_uncertainty_m=profile_uncertainty,
                        uncertainty_scope=(
                            "OWN_RAW_FACE_POSITION"
                            if qualified
                            else "PROFILE_BRACKET_NOT_CERTIFIED_STEEL_POSITION"
                        ),
                        source_point_indexes=ids_face.tolist(),
                        profile_source_point_indexes=support_ids.tolist(),
                        raw_index_digest=digest(ids_face),
                        profile_raw_index_digest=digest(support_ids),
                        observation_state=(
                            "OBSERVED_INNER_RIM"
                            if qualified
                            else (
                                "PARTIAL_INNER_RIM_SUPPORT"
                                if len(ids_face)
                                else "GEOMETRIC_TRANSITION_ONLY"
                            )
                        ),
                        independent_raw_sections=len(positive),
                        independently_observed_face_sections=sum(
                            bool(
                                row["vertical_face"]
                                and row["vertical_face"]["qualified"]
                            )
                            for row in roles
                        ),
                        physical_role_improved=qualified,
                        independent_opening_role_gain=qualified,
                        opening_assignment_state=(
                            "TRUSTED_INTERIOR_RAW_CHAIN"
                            if qualified
                            else "PARTIAL_OR_UNRESOLVED"
                        ),
                        physical_role=(
                            "OBSERVED_INNER_RIM"
                            if qualified
                            else (
                                "PARTIAL_INNER_RIM_SUPPORT"
                                if positive
                                else "RAW_TRANSITION_WITHOUT_COMPLETE_STEEL_ROLE"
                            )
                        ),
                        qualified_for_recovery=qualified,
                        measured_section_spans_m=observed_spans,
                        predicted_endpoints_uv=endpoints.tolist(),
                        observed_section_coverage=float(coverage),
                        profile_section_coverage=family["observed_section_coverage"],
                        raw_support_ids=[
                            row["raw_index_digest"] for row in family["sections"]
                        ],
                        section_evidence=roles,
                        transition_sections=family["sections"],
                        search_prior_is_measurement=False,
                        steel_certified=False,
                    )
                )
            report["candidates"][side] = dict(
                search_interval_inward_m=list(search),
                old_side_origin_uv=origin.tolist(),
                inward_normal_uv=(sign * inward).tolist(),
                along_tangent_uv=tangent.tolist(),
                length_m=length,
                state=(
                    "CANDIDATES_REQUIRE_AB" if candidates else "INNER_RIM_UNOBSERVABLE"
                ),
                sections=sections,
                candidates=candidates,
                surface_sources={
                    surface["surface_id"]: dict(
                        **{k: v for k, v in surface.items() if not k.startswith("_")},
                        source_point_indexes=raw_ids[masks[section_id]][
                            surface["_mask"]
                        ].tolist(),
                    )
                    for section_id, surfaces in surface_cache.items()
                    for surface in surfaces
                },
            )
        report["wall_s"] = time.perf_counter() - started
        return report
