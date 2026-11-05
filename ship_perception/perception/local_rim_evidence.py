"""Cell-local Raw plane, step and continuation evidence in one review frame."""

import hashlib
import time
import numpy as np

from .sections import physical_sections


def digest(indexes):
    return hashlib.sha256(
        np.sort(indexes).astype("<i8", copy=False).tobytes()
    ).hexdigest()


def plane(points, config, horizontal):
    minimum = config["geometry"]["normal_min_points"]
    if len(points) < minimum:
        return None
    center = np.median(points, axis=0)
    _, s, v = np.linalg.svd(points - center, full_matrices=False)
    if len(s) < 3 or s[1] <= np.finfo(float).eps:
        return None
    normal = v[-1]
    residual = float(np.median(abs((points - center) @ normal)))
    tilt = float(abs(normal[2]))
    angular = np.sin(np.deg2rad(config["geometry"]["normal_angle_deg"]))
    qualified = (
        tilt >= np.cos(np.deg2rad(config["boundary"]["profile_deck_slope_max_deg"]))
        if horizontal
        else tilt <= angular
    )
    return dict(
        normal_stz=normal.tolist(),
        residual_mad_m=residual,
        point_count=len(points),
        qualified=bool(qualified and residual <= config["geometry"]["plane_inlier_m"]),
        role="HORIZONTAL_DECK" if horizontal else "VERTICAL_FACE",
    )


from scipy.spatial import cKDTree


def gravity_surfaces(points, indexes, config):
    """Density-balanced nominations; every plane is rechecked on original Raw."""
    geom, boundary = config["geometry"], config["boundary"]
    if len(points) < geom["min_plane_points"]:
        return []
    step = geom["candidate_voxel_m"]
    origin = np.min(points, axis=0)
    cells = np.floor((points - origin) / step).astype(np.int64)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    _, positions = np.unique(cells[order], axis=0, return_index=True)
    coarse = points[order[positions]]
    tol = geom["plane_inlier_m"]
    keys = np.floor((coarse[:, 2] - origin[2]) / tol).astype(np.int64)
    levels, counts = np.unique(keys, return_counts=True)
    density = dict(zip(levels, counts))
    peaks = [
        k
        for k in levels
        if density[k] >= geom["normal_min_points"]
        and density[k] >= density.get(k - 1, 0)
        and density[k] > density.get(k + 1, 0)
    ]
    peaks = sorted(peaks, key=lambda k: (-density[k], k))[
        : geom["max_plane_hypotheses"]
    ]
    result = []
    for level in peaks:
        height = float(np.median(coarse[keys == level, 2]))
        coefficients = np.array([0.0, 0.0, height])
        mask = abs(coarse[:, 2] - height) <= tol
        fitted_valid = False
        for _ in range(geom["max_plane_hypotheses"]):
            if mask.sum() < geom["normal_min_points"]:
                break
            raw = coarse[mask]
            center = np.median(raw, axis=0)
            design = np.column_stack((raw[:, :2] - center[:2], np.ones(len(raw))))
            fitted, _, rank, _ = np.linalg.lstsq(
                design, raw[:, 2] - center[2], rcond=None
            )
            if rank < 3:
                fitted_valid = False
                break
            fitted[2] += center[2] - center[:2] @ fitted[:2]
            if np.linalg.norm(fitted[:2]) > np.tan(
                np.deg2rad(boundary["profile_deck_slope_max_deg"])
            ):
                fitted_valid = False
                break
            residual = (
                coarse[:, 2] - coarse[:, :2] @ fitted[:2] - fitted[2]
            ) / np.sqrt(1 + fitted[:2] @ fitted[:2])
            updated = abs(residual) <= tol
            coefficients = fitted
            fitted_valid = True
            if np.array_equal(mask, updated):
                mask = updated
                break
            mask = updated
        if not fitted_valid:
            continue
        normal = np.r_[-coefficients[:2], 1.0]
        normal /= np.linalg.norm(normal)
        residual = (
            points[:, 2] - points[:, :2] @ coefficients[:2] - coefficients[2]
        ) * normal[2]
        support = abs(residual) <= tol
        selected = points[support]
        if (
            len(selected) < geom["min_plane_points"]
            or np.ptp(selected[:, 0]) < boundary["line_min_length_m"]
            or np.ptp(selected[:, 1]) < boundary["profile_deck_support_span_m"]
        ):
            continue
        anchor = np.median(points[:, :2], axis=0)
        if any(
            abs(np.r_[anchor, 1.0] @ (coefficients - np.asarray(r["coefficients"])))
            <= tol
            and np.count_nonzero(support & r["_mask"])
            / min(support.sum(), r["_mask"].sum())
            >= config["frame"]["same_plane_overlap_min"]
            for r in result
        ):
            continue
        result.append(
            dict(
                surface_id="DECK:" + digest(indexes[support]),
                coefficients=coefficients.tolist(),
                normal_stz=normal.tolist(),
                raw_count=int(support.sum()),
                raw_index_digest=digest(indexes[support]),
                residual_mad_m=float(np.median(abs(residual[support]))),
                along_span_m=float(np.ptp(selected[:, 0])),
                cross_span_m=float(np.ptp(selected[:, 1])),
                _mask=support,
                qualified=True,
            )
        )
    return result


def isotropic_face(points, indexes, full_points, full_indexes, tree, offset, config):
    """A narrow band nominates samples; covariance sees a full 3-D neighbourhood."""
    geom, boundary = config["geometry"], config["boundary"]
    nominated = np.flatnonzero(abs(points[:, 1] - offset) <= geom["candidate_voxel_m"])
    empty = dict(
        qualified=False,
        method="ISOTROPIC_3D_NEIGHBOURHOOD",
        point_indexes=[],
        raw_index_digest=digest(np.array([], dtype=np.int64)),
        normal_sample_count=0,
    )
    if len(nominated) < geom["normal_min_points"]:
        return empty
    spacing = max(boundary["face_vertical_span_min_m"], geom["candidate_voxel_m"])
    sample_points = points[nominated]
    cells = np.floor(
        (sample_points[:, [0, 2]] - sample_points[:, [0, 2]].min(0)) / spacing
    ).astype(np.int64)
    order = np.lexsort((sample_points[:, 2], sample_points[:, 1], sample_points[:, 0]))
    _, where = np.unique(cells[order], axis=0, return_index=True)
    sample = nominated[order[where]]
    if len(sample) > geom["max_plane_hypotheses"]:
        sample = sample[
            np.linspace(0, len(sample) - 1, geom["max_plane_hypotheses"]).astype(int)
        ]
    distance, near = tree.query(
        points[sample],
        k=geom["normal_k"],
        distance_upper_bound=geom["normal_radius_m"],
        workers=1,
    )
    live = []
    normals = []
    for row, dist in zip(near, distance):
        row = row[np.isfinite(dist)]
        if len(row) < geom["normal_min_points"]:
            continue
        cloud = full_points[row]
        delta = cloud - cloud.mean(0)
        values, vectors = np.linalg.eigh(delta.T @ delta / len(delta))
        maximum = max(values[2], np.finfo(float).tiny)
        n = vectors[:, 0]
        if (
            values[1] / maximum < geom["normal_min_secondary_ratio"]
            or (values[1] - values[0]) / maximum < geom["planarity_min"]
            or abs(n[2]) > np.sin(np.deg2rad(geom["normal_angle_deg"]))
            or abs(n[1]) < np.cos(np.deg2rad(geom["normal_angle_deg"]))
        ):
            continue
        normal = n if n[1] > 0 else -n
        normals.append(normal)
        live.extend(
            row[
                abs((cloud - cloud.mean(0)) @ normal) <= geom["plane_inlier_m"]
            ].tolist()
        )
    if not live:
        return empty
    live = np.unique(live)
    observed = full_points[live]
    keep = (
        (observed[:, 0] >= points[:, 0].min())
        & (observed[:, 0] <= points[:, 0].max())
        & (abs(observed[:, 1] - offset) <= geom["candidate_voxel_m"])
    )
    live, observed = live[keep], observed[keep]
    if not len(live):
        return empty
    # Each exported support point must itself have a 3-D vertical normal;
    # points from an adjacent deck can also lie inside a wall's distance band.
    distance, near = tree.query(
        observed,
        k=geom["normal_k"],
        distance_upper_bound=geom["normal_radius_m"],
        workers=1,
    )
    valid = np.isfinite(distance)
    count = valid.sum(1)
    neighbours = full_points[np.where(valid, near, 0)]
    delta = neighbours - observed[:, None, :]
    mean = (delta * valid[:, :, None]).sum(1) / np.maximum(count, 1)[:, None]
    centered = (delta - mean[:, None, :]) * valid[:, :, None]
    covariance = (
        np.einsum("nki,nkj->nij", centered, centered)
        / np.maximum(count, 1)[:, None, None]
    )
    values, vectors = np.linalg.eigh(covariance)
    maximum = np.maximum(values[:, 2], np.finfo(float).tiny)
    n = vectors[:, :, 0]
    keep = (
        (count >= geom["normal_min_points"])
        & (values[:, 1] / maximum >= geom["normal_min_secondary_ratio"])
        & ((values[:, 1] - values[:, 0]) / maximum >= geom["planarity_min"])
        & (abs(n[:, 2]) <= np.sin(np.deg2rad(geom["normal_angle_deg"])))
        & (abs(n[:, 1]) >= np.cos(np.deg2rad(geom["normal_angle_deg"])))
    )
    live, observed = live[keep], observed[keep]
    if not len(live):
        return empty
    vertical = float(np.ptp(observed[:, 2]))
    along = float(np.ptp(observed[:, 0]))
    ids = full_indexes[live]
    qualified = bool(
        len(ids) >= geom["min_plane_points"]
        and vertical >= boundary["face_vertical_span_min_m"]
        and along >= boundary["line_min_length_m"]
        and len(normals) >= boundary["line_min_points"]
    )
    normal = np.median(normals, axis=0)
    normal /= np.linalg.norm(normal)
    residual = abs((observed - np.median(observed, axis=0)) @ normal)
    return dict(
        qualified=qualified,
        method="ISOTROPIC_3D_NEIGHBOURHOOD",
        raw_count=len(ids),
        normal_sample_count=len(normals),
        normal_stz=np.median(normals, axis=0).tolist(),
        residual_mad_m=float(np.median(residual)),
        vertical_span_m=vertical,
        along_span_m=along,
        raw_index_digest=digest(ids),
        point_indexes=ids.tolist() if qualified else [],
        offset_m=float(np.median(observed[:, 1])),
    )


class LocalRimEvidence:
    def __init__(self, measurement, config):
        self.measurement, self.config = measurement, config
        primary = getattr(measurement.local, "primary_cache", measurement.local)
        self.index = measurement.spatial
        self.xyz = measurement.raw_uvz
        self.ids = measurement.raw_point_indexes
        self.owned = np.isin(self.ids, primary.vessel.point_indexes)
        self._order = None

    def positions(self, ids):
        if self._order is None:
            self._order = np.argsort(self.ids, kind="stable")
        positions = np.searchsorted(self.ids[self._order], ids)
        if np.any(positions >= len(self.ids)) or not np.array_equal(
            self.ids[self._order][positions], ids
        ):
            raise ValueError("LOCAL_RIM_SOURCE_OUTSIDE_CURRENT_RAW")
        return self._order[positions]

    def side(self, polygon, side, members=None, other=None):
        started = time.perf_counter()
        pairs = {"U0": (0, 3), "U1": (1, 2), "V0": (0, 1), "V1": (3, 2)}
        vertices = np.asarray(polygon, float)
        a, b = vertices[list(pairs[side])]
        length = float(np.linalg.norm(b - a))
        tangent = (b - a) / length
        normal = np.array([-tangent[1], tangent[0]])
        if (vertices.mean(0) - a) @ normal < 0:
            normal = -normal
        radius = self.config["roi"]["support_search_m"]
        margin = self.config["boundary"]["profile_half_length_m"]
        corners = np.array(
            [
                a + normal * radius,
                a - normal * radius,
                b + normal * radius,
                b - normal * radius,
            ]
        )
        candidates = self.index.query_indexes(
            np.r_[corners.min(0) - margin, corners.max(0) + margin]
        )
        positives = candidates[self.owned[candidates]]
        delta = self.xyz[positives, :2] - a
        st = np.column_stack((delta @ tangent, delta @ normal, self.xyz[positives, 2]))
        keep = (
            (st[:, 0] >= -margin)
            & (st[:, 0] <= length + margin)
            & (abs(st[:, 1]) <= radius)
        )
        positives, st = positives[keep], st[keep]
        ids = self.ids[positives]
        tree = cKDTree(st)
        member = np.zeros(len(st), bool) if members is None else members[positives]
        foreign = np.zeros(len(st), bool) if other is None else other[positives]
        edges = physical_sections(
            (0.0, length), self.config["geometry"]["candidate_voxel_m"], self.config
        )
        sections = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            selected = (st[:, 0] >= lo) & (st[:, 0] < hi)
            sections.append(
                self._section(
                    st[selected],
                    ids[selected],
                    (lo, hi),
                    member[selected],
                    foreign[selected],
                    st,
                    ids,
                    tree,
                    members is not None,
                )
            )
        minimum = self.config["boundary"]["profile_min_sections"]
        nomination_elapsed = time.perf_counter() - started
        tolerance = self.config["boundary"]["corner_join_m"]
        nominations = [
            (i, m)
            for i, s in enumerate(sections)
            for m in s["modes"]
            if m["inner_rim_positive"]
        ]
        groups = []
        for i, mode in sorted(nominations, key=lambda r: r[1]["offset_m"]):
            if (
                not groups
                or mode["offset_m"] - groups[-1][0][1]["offset_m"] > tolerance
            ):
                groups.append([])
            groups[-1].append((i, mode))
        coherent = []
        for group in groups:
            unique = {i: m for i, m in group}
            if len(unique) < minimum:
                continue
            source_ids = np.unique(
                np.concatenate(
                    [
                        np.array(m["vertical_face"]["point_indexes"], dtype=np.int64)
                        for m in unique.values()
                    ]
                )
            )
            points = self.xyz[self.positions(source_ids), :2]
            order = np.lexsort((points[:, 1], points[:, 0]))
            points = points[order]
            center = np.median(points, axis=0)
            weight = np.ones(len(points))
            fitted_normal = normal.copy()
            for _ in range(self.config["geometry"]["max_plane_hypotheses"]):
                center = np.average(points, axis=0, weights=weight)
                delta = points - center
                _, _, v = np.linalg.svd(
                    delta * np.sqrt(weight[:, None]), full_matrices=False
                )
                fitted_normal = v[-1]
                residual = abs(delta @ fitted_normal)
                updated = np.minimum(
                    1.0,
                    self.config["boundary"]["line_inlier_m"]
                    / np.maximum(residual, np.finfo(float).eps),
                )
                if np.allclose(updated, weight, rtol=0, atol=np.finfo(float).eps):
                    break
                weight = updated
            axis = 0 if side.startswith("U") else 1
            if fitted_normal[axis] < 0:
                fitted_normal = -fitted_normal
            rho = float(center @ fitted_normal)
            direction = np.array([-fitted_normal[1], fitted_normal[0]])
            along = points @ direction
            uncertainty = max(
                self.config["geometry"]["plane_inlier_m"],
                float(np.quantile(abs(points @ fitted_normal - rho), 0.95)),
            )
            spans = [sections[i]["span_m"] for i in sorted(unique)]
            coverage = sum(hi - lo for lo, hi in spans) / length
            offset = float(np.median([m["offset_m"] for m in unique.values()]))
            qualified = bool(
                coverage >= self.config["boundary"]["min_edge_coverage"]
                and uncertainty <= self.config["boundary"]["profile_max_residual_m"]
            )
            coherent.append(
                dict(
                    mode_id="LOCAL_RIM:" + digest(source_ids),
                    offset_m=offset,
                    normal_uv=fitted_normal.tolist(),
                    rho=rho,
                    support_span=[float(along.min()), float(along.max())],
                    source="MEASURED_RAW",
                    state="MEASURED_WEAK",
                    uncertainty_m=uncertainty,
                    source_point_indexes=source_ids.tolist(),
                    raw_index_digest=digest(source_ids),
                    raw_support_ids=sorted(
                        {m["raw_index_digest"] for m in unique.values()}
                    ),
                    independent_raw_sections=len(unique),
                    independent_sections=len(unique),
                    section_ids=sorted(unique),
                    owned_raw_source_hashes=sorted(
                        {m["raw_index_digest"] for m in unique.values()}
                    ),
                    physical_role="DECK_FACE_OWNED_OPENING_CHAIN",
                    physical_role_improved=True,
                    independent_opening_role_gain=False,
                    opening_assignment_state="ASSIGNED_CURRENT_OPENING",
                    qualified_for_shadow=qualified,
                    observed_section_coverage=coverage,
                    measured_section_spans_m=spans,
                    steel_certified=False,
                    evidence_family="ONE_RAW_OBSERVATION_FAMILY_WITH_DISJOINT_PHYSICAL_SECTIONS",
                )
            )
        consensus_elapsed = time.perf_counter() - started - nomination_elapsed
        band = self.config["boundary"]["side_probe_m"]
        extension = (abs(st[:, 1]) <= band) & ((st[:, 0] < 0) | (st[:, 0] > length))
        left = extension & (st[:, 0] < 0)
        right = extension & (st[:, 0] > length)
        continued = (
            left.sum() >= self.config["boundary"]["profile_min_points"]
            and right.sum() >= self.config["boundary"]["profile_min_points"]
        )
        deck_sections = sum(bool(s["deck_surfaces"]) for s in sections)
        positive_sections = sum(
            any(m["inner_rim_positive"] for m in s["modes"]) for s in sections
        )
        if len(coherent) > 1:
            state = "MULTIMODE_COMPETING"
        elif coherent:
            state = "INNER_RIM_OBSERVED"
        elif not len(st):
            state = "UNOBSERVED_OR_CROPPED"
        elif continued and deck_sections < minimum:
            state = "STRUCTURAL_CONTINUATION_WITHOUT_DECK_CONFIRMATION"
        else:
            state = "RETAINED_LEGACY_REVIEW"
        return dict(
            state=state,
            measurement_state=(
                "measured" if coherent else "estimated" if deck_sections else "unknown"
            ),
            old_line=dict(
                origin_uv=a.tolist(),
                tangent_uv=tangent.tolist(),
                inward_normal_uv=normal.tolist(),
                length_m=length,
            ),
            new_line=None,
            selected_reason="AWAITING_SHADOW_RECTANGLE_AB",
            side_candidates=coherent,
            rejected_alternatives=coherent,
            sections=sections,
            independent_section_count=len(sections),
            independently_supported_section_count=positive_sections,
            current_owned_raw_count=len(st),
            raw_index_digest=digest(ids),
            contamination_raw_count=int((~self.owned[candidates]).sum()),
            continuation=dict(
                before_count=int(left.sum()),
                after_count=int(right.sum()),
                raw_index_digest=digest(ids[extension]),
            ),
            raw_support=[m["raw_index_digest"] for m in coherent],
            source="OWNED_RAW_MULTISURFACE_V2",
            reason="PHYSICAL_ROLE_REQUIRES_DECK_FACE_AND_FROZEN_OPENING_MEMBERS",
            phase_wall_s=dict(
                LOCAL_RIM_NOMINATION=nomination_elapsed,
                SIDE_ROLE_CONSENSUS=consensus_elapsed,
            ),
            frame_contract="CELL_EDGE_ST_CONTINUOUS_V2",
            steel_certified=False,
        )

    def _section(
        self, points, indexes, span, members, other, full, full_ids, tree, member_known
    ):
        cfg = self.config
        minimum = cfg["boundary"]["profile_min_points"]
        probe = cfg["boundary"]["side_probe_m"]
        step = cfg["geometry"]["candidate_voxel_m"]
        surfaces = gravity_surfaces(points, indexes, cfg)
        modes = []
        if len(points) >= 2 * minimum:
            bins = np.floor(points[:, 1] / step).astype(np.int64)
            keys, inverse, counts = np.unique(
                bins, return_inverse=True, return_counts=True
            )
            z = np.full(len(keys), np.nan)
            for k in np.flatnonzero(counts >= minimum):
                z[k] = np.median(points[inverse == k, 2])
            jumps = (
                np.flatnonzero(
                    (np.diff(keys) == 1)
                    & np.isfinite(z[1:])
                    & np.isfinite(z[:-1])
                    & (abs(np.diff(z)) >= cfg["roi"]["opening_drop_m"])
                )
                + 1
            )
            for k in jumps:
                rho = keys[k] * step
                close = abs(points[:, 1] - rho) <= step
                offset = float(np.median(points[close, 1]))
                outer = (points[:, 1] >= offset - probe) & (points[:, 1] < offset)
                inner = (points[:, 1] >= offset) & (points[:, 1] < offset + probe)
                eligible = []
                for surface in surfaces:
                    support = surface["_mask"] & outer
                    raw = points[support]
                    if (
                        len(raw) < cfg["geometry"]["min_plane_points"]
                        or np.ptp(raw[:, 1])
                        < cfg["boundary"]["profile_deck_support_span_m"]
                        or np.ptp(raw[:, 0]) < cfg["boundary"]["line_min_length_m"]
                    ):
                        continue
                    cells = np.floor((raw[:, :2] - raw[:, :2].min(0)) / step).astype(
                        np.int64
                    )
                    fill = len(np.unique(cells, axis=0)) / np.prod(cells.max(0) + 1)
                    if (
                        fill < cfg["roi"]["deck_patch_fill_min"]
                        or offset - raw[:, 1].max()
                        > cfg["boundary"]["side_inner_gap_max_m"]
                    ):
                        continue
                    eligible.append((surface, support, float(fill)))
                deck = None
                if eligible:
                    selected, support, fill = min(
                        eligible, key=lambda s: s[0]["residual_mad_m"]
                    )
                    deck = {k: v for k, v in selected.items() if not k.startswith("_")}
                    deck.update(
                        raw_index_digest=digest(indexes[support]),
                        raw_count=int(support.sum()),
                        fill=fill,
                    )
                ready = min(outer.sum(), inner.sum()) >= minimum
                contrast = (
                    float(np.median(points[inner, 2]) - np.median(points[outer, 2]))
                    if ready
                    else None
                )
                wall = (
                    isotropic_face(points, indexes, full, full_ids, tree, offset, cfg)
                    if deck
                    and ready
                    and abs(contrast) >= cfg["boundary"]["profile_min_drop_m"]
                    else None
                )
                inside_count = int(np.count_nonzero(inner & members & ~other))
                outside_count = int(np.count_nonzero(outer & members))
                assignment = bool(
                    member_known
                    and inside_count >= minimum
                    and outside_count < minimum
                    and not np.any(other & (outer | inner))
                )
                positive = bool(deck and wall and wall["qualified"] and assignment)
                modes.append(
                    dict(
                        offset_m=offset,
                        deck_plane=deck,
                        vertical_face=wall,
                        signed_height_change_m=contrast,
                        inner_rim_positive=positive,
                        raw_index_digest=digest(indexes[outer | inner]),
                        deck_raw_index_digest=(
                            deck["raw_index_digest"] if deck else None
                        ),
                        face_raw_index_digest=(
                            wall["raw_index_digest"] if wall else None
                        ),
                        inward_raw_index_digest=digest(indexes[inner]),
                        outward_count=int(outer.sum()),
                        inward_count=int(inner.sum()),
                        opening_attachment=dict(
                            known=member_known,
                            inside_owned_member_count=inside_count,
                            outside_owned_member_count=outside_count,
                            assigned=assignment,
                            core_promoted=False,
                        ),
                        reason=(
                            "VERIFIED_RAW_CHAIN"
                            if positive
                            else "MISSING_DECK_FACE_OR_ONE_SIDED_OPENING_ATTACHMENT"
                        ),
                    )
                )
        return dict(
            span_m=list(span),
            raw_count=len(points),
            raw_index_digest=digest(indexes),
            modes=modes,
            deck_surfaces=[
                {k: v for k, v in s.items() if not k.startswith("_")} for s in surfaces
            ],
            legacy_deck_observed=any(m["deck_plane"] for m in modes),
            legacy_inside_occupied=False,
            legacy_outside_occupied=False,
            legacy_both_sides_cargo=False,
            nomination_grid_origin_t_m=0.0,
            full_raw_retained=True,
        )


def shadow_rectangle_ab(
    region, candidate, evidence, measurement, assembler, config, bridge
):
    """Only a measured Raw chain can challenge the unchanged legacy solve."""
    import copy
    from .constrained_rectangle import constrain_review_geometry, RectangleInfeasible

    legacy_error = None
    try:
        legacy = constrain_review_geometry(region, candidate, config, bridge)
    except RectangleInfeasible as error:
        legacy_error = error
        legacy = None
    audit = dict(
        status="RETAINED_LEGACY",
        reason="NO_UNIQUE_QUALIFIED_RAW_CHAIN",
        sides={},
        steel_certified=False,
    )
    trial = copy.deepcopy(candidate)
    selected = {}
    for side, entry in evidence.items():
        modes = [m for m in entry["side_candidates"] if m["qualified_for_shadow"]]
        entry["selected_reason"] = "NO_UNIQUE_QUALIFIED_RAW_CHAIN"
        if len(modes) != 1 or len(entry["side_candidates"]) != 1:
            continue
        mode = modes[0]
        if abs(mode["offset_m"]) > min(
            config["boundary"]["corner_join_m"], config["roi"]["boundary_search_m"] / 2
        ):
            entry["selected_reason"] = (
                "INDEPENDENT_CHAIN_RETAINED_LARGE_MOVE_OUTSIDE_EXISTING_GUARD"
            )
            continue
        assembler._apply_mode(trial["sides"][side], mode)
        if trial["sides"][side]["selected_mode_id"] != mode["mode_id"]:
            entry["selected_reason"] = "EXISTING_SIDE_OWNERSHIP_OR_MODE_CONFLICT"
            continue
        selected[side] = mode
    if selected and legacy is not None:
        try:
            alternate = constrain_review_geometry(region, trial, config, bridge)
            before = np.asarray(legacy["boundary_xy"])
            after = np.asarray(alternate["boundary_xy"])
            jump = float(np.max(np.linalg.norm(after - before, axis=1)))
            audit["max_corner_jump_m"] = jump
            accepted = jump <= config["boundary"]["corner_join_m"]
            primary = getattr(measurement.local, "primary_cache", measurement.local)
            axes = primary.vessel.axes
            old_uv = before @ axes.T
            new_uv = after @ axes.T
            ids = measurement.spatial.query_indexes(np.r_[old_uv.min(0), old_uv.max(0)])
            from .opening_rim_source import member_mask

            members = member_mask(measurement.raw_uvz, region)
            owned = np.isin(
                measurement.raw_point_indexes[ids], primary.vessel.point_indexes
            )
            points = (
                measurement.raw_uvz[ids[owned & members[ids]], :2]
                if members is not None
                else np.empty((0, 2))
            )

            def inside(polygon):
                tolerance = 1e-8 * max(1.0, float(np.ptp(polygon, axis=0).max()))
                return np.all(
                    np.array(
                        [
                            np.cross(polygon[(i + 1) % 4] - p, points - p) >= -tolerance
                            for i, p in enumerate(polygon)
                        ]
                    ),
                    axis=0,
                )

            lost = (
                int(np.count_nonzero(inside(old_uv) & ~inside(new_uv)))
                if len(points)
                else 0
            )
            audit["lost_owned_opening_member_points"] = lost
            audit["owned_member_points_checked"] = len(points)
            accepted = (
                accepted
                and members is not None
                and len(points) >= config["boundary"]["profile_min_points"]
                and not lost
            )
            local = LocalRimEvidence(measurement, config)
            for side, mode in selected.items():
                raw = measurement.raw_uvz[
                    local.positions(
                        np.asarray(mode["source_point_indexes"], dtype=np.int64)
                    ),
                    :2,
                ]
                old_line = legacy["sides"][side]
                new_line = alternate["sides"][side]
                old_residual = float(
                    np.median(
                        abs(
                            raw @ np.asarray(old_line["normal_uv"])
                            - old_line["final_rho"]
                        )
                    )
                )
                new_residual = float(
                    np.median(
                        abs(
                            raw @ np.asarray(new_line["normal_uv"])
                            - new_line["final_rho"]
                        )
                    )
                )
                gain = old_residual - new_residual
                audit["sides"][side] = dict(
                    old_raw_residual_m=old_residual,
                    new_raw_residual_m=new_residual,
                    gain_m=gain,
                    required_gain_m=mode["uncertainty_m"],
                    raw_index_digest=mode["raw_index_digest"],
                )
                accepted = accepted and gain > mode["uncertainty_m"]
            if accepted:
                audit.update(
                    status="COMMITTED_GEOMETRY",
                    reason="UNIQUE_RAW_CHAIN_IMPROVES_RESIDUAL_WITH_MEMBER_AND_SEPARATOR_PROTECTION",
                )
                for side, mode in selected.items():
                    evidence[side].update(
                        new_line=dict(
                            normal_uv=alternate["sides"][side]["normal_uv"],
                            rho=alternate["sides"][side]["final_rho"],
                        ),
                        selected_reason=audit["reason"],
                    )
                alternate["local_rim_shadow_ab"] = audit
                return alternate
            audit["reason"] = "INSUFFICIENT_RESIDUAL_GAIN_OR_CORNER_MEMBER_PROTECTION"
        except RectangleInfeasible as error:
            audit["reason"] = str(error)
        for side in selected:
            evidence[side]["selected_reason"] = audit["reason"]
    elif selected:
        audit["reason"] = "LEGACY_INFEASIBLE_REVIEW_ANCHOR_RETAINED"
    for entry in evidence.values():
        entry["shadow_rectangle_ab"] = audit
    if legacy_error is not None:
        raise legacy_error
    legacy["local_rim_shadow_ab"] = audit
    return legacy
