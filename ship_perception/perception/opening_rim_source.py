"""Measurement-only rim surfaces attached to an established opening.

This cannot recall openings, move polygons, or extend the detection core. A
context plane remains unclassified until observed crest and Raw attachment
evidence qualifies its inner sampling strip for this opening alone.
"""

from collections import deque
import hashlib
import json
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from .sections import physical_sections


def member_mask(points, region):
    evidence = region.identity_evidence
    grid = evidence.get("member_support_grid")
    cells = evidence.get("member_support_cells_rc", [])
    if not grid or not cells:
        return None
    rows = np.floor((points[:, 1] - grid["y0"]) / grid["cell_m"]).astype(np.int64)
    cols = np.floor((points[:, 0] - grid["x0"]) / grid["cell_m"]).astype(np.int64)
    cells = np.asarray(cells, dtype=np.int64)
    keys = (rows << 32) | (cols & 0xFFFFFFFF)
    selected = (cells[:, 0] << 32) | (cells[:, 1] & 0xFFFFFFFF)
    return np.isin(keys, selected)


def support_fingerprint(region):
    evidence = region.identity_evidence
    return hashlib.sha256(
        json.dumps(
            {
                k: evidence.get(k)
                for k in ("member_support_grid", "member_support_cells_rc")
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def point_fingerprint(indexes):
    return hashlib.sha256(
        np.asarray(sorted(indexes), dtype="<i8").tobytes()
    ).hexdigest()


class OpeningRimSource:
    def __init__(self, measurement, config, plane_fitter, nonvertical):
        self.measurement, self.config = measurement, config
        self.fit, self.nonvertical = plane_fitter, nonvertical
        self.raw = measurement.raw_uvz
        self.original = measurement.raw_point_indexes
        self.order = np.argsort(self.original)
        self.sorted_original = self.original[self.order]
        self.raw_index_sha256 = hashlib.sha256(
            np.asarray(self.sorted_original, dtype="<i8").tobytes()
        ).hexdigest()
        primary = getattr(measurement.local, "primary_cache", measurement.local)
        self.owned = np.isin(self.original, primary.vessel.point_indexes)
        self.vessel_id = primary.vessel.vessel_id
        self.audit = dict(
            status="NO_QUALIFIED_ASSOCIATION",
            core_modified=False,
            geometry_modified=False,
            candidate_patches=0,
            attachment_queries=0,
            admitted_local_sources=0,
            admitted_persistent_sources=0,
            side_families=[],
        )

    def positions(self, indexes):
        indexes = np.asarray(indexes, dtype=np.int64)
        positions = np.searchsorted(self.sorted_original, indexes)
        if (positions >= len(self.sorted_original)).any():
            raise ValueError("RIM_SOURCE_OUTSIDE_CURRENT_RAW")
        if not np.array_equal(self.sorted_original[positions], indexes):
            raise ValueError("RIM_SOURCE_OUTSIDE_CURRENT_RAW")
        return self.order[positions]

    def checkpoint(self):
        budget = getattr(self.measurement, "budget", None)
        if budget is not None:
            budget.check()

    def _components(self, plane):
        ids = np.asarray(plane["source_point_indexes"], dtype=np.int64)
        points = self.raw[self.positions(ids)]
        edges = cKDTree(points).query_pairs(
            self.config["boundary"]["max_support_gap_m"], output_type="ndarray"
        )
        graph = coo_matrix(
            (np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
            shape=(len(ids), len(ids)),
        ).tocsr()
        from scipy.sparse.csgraph import connected_components

        count, labels = connected_components(graph, directed=False)
        for label in range(count):
            live = labels == label
            if int(live.sum()) >= self.config["geometry"]["min_plane_points"]:
                yield ids[live], points[live]

    def _patches(self, result, side, sign, incident):
        model = result["constrained_review_rectangle"]["sides"][side]
        normal, rho = np.asarray(model["normal_uv"]), model["final_rho"]
        tangent = np.array([-normal[1], normal[0]])
        polygon = np.asarray(result["boundary_uv"])
        span = sorted((polygon[incident] @ tangent).tolist())
        sections = physical_sections(
            span, self.config["geometry"]["refine_voxel_m"], self.config
        )
        search = self.config["roi"]["boundary_search_m"]
        extent = self.config["roi"]["support_search_m"]
        probe = self.config["boundary"]["side_probe_m"]
        interval = self.config["boundary"]["max_support_gap_m"]
        width = self.config["roi"]["deck_patch_width_m"]
        minimum = self.config["boundary"]["profile_min_points"]
        output = []
        for section, (lo, hi) in enumerate(zip(sections[:-1], sections[1:])):
            self.checkpoint()
            corners = np.array(
                [
                    normal * (rho + sign * d) + tangent * along
                    for d in (-search, extent + probe)
                    for along in (lo, hi)
                ]
            )
            ids = self.measurement.spatial.query_indexes(
                [*corners.min(0), *corners.max(0)]
            )
            points = self.raw[ids]
            along = points[:, :2] @ tangent
            distance = sign * (points[:, :2] @ normal - rho)
            keep = (
                (along >= lo)
                & (along <= hi)
                & (distance >= -search)
                & (distance <= extent + probe)
            )
            ids, points, along, distance = (
                ids[keep],
                points[keep],
                along[keep],
                distance[keep],
            )
            eligible = distance <= extent
            segment = (
                result["opening_identity"]
                + ":"
                + side
                + ":OBSERVED_CREST_SECTION:"
                + str(section)
            )
            planes = self.fit(
                points[eligible],
                self.original[ids[eligible]],
                segment,
                "UNCLASSIFIED_SURFACE",
            )
            for plane in planes:
                for raw_ids, patch in self._components(plane):
                    bins = np.floor(((patch[:, :2] @ tangent) - lo) / interval).astype(
                        int
                    )
                    raw_bins = np.floor((along - lo) / interval).astype(int)
                    d = sign * (patch[:, :2] @ normal - rho)
                    inner, outer = {}, {}
                    for bin_id in np.unique(bins):
                        live = bins == bin_id
                        inner[int(bin_id)] = float(d[live].min())
                        outer[int(bin_id)] = float(d[live].max())
                    if len(inner) < self.config["boundary"]["profile_min_sections"]:
                        continue
                    if abs(float(np.median(list(inner.values())))) > search:
                        continue
                    low = np.array([inner.get(int(k), np.nan) for k in raw_bins])
                    high = np.array([outer.get(int(k), np.nan) for k in raw_bins])
                    inside = (distance < low) & (distance >= low - probe)
                    outside = (distance > high) & (distance <= high + probe)
                    n = np.asarray(plane["normal_uvz"])
                    below = (
                        -(points[:, :2] @ n[:2] + plane["offset_m"]) / n[2]
                        - points[:, 2]
                    )
                    drop = self.config["roi"]["opening_drop_m"]
                    inside_drop = (
                        int(inside.sum()) >= minimum
                        and np.quantile(below[inside], 0.25) >= drop
                    )
                    outside_drop = (
                        int(outside.sum()) >= minimum
                        and np.quantile(below[outside], 0.25) >= drop
                    )
                    # Fit only the inner observed sampling strip. The measured
                    # crest width is not assumed equal to deck_patch_width_m.
                    position = d - np.array([inner[int(k)] for k in bins])
                    use = (position >= 0) & (position <= width)
                    selected = raw_ids[use]
                    if len(selected) < self.config["geometry"]["min_plane_points"]:
                        continue
                    canonical = np.lexsort(
                        (selected, patch[use, 2], patch[use, 1], patch[use, 0])
                    )
                    fitted = self.fit(
                        patch[use][canonical],
                        selected[canonical],
                        segment,
                        "UNCLASSIFIED_SURFACE",
                    )
                    for source in fitted:
                        near_ids = self.positions(source["source_point_indexes"])
                        q = self.raw[near_ids]
                        qb = np.floor(((q[:, :2] @ tangent) - lo) / interval).astype(
                            int
                        )
                        separation = sign * (q[:, :2] @ normal - rho) - np.array(
                            [inner.get(int(k), np.nan) for k in qb]
                        )
                        if (
                            int(
                                (
                                    separation
                                    <= self.config["boundary"]["side_inner_gap_max_m"]
                                ).sum()
                            )
                            < self.config["boundary"]["line_min_points"]
                        ):
                            continue
                        output.append(
                            dict(
                                plane=source,
                                section=section,
                                span=(lo, hi),
                                normal=normal,
                                tangent=tangent,
                                rho=rho,
                                sign=sign,
                                inner=inner,
                                inside_drop=bool(inside_drop),
                                outside_drop=bool(outside_drop),
                                outside_ids=ids[outside],
                                crest_ids=raw_ids,
                                query_truncated=bool(
                                    max(outer.values())
                                    >= extent
                                    - self.config["geometry"]["plane_inlier_m"]
                                ),
                            )
                        )
        self.audit["candidate_patches"] += len(output)
        return output

    def _low_attachment(self, patch, members, other, barriers):
        self.audit["attachment_queries"] += 1
        n, t, rho, sign = patch["normal"], patch["tangent"], patch["rho"], patch["sign"]
        lo, hi = patch["span"]
        extent = self.config["roi"]["support_search_m"]
        probe = self.config["boundary"]["side_probe_m"]
        inner = patch["inner"]
        corners = np.array(
            [
                n * (rho + sign * d) + t * a
                for d in (min(inner.values()) - extent, max(inner.values()))
                for a in (lo, hi)
            ]
        )
        ids = self.measurement.spatial.query_indexes([*corners.min(0), *corners.max(0)])
        points = self.raw[ids]
        along, distance = points[:, :2] @ t, sign * (points[:, :2] @ n - rho)
        interval = self.config["boundary"]["max_support_gap_m"]
        bins = np.floor((along - lo) / interval).astype(int)
        inward = np.array([inner.get(int(k), np.nan) for k in bins]) - distance
        plane = patch["plane"]
        normal = np.asarray(plane["normal_uvz"])
        below = (
            -(points[:, :2] @ normal[:2] + plane["offset_m"]) / normal[2] - points[:, 2]
        )
        keep = (along >= lo) & (along <= hi) & (inward > 0) & (inward <= extent)
        keep &= (
            (below >= self.config["roi"]["opening_drop_m"])
            & ~other[ids]
            & ~barriers[ids]
        )
        local, direct = ids[keep], inward[keep] <= probe
        result = dict(
            supported=False,
            owned_member_seed_count=0,
            attached_direct_low_count=0,
            witness_paths=[],
        )
        if not len(local):
            return result
        q = self.raw[local]
        cell = self.config["geometry"]["candidate_voxel_m"]
        gap = self.config["boundary"]["max_support_gap_m"]
        if math.sqrt(3) * cell > gap:
            result["reason"] = "NO_EXISTING_RAW_CONNECTIVITY_SCALE"
            return result
        order = np.lexsort((self.original[local], q[:, 2], q[:, 1], q[:, 0]))
        _, first, inverse_sorted = np.unique(
            np.floor(q[order] / cell).astype(np.int64),
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        rep = order[first]
        inverse = np.empty(len(q), dtype=np.int64)
        inverse[order] = inverse_sorted
        valid = self.nonvertical(local[rep])
        representatives = q[rep]
        distance, nn = cKDTree(representatives).query(
            representatives,
            k=min(self.config["geometry"]["normal_k"], len(rep)),
            distance_upper_bound=gap,
            workers=1,
        )
        if distance.ndim == 1:
            distance, nn = distance[:, None], nn[:, None]
        live = np.isfinite(distance)
        a = np.broadcast_to(np.arange(len(rep))[:, None], nn.shape)[live]
        b = nn[live]
        allowed = valid[a] & valid[b]
        a, b = a[allowed], b[allowed]
        graph = coo_matrix(
            (np.ones(len(a), dtype=np.uint8), (a, b)), shape=(len(rep), len(rep))
        ).tocsr()
        graph = graph.maximum(graph.T)
        seeds = np.flatnonzero(self.owned[local[rep]] & members[local[rep]] & valid)
        result["owned_member_seed_count"] = len(seeds)
        minimum = self.config["boundary"]["profile_min_points"]
        if len(seeds) < minimum:
            return result
        parent = np.full(len(rep), -2, dtype=np.int64)
        parent[seeds] = -1
        queue = deque(seeds.tolist())
        while queue:
            node = queue.popleft()
            for target in graph.indices[graph.indptr[node] : graph.indptr[node + 1]]:
                if parent[target] == -2:
                    parent[target] = node
                    queue.append(int(target))
        direct_ids = np.flatnonzero(direct)
        direct_valid = self.nonvertical(local[direct_ids])
        attached = direct_ids[direct_valid & (parent[inverse[direct_ids]] != -2)]
        result["attached_direct_low_count"] = len(attached)
        actual_bins = np.floor(((q[attached, :2] @ t) - lo) / interval).astype(int)
        unique = np.unique(actual_bins)
        result["attached_along_bin_count"] = len(unique)
        if (
            len(attached) < minimum
            or len(unique) < self.config["boundary"]["profile_min_sections"]
        ):
            return result
        # Witnesses retain actual returns. Neither representative discovery nor
        # these paths become new points or independent physical-section votes.
        chosen = unique[
            np.linspace(
                0, len(unique) - 1, self.config["boundary"]["profile_min_sections"]
            ).astype(int)
        ]
        for bin_id in chosen:
            i = int(attached[np.flatnonzero(actual_bins == bin_id)[0]])
            node = int(inverse[i])
            path = [int(self.original[local[i]])]
            while node != -1:
                path.append(int(self.original[local[rep[node]]]))
                node = int(parent[node])
            result["witness_paths"].append(
                dict(along_bin=int(bin_id), point_indexes=path)
            )
        result["supported"] = True
        return result

    def _contact(self, first, second):
        a, b = first["plane"], second["plane"]
        na, nb = np.asarray(a["normal_uvz"]), np.asarray(b["normal_uvz"])
        if abs(float(na @ nb)) < math.cos(
            math.radians(self.config["geometry"]["normal_angle_deg"])
        ):
            return None
        qa = self.raw[self.positions(a["source_point_indexes"])]
        qb = self.raw[self.positions(b["source_point_indexes"])]
        distance, nearest = cKDTree(qa).query(qb, workers=1)
        live = distance <= self.config["boundary"]["max_support_gap_m"]
        live &= (
            abs(qb @ na + a["offset_m"]) <= self.config["geometry"]["plane_inlier_m"]
        )
        live &= (
            abs(qa[nearest] @ nb + b["offset_m"])
            <= self.config["geometry"]["plane_inlier_m"]
        )
        minimum = self.config["boundary"]["line_min_points"]
        if int(live.sum()) < minimum or len(set(nearest[live].tolist())) < minimum:
            return None
        # Preserve the independent contacts that qualified this link. Dense
        # returns can share one nearest point, so the first live pairs need
        # not contain the required number of distinct witnesses on both sides.
        live_ids = np.flatnonzero(live)
        _, first = np.unique(nearest[live_ids], return_index=True)
        ids = live_ids[np.sort(first)[:minimum]]
        return dict(
            first_point_indexes=np.asarray(a["source_point_indexes"])[
                nearest[ids]
            ].tolist(),
            second_point_indexes=np.asarray(b["source_point_indexes"])[ids].tolist(),
        )

    def _emit(self, plane, side, normal, proof, tier, segments):
        return dict(
            plane,
            vessel_id=self.vessel_id,
            plane_role="RIM_TOP",
            role_confirmed=True,
            source_roles=["RIM_TOP"],
            side_id=side,
            rim_side_normal_uv=normal.tolist(),
            tier=tier,
            source_segment_ids=sorted(segments),
            segment_vote_count=len(segments),
            z_source_requires_estimation=True,
            source_ownership_state="CURRENT_OPENING_ATTACHED_RIM_ESTIMATED",
            source_ownership_evidence=proof,
            source_point_index_sha256=point_fingerprint(plane["source_point_indexes"]),
        )

    def collect(self, result):
        regions = [
            r
            for r in getattr(self.measurement.local, "opening_identities", ())
            if r.identity_role.startswith("CONFIRMED") and r.vessel_id == self.vessel_id
        ]
        region = next(
            (r for r in regions if r.region_id == result["opening_identity"]), None
        )
        if region is None or "constrained_review_rectangle" not in result:
            self.audit["status"] = "NO_CONFIRMED_CURRENT_REGION"
            return []
        frozen = support_fingerprint(region)
        members = member_mask(self.raw, region)
        if members is None:
            self.audit["status"] = "NO_FROZEN_CURRENT_MEMBER_SUPPORT"
            return []
        other = np.zeros(len(self.raw), dtype=bool)
        for r in regions:
            if r.region_id != region.region_id:
                mask = member_mask(self.raw, r)
                if mask is not None:
                    other |= mask
        patches = {
            side: self._patches(result, side, sign, incident)
            for side, sign, incident in (("V0", -1, [0, 1]), ("V1", 1, [3, 2]))
        }
        barriers = np.zeros(len(self.raw), dtype=bool)
        for rows in patches.values():
            for patch in rows:
                if patch["inside_drop"] and patch["outside_drop"]:
                    barriers[self.positions(patch["crest_ids"])] = True
        sources = []
        for side, nodes in patches.items():
            seeds = []
            seen = set()
            for i, patch in enumerate(nodes):
                self.checkpoint()
                source_ids = self.positions(patch["plane"]["source_point_indexes"])
                patch["propagation_eligible"] = (
                    not patch["query_truncated"] and not members[source_ids].any()
                )
                if (
                    patch["query_truncated"]
                    or not patch["inside_drop"]
                    or not patch["outside_drop"]
                ):
                    continue
                if members[patch["outside_ids"]].any():
                    continue
                proof = self._low_attachment(patch, members, other, barriers)
                patch["low_attachment"] = proof
                if proof["supported"]:
                    signature = point_fingerprint(
                        patch["plane"]["source_point_indexes"]
                    )
                    if signature not in seen:
                        seeds.append(i)
                        seen.add(signature)
            sections = {nodes[i]["section"] for i in seeds}
            if len(sections) < self.config["boundary"]["profile_min_sections"]:
                self.audit["side_families"].append(
                    dict(side=side, seed_sections=sorted(sections), admitted=0)
                )
                continue
            by_section = {}
            for i, patch in enumerate(nodes):
                by_section.setdefault(patch["section"], []).append(i)
            graph = [[] for _ in nodes]
            for i, patch in enumerate(nodes):
                for section in (patch["section"], patch["section"] + 1):
                    for j in by_section.get(section, ()):
                        if section == patch["section"] and j <= i:
                            continue
                        if not (nodes[i]["propagation_eligible"] or i in seeds) or not (
                            nodes[j]["propagation_eligible"] or j in seeds
                        ):
                            continue
                        contact = self._contact(patch, nodes[j])
                        if contact:
                            graph[i].append((j, contact))
                            graph[j].append((i, contact))
            parent = {i: None for i in seeds}
            queue = deque(seeds)
            while queue:
                i = queue.popleft()
                for j, contact in graph[i]:
                    if j not in parent:
                        parent[j] = (i, contact)
                        queue.append(j)
            local = []
            for i in parent:
                patch = nodes[i]
                chain, node = [], i
                while parent[node] is not None:
                    previous, contact = parent[node]
                    chain.append(dict(parent_node=previous, contact=contact))
                    node = previous
                proof = dict(
                    producer="CURRENT_CREST_AND_OPENING_LOW_ATTACHMENT",
                    opening_identity=region.region_id,
                    member_support_sha256=frozen,
                    root_seed_sections=sorted(sections),
                    root_low_attachment=nodes[node]["low_attachment"],
                    active_raw_point_index_sha256=self.raw_index_sha256,
                    root_upper_plane_uvz=dict(
                        normal=nodes[node]["plane"]["normal_uvz"],
                        offset_m=nodes[node]["plane"]["offset_m"],
                        source_point_indexes=nodes[node]["plane"][
                            "source_point_indexes"
                        ],
                    ),
                    root_side_reference=dict(
                        normal_uv=nodes[node]["normal"].tolist(),
                        rho=nodes[node]["rho"],
                        sign=nodes[node]["sign"],
                        support_span=list(nodes[node]["span"]),
                    ),
                    surface_attachment_chain=chain,
                    core_promoted=False,
                    original_primary_point_count=int(
                        self.owned[
                            self.positions(patch["plane"]["source_point_indexes"])
                        ].sum()
                    ),
                )
                local.append(
                    self._emit(
                        patch["plane"],
                        side,
                        patch["normal"],
                        proof,
                        "LOCAL_RIM_PLANE",
                        [patch["plane"]["segment_id"]],
                    )
                )
            sources.extend(local)
            self.audit["admitted_local_sources"] += len(local)
            # A persistent plane must explain each independent patch, rather
            # than collecting a few accidental fringe returns from each one.
            if local:
                ids = np.asarray(
                    sorted({i for s in local for i in s["source_point_indexes"]}),
                    dtype=np.int64,
                )
                merged = self.fit(
                    self.raw[self.positions(ids)],
                    ids,
                    region.region_id + ":" + side + ":OBSERVED_CREST_PERSISTENT",
                    "UNCLASSIFIED_SURFACE",
                )
                for plane in merged:
                    subset = set(plane["source_point_indexes"])
                    good = []
                    normal = np.asarray(plane["normal_uvz"])
                    for source in local:
                        if (
                            len(subset.intersection(source["source_point_indexes"]))
                            < self.config["boundary"]["line_min_points"]
                        ):
                            continue
                        if abs(float(normal @ source["normal_uvz"])) < math.cos(
                            math.radians(self.config["geometry"]["normal_angle_deg"])
                        ):
                            continue
                        points = self.raw[
                            self.positions(source["source_point_indexes"])
                        ]
                        if (
                            np.quantile(abs(points @ normal + plane["offset_m"]), 0.95)
                            <= self.config["geometry"]["plane_inlier_m"]
                        ):
                            good.append(source)
                    segments = {s["segment_id"] for s in good}
                    if len(segments) >= self.config["boundary"]["profile_min_sections"]:
                        proof = dict(
                            producer="CURRENT_CREST_CONSISTENT_PERSISTENT_PLANE",
                            opening_identity=region.region_id,
                            member_support_sha256=frozen,
                            parent_source_hashes=[
                                s["source_point_index_sha256"] for s in good
                            ],
                            core_promoted=False,
                        )
                        sources.append(
                            self._emit(
                                plane,
                                side,
                                nodes[0]["normal"],
                                proof,
                                "SAME_SIDE_RIM_PLANE",
                                segments,
                            )
                        )
                        self.audit["admitted_persistent_sources"] += 1
            self.audit["side_families"].append(
                dict(side=side, seed_sections=sorted(sections), admitted=len(local))
            )
        if support_fingerprint(region) != frozen:
            raise ValueError("RIM_ASSOCIATION_MODIFIED_FROZEN_MEMBER_SUPPORT")
        self.audit["status"] = (
            "CURRENT_MEASUREMENT_ONLY_ASSOCIATION"
            if sources
            else "NO_QUALIFIED_ASSOCIATION"
        )
        return sources
