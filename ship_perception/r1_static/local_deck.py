"""Candidate-local deck planes with physical support connectivity."""
import itertools
import math

import numpy as np
from scipy.spatial import cKDTree


def _plane_from_three(triangle, max_angle_rad):
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    length = float(np.linalg.norm(normal))
    if length < 1e-9:
        return None
    normal /= length
    if normal[2] < 0:
        normal = -normal
    if math.acos(float(np.clip(normal[2], -1, 1))) > max_angle_rad:
        return None
    return normal, -float(normal @ triangle[0])


def _huber_plane(points, initial, threshold, max_angle_rad):
    """Fit in a local frame so raw-coordinate translation preserves precision."""
    origin = np.median(points, axis=0)
    xy = points[:, :2].astype(float) - origin[:2]
    z = points[:, 2].astype(float) - origin[2]
    design = np.column_stack((xy, np.ones(len(points))))
    normal, offset = initial
    beta = np.array((-normal[0] / normal[2], -normal[1] / normal[2],
                     -(normal @ origin + offset) / normal[2]))
    for _ in range(8):
        residual = z - design @ beta
        weights = np.minimum(1.0, threshold / np.maximum(np.abs(residual), 1e-12))
        beta = np.linalg.lstsq(design * np.sqrt(weights[:, None]), z * np.sqrt(weights), rcond=None)[0]
    refined = np.array((-beta[0], -beta[1], 1.0))
    refined /= np.linalg.norm(refined)
    if math.acos(float(np.clip(refined[2], -1, 1))) > max_angle_rad:
        return None
    return refined, -float(refined @ origin + beta[2] * refined[2])


def _anchor_height(plane, anchor_xy):
    normal, offset = plane
    return -float(normal[:2] @ anchor_xy + offset) / normal[2]


def _plane_angle(first, second):
    return math.degrees(math.acos(float(np.clip(first[0] @ second[0], -1, 1))))


def _physical_components(points, radius):
    """Exact radius graph components without materializing dense point edges."""
    if not len(points):
        return []
    side = radius / math.sqrt(3.0)
    bins = np.floor((points - points.min(axis=0)) / side).astype(np.int64)
    keys, inverse = np.unique(bins, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1)
    lookup = {tuple(key): index for index, key in enumerate(keys)}
    parent = np.arange(len(keys))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    offsets = [delta for delta in itertools.product(range(-2, 3), repeat=3)
               if delta > (0, 0, 0) and
               sum(max(abs(value) - 1, 0) ** 2 for value in delta) * side * side <= radius * radius]
    for index, key in enumerate(keys):
        for delta in offsets:
            other = lookup.get(tuple(key + delta))
            if other is None or root(index) == root(other):
                continue
            left, right = members[index], members[other]
            if len(left) > len(right):
                left, right = right, left
            if np.isfinite(cKDTree(points[right]).query(points[left], distance_upper_bound=radius)[0]).any():
                parent[root(other)] = root(index)
    groups = {}
    for index, point_ids in enumerate(members):
        groups.setdefault(root(index), []).append(point_ids)
    return sorted((np.concatenate(group) for group in groups.values()),
                  key=lambda ids: (-len(ids), int(ids.min())))


def _ring_points(points, proposal, band):
    xy = points[:, :2]
    x0, y0, x1, y1 = proposal.bbox_xy
    outer = ((xy[:, 0] >= x0 - band) & (xy[:, 0] <= x1 + band) &
             (xy[:, 1] >= y0 - band) & (xy[:, 1] <= y1 + band))
    if not proposal.seed_components:
        inner = ((xy[:, 0] >= x0) & (xy[:, 0] <= x1) &
                 (xy[:, 1] >= y0) & (xy[:, 1] <= y1))
        return points[outer & ~inner].astype(float)
    subset = points[outer].astype(float)
    near = np.zeros(len(subset), dtype=bool)
    interior = np.zeros(len(subset), dtype=bool)
    for component in proposal.seed_components:
        cells = np.asarray(component.cells_rc, dtype=np.int64)
        if not len(cells):
            continue
        centers = np.column_stack((component.origin_xy[0] + (cells[:, 1] + .5) * component.cell_m,
                                   component.origin_xy[1] + (cells[:, 0] + .5) * component.cell_m))
        distance, nearest = cKDTree(centers).query(subset[:, :2])
        half = component.cell_m / 2
        offset = np.abs(subset[:, :2] - centers[nearest])
        interior |= (offset[:, 0] <= half) & (offset[:, 1] <= half)
        near |= distance <= band + math.sqrt(2) * half
    return subset[near & ~interior]


def estimate_local_deck(points, proposal, config):
    ring = _ring_points(points, proposal, config["roi"]["support_search_m"])
    minimum = config["geometry"]["min_plane_points"]
    result = dict(status="LOCAL_DECK_UNRESOLVED", normal_raw=None, offset=None,
                  ring_point_count=int(len(ring)), raw_ring_point_count=int(len(ring)),
                  plane_inlier_count=0, ransac_hypothesis_count=0, plane_cluster_count=0,
                  candidate_plane_support_counts=[], candidate_plane_anchor_heights=[],
                  candidate_plane_normal_angles=[], connectivity_component_count=0,
                  connectivity_component_sizes=[], accepted_component_ids=[],
                  support_count=0, support_sector_count=0, normal_dispersion_rad=None,
                  residual_p50_m=None, residual_p95_m=None, competing_plane_count=0,
                  reason="INSUFFICIENT_RING_SUPPORT", rejection_stage="RING_SUPPORT")
    empty = np.empty((0, 3))
    if len(ring) < minimum:
        return result, None, empty
    rng = np.random.default_rng(2026)
    sample = ring[rng.choice(len(ring), min(len(ring), 3000), replace=False)]
    threshold = config["geometry"]["plane_inlier_m"]
    max_angle = math.radians(config["geometry"]["normal_angle_deg"])
    x0, y0, x1, y1 = proposal.bbox_xy
    anchor = np.array(((x0 + x1) / 2, (y0 + y1) / 2))
    hypotheses = []
    for _ in range(config["geometry"]["ransac_iterations"]):
        plane = _plane_from_three(sample[rng.choice(len(sample), 3, replace=False)], max_angle)
        if plane is None:
            continue
        mask = np.abs(sample @ plane[0] + plane[1]) <= threshold
        if mask.any():
            hypotheses.append((int(mask.sum()), plane, mask))
    result["ransac_hypothesis_count"] = len(hypotheses)
    if not hypotheses:
        result.update(reason="NO_LOCAL_PLANAR_PATCH", rejection_stage="RANSAC")
        return result, None, empty
    hypotheses.sort(key=lambda row: -row[0])
    clusters = []
    overlap_gate = config["frame"]["same_plane_overlap_min"]
    angle_gate = config["frame"]["normal_refine_deg"]
    for count, plane, mask in hypotheses:
        height = _anchor_height(plane, anchor)
        if any((_plane_angle(plane, old[1]) <= angle_gate and
                abs(height - _anchor_height(old[1], anchor)) <= 3 * threshold) or
               (np.count_nonzero(mask & old[2]) / max(min(count, old[0]), 1) >= overlap_gate)
               for old in clusters):
            continue
        clusters.append((count, plane, mask))
    result["plane_cluster_count"] = len(clusters)
    candidates = clusters[:config["geometry"]["max_plane_hypotheses"]]
    result["candidate_plane_support_counts"] = [row[0] for row in candidates]
    result["candidate_plane_anchor_heights"] = [_anchor_height(row[1], anchor) for row in candidates]
    result["candidate_plane_normal_angles"] = [_plane_angle(row[1], candidates[0][1]) for row in candidates]
    best_count, initial, best_mask = candidates[0]
    competitors = [row for row in candidates[1:]
                   if row[0] >= best_count * overlap_gate and
                   np.count_nonzero(row[2] & ~best_mask) >= minimum and
                   (_plane_angle(row[1], initial) > angle_gate or
                    abs(_anchor_height(row[1], anchor) - _anchor_height(initial, anchor)) > 3 * threshold)]
    result["competing_plane_count"] = len(competitors)
    inliers = ring[np.abs(ring @ initial[0] + initial[1]) <= threshold]
    result["plane_inlier_count"] = len(inliers)
    if len(inliers) < minimum:
        result.update(reason="INSUFFICIENT_PLANAR_SUPPORT", rejection_stage="PLANE_INLIERS")
        return result, None, empty
    refined = _huber_plane(inliers, initial, threshold, max_angle)
    if refined is None:
        result.update(reason="DEGENERATE_PLANE_REFINE", rejection_stage="PLANE_REFINE")
        return result, None, empty
    inliers = ring[np.abs(ring @ refined[0] + refined[1]) <= threshold]
    components = _physical_components(inliers, config["roi"]["support_connectivity_m"])
    result["connectivity_component_count"] = len(components)
    result["connectivity_component_sizes"] = [len(ids) for ids in components]
    accepted = [index for index, ids in enumerate(components)
                if len(ids) >= config["geometry"]["normal_min_points"]]
    result["accepted_component_ids"] = accepted
    grown = inliers[np.concatenate([components[index] for index in accepted])] if accepted else empty
    if len(grown) < minimum:
        result.update(reason="INSUFFICIENT_CONNECTED_PLANAR_SUPPORT", rejection_stage="CONNECTIVITY")
        return result, None, grown
    final = _huber_plane(grown, refined, threshold, max_angle)
    if final is None:
        result.update(reason="CONNECTED_PATCH_TILT_OUT_OF_RANGE", rejection_stage="PLANE_REFINE")
        return result, None, grown
    normal, offset = final
    residual = np.abs(grown @ normal + offset)
    relative = grown[:, :2] - anchor
    sectors = np.floor((np.arctan2(relative[:, 1], relative[:, 0]) + np.pi) * 8 / (2 * np.pi)).astype(int) % 8
    sector_count = int(len(np.unique(sectors)))
    angles = []
    for index in accepted:
        ids = components[index]
        if len(ids) >= minimum:
            local = _huber_plane(inliers[ids], final, threshold, max_angle)
            if local is not None:
                angles.append(math.radians(_plane_angle(local, final)))
    result.update(normal_raw=normal.tolist(), offset=float(offset), support_count=int(len(grown)),
                  support_sector_count=sector_count, normal_dispersion_rad=float(np.median(angles)) if angles else 0.0,
                  residual_p50_m=float(np.percentile(residual, 50)),
                  residual_p95_m=float(np.percentile(residual, 95)))
    if competitors:
        result.update(status="LOCAL_DECK_AMBIGUOUS", reason="COMPETING_PLANAR_PATCHES",
                      rejection_stage="PLANE_COMPETITION")
        return result, None, grown
    if sector_count < math.ceil(8 * config["roi"]["min_enclosure_ratio"]):
        result.update(reason="INSUFFICIENT_SECTOR_SUPPORT", rejection_stage="SECTOR_SUPPORT")
        return result, None, grown
    if (not (np.min(grown[:, 0]) < anchor[0] < np.max(grown[:, 0])) or
            not (np.min(grown[:, 1]) < anchor[1] < np.max(grown[:, 1]))):
        result.update(reason="ONE_SIDED_LOCAL_DECK_SUPPORT", rejection_stage="ENCLOSURE")
        return result, None, grown
    result.update(status="RESOLVED", reason="MULTI_PATCH_LOCAL_PLANAR_SUPPORT", rejection_stage=None)
    return result, final, grown
