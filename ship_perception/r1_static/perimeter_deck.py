"""Independent gravity-compatible deck evidence outside each rough perimeter side."""
import math

import numpy as np
from scipy.spatial import cKDTree

from .local_deck import _anchor_height, _huber_plane, _plane_angle, _plane_from_three


def estimate_segment_deck(points, segment, config, tree=None):
    """Return a side-local plane and exact raw-point ownership for every candidate."""
    result = dict(segment_id=segment.segment_id, status="SEGMENT_DECK_UNRESOLVED",
                  reason="INSUFFICIENT_SEGMENT_SPAN", fov_status=segment.fov_status,
                  support_strip_point_count=0, support_raw_point_count=0,
                  plane_candidate_count=0, candidate_plane_normals=[],
                  candidate_plane_heights=[], candidate_support_counts=[],
                  candidate_outward_widths=[], candidate_along_coverages=[],
                  selected_plane=None, selected_candidate_index=None,
                  outward_width_support_m=0.0, along_segment_coverage=0.0,
                  residual_p95_m=None)
    forensic = dict(strip_raw_ids=np.empty(0, dtype=np.int64),
                    candidate_raw_ids=[], selected_raw_ids=np.empty(0, dtype=np.int64))
    if segment.length_m < config["roi"]["min_deck_span_m"]:
        return result, None, forensic
    if tree is None:
        tree = cKDTree(points[:, :2])
    start, end = np.asarray(segment.rough_start), np.asarray(segment.rough_end)
    tangent = np.asarray(segment.tangent)
    outward = np.asarray(segment.outward_normal)
    band = config["roi"]["support_search_m"]
    nearby = np.asarray(tree.query_ball_point((start + end) / 2,
                                               segment.length_m / 2 + band), dtype=np.int64)
    if not len(nearby):
        result["reason"] = "NO_OUTWARD_RAW_SUPPORT"
        return result, None, forensic
    displacement = points[nearby, :2].astype(float) - start
    along = displacement @ tangent
    distance = displacement @ outward
    strip = ((along >= 0) & (along <= segment.length_m) &
             (distance > 0) & (distance <= band))
    raw_ids = nearby[strip]
    forensic["strip_raw_ids"] = raw_ids
    result["support_strip_point_count"] = int(len(raw_ids))
    minimum = config["geometry"]["min_plane_points"]
    if len(raw_ids) < minimum:
        result["reason"] = "INSUFFICIENT_OUTWARD_RAW_SUPPORT"
        return result, None, forensic
    raw = points[raw_ids].astype(float)
    rng = np.random.default_rng(2026)
    sample_ids = rng.choice(len(raw), min(len(raw), 3000), replace=False)
    sample = raw[sample_ids]
    threshold = config["geometry"]["plane_inlier_m"]
    max_angle = math.radians(config["geometry"]["normal_angle_deg"])
    hypotheses = []
    for _ in range(config["geometry"]["ransac_iterations"]):
        plane = _plane_from_three(sample[rng.choice(len(sample), 3, replace=False)], max_angle)
        if plane is None:
            continue
        mask = np.abs(sample @ plane[0] + plane[1]) <= threshold
        if int(mask.sum()) >= minimum:
            hypotheses.append((int(mask.sum()), plane, mask))
    if not hypotheses:
        result["reason"] = "NO_GRAVITY_COMPATIBLE_PLANE"
        return result, None, forensic
    hypotheses.sort(key=lambda row: -row[0])
    angle_gate = config["frame"]["normal_refine_deg"]
    overlap_gate = config["frame"]["same_plane_overlap_min"]
    anchor = (start + end) / 2
    unique = []
    for count, plane, mask in hypotheses:
        height = _anchor_height(plane, anchor)
        if any((_plane_angle(plane, old[1]) <= angle_gate and
                abs(height - _anchor_height(old[1], anchor)) <= 3 * threshold) or
               (np.count_nonzero(mask & old[2]) / max(min(count, old[0]), 1) >= overlap_gate)
               for old in unique):
            continue
        unique.append((count, plane, mask))
        if len(unique) >= config["geometry"]["max_plane_hypotheses"]:
            break
    candidates = []
    outward_distance = (raw[:, :2] - start) @ outward
    along_distance = (raw[:, :2] - start) @ tangent
    bin_m = config["roi"]["support_connectivity_m"]
    possible_bins = max(1, int(math.ceil(segment.length_m / bin_m)))
    for sample_count, plane, _ in unique:
        mask = np.abs(raw @ plane[0] + plane[1]) <= threshold
        if int(mask.sum()) < minimum:
            continue
        refined = _huber_plane(raw[mask], plane, threshold, max_angle)
        if refined is None:
            continue
        mask = np.abs(raw @ refined[0] + refined[1]) <= threshold
        ids = raw_ids[mask]
        if len(ids) < minimum:
            continue
        local_along = along_distance[mask]
        local_outward = outward_distance[mask]
        width = float(np.percentile(local_outward, 95) -
                      np.percentile(local_outward, 5))
        span = float(np.max(local_along) - np.min(local_along))
        occupied_bins = np.floor(local_along / bin_m).astype(int)
        coverage = min(1.0, len(np.unique(occupied_bins)) / possible_bins)
        residual = np.abs(raw[mask] @ refined[0] + refined[1])
        qualified = (width >= config["roi"]["deck_patch_width_m"] and
                     span >= config["roi"]["min_deck_span_m"] and
                     coverage >= config["roi"]["deck_patch_fill_min"])
        candidates.append(dict(plane=refined, raw_ids=ids, sample_count=sample_count,
                               width=width, span=span, coverage=float(coverage),
                               residual_p95=float(np.percentile(residual, 95)),
                               qualified=qualified))
    result["plane_candidate_count"] = len(candidates)
    result["candidate_plane_normals"] = [row["plane"][0].tolist() for row in candidates]
    result["candidate_plane_heights"] = [_anchor_height(row["plane"], anchor) for row in candidates]
    result["candidate_support_counts"] = [int(len(row["raw_ids"])) for row in candidates]
    result["candidate_outward_widths"] = [row["width"] for row in candidates]
    result["candidate_along_coverages"] = [row["coverage"] for row in candidates]
    forensic["candidate_raw_ids"] = [row["raw_ids"] for row in candidates]
    qualified_ids = [index for index, row in enumerate(candidates) if row["qualified"]]
    if not qualified_ids:
        result["reason"] = ("NO_GRAVITY_COMPATIBLE_PLANE" if not candidates else
                            "INSUFFICIENT_DECK_WIDTH_OR_ALONG_COVERAGE")
        return result, None, forensic
    qualified_ids.sort(key=lambda index: (-candidates[index]["coverage"],
                                          -candidates[index]["width"],
                                          -len(candidates[index]["raw_ids"]),
                                          candidates[index]["residual_p95"], index))
    selected_index = qualified_ids[0]
    selected = candidates[selected_index]
    independent = [index for index in qualified_ids[1:]
                   if len(candidates[index]["raw_ids"]) >=
                   len(selected["raw_ids"]) * overlap_gate and
                   np.setdiff1d(candidates[index]["raw_ids"], selected["raw_ids"],
                                assume_unique=True).size >= minimum and
                   (_plane_angle(candidates[index]["plane"], selected["plane"]) > angle_gate or
                    abs(_anchor_height(candidates[index]["plane"], anchor) -
                        _anchor_height(selected["plane"], anchor)) > 3 * threshold)]
    if independent:
        result.update(status="SEGMENT_DECK_AMBIGUOUS", reason="MULTIPLE_DECK_LIKE_PLANES")
        return result, None, forensic
    forensic["selected_raw_ids"] = selected["raw_ids"]
    normal, offset = selected["plane"]
    result.update(status="RESOLVED", reason="BROAD_OUTWARD_DECK_SUPPORT",
                  support_raw_point_count=int(len(selected["raw_ids"])),
                  selected_candidate_index=selected_index,
                  selected_plane=dict(normal_raw=normal.tolist(), offset=float(offset)),
                  outward_width_support_m=selected["width"],
                  along_segment_coverage=selected["coverage"],
                  residual_p95_m=selected["residual_p95"])
    return result, selected["plane"], forensic
