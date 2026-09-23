"""Observed profile breaks and vertical faces in raw 3D; no edge completion."""
import math

import numpy as np
from scipy.spatial import cKDTree


def _profile(points, tree, center, tangent, inward, plane, config, profile_id):
    b = config["boundary"]
    half, width, step = b["profile_half_length_m"], b["profile_half_width_m"], b["profile_step_m"]
    ids = tree.query_ball_point(center, half + width)
    row = dict(profile_id=profile_id, sample_count=0, deck_support_count=0,
               lower_support_count=0, break_position=None, break_strength=None,
               residual=None, valid=False, reason="INSUFFICIENT_PROFILE_POINTS", face_position=None)
    if not ids:
        return row
    local = points[ids].astype(float)
    delta = local[:, :2] - center
    along = delta @ inward
    use = (np.abs(delta @ tangent) <= width) & (np.abs(along) <= half)
    if int(use.sum()) < b["profile_min_points"]:
        return row
    local, along = local[use], along[use]
    residual = local @ plane[0] + plane[1]
    bins = np.floor(along / step).astype(int)
    samples = []
    face_options = []
    for index in np.unique(bins):
        values = residual[bins == index]
        position = (index + 0.5) * step
        samples.append((position, float(np.quantile(values, b["profile_lower_quantile"])), len(values)))
        if len(values) >= max(3, config["geometry"]["normal_min_points"] // 2) and np.ptp(values) >= b["face_vertical_span_min_m"]:
            if np.max(values) >= -config["roi"]["support_band_m"] and np.min(values) <= -b["profile_min_drop_m"]:
                face_options.append(position)
    row["sample_count"] = len(samples)
    if face_options:
        face_at = min(face_options, key=abs)
        row["face_position"] = [float(center[0] + face_at * inward[0]),
                                float(center[1] + face_at * inward[1])]
    deck = [s for s in samples if abs(s[1]) <= config["roi"]["support_band_m"]]
    lower = [s for s in samples if s[1] < -b["profile_min_drop_m"]]
    row["deck_support_count"], row["lower_support_count"] = len(deck), len(lower)
    if len(samples) < b["profile_min_points"] or len(deck) < 3 or len(lower) < 3:
        row["reason"] = "DECK_OR_LOWER_NOT_OBSERVED"
        return row
    options = []
    for index, inside in enumerate(samples):
        if inside[1] >= -b["profile_min_drop_m"]:
            continue
        prior = [(j, outer) for j, outer in enumerate(samples[:index])
                 if 0 < inside[0] - outer[0] <= b["profile_max_bracket_m"]
                 and abs(outer[1]) <= config["roi"]["support_band_m"]]
        if not prior:
            continue
        j, outside = prior[-1]
        preceding = [s for s in samples[:j + 1] if abs(s[1]) <= config["roi"]["support_band_m"]]
        following = [s for s in samples[index:] if s[1] < -b["profile_min_drop_m"]]
        if len(preceding) < 3 or len(following) < 3:
            continue
        if preceding[-1][0] - preceding[0][0] < b["profile_deck_support_span_m"]:
            continue
        if any(s[1] < -b["profile_min_drop_m"] for s in samples[j + 1:index]):
            continue
        strength = (outside[1] - inside[1]) / (inside[0] - outside[0])
        if strength < b["profile_step_slope_min"]:
            continue
        options.append(((outside[0] + inside[0]) / 2, strength,
                        abs(inside[0] - outside[0])))
    if not options:
        row["reason"] = "NO_OBSERVED_DECK_TO_LOWER_BREAK"
        return row
    options.sort(key=lambda item: (abs(item[0]), -item[1]))
    best = options[0]
    if any(abs(other[0] - best[0]) > b["profile_breakpoint_stability_m"] for other in options):
        row["reason"] = "COMPETING_BREAKS"
        return row
    row.update(break_position=[float(center[0] + best[0] * inward[0]),
                               float(center[1] + best[0] * inward[1])],
               break_strength=float(best[1]), residual=float(best[2] / 2),
               valid=True, reason=None)
    return row


def _fit_line(xy, config):
    """Deterministic pair RANSAC followed by Huber weighted orthogonal fit."""
    b = config["boundary"]
    if len(xy) < b["line_min_points"]:
        return None
    rng = np.random.default_rng(2026)
    best = np.zeros(len(xy), dtype=bool)
    for _ in range(min(128, len(xy) * 4)):
        a, c = xy[rng.choice(len(xy), 2, replace=False)]
        direction = c - a
        length = np.linalg.norm(direction)
        if length < 1e-9:
            continue
        direction /= length
        mask = np.abs(np.cross(direction, xy - a)) <= b["line_inlier_m"]
        if int(mask.sum()) > int(best.sum()):
            best = mask
    if int(best.sum()) < b["line_min_points"]:
        return None
    fit = xy[best]
    weights = np.ones(len(fit))
    for _ in range(6):
        center = np.average(fit, axis=0, weights=weights)
        covariance = ((fit - center) * weights[:, None]).T @ (fit - center)
        _, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, -1]
        residual = np.abs(np.cross(direction, fit - center))
        weights = np.minimum(1.0, b["line_inlier_m"] / np.maximum(residual, 1e-12))
    projection = (fit - center) @ direction
    order = np.sort(projection)
    support_length = float(np.diff(order)[np.diff(order) <= b["max_support_gap_m"]].sum())
    if support_length < b["line_min_length_m"]:
        return None
    endpoints = [center + order[0] * direction, center + order[-1] * direction]
    return dict(a_xy=endpoints[0], b_xy=endpoints[1], support_count=int(len(fit)),
                support_length=support_length, residual_p50=float(np.percentile(residual, 50)),
                residual_p95=float(np.percentile(residual, 95)),
                normal_uncertainty=float(np.percentile(residual, 95) / math.sqrt(len(fit))))


def _raw3(xy, plane):
    normal, offset = plane
    return [float(xy[0]), float(xy[1]), float(-(normal[0] * xy[0] + normal[1] * xy[1] + offset) / normal[2])]


def _face_normal_supported(points, line, plane, config):
    """Require a long near-vertical raw 3D patch, not only stacked heights."""
    a, c = line["a_xy"], line["b_xy"]
    direction = c - a
    length = np.linalg.norm(direction)
    if length < config["boundary"]["line_min_length_m"]:
        return False
    direction /= length
    delta = points[:, :2] - a
    along = delta @ direction
    distance = np.abs(np.cross(direction, delta))
    mask = (along >= 0) & (along <= length)
    mask &= distance <= max(config["boundary"]["profile_step_m"], config["geometry"]["refine_voxel_m"])
    mask &= ((points @ plane[0] + plane[1]) >= -config["roi"]["max_opening_depth_m"])
    support = points[mask].astype(float)
    if len(support) < config["geometry"]["min_plane_points"]:
        return False
    if np.ptp(support @ plane[0] + plane[1]) < config["boundary"]["face_vertical_span_min_m"]:
        return False
    _, _, vectors = np.linalg.svd(support - support.mean(0), full_matrices=False)
    return abs(float(vectors[-1, 2])) <= math.sin(math.radians(config["geometry"]["normal_angle_deg"]))


def _intersect(first, second):
    a, b = np.asarray(first["a_raw"][:2]), np.asarray(first["b_raw"][:2])
    c, d = np.asarray(second["a_raw"][:2]), np.asarray(second["b_raw"][:2])
    matrix = np.column_stack((b - a, c - d))
    if abs(np.linalg.det(matrix)) < 1e-9:
        return None
    return a + np.linalg.solve(matrix, c - a)[0] * (b - a)


def refine_boundaries(points, opening, plane, config, tree=None):
    if tree is None:
        tree = cKDTree(points[:, :2])
    x0, y0, x1, y1 = opening.bbox_xy
    rough = np.array(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=float)
    b = config["boundary"]
    spacing = max(b["profile_step_m"], min(b["max_support_gap_m"] * 0.8,
                                            config["geometry"]["coarse_voxel_m"] / 2))
    edges, profiles = [], []
    for edge_id, (a, c) in enumerate(zip(rough, np.roll(rough, -1, axis=0))):
        length = float(np.linalg.norm(c - a))
        if length < b["line_min_length_m"]:
            continue
        tangent = (c - a) / length
        inward = np.array((-tangent[1], tangent[0]))
        positions = np.arange(spacing / 2, length, spacing)
        rows = [_profile(points, tree, a + tangent * pos, tangent, inward, plane, config,
                         "%s-e%d-p%d" % (opening.opening_id, edge_id, index))
                for index, pos in enumerate(positions)]
        profiles.extend(rows)
        breaks = np.array([r["break_position"] for r in rows if r["valid"]], dtype=float).reshape(-1, 2)
        faces = np.array([r["face_position"] for r in rows if r["face_position"] is not None], dtype=float).reshape(-1, 2)
        break_line = _fit_line(breaks, config)
        face_line = _fit_line(faces, config)
        if face_line is not None and not _face_normal_supported(points, face_line, plane, config):
            face_line = None
        if break_line and face_line:
            separation = np.linalg.norm((break_line["a_xy"] + break_line["b_xy"] -
                                         face_line["a_xy"] - face_line["b_xy"]) / 2)
            if separation <= max(b["line_inlier_m"], config["roi"]["support_band_m"]):
                chosen = _fit_line(np.vstack((breaks, faces)), config) or break_line
                evidence = "OBSERVED_PROFILE_BREAK+OBSERVED_3D_FACE"
            else:
                chosen, evidence = break_line, "OBSERVED_PROFILE_BREAK"
        elif break_line:
            chosen, evidence = break_line, "OBSERVED_PROFILE_BREAK"
        elif face_line:
            chosen, evidence = face_line, "OBSERVED_3D_FACE"
        else:
            continue
        coverage = min(1.0, chosen["support_length"] / length)
        edges.append(dict(edge_id="e%d" % edge_id, a_raw=_raw3(chosen["a_xy"], plane),
                          b_raw=_raw3(chosen["b_xy"], plane), evidence_type=evidence,
                          support_count=chosen["support_count"],
                          observed_support_length=chosen["support_length"],
                          fit_residual_p50_m=chosen["residual_p50"],
                          fit_residual_p95_m=chosen["residual_p95"],
                          visibility="VISIBLE", uncertainty_m=chosen["normal_uncertainty"],
                          coverage=coverage, reason="RAW_3D_OBSERVED_SUPPORT"))
    polygon = None
    if len(edges) == 4 and {edge["edge_id"] for edge in edges} == {"e0", "e1", "e2", "e3"}:
        by_id = {edge["edge_id"]: edge for edge in edges}
        corners = []
        for previous, current in (("e3", "e0"), ("e0", "e1"), ("e1", "e2"), ("e2", "e3")):
            intersection = _intersect(by_id[previous], by_id[current])
            if intersection is None:
                break
            endpoints = [np.asarray(by_id[previous][side][:2]) for side in ("a_raw", "b_raw")]
            endpoints += [np.asarray(by_id[current][side][:2]) for side in ("a_raw", "b_raw")]
            if max(min(np.linalg.norm(intersection - endpoint) for endpoint in endpoints[:2]),
                   min(np.linalg.norm(intersection - endpoint) for endpoint in endpoints[2:])) > b["corner_join_m"]:
                break
            corners.append(_raw3(intersection, plane))
        if len(corners) == 4:
            polygon = corners
    status = "COMPLETE_OBSERVED" if polygon is not None else "PARTIAL" if edges else "UNRESOLVED"
    return dict(status=status, boundaries=edges, polygon_raw=polygon,
                center_raw=_raw3(((x0 + x1) / 2, (y0 + y1) / 2), plane),
                center_source="COARSE_OPENING_SEED", profiles=profiles)
