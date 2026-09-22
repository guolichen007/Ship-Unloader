"""R0 oracle-only local geometry diagnosis. Never imported into production.

The oracle rough hatch/vessel polygons enter *only* this evaluator. A profile
break is a measured deck-to-lower transition, not a fixed-height contour or a
CNN rectangle. The returned edges remain diagnostic until independently scored
against adjudicated physical opening-side edge observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ship_perception.tools.v15_pcd import decode


def inside_xy(xy: np.ndarray, polygon) -> np.ndarray:
    """Vectorized even-odd containment; no AABB is used as a vessel mask."""
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
        raise ValueError("INVALID_ORACLE_POLYGON")
    x, y = xy[:, 0], xy[:, 1]
    out = np.zeros(len(xy), dtype=bool)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        cross = ((a[1] > y) != (b[1] > y)) & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1]+1e-300)+a[0])
        out ^= cross
    return out


def distance_to_edges(xy: np.ndarray, polygon) -> np.ndarray:
    poly = np.asarray(polygon, dtype=float)
    distances = np.full(len(xy), np.inf)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        d = b-a
        norm2 = float(d @ d)
        if norm2 < 1e-12:
            continue
        t = np.clip(((xy-a) @ d)/norm2, 0, 1)
        distances = np.minimum(distances, np.linalg.norm(xy-a-t[:, None]*d, axis=1))
    return distances


def estimate_local_deck(points: np.ndarray, hatch, vessel, config, other_hatches=()):
    roi = config["roi"]
    xy = points[:, :2].astype(float)
    vessel_mask = inside_xy(xy, vessel)
    ring = vessel_mask & ~inside_xy(xy, hatch) & (distance_to_edges(xy, hatch) <= roi["support_search_m"])
    for other in other_hatches:
        ring &= ~inside_xy(xy, other)
    ring_points = points[ring].astype(float)
    result = {"status": "LOCAL_DECK_UNRESOLVED", "ring_point_count": int(len(ring_points)),
              "support_point_count": 0, "support_azimuth_count": 0,
              "plane_normal": None, "plane_offset": None, "plane_residual_p50": None,
              "plane_residual_p95": None, "competing_plane_count": 0}
    if len(ring_points) < roi["min_deck_support_cells"]:
        result["reason"] = "INSUFFICIENT_RING_SUPPORT"
        return result, None
    # Bounded deterministic sample, fixed pre-existing V1.5 plane threshold.
    rng = np.random.default_rng(2026)
    sample = ring_points[rng.choice(len(ring_points), min(len(ring_points), 20000), replace=False)]
    threshold = config["geometry"]["plane_inlier_m"]
    max_tilt = np.deg2rad(config["geometry"]["normal_angle_deg"])
    hypotheses = []
    for _ in range(config["geometry"]["ransac_iterations"]):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        magnitude = np.linalg.norm(normal)
        if magnitude < 1e-9:
            continue
        normal /= magnitude
        if normal[2] < 0:
            normal = -normal
        if np.arccos(np.clip(normal[2], -1, 1)) > max_tilt:
            continue
        offset = -float(normal @ tri[0])
        support = np.abs(sample @ normal + offset) <= threshold
        count = int(support.sum())
        if count:
            hypotheses.append((count, normal.copy(), offset))
    if not hypotheses:
        result["reason"] = "NO_GRAVITY_CONSISTENT_LOCAL_PLANE"
        return result, None
    hypotheses.sort(key=lambda h: -h[0])
    best_count, normal, offset = hypotheses[0]
    # Audit competing distinct offsets; near-identical RANSAC hypotheses do not count.
    competing = []
    for count, n, d in hypotheses[1:]:
        if count < best_count*0.6:
            break
        if abs(d-offset) > 3*threshold and all(abs(d-old) > 3*threshold for old in competing):
            competing.append(d)
    result["competing_plane_count"] = len(competing)
    support = np.abs(ring_points @ normal + offset) <= threshold
    fit = ring_points[support]
    if len(fit) < roi["min_deck_support_cells"]:
        result["reason"] = "TOO_FEW_LOCAL_INLIERS"
        return result, None
    centroid = fit.mean(0)
    _, _, vh = np.linalg.svd(fit-centroid, full_matrices=False)
    refined = vh[-1]
    if refined[2] < 0:
        refined = -refined
    if np.arccos(np.clip(refined[2], -1, 1)) > max_tilt:
        result["reason"] = "REFINED_DECK_NORMAL_OUT_OF_RANGE"
        return result, None
    refined_offset = -float(refined @ centroid)
    residual = np.abs(ring_points @ refined + refined_offset)
    support = residual <= threshold
    center = np.asarray(hatch, dtype=float).mean(0)
    relative = ring_points[support, :2]-center
    sectors = np.floor(((np.arctan2(relative[:, 1], relative[:, 0])+np.pi)/(2*np.pi))*8).astype(int) % 8
    azimuth = len(set(sectors.tolist()))
    support_xy = ring_points[support, :2]
    span = np.ptp(support_xy, axis=0) if len(support_xy) else np.zeros(2)
    result.update(support_point_count=int(support.sum()), support_azimuth_count=azimuth,
                  plane_normal=refined.tolist(), plane_offset=refined_offset,
                  plane_residual_p50=float(np.percentile(residual[support], 50)),
                  plane_residual_p95=float(np.percentile(residual[support], 95)))
    if len(competing) or azimuth < 3 or max(span) < roi["min_deck_span_m"]:
        result["reason"] = "COMPETING_OR_ONE_SIDED_DECK_SUPPORT"
        return result, None
    result["status"] = "RESOLVED"
    result["reason"] = "LOCAL_RING_PLANE"
    return result, (refined, refined_offset)


def _profile(points, tree, center, tangent, inward, plane, config, profile_id):
    boundary = config["boundary"]
    half = boundary["profile_half_length_m"]
    width = boundary["profile_half_width_m"]
    ids = tree.query_ball_point(center, half+width)
    row = {"profile_id": profile_id, "sample_count": len(ids), "deck_support": 0,
           "break_position": None, "break_strength": None, "valid": False,
           "failure_reason": "INSUFFICIENT_PROFILE_POINTS"}
    if not ids:
        return row
    p = points[ids].astype(float)
    delta = p[:, :2]-center
    along = delta @ inward
    use = (np.abs(delta @ tangent) <= width) & (np.abs(along) <= half)
    if use.sum() < boundary["profile_min_points"]:
        return row
    p, along = p[use], along[use]
    z = p @ plane[0] + plane[1]
    step = boundary["profile_step_m"]
    bins = np.floor(along/step).astype(int)
    samples = []
    for b in np.unique(bins):
        values = z[bins == b]
        samples.append(((b+0.5)*step, float(np.quantile(values, boundary["profile_lower_quantile"])), len(values)))
    row["sample_count"] = len(samples)
    if len(samples) < boundary["profile_min_points"]:
        return row
    deck = [s for s in samples if s[0] < 0 and abs(s[1]) <= config["roi"]["support_band_m"]]
    row["deck_support"] = len(deck)
    if len(deck) < 3 or deck[-1][0]-deck[0][0] < boundary["profile_deck_support_span_m"]:
        row["failure_reason"] = "DECK_NOT_BRACKETED"
        return row
    possibilities = []
    for i, inside in enumerate(samples):
        if inside[0] <= 0 or inside[1] >= -boundary["profile_min_drop_m"]:
            continue
        preceding = [(j, s) for j, s in enumerate(samples[:i])
                     if 0 < inside[0]-s[0] <= boundary["profile_max_bracket_m"] and
                     abs(s[1]) <= config["roi"]["support_band_m"]]
        if not preceding:
            continue
        j, outer = preceding[-1]
        if any(s[1] > config["roi"]["support_band_m"] for s in samples[j+1:i]):
            continue
        if any(s[1] < -boundary["profile_min_drop_m"] for s in samples[j+1:i]):
            continue  # this is the same descent's interior, not a second breakpoint
        low_support = sum(s[1] < -boundary["profile_min_drop_m"] for s in samples[i:])
        if low_support < 3:
            continue
        strength = (outer[1]-inside[1])/(inside[0]-outer[0])
        if strength < boundary["profile_step_slope_min"]:
            continue
        possibilities.append(((outer[0]+inside[0])/2, strength))
    if not possibilities:
        row["failure_reason"] = "NO_OBSERVED_DECK_TO_LOWER_BREAK"
        return row
    if max(x for x, _ in possibilities)-min(x for x, _ in possibilities) > boundary["profile_breakpoint_stability_m"]:
        row["failure_reason"] = "COMPETING_BREAKS"
        return row
    pos, strength = possibilities[0]
    row.update(break_position=[float(center[0]+pos*inward[0]), float(center[1]+pos*inward[1])],
               break_strength=float(strength), valid=True, failure_reason=None)
    return row


def diagnose(points: np.ndarray, hatch, vessel, config, other_hatches=()):
    if not np.isfinite(points).all() or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("INVALID_XYZ")
    deck, plane = estimate_local_deck(points, hatch, vessel, config, other_hatches)
    result = {"status": "LOCAL_DECK" if plane is None else "BOUNDARY",
              "local_deck": deck, "edges": [], "failure_layer": "LOCAL_DECK" if plane is None else "BOUNDARY"}
    if plane is None:
        return result
    poly = np.asarray(hatch, dtype=float)
    tree = cKDTree(points[:, :2])
    for edge_id, (a, b) in enumerate(zip(poly, np.roll(poly, -1, axis=0))):
        length = np.linalg.norm(b-a)
        if length < 1e-9:
            result["edges"].append({"edge_id": f"e{edge_id}", "evidence_type": None,
                                    "reason": "DEGENERATE_ROUGH_ROI_EDGE", "profiles": []})
            continue
        tangent = (b-a)/length
        inward = np.array([-tangent[1], tangent[0]])
        mid = (a+b)/2
        side = inside_xy(np.stack((mid+inward*0.1, mid-inward*0.1)), poly)
        if not side[0] and side[1]:
            inward = -inward
        elif side[0] == side[1]:
            result["edges"].append({"edge_id": f"e{edge_id}", "evidence_type": None,
                                    "reason": "ROUGH_ROI_SIDE_UNRESOLVED", "profiles": []})
            continue
        step = max(0.5, length/30)
        positions = np.arange(step/2, length, step)
        profiles = [_profile(points, tree, a+tangent*d, tangent, inward, plane, config,
                             f"e{edge_id}-p{i}") for i, d in enumerate(positions)]
        good = np.array([p["break_position"] for p in profiles if p["valid"]], dtype=float).reshape(-1, 2)
        edge = {"edge_id": f"e{edge_id}", "profile_count": len(profiles),
                "valid_profile_count": len(good), "break_support_points": len(good),
                "vertical_face_support_points": 0, "fitted_line": None,
                "line_residual_p50": None, "line_residual_p95": None,
                "evidence_type": None, "visibility": "UNCERTAIN",
                "opening_side_status": "SIDE_UNRESOLVED", "reason": "TOO_FEW_VALID_PROFILES",
                "profiles": profiles}
        if len(good) >= config["boundary"]["profile_min_sections"]:
            centroid = good.mean(0)
            _, _, vh = np.linalg.svd(good-centroid, full_matrices=False)
            direction = vh[0]
            residual = np.abs(np.cross(direction, good-centroid))
            if np.percentile(residual, 95) <= config["boundary"]["line_inlier_m"]:
                projection = (good-centroid) @ direction
                lo, hi = centroid+projection.min()*direction, centroid+projection.max()*direction
                if np.linalg.norm(hi-lo) >= config["boundary"]["line_min_length_m"]:
                    edge.update(fitted_line=[lo.tolist(), hi.tolist()],
                                line_residual_p50=float(np.percentile(residual, 50)),
                                line_residual_p95=float(np.percentile(residual, 95)),
                                evidence_type="OBSERVED_PROFILE_BREAK", visibility="VISIBLE",
                                opening_side_status="INNER_OPENING_FACE", reason="MEASURED_LOCAL_BREAK")
        result["edges"].append(edge)
    observed = sum(e["evidence_type"] == "OBSERVED_PROFILE_BREAK" for e in result["edges"])
    result["observed_edge_count"] = observed
    result["status"] = "COMPLETE_OBSERVED_DIAGNOSTIC" if observed == len(poly) else "PARTIAL" if observed else "BOUNDARY_UNRESOLVED"
    result["failure_layer"] = None if observed == len(poly) else "BOUNDARY"
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pcd", type=Path, required=True)
    ap.add_argument("--oracle-roi", type=Path, required=True,
                    help="Evaluation-only JSON with rough_hatch_polygon, vessel_polygon; never product config")
    ap.add_argument("--config", type=Path, default=Path("ship_perception/config/v15.json"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    labels = json.loads(args.oracle_roi.read_text(encoding="utf-8"))
    points, _ = decode(args.pcd)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = diagnose(points, labels["rough_hatch_polygon"], labels["vessel_polygon"], config,
                      labels.get("other_hatch_polygons", []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
