"""Observed-only rectangular topology from S4-R1 raw profile observations.

This is an offline R1 review stage. A height region supplies an opening seed,
never a steel boundary. Every reported side is fitted to current-scan profile
break or vertical-face positions. No expected count or hatch dimensions enter
the detector.
"""

import argparse
import hashlib
import json
import math
import os
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

from .run import _atomic_json

FORBIDDEN_REFERENCE_ROLES = frozenset(("INTERIOR_SURFACE", "ROLE_AMBIGUOUS_OVERFLOW"))


def _unit(vector):
    length = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=float) / length if length > 1e-9 else None


def structural_axes(edges, max_angle_deg):
    """Estimate two orthogonal axes with equal maximum vote per Height node."""
    by_node = defaultdict(list)
    for edge in edges:
        a = np.asarray(edge["a_raw"][:2], dtype=float)
        b = np.asarray(edge["b_raw"][:2], dtype=float)
        direction = _unit(b - a)
        if direction is None:
            continue
        angle = math.atan2(direction[1], direction[0])
        weight = min(float(edge["observed_support_length"]), 10.0)
        by_node[edge["node_id"]].append((angle, weight))
    votes = []
    for rows in by_node.values():
        total = sum(weight for _, weight in rows)
        if total <= 0:
            continue
        votes.extend((angle, weight / total) for angle, weight in rows)
    if not votes:
        return None
    tolerance = math.radians(max_angle_deg)

    def separation(a, b):
        return abs((a - b + math.pi / 4) % (math.pi / 2) - math.pi / 4)

    center = max(
        (angle for angle, _ in votes),
        key=lambda angle: sum(
            weight
            for candidate, weight in votes
            if separation(angle, candidate) <= tolerance
        ),
    )
    inliers = [
        (angle, weight)
        for angle, weight in votes
        if separation(angle, center) <= tolerance
    ]
    if sum(weight for _, weight in inliers) < 0.5 * sum(weight for _, weight in votes):
        return None
    phase = sum(weight * np.exp(4j * angle) for angle, weight in inliers)
    angle = float(np.angle(phase) / 4)
    axis_a = np.array([math.cos(angle), math.sin(angle)])
    axis_b = np.array([-axis_a[1], axis_a[0]])
    return axis_a, axis_b


def _observations(segments):
    records = []
    executed = 0
    for segment in segments:
        if not segment["profile_executed_after"]:
            continue
        executed += 1
        for key, evidence in (
            ("profile_break_positions", "OBSERVED_PROFILE_BREAK"),
            ("face_positions", "OBSERVED_3D_FACE"),
        ):
            for index, xy in enumerate(segment.get(key, [])):
                if len(xy) != 2 or not np.isfinite(xy).all():
                    raise ValueError("NONFINITE_PROFILE_OBSERVATION")
                records.append(
                    dict(
                        observation_id="O%06d" % (len(records) + 1),
                        node_id=segment["node_id"],
                        seed_id=segment["seed_id"],
                        segment_id=segment["segment_id"],
                        evidence_type=evidence,
                        segment_start_xy=segment.get("segment_start_xy"),
                        segment_end_xy=segment.get("segment_end_xy"),
                        segment_tangent=segment.get("segment_tangent"),
                        segment_outward_normal=segment.get("segment_outward_normal"),
                        reference_family_id=segment.get("reference_family_id"),
                        reference_candidate_index=segment.get(
                            "reference_candidate_index"
                        ),
                        reference_role=segment.get("reference_role"),
                        via_family_reference=segment.get("reference_resolution_status")
                        == "FAMILY_REFERENCE_RESOLVED"
                        and not segment.get("profile_executed_before", False),
                        source_index=index,
                        xy=[float(xy[0]), float(xy[1])],
                        status="REJECTED",
                        reject_reason="REJECT_AXIS_INCONSISTENT",
                    )
                )
    return records, executed


def _group_orientation(points):
    if len(points) < 3:
        return None
    centered = points - np.median(points, axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    if values[1] <= 0 or values[0] / values[1] > 0.1:
        return None
    return _unit(vectors[:, 1])


def _candidate_modes(observations, axes, config):
    boundary = config["boundary"]
    groups = defaultdict(list)
    for index, row in enumerate(observations):
        groups[(row["node_id"], row["segment_id"], row["evidence_type"])].append(index)
    contributions = defaultdict(list)
    angle_gate = math.sin(math.radians(2 * boundary["merge_angle_deg"]))
    for (node_id, _, _), ids in groups.items():
        if observations[ids[0]]["reference_role"] in FORBIDDEN_REFERENCE_ROLES:
            for index in ids:
                observations[index][
                    "reject_reason"
                ] = "REJECT_INTERIOR_OR_OVERFLOW_ROLE"
            continue
        points = np.asarray([observations[index]["xy"] for index in ids])
        direction = _group_orientation(points)
        if direction is None:
            for index in ids:
                observations[index][
                    "reject_reason"
                ] = "REJECT_TOO_FEW_OR_NONLINEAR_POINTS"
            continue
        alignment = [abs(float(np.cross(direction, axis))) for axis in axes]
        axis_index = int(np.argmin(alignment))
        if alignment[axis_index] > angle_gate:
            continue
        normal = axes[1 - axis_index]
        rho = points @ normal
        median = float(np.median(rho))
        inliers = np.abs(rho - median) <= 2 * boundary["line_inlier_m"]
        if (
            int(inliers.sum()) < 2
            or np.ptp(points[inliers] @ axes[axis_index])
            < boundary["line_min_length_m"]
        ):
            continue
        inlier_ids = {ids[i] for i in np.flatnonzero(inliers)}
        contributions[(node_id, axis_index)].append((median, sorted(inlier_ids)))
        for index in ids:
            if index not in inlier_ids:
                observations[index]["reject_reason"] = "REJECT_LINE_RESIDUAL"

    modes = defaultdict(list)
    for key, rows in contributions.items():
        # This threshold clusters raw point groups, not fitted fragment lines.
        rows.sort(key=lambda row: row[0])
        clusters = []
        for rho, ids in rows:
            if (
                clusters
                and abs(rho - np.median([entry[0] for entry in clusters[-1]]))
                <= 2 * boundary["line_inlier_m"]
            ):
                clusters[-1].append((rho, ids))
            else:
                clusters.append([(rho, ids)])
        direction = axes[key[1]]
        normal = axes[1 - key[1]]
        for cluster in clusters:
            ids = sorted(set(index for _, indexes in cluster for index in indexes))
            points = np.asarray([observations[index]["xy"] for index in ids])
            if len(ids) < boundary["line_min_points"]:
                continue
            along = points @ direction
            if np.ptp(along) < boundary["line_min_length_m"]:
                continue
            rho = float(np.median(points @ normal))
            residual = np.abs(points @ normal - rho)
            valid = residual <= 2 * boundary["line_inlier_m"]
            ids = [ids[index] for index in np.flatnonzero(valid)]
            along = along[valid]
            if len(ids) < boundary["line_min_points"]:
                continue
            support_segments = sorted(
                {observations[index]["segment_id"] for index in ids}
            )
            modes[key].append(
                dict(
                    rho=rho,
                    observation_ids=[
                        observations[index]["observation_id"] for index in ids
                    ],
                    observation_indexes=ids,
                    source_segment_ids=support_segments,
                    along_min=float(along.min()),
                    along_max=float(along.max()),
                    along_values=along.tolist(),
                    support_count=len(ids),
                    residual_p95_m=float(np.percentile(residual[valid], 95)),
                )
            )
    return modes


def _observed_intervals(values, max_gap_m):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return []
    intervals = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value - previous > max_gap_m:
            intervals.append((start, previous))
            start = value
        previous = value
    intervals.append((start, previous))
    return intervals


def _interval_coverage(intervals, start, stop):
    if stop <= start:
        return 0.0
    covered = sum(
        max(0.0, min(high, stop) - max(low, start)) for low, high in intervals
    )
    return min(1.0, covered / (stop - start))


def _choose_side(modes, target, opposite_span, config, search_m):
    if not modes:
        return None
    candidates = []
    for mode in modes:
        distance = abs(mode["rho"] - target)
        if distance > search_m:
            continue
        intervals = _observed_intervals(
            mode["along_values"], config["boundary"]["max_support_gap_m"]
        )
        coverage = _interval_coverage(intervals, *opposite_span)
        if (
            coverage * (opposite_span[1] - opposite_span[0])
            < config["boundary"]["line_min_length_m"]
        ):
            continue
        score = (
            2 * coverage
            + min(len(mode["source_segment_ids"]), 3) * 0.1
            - distance / search_m
        )
        candidates.append((score, mode))
    return max(candidates, key=lambda row: row[0])[1] if candidates else None


def _corner_support(mode, corner_along, config):
    if mode is None:
        return False
    tolerance = (
        config["boundary"]["corner_join_m"] + config["boundary"]["max_support_gap_m"]
    )
    return min(abs(corner_along - value) for value in mode["along_values"]) <= tolerance


def topology_candidates(height_regions, modes, axes, observations, config):
    results = []
    for node in height_regions:
        node_id = node["node_id"]
        x0, y0, x1, y1 = node["bbox_xy"]
        bbox = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=float)
        spans = [
            (float(np.min(bbox @ axis)), float(np.max(bbox @ axis))) for axis in axes
        ]
        # Axis-aligned Height boxes grow when the current ship is rotated.
        # Account for this purely geometric AABB inflation; no ship dimensions
        # or scene-specific angles are assumed.
        box_size = np.array([x1 - x0, y1 - y0])
        inflation = abs(float(axes[0][0] * axes[0][1])) * float(max(box_size))
        search_m = config["roi"]["support_search_m"] + inflation
        # A side directed along axis A has normal axis B, and vice versa.
        side_modes = [
            _choose_side(
                modes.get((node_id, 1), []), spans[0][0], spans[1], config, search_m
            ),
            _choose_side(
                modes.get((node_id, 1), []), spans[0][1], spans[1], config, search_m
            ),
            _choose_side(
                modes.get((node_id, 0), []), spans[1][0], spans[0], config, search_m
            ),
            _choose_side(
                modes.get((node_id, 0), []), spans[1][1], spans[0], config, search_m
            ),
        ]
        if side_modes[0] is side_modes[1]:
            side_modes[1] = None
        if side_modes[2] is side_modes[3]:
            side_modes[3] = None
        u = [side_modes[i]["rho"] if side_modes[i] else spans[0][i] for i in (0, 1)]
        v = [
            side_modes[i + 2]["rho"] if side_modes[i + 2] else spans[1][i]
            for i in (0, 1)
        ]
        corners = [
            u[i] * axes[0] + v[j] * axes[1] for i, j in ((0, 0), (1, 0), (1, 1), (0, 1))
        ]
        if (
            u[1] - u[0] <= config["geometry"]["coarse_voxel_m"]
            or v[1] - v[0] <= config["geometry"]["coarse_voxel_m"]
        ):
            continue
        sides = []
        for index, mode in enumerate(side_modes):
            if index < 2:
                ends = v
            else:
                ends = u
            observed = mode is not None
            corner_flags = [_corner_support(mode, value, config) for value in ends]
            intervals = (
                _observed_intervals(
                    mode["along_values"], config["boundary"]["max_support_gap_m"]
                )
                if observed
                else []
            )
            coverage = _interval_coverage(intervals, ends[0], ends[1])
            side = dict(
                side_id=("U0", "U1", "V0", "V1")[index],
                evidence_types=(
                    sorted(
                        {
                            observations[observation_index]["evidence_type"]
                            for observation_index in mode["observation_indexes"]
                        }
                    )
                    if observed
                    else ["UNKNOWN"]
                ),
                visibility="VISIBLE" if observed else "UNKNOWN",
                opening_side_status="OBSERVED_CANDIDATE" if observed else "UNRESOLVED",
                rho=mode["rho"] if observed else None,
                support_count=mode["support_count"] if observed else 0,
                uncertainty_m=mode["residual_p95_m"] if observed else None,
                source_segment_ids=mode["source_segment_ids"] if observed else [],
                observation_ids=mode["observation_ids"] if observed else [],
                observed_intervals=intervals,
                corner_supported=corner_flags,
                coverage=float(coverage),
                observed=observed,
            )
            sides.append(side)
        observed_count = sum(side["observed"] for side in sides)
        complete = observed_count == 4 and all(
            all(side["corner_supported"])
            and side["coverage"] >= config["boundary"]["min_edge_coverage"]
            for side in sides
        )
        status = (
            "COMPLETE_OBSERVED_RECTANGLE"
            if complete
            else "PARTIAL_RECTANGLE" if observed_count >= 2 else "INSUFFICIENT_TOPOLOGY"
        )
        for side in sides:
            for observation_id in side["observation_ids"]:
                row = observations[int(observation_id[1:]) - 1]
                row["status"] = "USED_IN_BOUNDARY_LINE"
                row["reject_reason"] = None
        results.append(
            dict(
                node_id=node_id,
                status=status,
                polygon_xy=(
                    [corner.tolist() for corner in corners] if complete else None
                ),
                hypothesis_corners_xy=[corner.tolist() for corner in corners],
                observed_side_count=observed_count,
                corner_supported_side_count=sum(
                    all(side["corner_supported"]) for side in sides
                ),
                reason=(
                    "RAW_OBSERVATION_CORNERS_AND_COVERAGE"
                    if complete
                    else "MISSING_SIDE_OR_CORNER_SUPPORT"
                ),
                sides=sides,
                bbox_seed_xy=node["bbox_xy"],
            )
        )
    for row in observations:
        if row["status"] == "REJECTED" and row["reject_reason"] is None:
            row["reject_reason"] = "REJECT_TOPOLOGY_INCONSISTENT"
    return results


def read_xyzrgb_ply(path):
    """Strictly decode the narrow binary PLY format used by review artifacts."""
    with Path(path).open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line or len(header) > 4096:
                raise ValueError("INVALID_PLY_HEADER")
            header += line
        lines = header.decode("ascii").splitlines()
        if lines[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError("UNSUPPORTED_PLY_FORMAT")
        vertex = [line for line in lines if line.startswith("element vertex ")]
        properties = [line for line in lines if line.startswith("property ")]
        if len(vertex) != 1 or properties != [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]:
            raise ValueError("UNSUPPORTED_PLY_SCHEMA")
        count = int(vertex[0].split()[-1])
        payload = stream.read()
    if len(payload) != count * struct.calcsize("<fffBBB"):
        raise ValueError("PLY_LENGTH_MISMATCH")
    array = np.frombuffer(
        payload,
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("r", "u1"),
                ("g", "u1"),
                ("b", "u1"),
            ]
        ),
    )
    xyz = np.column_stack((array["x"], array["y"], array["z"]))
    if not np.isfinite(xyz).all():
        raise ValueError("NONFINITE_PLY_VERTEX")
    return xyz


def analyze(segments, edges, height_regions, config):
    observations, executed = _observations(segments)
    forbidden_segments = {
        row["segment_id"]
        for row in segments
        if row.get("reference_role") in FORBIDDEN_REFERENCE_ROLES
    }
    eligible_edges = [
        edge for edge in edges if edge.get("segment_id") not in forbidden_segments
    ]
    axes = structural_axes(eligible_edges, 2 * config["boundary"]["merge_angle_deg"])
    if axes is None:
        for row in observations:
            row["reject_reason"] = "REJECT_STRUCTURAL_AXES_UNRESOLVED"
        return dict(
            status="STRUCTURAL_AXES_UNRESOLVED",
            candidates=[],
            profile_executed_segments=executed,
            observation_count=len(observations),
            used_count=0,
            rejected_count=len(observations),
            observation_accounting_pass=True,
            observations=observations,
        )
    modes = _candidate_modes(observations, axes, config)
    candidates = topology_candidates(height_regions, modes, axes, observations, config)
    used = {
        row["observation_id"]
        for row in observations
        if row["status"] == "USED_IN_BOUNDARY_LINE"
    }
    rejected = {
        row["observation_id"] for row in observations if row["status"] == "REJECTED"
    }
    all_ids = {row["observation_id"] for row in observations}
    if used & rejected or used | rejected != all_ids:
        raise ValueError("OBSERVATION_ACCOUNTING_FAILED")
    ready = any(
        candidate["status"] != "INSUFFICIENT_TOPOLOGY" for candidate in candidates
    )
    return dict(
        status="CANDIDATE_READY" if ready else "NOT_READY",
        coordinate_frame="RAW_INPUT_FRAME",
        axes=[axis.tolist() for axis in axes],
        profile_executed_segments=executed,
        observation_count=len(observations),
        used_count=len(used),
        rejected_count=len(rejected),
        observation_accounting_pass=True,
        observations=observations,
        candidates=candidates,
    )


def annotate_ply_support(result, xyz, config):
    """Check review PLY against selected lines without treating it as raw GT."""
    if "axes" not in result:
        return
    axes = [np.asarray(axis) for axis in result["axes"]]
    xy = xyz[:, :2]
    tolerance = config["boundary"]["corner_join_m"]
    for candidate in result["candidates"]:
        for index, side in enumerate(candidate["sides"]):
            if not side["observed"]:
                side["ply_near_line_count"] = 0
                continue
            direction = axes[1] if index < 2 else axes[0]
            normal = axes[0] if index < 2 else axes[1]
            along = xy @ direction
            near = np.abs(xy @ normal - side["rho"]) <= tolerance
            near &= (
                along >= min(low for low, _ in side["observed_intervals"]) - tolerance
            )
            near &= (
                along <= max(high for _, high in side["observed_intervals"]) + tolerance
            )
            side["ply_near_line_count"] = int(near.sum())


def render_review(result, xyz, path, node_id=None):
    """Draw review PLY and observed side intervals; dashed boxes are hypotheses."""
    cache = Path(path).parent / ".mplconfig"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    ax.scatter(xyz[:, 0], xyz[:, 1], s=1.5, c="#667b9b", alpha=0.35, rasterized=True)
    if "axes" in result:
        axes = [np.asarray(axis) for axis in result["axes"]]
        candidates = (
            result["candidates"]
            if node_id is None
            else [row for row in result["candidates"] if row["node_id"] == node_id]
        )
        ranked = sorted(
            candidates,
            key=lambda row: (
                -row["observed_side_count"],
                -sum(side["coverage"] for side in row["sides"]),
            ),
        )[:8]
        for candidate in ranked:
            corner = np.asarray(candidate["hypothesis_corners_xy"])
            cycle = np.vstack((corner, corner[0]))
            ax.plot(
                cycle[:, 0], cycle[:, 1], color="#d7a21b", ls=":", lw=0.8, alpha=0.6
            )
            center = np.mean(corner, axis=0)
            ax.text(
                center[0],
                center[1],
                candidate["node_id"]
                + "\n"
                + candidate["status"].replace("_RECTANGLE", ""),
                ha="center",
                va="center",
                fontsize=7,
                color="#f2d77a",
                bbox=dict(facecolor="#101820", alpha=0.8, edgecolor="none"),
            )
            for index, side in enumerate(candidate["sides"]):
                if not side["observed"]:
                    continue
                direction = axes[1] if index < 2 else axes[0]
                normal = axes[0] if index < 2 else axes[1]
                for low, high in side["observed_intervals"]:
                    endpoints = np.asarray(
                        [
                            normal * side["rho"] + direction * low,
                            normal * side["rho"] + direction * high,
                        ]
                    )
                    ax.plot(endpoints[:, 0], endpoints[:, 1], color="#49e0a5", lw=2.1)
        if node_id is not None and ranked:
            x0, y0, x1, y1 = ranked[0]["bbox_seed_xy"]
            margin = 3.0
            ax.set_xlim(x0 - margin, x1 + margin)
            ax.set_ylim(y0 - margin, y1 + margin)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Raw X (m)")
    ax.set_ylabel("Raw Y (m)")
    ax.set_title(
        result["scene_id"]
        + " — observed boundary topology; dotted = unconfirmed hypothesis"
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--s4r1-root", type=Path)
    source.add_argument("--pcd", type=Path)
    parser.add_argument("--height-root", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from .run import DEFAULT_CONFIG, resolve_config

    config, _ = resolve_config(DEFAULT_CONFIG)
    height = json.loads(
        (args.height_root / args.scene / "hatch_regions.json").read_text(
            encoding="utf-8"
        )
    )
    input_sha = None
    if args.pcd is not None:
        from ship_perception.tools.v15_pcd import decode
        from .family_reference import analyze_scene_s4r1
        from .heightmap_batch import RESEARCH_CONFIG

        input_sha = hashlib.sha256(args.pcd.read_bytes()).hexdigest()
        if input_sha != height["input_sha256"]:
            raise ValueError("BASELINE_INPUT_MISMATCH")
        points, _ = decode(args.pcd)
        research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
        reference = analyze_scene_s4r1(points, config, research, height)
        segments, edge_rows = reference["segment_resolutions"], reference["edges"]
    else:
        reference = json.loads(
            (
                args.s4r1_root / args.scene / "segment_reference_resolution.json"
            ).read_text(encoding="utf-8")
        )
        edges = json.loads(
            (args.s4r1_root / args.scene / "structural_edges.json").read_text(
                encoding="utf-8"
            )
        )
        segments, edge_rows = reference["segments"], edges["edges"]
    regions = [
        dict(node_id=row["node_id"], bbox_xy=row["bbox_xy"])
        for row in height["level_hierarchy"]
    ]
    result = analyze(segments, edge_rows, regions, config)
    xyz = read_xyzrgb_ply(args.ply)
    result["scene_id"] = args.scene
    result["input_pcd_sha256"] = input_sha
    result["ply_vertex_count"] = len(xyz)
    result["ply_bbox_xyz"] = [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()]
    annotate_ply_support(result, xyz, config)
    _atomic_json(args.output, result)
    render_review(result, xyz, args.output.with_suffix(".png"))
    for candidate in result.get("candidates", []):
        if candidate["observed_side_count"] >= 2:
            path = args.output.with_name(
                args.output.stem + "_node_" + candidate["node_id"] + ".png"
            )
            render_review(result, xyz, path, candidate["node_id"])
    print(
        json.dumps(
            dict(
                scene=args.scene,
                status=result["status"],
                ply_vertex_count=len(xyz),
                complete=sum(
                    row["status"] == "COMPLETE_OBSERVED_RECTANGLE"
                    for row in result["candidates"]
                ),
                partial=sum(
                    row["status"] == "PARTIAL_RECTANGLE" for row in result["candidates"]
                ),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
