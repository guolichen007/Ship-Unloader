"""Research-only P1.5 plane-family consensus and spatial-role forensics.

This module never re-runs ``estimate_segment_deck``. It consumes the already
frozen ``baseline_006_p1`` plane candidates + raw IDs and re-organizes them
into cross-segment coplanar families, then attaches a spatial ROLE_HYPOTHESIS
(never a physical Deck / Coaming / Cargo label). It is never imported by the
detector or the structural audit path.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ship_perception.tools.v15_pcd import decode

from .batch import NORMAL6
from .local_deck import _physical_components
from .run import REPO, _atomic_json
from .visualization import write_colored_ply

ROLES = (
    "BROAD_PERIMETER_SUPPORT",
    "NARROW_BOUNDARY_STRIP",
    "INTERIOR_SURFACE",
    "REMOTE_PLANAR_STRUCTURE",
    "ROLE_AMBIGUOUS",
    "ROLE_AMBIGUOUS_OVERFLOW",
    "UNKNOWN",
)

PALETTE = (
    (36, 190, 220), (255, 188, 53), (123, 214, 85), (177, 116, 238),
    (255, 110, 192), (80, 140, 250), (223, 220, 75), (245, 145, 35),
)


def normalize_normal(normal):
    vector = np.asarray(normal, dtype=float)
    length = np.linalg.norm(vector)
    if length < 1e-12:
        return vector
    vector = vector / length
    if vector[2] < 0:
        vector = -vector
    return vector


def centered_offset(offset, normal, centroid):
    return float(offset + float(np.asarray(normal) @ np.asarray(centroid)))


def _corners_of(bbox):
    values = list(bbox)
    if len(values) == 4 and all(np.isscalar(value) for value in values):
        x0, y0, x1, y1 = values
        return np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=float)
    return np.asarray(values, dtype=float)


def _point_in_polygon(xy, corners):
    xy = np.asarray(xy, dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    for index in range(len(corners)):
        x1, y1 = corners[index]
        x2, y2 = corners[(index + 1) % len(corners)]
        straddles = (y1 > y) != (y2 > y)
        if not np.any(straddles):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (y - y1) / (y2 - y1)
        t = np.where(straddles, t, 0.0)
        inside ^= straddles & (x < x1 + t * (x2 - x1))
    return inside


def _distance_to_polygon(xy, corners):
    xy = np.asarray(xy, dtype=float)
    best = np.full(len(xy), np.inf)
    for index in range(len(corners)):
        a = corners[index]
        b = corners[(index + 1) % len(corners)]
        ab = b - a
        denominator = float(ab @ ab)
        if denominator < 1e-12:
            continue
        t = np.clip(((xy - a) @ ab) / denominator, 0.0, 1.0)
        projection = a + t[:, None] * ab
        best = np.minimum(best, np.linalg.norm(xy - projection, axis=1))
    return best


def classify_support(xy, bbox, band_m):
    """Classify support points into inside / boundary / outside of a node box.

    ``bbox`` is either an axis-aligned ``(x0, y0, x1, y1)`` tuple or a list of
    polygon corners, so a rigidly transformed (rotated) node stays correct.
    """
    xy = np.asarray(xy, dtype=float)
    corners = _corners_of(bbox)
    inside = _point_in_polygon(xy, corners)
    distance = _distance_to_polygon(xy, corners)
    boundary = ~inside & (distance <= band_m)
    outside = ~inside & (distance > band_m)
    return inside, boundary, outside


def cluster_candidates(candidates, angle_deg, separation_m):
    """Greedy coplanar clustering on normalized plane equation (rotation safe).

    Each candidate keeps its ``segment_id``; a segment votes at most once per
    family because membership is per candidate, and vote counting dedups by
    segment later. The comparison uses the angular difference between unit
    normals and the centered-offset separation.
    """
    families = []
    order = sorted(range(len(candidates)), key=lambda index: -candidates[index]["support_count"])
    for index in order:
        candidate = candidates[index]
        normal = normalize_normal(candidate["normal"])
        d_center = candidate["d_centered"]
        placed = False
        for family in families:
            angle = math.degrees(math.acos(float(np.clip(
                np.dot(family["rep_normal"], normal), -1.0, 1.0))))
            separation = abs(d_center - family["rep_d_centered"])
            if angle <= angle_deg and separation <= separation_m:
                family["members"].append(index)
                placed = True
                break
        if not placed:
            families.append(dict(members=[index], rep_normal=normal,
                                 rep_d_centered=d_center))
    # Recompute representatives after assignment.
    for family in families:
        normals = np.vstack([normalize_normal(candidates[index]["normal"])
                             for index in family["members"]])
        representative = normals.mean(axis=0)
        representative /= np.linalg.norm(representative)
        if representative[2] < 0:
            representative = -representative
        d_centers = [candidates[index]["d_centered"] for index in family["members"]]
        family["rep_normal"] = representative
        family["rep_d_centered"] = float(np.median(d_centers))
    return families


def _support_xy(points, raw_ids):
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    if not len(raw_ids):
        return np.empty((0, 2))
    return points[raw_ids, :2].astype(float)


def family_role(family, candidates, points, node_bbox_map, band_m, narrow_threshold):
    """Aggregate spatial support across members and emit a ROLE_HYPOTHESIS."""
    inside = boundary = outside = 0
    widths = []
    for index in family["members"]:
        candidate = candidates[index]
        node_id = candidate["node_id"]
        bbox = node_bbox_map.get(node_id)
        if bbox is None:
            continue
        xy = _support_xy(points, candidate["raw_ids"])
        if not len(xy):
            continue
        ins, bnd, out = classify_support(xy, bbox, band_m)
        inside += int(ins.sum())
        boundary += int(bnd.sum())
        outside += int(out.sum())
        widths.append(candidate["outward_width"])
    total = inside + boundary + outside
    median_width = float(np.median(widths)) if widths else 0.0
    if total == 0:
        return dict(role_hypothesis="UNKNOWN", support_inside_fraction=0.0,
                    support_boundary_fraction=0.0, support_outside_fraction=0.0,
                    median_transverse_width=median_width, support_point_count=0)
    inside_frac = inside / total
    boundary_frac = boundary / total
    outside_frac = outside / total
    is_narrow = median_width <= narrow_threshold
    spans_all = (inside_frac >= 0.1 and boundary_frac >= 0.1 and outside_frac >= 0.1)
    if spans_all and not is_narrow:
        role = "ROLE_AMBIGUOUS_OVERFLOW"
    elif inside_frac >= 0.5:
        role = "INTERIOR_SURFACE"
    elif boundary_frac >= 0.4:
        role = "NARROW_BOUNDARY_STRIP" if is_narrow else "BROAD_PERIMETER_SUPPORT"
    elif outside_frac >= 0.5:
        role = "REMOTE_PLANAR_STRUCTURE"
    else:
        role = "ROLE_AMBIGUOUS"
    return dict(role_hypothesis=role, support_inside_fraction=float(inside_frac),
                support_boundary_fraction=float(boundary_frac),
                support_outside_fraction=float(outside_frac),
                median_transverse_width=median_width, support_point_count=total)


def spatial_connectedness(family, candidates, points, config):
    union = np.concatenate([candidates[index]["raw_ids"] for index in family["members"]]) \
        if family["members"] else np.empty(0, dtype=np.int64)
    union = np.unique(union)
    if not len(union):
        return 0.0
    components = _physical_components(points[union].astype(float),
                                      config["roi"]["support_connectivity_m"])
    if not components:
        return 0.0
    largest = max(len(component) for component in components)
    return float(largest / len(union))


def load_scene_candidates(scene, p1_root, p5_root, data_root, manifest_path, config):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scans = {scan["scan_id"]: scan for scan in manifest["scans"]}
    scan = scans[NORMAL6[scene]]
    pcd = Path(data_root) / scan["pcd_path"]
    points, _ = decode(pcd)
    points = np.asarray(points, dtype=np.float32)
    centroid = points.mean(axis=0)

    ambiguity = json.loads(
        (Path(p1_root) / scene / "p1_ambiguity_forensics.json").read_text(encoding="utf-8"))
    npz = np.load(Path(p1_root) / scene / "ambiguity_ownership.npz")
    audit5 = json.loads(
        (Path(p5_root) / scene / "node_structure_audit.json").read_text(encoding="utf-8"))
    node_bbox_map = {node["node_id"]: tuple(node["bbox_xy"]) for node in audit5["nodes"]}

    candidates = []
    for segment in ambiguity["segments"]:
        for candidate in segment["plane_candidates"]:
            if not candidate["qualified"]:
                continue
            key = "%s_candidate_%02d" % (segment["segment_id"], candidate["candidate_index"])
            raw_ids = np.asarray(npz[key], dtype=np.int64) if key in npz else np.empty(0, dtype=np.int64)
            normal = normalize_normal(candidate["normal"])
            offset = float(candidate["offset"])
            candidates.append(dict(
                scene_id=scene,
                node_id=segment["node_id"],
                segment_id=segment["segment_id"],
                candidate_index=candidate["candidate_index"],
                baseline_selected=segment.get("baseline_selected", False),
                normal=normal,
                offset=offset,
                d_centered=centered_offset(offset, normal, centroid),
                support_count=candidate["support_count"],
                outward_width=candidate["outward_width_m"],
                along_span=candidate["along_span_m"],
                along_coverage=candidate["along_coverage"],
                raw_ids=raw_ids,
            ))
    return dict(points=points, centroid=centroid, node_bbox_map=node_bbox_map,
                candidates=candidates, scan_id=scan["scan_id"],
                source_p1_sha=ambiguity["software_git_sha"],
                input_sha256=ambiguity["input_sha256"],
                selected_nodes=[node["node_id"] for node in audit5["nodes"]
                                if node.get("baseline_selected")])


def run_consensus(candidates, points, node_bbox_map, config, angle_deg, separation_m):
    """Cluster candidates and attach spatial roles; returns (family_records, narrow)."""
    families = cluster_candidates(candidates, angle_deg, separation_m)
    widths = [float(np.median([candidates[i]["outward_width"] for i in f["members"]]))
              for f in families]
    narrow_threshold = 0.5 * float(np.median(widths)) if widths else 0.0
    total_segments = len({candidate["segment_id"] for candidate in candidates})
    family_records = []
    for family_id, family in enumerate(families):
        members = family["members"]
        segments = [candidates[index]["segment_id"] for index in members]
        nodes = [candidates[index]["node_id"] for index in members]
        role = family_role(family, candidates, points, node_bbox_map,
                           config["roi"]["support_search_m"], narrow_threshold)
        normal_dispersion = max(
            (math.degrees(math.acos(float(np.clip(
                np.dot(family["rep_normal"], normalize_normal(candidates[index]["normal"])),
                -1.0, 1.0)))) for index in members), default=0.0)
        separation_dispersion = max(
            (abs(candidates[index]["d_centered"] - family["rep_d_centered"])
             for index in members), default=0.0)
        family_records.append(dict(
            family_id="f%03d" % family_id,
            unique_segment_votes=len(set(segments)),
            unique_node_votes=len(set(nodes)),
            segment_vote_ratio=float(len(set(segments)) / max(total_segments, 1)),
            member_candidate_count=len(members),
            member_candidate_indexes=[int(index) for index in members],
            representative_normal=family["rep_normal"].tolist(),
            representative_offset=family["rep_d_centered"],
            normal_dispersion_deg=float(normal_dispersion),
            plane_separation_dispersion_m=float(separation_dispersion),
            support_point_count=int(sum(candidates[index]["support_count"] for index in members)),
            median_transverse_width=role["median_transverse_width"],
            total_along_span=float(sum(candidates[index]["along_span"] for index in members)),
            spatial_connectedness=spatial_connectedness(family, candidates, points, config),
            support_inside_fraction=role["support_inside_fraction"],
            support_boundary_fraction=role["support_boundary_fraction"],
            support_outside_fraction=role["support_outside_fraction"],
            role_hypothesis=role["role_hypothesis"],
            node_ids=sorted(set(nodes)),
            segment_ids=sorted(set(segments)),
            baseline_selected_votes=sum(candidates[index].get("baseline_selected", False)
                                        for index in members),
        ))
    family_records.sort(key=lambda row: (-row["unique_segment_votes"], -row["support_point_count"]))
    return family_records, narrow_threshold


def analyze_scene(scene, p1_root, p5_root, data_root, manifest_path, config,
                  angle_deg, separation_m):
    loaded = load_scene_candidates(scene, p1_root, p5_root, data_root, manifest_path, config)
    candidates = loaded["candidates"]
    family_records, narrow_threshold = run_consensus(
        candidates, loaded["points"], loaded["node_bbox_map"], config, angle_deg, separation_m)
    return dict(
        scene_id=scene,
        centroid=loaded["centroid"].tolist(),
        source_p1_sha=loaded["source_p1_sha"],
        input_sha256=loaded["input_sha256"],
        scan_id=loaded["scan_id"],
        candidate_count=len(candidates),
        total_segments_with_candidates=len({candidate["segment_id"] for candidate in candidates}),
        selected_nodes=loaded["selected_nodes"],
        angle_threshold_deg=angle_deg,
        separation_threshold_m=separation_m,
        narrow_width_threshold_m=float(narrow_threshold),
        family_count=len(family_records),
        families=family_records,
        candidates=candidates,
        points=loaded["points"],
        node_bbox_map=loaded["node_bbox_map"],
    )


def cross_hatch_consensus(analysis, config):
    """Same family recurring around multiple openings is ship-internal consensus."""
    cross = [family for family in analysis["families"] if family["unique_node_votes"] >= 2]
    structural_cross = [family for family in cross if family["role_hypothesis"]
                        in ("BROAD_PERIMETER_SUPPORT", "NARROW_BOUNDARY_STRIP")]
    return dict(
        schema="ship_perception.v15r.p15_cross_hatch_consensus.1",
        scene_id=analysis["scene_id"],
        cross_hatch_family_count=len(cross),
        structural_cross_hatch_family_count=len(structural_cross),
        current_ship_internal_consensus=bool(structural_cross),
        cross_hatch_families=[
            dict(family_id=family["family_id"], unique_node_votes=family["unique_node_votes"],
                 node_ids=family["node_ids"], role_hypothesis=family["role_hypothesis"],
                 median_transverse_width=family["median_transverse_width"],
                 representative_normal=family["representative_normal"])
            for family in structural_cross
        ],
        possible_missing_hatch_regions=[],
        caveat="POSSIBLE_MISSING_HATCH_REGION_IS_DIAGNOSTIC_ONLY_NO_NEW_HATCH",
    )


def sensitivity_sweep(candidates, config, points, node_bbox_map):
    """Diagnostic 0.5x / 1x / 2x family-count sweep; never used to pick a threshold."""
    base_angle = config["frame"]["normal_refine_deg"]
    base_sep = config["geometry"]["plane_inlier_m"]
    rows = []
    for factor in (0.5, 1.0, 2.0):
        families, _ = run_consensus(candidates, points, node_bbox_map, config,
                                    base_angle * factor, base_sep * factor)
        rows.append(dict(
            factor=factor,
            angle_threshold_deg=base_angle * factor,
            separation_threshold_m=base_sep * factor,
            family_count=len(families),
            cross_segment_families=sum(1 for family in families
                                       if family["unique_segment_votes"] >= 2),
            stable_families=sum(1 for family in families
                                if family["unique_segment_votes"] >= 3),
            top_family_segment_votes=max((family["unique_segment_votes"]
                                          for family in families), default=0),
        ))
    return rows


def _write_candidates_csv(path, analyses):
    rows = []
    for analysis in analyses:
        for candidate in analysis["candidates"]:
            rows.append(dict(
                scene_id=analysis["scene_id"], node_id=candidate["node_id"],
                segment_id=candidate["segment_id"],
                candidate_index=candidate["candidate_index"],
                normal_x=candidate["normal"][0], normal_y=candidate["normal"][1],
                normal_z=candidate["normal"][2], offset=candidate["offset"],
                d_centered=candidate["d_centered"], support_count=candidate["support_count"],
                outward_width=candidate["outward_width"],
                along_span=candidate["along_span"],
                along_coverage=candidate["along_coverage"],
                baseline_selected=candidate["baseline_selected"],
            ))
    fieldnames = ["scene_id", "node_id", "segment_id", "candidate_index", "normal_x",
                  "normal_y", "normal_z", "offset", "d_centered", "support_count",
                  "outward_width", "along_span", "along_coverage", "baseline_selected"]
    _write_csv(path, rows, fieldnames)


def _write_csv(path, rows, fieldnames=None):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = fieldnames or sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_family_overlay(directory, analysis):
    clouds, colors = [], []
    palette = PALETTE
    for family in analysis["families"]:
        ids = np.concatenate([analysis["candidates"][member]["raw_ids"]
                              for member in family["member_candidate_indexes"]]) \
            if family["member_candidate_indexes"] else np.empty(0, dtype=np.int64)
        ids = np.unique(ids)
        if not len(ids):
            continue
        clouds.append(analysis["points"][ids])
        colors.append(np.tile(np.array(palette[int(family["family_id"][1:]) % len(palette)],
                                       dtype=np.uint8), (len(ids), 1)))
    if not clouds:
        return None
    write_colored_ply(directory / "plane_family_overlay.ply",
                      np.vstack(clouds), np.vstack(colors).astype(np.uint8))
    return "plane_family_overlay.ply"


def _write_family_ownership(directory, analysis):
    payload = {}
    for family in analysis["families"]:
        ids = np.concatenate([analysis["candidates"][member]["raw_ids"]
                              for member in family["member_candidate_indexes"]]) \
            if family["member_candidate_indexes"] else np.empty(0, dtype=np.int64)
        payload[family["family_id"]] = np.unique(ids)
    np.savez_compressed(directory / "plane_family_ownership.npz", **payload)
    return "plane_family_ownership.npz"


def run_batch_p15(run_id, p1_root, p5_root, data_root, manifest_path, output_root,
                  angle_deg=3.0, separation_m=0.05):
    if not run_id or run_id in (".", "..") or any(c in run_id for c in "/\\:"):
        raise ValueError("INVALID_RUN_ID")
    destination = Path(output_root) / run_id
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("P15_RUN_ALREADY_EXISTS:" + str(destination))
    config = json.loads((REPO / "ship_perception/config/v15.json").read_text(encoding="utf-8"))
    analyses = []
    for scene in NORMAL6:
        analysis = analyze_scene(scene, p1_root, p5_root, data_root, manifest_path,
                                 config, angle_deg, separation_m)
        directory = destination / scene
        directory.mkdir(parents=True, exist_ok=False)
        _write_family_overlay(directory, analysis)
        _write_family_ownership(directory, analysis)
        _atomic_json(directory / "plane_families.json", dict(
            schema="ship_perception.v15r.p15_plane_families.1",
            scene_id=scene, source_p1_sha=analysis["source_p1_sha"],
            input_sha256=analysis["input_sha256"], scan_id=analysis["scan_id"],
            angle_threshold_deg=angle_deg, separation_threshold_m=separation_m,
            narrow_width_threshold_m=analysis["narrow_width_threshold_m"],
            candidate_count=analysis["candidate_count"],
            total_segments_with_candidates=analysis["total_segments_with_candidates"],
            family_count=analysis["family_count"],
            selected_nodes=analysis["selected_nodes"],
            semantics="ROLE_IS_HYPOTHESIS_NOT_PHYSICAL_TRUTH; SEGMENT_VOTES_AT_MOST_ONE_PER_FAMILY",
            families=analysis["families"],
        ))
        _atomic_json(directory / "plane_family_roles.json", dict(
            schema="ship_perception.v15r.p15_plane_family_roles.1",
            scene_id=scene,
            roles=[dict(family_id=family["family_id"],
                        role_hypothesis=family["role_hypothesis"],
                        support_inside_fraction=family["support_inside_fraction"],
                        support_boundary_fraction=family["support_boundary_fraction"],
                        support_outside_fraction=family["support_outside_fraction"],
                        median_transverse_width=family["median_transverse_width"],
                        unique_segment_votes=family["unique_segment_votes"],
                        unique_node_votes=family["unique_node_votes"])
                   for family in analysis["families"]],
        ))
        _atomic_json(directory / "cross_hatch_consensus.json",
                     cross_hatch_consensus(analysis, config))
        print("%s: %d candidates -> %d families"
              % (scene, analysis["candidate_count"], analysis["family_count"]), flush=True)
        analyses.append(analysis)
    destination.mkdir(parents=True, exist_ok=True)
    _write_candidates_csv(destination / "plane_candidates_normalized.csv", analyses)
    summary_rows = []
    for analysis in analyses:
        role_counts = Counter(family["role_hypothesis"] for family in analysis["families"])
        summary_rows.append(dict(
            scene_id=analysis["scene_id"], candidate_count=analysis["candidate_count"],
            total_segments_with_candidates=analysis["total_segments_with_candidates"],
            family_count=analysis["family_count"],
            cross_segment_families=sum(1 for family in analysis["families"]
                                       if family["unique_segment_votes"] >= 2),
            structural_families=sum(1 for family in analysis["families"]
                                    if family["role_hypothesis"] in
                                    ("BROAD_PERIMETER_SUPPORT", "NARROW_BOUNDARY_STRIP")),
            **{role.lower(): role_counts.get(role, 0) for role in ROLES},
        ))
    _write_csv(destination / "normal6_plane_consensus_summary.csv", summary_rows)
    _atomic_json(destination / "batch_summary.json", dict(
        schema="ship_perception.v15r.p15_batch.1", run_id=run_id,
        source_p1_sha=analyses[0]["source_p1_sha"] if analyses else None,
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        scipy_version=__import__("scipy").__version__,
        sensitivity_sweep={
            analysis["scene_id"]: sensitivity_sweep(
                analysis["candidates"], config, analysis["points"], analysis["node_bbox_map"])
            for analysis in analyses
        },
        scenes={analysis["scene_id"]: dict(candidate_count=analysis["candidate_count"],
                                           family_count=analysis["family_count"])
                for analysis in analyses},
    ))
    return analyses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--p1-root", type=Path,
                        default=REPO / "Ship-Unloader-Work/r1_height_structure_forensics/baseline_006_p1")
    parser.add_argument("--p5-root", type=Path,
                        default=REPO / "Ship-Unloader-Work/r1_height_structure_forensics/baseline_005_p0")
    parser.add_argument("--dataset-manifest", type=Path,
                        default=REPO / "ship_perception/datasets/v15_dataset_manifest.json")
    parser.add_argument("--data-root", type=Path,
                        default=REPO / "Ship-Unloader-Data/legacy/hold_detector")
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "Ship-Unloader-Work/r1_plane_consensus")
    args = parser.parse_args()
    run_batch_p15(args.run_id, args.p1_root, args.p5_root, args.data_root,
                  args.dataset_manifest, args.output_root)


if __name__ == "__main__":
    main()
