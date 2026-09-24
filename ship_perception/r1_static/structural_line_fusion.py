"""S4-R2: fuse per-node structural primitives into observed Structural Lines.

The S4-R1 chain leaves tens of per-segment edge fragments. Here we group the
PROFILE_BREAK / OBSERVED_3D_FACE observations of one Height node into a small
number of stable 2D structural lines (normal form ``n2 . p = rho``), preserving
occlusion gaps as UNOBSERVED and never merging parallel-but-separate edges.
``NARROW_BOUNDARY_STRIP`` only corroborates; its centerline is never a Hatch
edge.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ship_perception.tools.v15_config_identity import config_hash
from ship_perception.tools.v15_pcd import decode

from .batch import NORMAL6
from .boundary_refinement import _fit_line
from .family_reference import analyze_scene_s4r1
from .heightmap_batch import RESEARCH_CONFIG
from .heightmap_render import _draw_line, _draw_text, _pixel, _save_png, _surface_image
from .run import DEFAULT_CONFIG, REPO, _atomic_json, _git_sha, resolve_config
from .visualization import write_colored_ply


PROFILE_COLOR = (245, 145, 35)
FACE_COLOR = (35, 215, 220)
MULTI_COLOR = (250, 220, 40)
BAND_COLOR = (177, 116, 238)
LABEL_COLOR = (255, 255, 255)


def _line_2d_normal(positions):
    """Canonical 2D line (normal, rho) with normal.y >= 0."""
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2:
        return None
    center = positions.mean(axis=0)
    _, _, vh = np.linalg.svd(positions - center, full_matrices=False)
    direction = vh[0]
    normal = np.array((-direction[1], direction[0]))
    rho = float(normal @ center)
    if normal[1] < 0 or (abs(normal[1]) < 1e-12 and normal[0] < 0):
        normal, rho = -normal, -rho
    return normal, rho


def _angle_deg(first, second):
    return math.degrees(math.acos(float(np.clip(float(first @ second), -1.0, 1.0))))


def _fit_line_normal(positions, config):
    """Robust 2D line (normal, rho) via the existing _fit_line, canonical sign."""
    fit = _fit_line(positions, config)
    if fit is None:
        return None
    a = np.asarray(fit["a_xy"], dtype=float)
    b = np.asarray(fit["b_xy"], dtype=float)
    direction = b - a
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return None
    direction /= length
    normal = np.array((-direction[1], direction[0]))
    rho = float(normal @ a)
    if normal[1] < 0 or (abs(normal[1]) < 1e-12 and normal[0] < 0):
        normal, rho = -normal, -rho
    return normal, rho


def _projection_intervals(projections, gap_m):
    projections = np.sort(np.asarray(projections, dtype=float))
    if not len(projections):
        return []
    intervals = []
    start = prev = projections[0]
    for value in projections[1:]:
        if value - prev > gap_m:
            intervals.append([float(start), float(prev)])
            start = value
        prev = value
    intervals.append([float(start), float(prev)])
    return intervals


def split_primitives(primitives, config):
    """Split segment primitives into per-modality observations with raw points."""
    minimum = config["boundary"]["line_min_points"]
    observations = []
    rejected = []
    for primitive in primitives:
        evidence = primitive["evidence_type"]
        breaks = np.asarray(primitive["profile_break_positions"], dtype=float)
        faces = np.asarray(primitive["face_positions"], dtype=float)
        has_profile = "OBSERVED_PROFILE_BREAK" in evidence and len(breaks) >= minimum
        has_face = "OBSERVED_3D_FACE" in evidence and len(faces) >= minimum
        if has_profile:
            observations.append(dict(
                node_id=primitive["node_id"], segment_id=primitive["segment_id"],
                reference_family_id=primitive["reference_family_id"],
                primitive_type="OBSERVED_PROFILE_BREAK", positions=breaks))
        if has_face:
            observations.append(dict(
                node_id=primitive["node_id"], segment_id=primitive["segment_id"],
                reference_family_id=primitive["reference_family_id"],
                primitive_type="OBSERVED_3D_FACE", positions=faces))
        if not (has_profile or has_face):
            rejected.append(dict(
                node_id=primitive["node_id"], segment_id=primitive["segment_id"],
                evidence_type=evidence, reject_reason="INSUFFICIENT_LINE_POINTS"))
    return observations, rejected


def fuse_node_lines(observations, config):
    """Fuse one node's observations into Structural Lines."""
    b = config["boundary"]
    merge_angle = b["merge_angle_deg"]
    merge_distance = b["merge_distance_m"]
    families = []
    rejected = []
    for observation in observations:
        fit = _fit_line_normal(observation["positions"], config)
        if fit is None:
            rejected.append(dict(
                node_id=observation["node_id"], segment_id=observation["segment_id"],
                primitive_type=observation["primitive_type"],
                reject_reason="NO_STABLE_LINE_FIT"))
            continue
        normal, rho = fit
        placed = False
        for family in families:
            if (_angle_deg(normal, family["normal"]) <= merge_angle and
                    abs(rho - family["rho"]) <= merge_distance):
                family["observations"].append(observation)
                placed = True
                break
        if not placed:
            families.append(dict(normal=normal, rho=rho, observations=[observation]))
    lines = []
    for family in families:
        positions = np.vstack([obs["positions"] for obs in family["observations"]])
        refit = _fit_line(positions, config)
        if refit is None:
            continue
        normal, rho = _fit_line_normal(positions, config)
        direction = np.array((-normal[1], normal[0]))
        projections = positions @ direction
        intervals = _projection_intervals(projections, b["max_support_gap_m"])
        span = float(projections.max() - projections.min()) if len(projections) else 0.0
        observed_length = sum(end - start for start, end in intervals)
        coverage = observed_length / span if span > 0 else 0.0
        evidence_types = sorted({obs["primitive_type"] for obs in family["observations"]})
        source_segments = sorted({obs["segment_id"] for obs in family["observations"]})
        source_families = sorted({obs["reference_family_id"] for obs in family["observations"]
                                  if obs["reference_family_id"]})
        profile_support = sum(len(obs["positions"]) for obs in family["observations"]
                              if obs["primitive_type"] == "OBSERVED_PROFILE_BREAK")
        face_support = sum(len(obs["positions"]) for obs in family["observations"]
                           if obs["primitive_type"] == "OBSERVED_3D_FACE")
        multi_modal = len(evidence_types) >= 2
        status = "OBSERVED_MULTI_MODAL_LINE" if multi_modal else "OBSERVED_SINGLE_MODAL_LINE"
        lines.append(dict(
            line_id=None, node_id=None, normal=normal.tolist(), rho=rho,
            direction=direction.tolist(), evidence_types=evidence_types,
            source_segment_ids=source_segments, source_family_ids=source_families,
            profile_support_count=profile_support, face_support_count=face_support,
            top_strip_support=0, observed_support_length=observed_length,
            span_length=span, coverage=coverage,
            observed_intervals=intervals, observed_interval_count=len(intervals),
            fit_residual_p50=refit["residual_p50"], fit_residual_p95=refit["residual_p95"],
            visibility="VISIBLE", status=status, member_count=len(family["observations"]),
        ))
    return lines, rejected


def fuse_lines(observations, config):
    """Fuse per Height node, then assign global line ids."""
    by_node = {}
    for observation in observations:
        by_node.setdefault(observation["node_id"], []).append(observation)
    lines = []
    rejected = []
    for node_id in sorted(by_node):
        node_lines, node_rejected = fuse_node_lines(by_node[node_id], config)
        for line in node_lines:
            line["node_id"] = node_id
            lines.append(line)
        rejected.extend(node_rejected)
    lines.sort(key=lambda line: (line["node_id"], -line["observed_support_length"]))
    for index, line in enumerate(lines):
        line["line_id"] = "L%03d" % index
    return lines, rejected


def corroborate_strips(lines, narrow_strips):
    """Mark a line as top-strip corroborated when it shares segments with a strip."""
    strip_by_segment = {}
    for strip in narrow_strips:
        for segment_id in strip["member_segment_ids"]:
            strip_by_segment.setdefault(segment_id, set()).add(strip["family_id"])
    corroborated_strip_ids = set()
    for line in lines:
        families = set()
        for segment_id in line["source_segment_ids"]:
            families |= strip_by_segment.get(segment_id, set())
        if families:
            line["top_strip_support"] = len(families)
            line["top_strip_corroborated"] = True
            line["evidence_types"] = sorted(set(line["evidence_types"]) | {"TOP_STRIP_SUPPORT"})
            corroborated_strip_ids |= families
        else:
            line["top_strip_corroborated"] = False
    band_only = [strip for strip in narrow_strips if strip["family_id"] not in corroborated_strip_ids]
    return lines, band_only


def _line_color(line):
    if line["status"] == "OBSERVED_MULTI_MODAL_LINE":
        return MULTI_COLOR
    if "OBSERVED_PROFILE_BREAK" in line["evidence_types"]:
        return PROFILE_COLOR
    if "OBSERVED_3D_FACE" in line["evidence_types"]:
        return FACE_COLOR
    return LABEL_COLOR


def render_lines_png(path, grid, valid, lines, config):
    scale = 8
    overlay = _surface_image(grid, valid, scale)
    for line in lines:
        normal = np.asarray(line["normal"], dtype=float)
        direction = np.array((-normal[1], normal[0]))
        color = _line_color(line)
        for start, end in line["observed_intervals"]:
            a = line["rho"] * normal + start * direction
            b = line["rho"] * normal + end * direction
            _draw_line(overlay, _pixel(grid, a, scale), _pixel(grid, b, scale), color, width=3)
        middle = line["rho"] * normal + line["observed_intervals"][0][0] * direction
        px, py = _pixel(grid, middle, scale)
        _draw_text(overlay, line["line_id"], int(px), int(py), LABEL_COLOR, scale=2)
    _save_png(path, overlay)
    return path


def render_lines_ply(path, lines, points, config):
    z_center = float(np.median(points[:, 2])) if len(points) else 0.0
    spacing = config["geometry"]["coarse_voxel_m"] / 2
    xyz, colors = [], []
    for line in lines:
        normal = np.asarray(line["normal"], dtype=float)
        direction = np.array((-normal[1], normal[0]))
        color = np.asarray(_line_color(line), dtype=np.uint8)
        for start, end in line["observed_intervals"]:
            count = max(2, int(np.ceil((end - start) / spacing)) + 1)
            samples = np.linspace(start, end, count)
            points_xy = line["rho"] * normal + samples[:, None] * direction
            xyz.append(np.column_stack((points_xy, np.full(len(samples), z_center))))
            colors.append(np.tile(color, (len(samples), 1)))
    if not xyz:
        write_colored_ply(path, np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8))
        return path
    write_colored_ply(path, np.vstack(xyz), np.vstack(colors))
    return path


def run_scene_s4r2(points, config, research, baseline_result):
    analysis = analyze_scene_s4r1(points, config, research, baseline_result)
    observations, rejected_primitives = split_primitives(analysis["primitives"], config)
    lines, rejected_observations = fuse_lines(observations, config)
    lines, band_only = corroborate_strips(lines, analysis["narrow_strips"])
    node_evidence = {}
    for line in lines:
        node_id = line["node_id"]
        summary = node_evidence.setdefault(node_id, dict(
            observed_line_count=0, multi_segment_line_count=0, multi_modal_line_count=0,
            top_strip_corroborated_count=0, observed_support_length=0.0))
        summary["observed_line_count"] += 1
        if len(line["source_segment_ids"]) >= 2:
            summary["multi_segment_line_count"] += 1
        if len([t for t in line["evidence_types"]
                if t in ("OBSERVED_PROFILE_BREAK", "OBSERVED_3D_FACE")]) >= 2:
            summary["multi_modal_line_count"] += 1
        if line["top_strip_corroborated"]:
            summary["top_strip_corroborated_count"] += 1
        summary["observed_support_length"] += line["observed_support_length"]
    return dict(
        primitives=analysis["primitives"], observations=observations, lines=lines,
        structure_bands=band_only, node_evidence=node_evidence,
        rejected_primitives=rejected_primitives, rejected_observations=rejected_observations,
        grid=analysis["grid"], valid=analysis["valid"],
    )


def run_batch_s4r2(run_id, baseline_root, manifest_path, data_root, output_root):
    if not run_id or run_id in (".", "..") or any(c in run_id for c in "/\\:"):
        raise ValueError("INVALID_RUN_ID")
    destination = Path(output_root) / run_id
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("S4R2_RUN_ALREADY_EXISTS:" + str(destination))
    config, _ = resolve_config(DEFAULT_CONFIG)
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    research_sha = hashlib.sha256(
        json.dumps(research, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scans = {scan["scan_id"]: scan for scan in manifest["scans"]}
    summary_rows = []
    for scene, scan_id in NORMAL6.items():
        scan = scans[scan_id]
        if scan["split_role"] != "DEVELOPMENT":
            raise ValueError("NON_DEVELOPMENT_INPUT:" + scene)
        pcd = Path(data_root) / scan["pcd_path"]
        input_sha = hashlib.sha256(pcd.read_bytes()).hexdigest()
        if input_sha != scan["pcd_sha256"]:
            raise ValueError("INPUT_SHA_MISMATCH:" + scene)
        frozen = json.loads(
            (Path(baseline_root) / scene / "hatch_regions.json").read_text(encoding="utf-8"))
        if input_sha != frozen["input_sha256"]:
            raise ValueError("BASELINE_INPUT_MISMATCH:" + scene)
        if (config_hash(config) != frozen["v15_config_hash"]
                or research_sha != frozen["research_config_sha256"]):
            raise ValueError("BASELINE_CONFIG_MISMATCH:" + scene)
        points, _ = decode(pcd)
        result = run_scene_s4r2(points, config, research, frozen)
        directory = destination / scene
        directory.mkdir(parents=True, exist_ok=False)
        _atomic_json(directory / "structural_primitives.json", dict(
            schema="ship_perception.v15r.s4r2_primitives.1", scene_id=scene,
            primitives=result["primitives"]))
        _atomic_json(directory / "rejected_evidence.json", dict(
            schema="ship_perception.v15r.s4r2_rejected.1", scene_id=scene,
            rejected_primitives=result["rejected_primitives"],
            rejected_observations=result["rejected_observations"],
            semantics="GATE_B_TRACEABILITY_NO_SILENT_DROP"))
        _atomic_json(directory / "structural_line_families.json", dict(
            schema="ship_perception.v15r.s4r2_line_families.1", scene_id=scene,
            lines=result["lines"],
            semantics="2D_NORMAL_FORM; OCCLUSION_GAPS_ARE_UNOBSERVED"))
        _atomic_json(directory / "structure_bands.json", dict(
            schema="ship_perception.v15r.s4r2_structure_bands.1", scene_id=scene,
            structure_bands=result["structure_bands"],
            semantics="STRUCTURE_BAND_ONLY_NOT_HATCH_EDGE"))
        _atomic_json(directory / "node_line_evidence.json", dict(
            schema="ship_perception.v15r.s4r2_node_evidence.1", scene_id=scene,
            nodes=result["node_evidence"]))
        render_lines_ply(directory / "structural_lines.ply", result["lines"], points, config)
        render_lines_png(directory / "structural_lines.png", result["grid"],
                         result["valid"], result["lines"], config)
        for node_id, node_lines in _lines_by_node(result["lines"]).items():
            render_lines_png(directory / ("node_%s_structural_lines.png" % node_id),
                             result["grid"], result["valid"], node_lines, config)
        for line in result["lines"]:
            summary_rows.append(dict(scene_id=scene, **{
                key: value for key, value in line.items() if key not in ("observed_intervals",)}))
        print("%s: %d primitives, %d lines" % (scene, len(result["primitives"]),
                                                len(result["lines"])), flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene_id", "line_id", "node_id", "status", "evidence_types",
                  "source_segment_ids", "source_family_ids", "profile_support_count",
                  "face_support_count", "top_strip_support", "observed_support_length",
                  "span_length", "coverage", "observed_interval_count", "fit_residual_p50",
                  "fit_residual_p95", "member_count"]
    _write_csv(destination / "normal6_s4r2_summary.csv", summary_rows, fieldnames)
    _atomic_json(destination / "batch_summary.json", dict(
        schema="ship_perception.v15r.s4r2_batch.1", run_id=run_id,
        software_git_sha=_git_sha(), scenes={scene: dict() for scene in NORMAL6}))
    return summary_rows


def _lines_by_node(lines):
    grouped = {}
    for line in lines:
        grouped.setdefault(line["node_id"], []).append(line)
    return grouped


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-root", type=Path,
                        default=REPO / "Ship-Unloader-Work/r1_heightmap/baseline_004_heightmap")
    parser.add_argument("--dataset-manifest", type=Path,
                        default=REPO / "ship_perception/datasets/v15_dataset_manifest.json")
    parser.add_argument("--data-root", type=Path,
                        default=REPO / "Ship-Unloader-Data/legacy/hold_detector")
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "Ship-Unloader-Work/r1_structural_lines")
    args = parser.parse_args()
    run_batch_s4r2(args.run_id, args.baseline_root, args.dataset_manifest,
                   args.data_root, args.output_root)


if __name__ == "__main__":
    main()
