"""Post-freeze P1 evaluator: weak-label Oracle boundary-band probe and the
forensic decision table. This module is never imported by the detector or the
structural audit path, and annotation is opened only after the detector-side
P1 outputs have been SHA-frozen."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ship_perception.tools.v15_pcd import decode

from .batch import NORMAL6
from .boundary_refinement import _face_normal_supported, _fit_line, _profile
from .model import PerimeterSegment
from .perimeter_deck import estimate_segment_deck
from .run import REPO, _atomic_json

ORACLE_OFFSETS_M = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)

PIPELINE_ORDER = (
    "DECK_REFERENCE_AMBIGUOUS",
    "PROFILE_NOT_EXECUTED",
    "PROFILE_MODEL_INSUFFICIENT",
    "FACE_MODEL_INSUFFICIENT",
    "LINE_AGGREGATION_FAILURE",
)


def _orientation_difference_deg(first_deg, second_deg):
    difference = abs(first_deg - second_deg) % 180.0
    return float(min(difference, 180.0 - difference))


def _line_orientation_deg(line):
    direction = np.asarray(line["b_xy"], dtype=float) - np.asarray(line["a_xy"], dtype=float)
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return None
    return float(math.degrees(math.atan2(direction[1], direction[0])))


def oracle_band_probe(points, annotation, config, tree):
    """Sweep each weak-box edge along its normal and run the existing chain."""
    edges = []
    for annotation_id, box in enumerate(annotation["annotations"]):
        corners = [np.asarray(point, dtype=float) for point in box["corners"]]
        if len(corners) < 4:
            continue
        center = np.mean(corners, axis=0)
        for index in range(4):
            start = corners[index]
            end = corners[(index + 1) % 4]
            direction = end - start
            length = float(np.linalg.norm(direction))
            if length < 1e-9:
                continue
            tangent = direction / length
            inward = np.array((-tangent[1], tangent[0]))
            if inward @ (center - (start + end) / 2) < 0:
                inward = -inward
            edge_id = "w%02d_e%d" % (annotation_id, index)
            weak_orientation = float(math.degrees(math.atan2(tangent[1], tangent[0])))
            probes = _probe_edge(points, config, tree, start, end, tangent, inward,
                                 length, edge_id, weak_orientation)
            best = _best_probe(probes)
            edges.append(dict(
                oracle_edge_id=edge_id,
                annotation_index=annotation_id,
                weak_edge_orientation_deg=weak_orientation,
                best_observed_offset_m=best["offset_m"],
                observed_line_orientation_deg=best["orientation_deg"],
                orientation_difference_deg=best["orientation_difference_deg"],
                probes=probes,
            ))
    return dict(
        schema="ship_perception.v15r.p1_oracle_band.1",
        semantics="WEAK_LABEL_ORACLE_DIAGNOSTIC_ONLY",
        caveat="PARTIAL_SINGLE_HATCH_LABEL_IS_NOT_GOLDEN; SWEEP_VALUES_ARE_FORENSIC_ONLY",
        offset_values_m=list(ORACLE_OFFSETS_M),
        boundary_search_m=config["roi"]["boundary_search_m"],
        edges=edges,
    )


def _probe_edge(points, config, tree, start, end, tangent, inward, length,
                edge_id, weak_orientation):
    boundary = config["boundary"]
    spacing = max(boundary["profile_step_m"],
                  min(boundary["max_support_gap_m"] * 0.8,
                      config["geometry"]["coarse_voxel_m"] / 2))
    probes = []
    for offset in ORACLE_OFFSETS_M:
        shift = offset * inward
        shifted_start = start + shift
        shifted_end = end + shift
        segment = PerimeterSegment(
            "%s-o%+.1f" % (edge_id, offset), edge_id,
            tuple(shifted_start), tuple(shifted_end),
            tuple(tangent), tuple(-inward), float(np.linalg.norm(shifted_end - shifted_start)),
            "FULLY_OBSERVED")
        deck, plane, forensic = estimate_segment_deck(points, segment, config, tree)
        probe = dict(
            offset_m=float(offset),
            deck_status=deck["status"],
            deck_reason=deck["reason"],
            deck_candidate_count=deck["plane_candidate_count"],
            profile_executed=False,
            valid_profile_sections=0,
            profile_break_count=0,
            face_candidate_count=0,
            face_valid_count=0,
            line_fit_status=None,
            observed_support_length=0.0,
            coverage=0.0,
            residual_p95=None,
            raw_support_count=int(len(forensic["strip_raw_ids"])),
            raw_support_ids_stored_in="oracle_ownership.npz",
            orientation_difference_deg=None,
        )
        if deck["status"] == "RESOLVED" and plane is not None:
            probe["profile_executed"] = True
            positions = np.arange(spacing / 2, length, spacing)
            rows = [_profile(points, tree, shifted_start + tangent * position, tangent,
                             inward, plane, config, "%s-p%d" % (segment.segment_id, index))
                    for index, position in enumerate(positions)]
            breaks = np.asarray([row["break_position"] for row in rows if row["valid"]],
                                dtype=float).reshape(-1, 2)
            faces = np.asarray([row["face_position"] for row in rows
                                if row["face_position"] is not None], dtype=float).reshape(-1, 2)
            probe["valid_profile_sections"] = int(sum(1 for row in rows if row["valid"]))
            probe["profile_break_count"] = int(len(breaks))
            probe["face_candidate_count"] = int(len(faces))
            break_line = _fit_line(breaks, config) if len(breaks) else None
            face_line = _fit_line(faces, config) if len(faces) else None
            if face_line is not None:
                if _face_normal_supported(points, face_line, plane, config):
                    probe["face_valid_count"] = 1
                else:
                    face_line = None
            chosen = None
            if break_line is not None and face_line is not None:
                separation = float(np.linalg.norm(
                    (break_line["a_xy"] + break_line["b_xy"] -
                     face_line["a_xy"] - face_line["b_xy"]) / 2))
                if separation <= max(boundary["line_inlier_m"], config["roi"]["support_band_m"]):
                    chosen = _fit_line(np.vstack((breaks, faces)), config) or break_line
                    probe["line_fit_status"] = "PROFILE_BREAK+OBSERVED_3D_FACE"
                else:
                    chosen = break_line
                    probe["line_fit_status"] = "OBSERVED_PROFILE_BREAK"
            elif break_line is not None:
                chosen = break_line
                probe["line_fit_status"] = "OBSERVED_PROFILE_BREAK"
            elif face_line is not None:
                chosen = face_line
                probe["line_fit_status"] = "OBSERVED_3D_FACE"
            if chosen is not None:
                probe["observed_support_length"] = float(chosen["support_length"])
                probe["coverage"] = float(min(1.0, chosen["support_length"] / length))
                probe["residual_p95"] = float(chosen["residual_p95"])
                line_orientation = _line_orientation_deg(chosen)
                probe["line_orientation_deg"] = line_orientation
                if line_orientation is not None:
                    probe["orientation_difference_deg"] = _orientation_difference_deg(
                        weak_orientation, line_orientation)
        probes.append(probe)
    return probes


def _best_probe(probes):
    best = dict(offset_m=None, orientation_deg=None, orientation_difference_deg=None)
    candidates = [row for row in probes if row["line_fit_status"] is not None]
    if not candidates:
        return best
    chosen = max(candidates, key=lambda row: row["observed_support_length"])
    best["offset_m"] = chosen["offset_m"]
    best["orientation_deg"] = chosen.get("line_orientation_deg")
    best["orientation_difference_deg"] = chosen.get("orientation_difference_deg")
    return best


def segment_first_bad_stage(lineage, ambiguity_lookup):
    """FIRST_BAD_STAGE for one perimeter segment in pipeline order."""
    if lineage["deck_status"] == "SEGMENT_DECK_AMBIGUOUS":
        return "DECK_REFERENCE_AMBIGUOUS"
    if lineage["deck_status"] != "RESOLVED" or not lineage["profile_executed"]:
        return "PROFILE_NOT_EXECUTED"
    if lineage["line_fit_result"] is not None:
        return None
    if lineage["break_candidate_count"] > 0:
        return "LINE_AGGREGATION_FAILURE"
    if lineage["face_option_count"] > 0:
        return "FACE_MODEL_INSUFFICIENT"
    return "PROFILE_MODEL_INSUFFICIENT"


def _pair_geometric_coaming(pair):
    """The geometric coaming signature: narrow elevated plane adjacent to the
    boundary, independent of whether the vertical wall returns were observed."""
    if pair["classification"] == "DECK_COAMING_LIKE_PAIR":
        return True
    return pair["classification_reason"] == "NO_VERTICAL_CONNECTOR_RETURNS_BETWEEN_PLANES"


def _deck_coaming_hypothesis(ambiguity_records):
    """SUPPORTED / REFUTED / INCONCLUSIVE for one scene's ambiguous segments.

    ``geometric`` counts segments whose strip contains the broad-low +
    narrow-elevated-adjacent pair pattern; ``strict`` additionally requires the
    vertical connector returns. The vertical coaming wall is frequently not
    directly observed, so the geometric fraction is the primary signal.
    """
    if not ambiguity_records:
        return dict(status="REFUTED", reason="NO_AMBIGUOUS_SEGMENTS",
                    ambiguous_segment_count=0, geometric_coaming_like_fraction=None,
                    strict_coaming_like_fraction=None)
    heights = []
    geometric = 0
    strict = 0
    for record in ambiguity_records:
        if any(_pair_geometric_coaming(pair) for pair in record["plane_pairs"]):
            geometric += 1
        strict_pairs = [pair for pair in record["plane_pairs"]
                        if pair["classification"] == "DECK_COAMING_LIKE_PAIR"]
        if strict_pairs:
            strict += 1
            heights.extend(pair["height_difference_m"] for pair in strict_pairs)
    geometric_fraction = geometric / len(ambiguity_records)
    strict_fraction = strict / len(ambiguity_records)
    median_height = float(np.median(heights)) if heights else None
    if geometric_fraction >= 0.5 and median_height is not None and 0.2 <= median_height <= 1.5:
        status = "SUPPORTED"
    elif geometric_fraction < 0.2:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    return dict(status=status, ambiguous_segment_count=len(ambiguity_records),
                geometric_coaming_like_segment_count=geometric,
                geometric_coaming_like_fraction=float(geometric_fraction),
                strict_coaming_like_segment_count=strict,
                strict_coaming_like_fraction=float(strict_fraction),
                coaming_like_median_height_difference_m=median_height)


def _next_decision(deck_coaming, oracle, selected_lineage, contour_audit):
    """Map P1 evidence to the next algorithm decision using the frozen table."""
    oracle_has_line = any(
        any(probe["line_fit_status"] is not None for probe in edge["probes"])
        for edge in (oracle["edges"] if oracle else [])
    )
    selected_has_line = any(row["line_fit_result"] is not None for row in selected_lineage)
    geometric_fraction = deck_coaming.get("geometric_coaming_like_fraction") or 0.0
    ambiguous_present = deck_coaming["ambiguous_segment_count"] > 0
    profile_executed_failed = any(
        row["profile_executed"] and row["line_fit_result"] is None for row in selected_lineage)
    if ambiguous_present and geometric_fraction >= 0.5:
        return "MULTI_PLANE_STRUCTURAL_INTERPRETATION"
    if oracle_has_line and not selected_has_line:
        return "BOUNDARY_MEASUREMENT_PLACEMENT_FIX"
    if ambiguous_present:
        return "MULTI_PLANE_STRUCTURAL_INTERPRETATION"
    if profile_executed_failed:
        return "GENERALIZED_PROFILE"
    if not oracle_has_line and not selected_has_line:
        return "STATIC_PLY_BOUNDARY_UNOBSERVABLE"
    return "INCONCLUSIVE"


def evaluate_run_p1(run_root, manifest_path, data_root):
    run_root = Path(run_root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scans = {scan["scan_id"]: scan for scan in manifest["scans"]}
    frozen = {}
    for scene in NORMAL6:
        directory = run_root / scene
        names = ("p1_ambiguity_forensics.json", "p1_segment_lineage.json",
                 "p1_contour_audit.json", "ambiguity_ownership.npz")
        if not all((directory / name).is_file() for name in names):
            raise ValueError("P1_AUDIT_OUTPUT_NOT_COMPLETE:" + scene)
        frozen[scene] = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in names
        }
    _atomic_json(run_root / "frozen_audit_outputs.json",
                 dict(schema="ship_perception.v15r.frozen_audit.1", files=frozen))
    decisions = []
    for scene in NORMAL6:
        directory = run_root / scene
        scan = scans[NORMAL6[scene]]
        scan_id = scan["scan_id"]
        ambiguity = json.loads((directory / "p1_ambiguity_forensics.json").read_text(encoding="utf-8"))
        lineage = json.loads((directory / "p1_segment_lineage.json").read_text(encoding="utf-8"))
        contour = json.loads((directory / "p1_contour_audit.json").read_text(encoding="utf-8"))
        ambiguity_records = ambiguity["segments"]
        lineage_records = lineage["segments"]
        selected_lineage = [row for row in lineage_records if row.get("baseline_selected")]
        deck_coaming = _deck_coaming_hypothesis(ambiguity_records)
        oracle = None
        if scan["annotation_scope"] != "UNLABELED":
            points, _ = decode(Path(data_root) / scan["pcd_path"])
            tree = cKDTree(points[:, :2])
            config = json.loads((REPO / "ship_perception/config/v15.json").read_text(encoding="utf-8"))
            canonical = []
            for name, digest in zip(scan["annotation_files"], scan["annotation_sha256"]):
                path = Path(data_root) / name
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("WEAK_ANNOTATION_SHA_MISMATCH:" + scene)
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("cloud_file") == scan["pcd_filename"]:
                    canonical.append(data)
            if len(canonical) != 1:
                raise ValueError("CANONICAL_WEAK_ANNOTATION_NOT_UNIQUE:" + scene)
            oracle = oracle_band_probe(points, canonical[0], config, tree)
            oracle.update(scene_id=scene, scan_id=scan_id, annotation_file=canonical[0])
            _atomic_json(directory / "p1_oracle_band.json", oracle)
        bad_stages = [segment_first_bad_stage(row, {r["segment_id"]: r for r in ambiguity_records})
                      for row in selected_lineage]
        bad_stages = [stage for stage in bad_stages if stage]
        first_bad = next((stage for stage in PIPELINE_ORDER if stage in bad_stages), None)
        decision = dict(
            scene_id=scene,
            deck_coaming_hypothesis=deck_coaming,
            oracle_has_line=bool(oracle and any(
                any(probe["line_fit_status"] is not None for probe in edge["probes"])
                for edge in oracle["edges"])),
            selected_segment_count=len(selected_lineage),
            selected_segment_bad_stage_counts={
                stage: bad_stages.count(stage) for stage in PIPELINE_ORDER},
            first_bad_stage=first_bad or "NO_SELECTED_SEGMENT",
            next_algorithm_decision=_next_decision(
                deck_coaming, oracle, selected_lineage, contour),
        )
        decisions.append(decision)
    _atomic_json(run_root / "decision_matrix_p1.json", dict(
        schema="ship_perception.v15r.p1_decision.1",
        evaluator_git_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip(),
        evidence_only=True,
        decisions=decisions,
    ))
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path,
                        default=REPO / "ship_perception/datasets/v15_dataset_manifest.json")
    parser.add_argument("--data-root", type=Path,
                        default=REPO / "Ship-Unloader-Data/legacy/hold_detector")
    args = parser.parse_args()
    for row in evaluate_run_p1(args.run_root, args.dataset_manifest, args.data_root):
        print("%s: %s -> %s" % (row["scene_id"], row["first_bad_stage"],
                                row["next_algorithm_decision"]))


if __name__ == "__main__":
    main()
