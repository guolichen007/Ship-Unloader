"""Offline R2H-B2 per-vessel proposal review on static PCD/PLY scans."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .hatch_proposal import discover_scene_hatches, propose_for_vessel
from .heightmap_batch import RESEARCH_CONFIG
from .rectangle_refinement_review import _read_validation_xyz
from .run import DEFAULT_CONFIG, _atomic_json, resolve_config
from .scene_vessel import infer_scene_vessels


def _old_associations(result, old):
    def overlap_metrics(first, second):
        intersection = (max(0, min(first[2], second[2])-max(first[0], second[0])) *
                        max(0, min(first[3], second[3])-max(first[1], second[1])))
        first_area = max(0, first[2]-first[0]) * max(0, first[3]-first[1])
        second_area = max(0, second[2]-second[0]) * max(0, second[3]-second[1])
        return (float(intersection / max(first_area + second_area - intersection, 1e-9)),
                float(intersection / max(first_area, 1e-9)))

    matches = []
    for rectangle in old.get("rectangles", []):
        polygon = np.asarray(rectangle.get("polygon_after", []), dtype=float)
        if polygon.shape != (4, 2):
            continue
        choices = []
        for per_vessel in result["per_vessel"]:
            vessel = next(row for row in result["scene"]["vessel_hypotheses"]
                          if row["vessel_hypothesis_id"] == per_vessel["vessel_hypothesis_id"])
            axial = polygon @ np.asarray(vessel["local_axes"]).T
            bounds = [float(axial[:, 0].min()), float(axial[:, 1].min()),
                      float(axial[:, 0].max()), float(axial[:, 1].max())]
            for candidate in per_vessel["hatch_hypotheses"]:
                if candidate["bounds_axial"] is not None:
                    iou, old_coverage = overlap_metrics(bounds, candidate["bounds_axial"])
                    choices.append((iou, old_coverage, candidate))
        if choices:
            score, old_coverage, match = max(choices, key=lambda row: row[0])
            matches.append(dict(old_hatch_id=rectangle["hatch_id"],
                                new_hatch_hypothesis_id=match["hatch_hypothesis_id"],
                                new_state=match["state"],
                                intersection_over_union=score,
                                old_rectangle_coverage=old_coverage))
    return matches


def review_file(scene_id, path, config, research, prior_root=None,
                force_height_miss=False):
    path = Path(path)
    points = _read_validation_xyz(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = discover_scene_hatches(points, config, research)
    result.update(scene_id=scene_id, input_file=path.name, input_sha256=digest,
                  raw_point_count=int(len(points)))
    if prior_root is not None:
        old_path = Path(prior_root) / scene_id / "rectangle_refinement.json"
        if old_path.exists():
            old = json.loads(old_path.read_text(encoding="utf-8"))
            if old.get("input_pcd_sha256") == digest:
                result["prior_r2f_r2g"] = dict(
                    source=str(old_path), input_sha256_verified=True,
                    r2f_count=old["rectangle_count_before"],
                    r2g_count=old["rectangle_count_after"],
                    associations=_old_associations(result, old))
    if force_height_miss:
        scene, auxiliary = infer_scene_vessels(points, config, research)
        forced = []
        for vessel in scene["vessel_hypotheses"]:
            if vessel["classification_status"] != "VESSEL_HYPOTHESIS":
                continue
            indexes = auxiliary["vessel_point_indexes"][vessel["vessel_hypothesis_id"]]
            run = propose_for_vessel(points[indexes], vessel, config, research,
                                     height_override=[])
            forced.append(dict(vessel_hypothesis_id=vessel["vessel_hypothesis_id"],
                               provider_summary=run["provider_summary"],
                               state_counts=dict(Counter(row["state"] for row in
                                                         run["hatch_hypotheses"]))))
        result["height_provider_forced_miss_review"] = forced
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", required=True,
                        help="SCENE_ID=absolute-or-relative-PCD/PLY-path")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prior-r2g-root", type=Path)
    parser.add_argument("--force-height-miss-scene", action="append", default=[])
    args = parser.parse_args()
    config, _ = resolve_config(DEFAULT_CONFIG)
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    summaries = {}
    for spec in args.scene:
        scene_id, path = spec.split("=", 1)
        if not scene_id or any(char in scene_id for char in "/\\:"):
            raise ValueError("INVALID_SCENE_ID")
        result = review_file(scene_id, path, config, research, args.prior_r2g_root,
                             scene_id in args.force_height_miss_scene)
        directory = args.output_root / scene_id
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(directory / "hatch_proposals.json", result)
        counts = Counter(row["state"] for run in result["per_vessel"]
                         for row in run["hatch_hypotheses"])
        summary = dict(
                       vessel_hypotheses=sum(
                           row["classification_status"] == "VESSEL_HYPOTHESIS"
                           for row in result["scene"]["vessel_hypotheses"]),
                       unresolved_scene_hypotheses=sum(
                           row["classification_status"] == "UNRESOLVED"
                           for row in result["scene"]["vessel_hypotheses"]),
                       provider_context_count=len(result["per_vessel"]),
                       state_counts=dict(counts),
                       provider_counts={run["vessel_hypothesis_id"]: run["provider_summary"]
                                        for run in result["per_vessel"]},
                       prior_r2f_r2g=result.get("prior_r2f_r2g"),
                       height_provider_forced_miss_review=result.get(
                           "height_provider_forced_miss_review"))
        summaries[scene_id] = summary
        print(scene_id, summary["vessel_hypotheses"], dict(counts), flush=True)
    _atomic_json(args.output_root / "summary.json", summaries)


if __name__ == "__main__":
    main()
