"""Run V1.5-R1-S0 observed-only static calibration on one strict binary PCD."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from jsonschema import Draft202012Validator
from scipy.spatial import cKDTree

from ship_perception.tools.v15_config_identity import config_hash
from ship_perception.tools.v15_pcd import decode

from .boundary_refinement import refine_boundaries
from .hypothesis_solver import solve
from .local_deck import estimate_local_deck
from .local_opening import find_openings
from .structural_proposal import propose
from .visualization import write_debug_clouds


SOURCE = Path(__file__).resolve().parents[1]
REPO = SOURCE.parent
DEFAULT_CONFIG = SOURCE / "config/v15.json"
SCHEMA = SOURCE / "config/v15.schema.json"


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _apply_override(config, key, value):
    parts = key.split(".")
    if len(parts) != 2 or parts[0] not in ("geometry", "frame", "roi", "boundary"):
        raise ValueError("R1_OVERRIDE_FORBIDDEN:" + key)
    if parts[1] not in config[parts[0]]:
        raise ValueError("R1_OVERRIDE_UNKNOWN:" + key)
    if type(value) is not type(config[parts[0]][parts[1]]):
        raise ValueError("R1_OVERRIDE_TYPE:" + key)
    config[parts[0]][parts[1]] = value


def resolve_config(path=DEFAULT_CONFIG, overrides=(), override_file=None):
    config = copy.deepcopy(json.loads(Path(path).read_text(encoding="utf-8")))
    if override_file is not None:
        target = Path(override_file).resolve()
        work = (REPO / "Ship-Unloader-Work").resolve()
        if work not in target.parents:
            raise ValueError("R1_OVERRIDE_FILE_OUTSIDE_WORK")
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("R1_OVERRIDE_FILE_OBJECT_REQUIRED")
        for key, value in data.items():
            _apply_override(config, key, value)
    for entry in overrides:
        if "=" not in entry:
            raise ValueError("R1_OVERRIDE_KEY_VALUE_REQUIRED")
        key, raw = entry.split("=", 1)
        _apply_override(config, key, json.loads(raw))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(config)
    geometry = config["geometry"]
    if not 0 < geometry["refine_voxel_m"] <= geometry["candidate_voxel_m"] <= geometry["coarse_voxel_m"]:
        raise ValueError("R1_RESOLUTION_ORDER")
    for section in ("geometry", "frame", "roi", "boundary"):
        for key, value in config[section].items():
            if key.startswith("grid_phase_") or key.endswith("_weight") or key in ("max_inferred_edges", "profile_lower_quantile"):
                continue
            if value <= 0:
                raise ValueError("R1_NONPOSITIVE:" + section + "." + key)
    return config, config_hash(config)


def _git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()


def analyze_points(points, config, *, software_git_sha="UNCOMMITTED", input_sha256="SYNTHETIC",
                   run_id="synthetic", coordinate_frame="RAW_INPUT_FRAME"):
    """Pure geometry path; scene name and expected hatch count are absent."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("INVALID_RAW_POINTS")
    started = time.perf_counter()
    proposals, _ = propose(points, config)
    proposal_ms = (time.perf_counter() - started) * 1000
    deck_ms = boundary_ms = 0.0
    rows, deck_supports, diagnostics = [], [], []
    tree = cKDTree(points[:, :2])
    for proposal in proposals:
        t = time.perf_counter()
        deck, plane, support = estimate_local_deck(points, proposal, config)
        deck_ms += (time.perf_counter() - t) * 1000
        deck_supports.append((proposal, deck, plane))
        detail = dict(proposal_id=proposal.proposal_id, local_deck=deck, openings=[])
        if plane is not None:
            t = time.perf_counter()
            openings, _ = find_openings(points, proposal, plane, config)
            deck_ms += (time.perf_counter() - t) * 1000
            for opening in openings:
                t = time.perf_counter()
                refined = refine_boundaries(points, opening, plane, config, tree)
                boundary_ms += (time.perf_counter() - t) * 1000
                detail["openings"].append({**opening.record(), "status": refined["status"],
                                           "observed_edge_count": len(refined["boundaries"])})
                rows.append(dict(opening_id=opening.opening_id, proposal_id=proposal.proposal_id,
                                 bbox_xy=list(opening.bbox_xy), status=refined["status"],
                                 local_deck=deck, boundaries=refined["boundaries"],
                                 polygon_raw=refined["polygon_raw"], center_raw=refined["center_raw"],
                                 center_source=refined["center_source"], support_cells=opening.support_cells,
                                 area_m2=opening.area_m2, mean_drop_m=opening.mean_drop_m,
                                 profiles=refined["profiles"]))
        diagnostics.append(detail)
    t = time.perf_counter()
    decision = solve(rows, config)
    hypothesis_ms = (time.perf_counter() - t) * 1000
    selected = decision["selected"]
    hatches = []
    for index, row in enumerate(selected):
        hatch = {key: value for key, value in row.items() if key not in ("profiles", "bbox_xy")}
        hatch["hatch_id"] = "h%d" % (index + 1)
        hatch["extent"] = row["bbox_xy"]
        hatches.append(hatch)
    warnings = ["%s:%s" % (row["proposal_id"], row["local_deck"]["reason"])
                for row in diagnostics if row["local_deck"]["status"] != "RESOLVED"]
    if decision["warning"]:
        warnings.append(decision["warning"])
    limitations = ["STATIC_SINGLE_FRAME", "PYTHON_REFERENCE_IMPLEMENTATION", "OBSERVED_ONLY_NO_EDGE_COMPLETION",
                   "NO_V14_TRACKING", "HUMAN_REVIEW_NOT_COMMERCIAL_GOLDEN",
                   "SHARED_LOCAL_DECK_WITHIN_PROPOSAL"]
    timing = dict(decode_ms=0.0, proposal_ms=proposal_ms, local_deck_ms=deck_ms,
                  boundary_ms=boundary_ms, hypothesis_ms=hypothesis_ms,
                  visualization_ms=0.0, total_ms=(time.perf_counter() - started) * 1000)
    result = dict(schema_version="ship_perception.v15r.static_result.1",
                  software_git_sha=software_git_sha, input_sha256=input_sha256,
                  config_hash=config_hash(config), run_id=run_id, coordinate_frame=coordinate_frame,
                  point_count=int(len(points)), bbox_raw=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                  scene_status=decision["scene_status"], confirmed_hatch_count=len(hatches),
                  observed_only=True, proposals=[proposal.record() for proposal in proposals],
                  hypotheses=decision["hypotheses"], hatches=hatches,
                  warnings=warnings, known_limitations=limitations, timing=timing)
    auxiliary = dict(proposals=proposals, proposal_debug=diagnostics,
                     profile_debug={row["opening_id"]: row["profiles"] for row in rows},
                     decision_debug={key: value for key, value in decision.items() if key != "selected"},
                     deck_supports=deck_supports)
    return result, auxiliary


def run_file(input_path, output_root, run_id, *, scene_id=None, config_path=DEFAULT_CONFIG,
             overrides=(), override_file=None):
    input_path = Path(input_path)
    if not run_id or run_id in (".", "..") or any(c in run_id for c in "/\\:"):
        raise ValueError("INVALID_RUN_ID")
    scene_id = scene_id or input_path.stem
    if not scene_id or scene_id in (".", "..") or any(c in scene_id for c in "/\\:"):
        raise ValueError("INVALID_SCENE_ID")
    directory = Path(output_root) / run_id / scene_id
    if (directory / "result.json").exists():
        raise FileExistsError("R1_RUN_RESULT_ALREADY_EXISTS:" + str(directory))
    config, _ = resolve_config(config_path, overrides, override_file)
    t0 = time.perf_counter()
    points, _ = decode(input_path)
    decode_ms = (time.perf_counter() - t0) * 1000
    sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    result, aux = analyze_points(points, config, software_git_sha=_git_sha(), input_sha256=sha, run_id=run_id)
    result["scene_id"] = scene_id
    result["timing"]["decode_ms"] = decode_ms
    t = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    cloud = write_debug_clouds(directory, points, aux["proposals"], aux["deck_supports"], result, config)
    result["timing"]["visualization_ms"] = (time.perf_counter() - t) * 1000
    result["timing"]["total_ms"] += decode_ms + result["timing"]["visualization_ms"]
    _atomic_json(directory / "resolved_config.json", config)
    _atomic_json(directory / "result.json", result)
    _atomic_json(directory / "proposal_debug.json", dict(proposals=aux["proposal_debug"],
                                                     decision=aux["decision_debug"], cloud=cloud))
    _atomic_json(directory / "profile_debug.json", aux["profile_debug"])
    _atomic_json(directory / "timing.json", result["timing"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO / "Ship-Unloader-Work/r1_static")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scene-id")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--observed-only", action="store_true", required=False)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--override-file", type=Path)
    args = parser.parse_args()
    result = run_file(args.input, args.output_root, args.run_id, scene_id=args.scene_id,
                      config_path=args.config, overrides=args.override, override_file=args.override_file)
    print(json.dumps({"scene_status": result["scene_status"],
                      "confirmed_hatch_count": result["confirmed_hatch_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
