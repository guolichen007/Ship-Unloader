"""Run research-only heightmap region boxes on the six Development scans."""
import argparse
import hashlib
import json
from pathlib import Path

from ship_perception.tools.v15_config_identity import config_hash
from ship_perception.tools.v15_pcd import decode

from .batch import NORMAL6
from .heightmap_render import render_regions
from .heightmap_topology import detect_regions
from .run import DEFAULT_CONFIG, REPO, _atomic_json, _git_sha, resolve_config


RESEARCH_CONFIG = REPO / "ship_perception/config/r1_heightmap_research.json"


def run_file(pcd, output_root, run_id, scene_id, config, research):
    directory = Path(output_root) / run_id / scene_id
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("HEIGHTMAP_RUN_ALREADY_EXISTS:" + str(directory))
    points, _ = decode(pcd)
    result, auxiliary = detect_regions(points, config, research)
    result.update(scene_id=scene_id, run_id=run_id,
                  software_git_sha=_git_sha(),
                  input_sha256=hashlib.sha256(Path(pcd).read_bytes()).hexdigest(),
                  v15_config_hash=config_hash(config),
                  research_config_sha256=hashlib.sha256(
                      json.dumps(research, sort_keys=True, separators=(",", ":")).encode("utf-8")
                  ).hexdigest(),
                  point_count=int(len(points)))
    directory.mkdir(parents=True, exist_ok=True)
    result["visualization"] = render_regions(directory, points, result, auxiliary, config)
    _atomic_json(directory / "hatch_regions.json", result)
    return result


def run_batch(run_id, manifest_path, data_root, output_root):
    if not run_id or run_id in (".", "..") or any(char in run_id for char in "/\\:"):
        raise ValueError("INVALID_RUN_ID")
    destination = Path(output_root) / run_id
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("HEIGHTMAP_BATCH_ALREADY_EXISTS:" + str(destination))
    config, _ = resolve_config(DEFAULT_CONFIG)
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scans = {scan["scan_id"]: scan for scan in manifest["scans"]}
    summaries = {}
    for scene_id, scan_id in NORMAL6.items():
        scan = scans[scan_id]
        if scan["split_role"] != "DEVELOPMENT":
            raise ValueError("NON_DEVELOPMENT_INPUT:" + scan_id)
        pcd = Path(data_root) / scan["pcd_path"]
        if hashlib.sha256(pcd.read_bytes()).hexdigest() != scan["pcd_sha256"]:
            raise ValueError("INPUT_SHA_MISMATCH:" + scan_id)
        result = run_file(pcd, output_root, run_id, scene_id, config, research)
        summaries[scene_id] = dict(region_count=result["region_count"],
                                   status=result["status"],
                                   possible_merges=sum(row["topology_review"] == "POSSIBLE_MERGE"
                                                       for row in result["regions"]),
                                   input_sha256=result["input_sha256"])
        print("%s: %d coarse regions" % (scene_id, result["region_count"]), flush=True)
    _atomic_json(destination / "batch_summary.json",
                 dict(schema="ship_perception.v15r.heightmap_batch.1", run_id=run_id,
                      software_git_sha=_git_sha(), scenes=summaries))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-set", choices=("normal6",), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-manifest", type=Path,
                        default=REPO / "ship_perception/datasets/v15_dataset_manifest.json")
    parser.add_argument("--data-root", type=Path,
                        default=REPO / "Ship-Unloader-Data/legacy/hold_detector")
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "Ship-Unloader-Work/r1_heightmap")
    args = parser.parse_args()
    run_batch(args.run_id, args.dataset_manifest, args.data_root, args.output_root)


if __name__ == "__main__":
    main()
