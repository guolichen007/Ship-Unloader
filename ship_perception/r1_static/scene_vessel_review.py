"""Offline R2H-B1 scene review from static PCD/PLY; no target selection."""

import argparse
import hashlib
import json
from pathlib import Path

from .heightmap_batch import RESEARCH_CONFIG
from .rectangle_refinement_review import _read_validation_xyz
from .run import DEFAULT_CONFIG, _atomic_json, resolve_config
from .scene_vessel import infer_scene_vessels


def review_file(path, config, research):
    path = Path(path)
    points = _read_validation_xyz(path)
    result, _ = infer_scene_vessels(points, config, research)
    result.update(input_file=path.name,
                  input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  raw_point_count=int(len(points)), frame_id="STATIC_SINGLE_FRAME",
                  sensor_id="UNKNOWN")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, _ = resolve_config(DEFAULT_CONFIG)
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    result = {path.stem: review_file(path, config, research) for path in args.input}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    for name, scene in result.items():
        counts = {}
        for row in scene["vessel_hypotheses"]:
            key = row["classification_status"]
            counts[key] = counts.get(key, 0) + 1
        print(name, counts)


if __name__ == "__main__":
    main()
