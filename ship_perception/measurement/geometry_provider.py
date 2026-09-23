"""Minimal label-free R0 proposal diagnostic, not an R1 detector.

It deliberately uses the frozen V1.5 coarse grid and opening thresholds without
an oracle crop, target-vessel seed, learned parameters, or boundary refinement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from ship_perception.measurement.proposal import Candidate, nms
from ship_perception.tools.v15_pcd import decode


def candidates_from_grid(surf, valid, meta, levels, config):
    x0, y0, _, _, cs = meta
    roi = config["roi"]
    found = []
    for level_id, level in enumerate(levels):
        mask = valid & (surf < level-roi["opening_drop_m"]) & (surf > level-roi["max_opening_depth_m"])
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        slices = ndimage.find_objects(labels)
        for label_id, sl in enumerate(slices, 1):
            if sl is None:
                continue
            area_cells = int(np.count_nonzero(labels[sl] == label_id))
            area = area_cells * cs * cs
            if area_cells < roi["min_opening_cells"] or not roi["opening_min_area_m2"] <= area <= roi["opening_max_area_m2"]:
                continue
            rr, cc = sl
            rect = (x0 + cc.start*cs, y0 + rr.start*cs, x0 + cc.stop*cs, y0 + rr.stop*cs)
            box_area = (rect[2]-rect[0])*(rect[3]-rect[1])
            score = area/box_area
            found.append(Candidate("GEOMETRY_MINIMAL", f"g-{level_id:02d}-{label_id:05d}",
                                   float(score), rect, f"coarse_low_component@level_{level_id}"))
    return found


def run(pcd: Path, config_path: Path, legacy_dir: Path):
    # Reuse the archived height-map construction as a read-only research provider.
    # Its PCD reader is bypassed: all points come from the strict V1.5 decoder.
    import sys
    legacy = legacy_dir
    if not legacy.is_dir():
        raise FileNotFoundError("LEGACY_HEIGHTMAP_PROVIDER_MISSING")
    sys.path.insert(0, str(legacy))
    from analyze_pcd import build_grid, extract_levels

    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    points, _ = decode(pcd)
    cs = config["geometry"]["coarse_voxel_m"]
    surf, _, valid, meta = build_grid(points, cs, 5, 0.5)
    levels = extract_levels(surf, valid)
    raw = candidates_from_grid(surf, valid, meta, levels, config)
    audited = nms(raw, threshold=0.3, top_k=50)
    return {"provider": "GEOMETRY_MINIMAL", "diagnostic_only": True,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "input_pcd_sha256": hashlib.sha256(pcd.read_bytes()).hexdigest(),
            "height_levels_m": [float(z) for z in levels],
            "candidate_count": len(raw), "candidates": [c.record() for c in audited]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pcd", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("ship_perception/config/v15.json"))
    ap.add_argument("--legacy-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = run(args.pcd, args.config, args.legacy_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
