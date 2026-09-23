"""Run R0 providers on the frozen Development scans and report only attributable metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from ship_perception.measurement.proposal import Candidate, evaluate_instances, nms
from ship_perception.tools.validate_v15r_scene_manifest import read_json, validate, DATASET, SCHEMA

NORMAL = ("2026-08-01-05-09-32", "2026-08-08-02-33-57",
          "final_map_4_16_2d_005", "final_map_6_8_2d_005",
          "final_map_7_16_2d_005", "final_map_8_22_cut")
STRESS = ("final_8_17_005", "final_map_5_29_2d_005", "final_map_7_2_005")
EXPECTED_HATCHES = dict(zip(NORMAL, (2, 1, 1, 1, 1, 3)))


def candidate(row):
    return Candidate(row["provider"], row["candidate_id"], row["score"],
                     tuple(row["rectangle"]), row["source"], row["rank_pre_nms"],
                     row["rank_post_nms"], row["rejection_reason"])


def scoring_rois(manifest_path):
    if manifest_path is None:
        return {}, "NO_ADJUDICATED_SCENE_MANIFEST"
    manifest = read_json(manifest_path)
    errors, ready = validate(manifest, read_json(DATASET), read_json(SCHEMA), True)
    if errors:
        raise ValueError("SCENE_MANIFEST_NOT_SCORING_READY:"+";".join(errors))
    scans = {s["scan_id"]: s for s in manifest["scans"]}
    if not set(NORMAL).issubset(ready):
        raise ValueError("NORMAL_SCANS_NOT_ADJUDICATED")
    if any(s["split_role"] != "DEVELOPMENT" for s in manifest["scans"]):
        raise ValueError("SEALED_HOLDOUT_ACCESS_FORBIDDEN")
    rois = {}
    for sid in NORMAL:
        hatches = [(v["vessel_local_id"], h) for v in scans[sid]["vessels"]
                   for h in v["hatches"] if v["vessel_role"] == "PRIMARY_DEVELOPMENT"]
        if len(hatches) != EXPECTED_HATCHES[sid]:
            raise ValueError(f"HATCH_COUNT_MISMATCH:{sid}")
        rois[sid] = hatches
    return rois, "ADJUDICATED"


def run(data_root: Path, out_dir: Path, report_path: Path, manifest_path=None):
    dataset = read_json(DATASET)
    sources = {s["scan_id"]: s for s in dataset["scans"]}
    rois, annotation_status = scoring_rois(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy = data_root / "detect"
    rows = []
    aggregate = {key: [0, 0] for key in ("cnn_pre_nms", "cnn_post_nms", "geometry", "union")}
    for sid in NORMAL + STRESS:
        source = sources[sid]
        if source["split_role"] != "DEVELOPMENT":
            raise ValueError(f"NON_DEVELOPMENT_SCAN:{sid}")
        pcd = data_root / source["pcd_path"]
        if hashlib.sha256(pcd.read_bytes()).hexdigest() != source["pcd_sha256"]:
            raise ValueError(f"PCD_HASH_MISMATCH:{sid}")
        scan_dir = out_dir / sid
        scan_dir.mkdir(parents=True, exist_ok=True)
        cnn_path, geo_path = scan_dir / "legacy_cnn.json", scan_dir / "geometry.json"
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
        for module, output, extra in (
            ("ship_perception.measurement.legacy_cnn_provider", cnn_path,
             ["--legacy-dir", str(legacy)]),
            ("ship_perception.measurement.geometry_provider", geo_path,
             ["--legacy-dir", str(legacy)]),
        ):
            command = [sys.executable, "-m", module, "--pcd", str(pcd),
                       "--output", str(output), *extra]
            subprocess.run(command, check=True, env=env)
        cnn = read_json(cnn_path)
        geo = read_json(geo_path)
        c_all = [candidate(c) for c in cnn["candidates"]]
        g_all = [candidate(c) for c in geo["candidates"]]
        selected = [c for c in c_all if c.rank_post_nms is not None]
        g_selected = [c for c in g_all if c.rank_post_nms is not None]
        union = nms(selected + g_selected, threshold=0.3, top_k=55)
        (scan_dir / "union.json").write_text(json.dumps([c.record() for c in union],
                                               allow_nan=False, indent=2), encoding="utf-8")
        row = {"scan_id": sid, "lane": "NORMAL" if sid in NORMAL else "STRESS",
               "expected_hatches_from_review_a_count_only": EXPECTED_HATCHES.get(sid),
               "cnn_pre_nms_count": len(c_all), "cnn_post_nms_count": len(selected),
               "geometry_raw_count": len(g_all), "geometry_post_nms_count": len(g_selected),
               "union_post_nms_count": sum(c.rank_post_nms is not None for c in union),
               "candidate_files": {"cnn": str(cnn_path), "geometry": str(geo_path),
                                   "union": str(scan_dir / "union.json")},
               "target_assignment": "UNRESOLVED" if sid in STRESS else "SINGLE_VESSEL_SCENE"}
        if sid in rois:
            gt = [dict(h, local_id=f"{vid}/{h['local_id']}") for vid, h in rois[sid]]
            for name, candidates in (("cnn_pre_nms", c_all), ("cnn_post_nms", selected),
                                     ("geometry", g_selected),
                                     ("union", [c for c in union if c.rank_post_nms is not None])):
                result = evaluate_instances(candidates, gt)
                row[name] = result
                aggregate[name][0] += result["matched"]
                aggregate[name][1] += result["total"]
        else:
            row["instance_metrics_status"] = "NOT_MEASURABLE_NO_ADJUDICATED_ROI"
        rows.append(row)
    report = {"schema": "ship_perception.r0.proposal_baseline.1",
              "annotation_status": annotation_status,
              "metric_definition": "one-to-one axis-aligned coarse envelope IoU >= 0.1; merge covering >= 0.5 of two GT boxes is rejected",
              "provider_caveats": ["legacy weights may have seen weak labels from all 13 scans",
                                   "geometry provider is minimal diagnostic, not product detector"],
              "recall": {k: v[0]/v[1] if v[1] else None for k, v in aggregate.items()},
              "recall_counts": {k: {"matched": v[0], "total": v[1]} for k, v in aggregate.items()},
              "scans": rows}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, default=Path("Ship-Unloader-Work/validation/r0_measurement"))
    ap.add_argument("--report", type=Path, default=Path("Ship-Unloader-Work/reports/proposal_baseline_report.json"))
    ap.add_argument("--scene-manifest", type=Path)
    args = ap.parse_args()
    run(args.data_root, args.output_root, args.report, args.scene_manifest)


if __name__ == "__main__":
    main()
