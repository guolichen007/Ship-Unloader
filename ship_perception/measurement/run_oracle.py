"""Isolated R0 Oracle ROI runner for the *existing* V1.5 geometry executable.

No oracle polygon is passed to a product config or used by runtime proposal.
Rough ROI only crops an evaluation cloud. Its coordinates are not steel-edge GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

from ship_perception.measurement.oracle_local_geometry import diagnose, distance_to_edges, inside_xy
from ship_perception.measurement.run_proposals import NORMAL, EXPECTED_HATCHES, scoring_rois
from ship_perception.tools.v15_pcd import decode
from ship_perception.tools.validate_v15r_scene_manifest import read_json, DATASET


def write_xyz_cache(path: Path, points: np.ndarray):
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("INVALID_ORACLE_XYZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"SXYZV15\0")
        stream.write(struct.pack("<Q", len(points)))
        stream.write(np.asarray(points, dtype="<f4").tobytes(order="C"))


def crop_oracle_cloud(points, hatch, vessel, ring_m):
    xy = points[:, :2].astype(float)
    # Both polygons are evaluation-only. A blank cell is never synthesized.
    mask = inside_xy(xy, vessel) & (inside_xy(xy, hatch) | (distance_to_edges(xy, hatch) <= ring_m))
    return points[mask]


def classify(local, model):
    """Layer attribution only; physical success needs adjudicated edge support."""
    if local["local_deck"]["status"] != "RESOLVED" or not model.get("deck", {}).get("valid"):
        return "LOCAL_DECK"
    if not model.get("frame_resolved"):
        return "LOCAL_DECK"
    hatches = model.get("hatches", [])
    if len(hatches) > 1:
        return "TOPOLOGY"
    if not hatches or not any(h.get("observed_edge_count", 0) for h in hatches):
        return "BOUNDARY"
    if local["status"] == "PARTIAL" or any(h.get("status") == "PARTIAL" for h in hatches):
        return "INSUFFICIENT_OBSERVATION"
    return None


def score_physical_edges(hatch, model, tolerance_m):
    """GT_SCORER lane only. Rough ROI is never substituted for an observed edge."""
    gt_all = hatch.get("edges", [])
    gt = [e for e in gt_all if e["visibility"] == "VISIBLE" and
          e["opening_side_status"] == "OBSERVABLE" and e["observed_support"] is not None]
    predicted = [e for h in model.get("hatches", []) for e in h.get("boundaries", [])
                 if e.get("side") == "INNER_OPENING_FACE" and
                 e.get("visibility") == "VISIBLE" and (e.get("evidence_flags", 0) & 1)]
    if not gt:
        return {"status": "NO_ADJUDICATED_OBSERVED_EDGE_SUPPORT", "success": None,
                "edges": [], "predicted_observed_inner_edge_count": len(predicted)}

    def samples(polyline):
        chain = np.asarray(polyline, dtype=float)
        if chain.ndim != 2 or chain.shape[1] != 2 or len(chain) < 2 or not np.isfinite(chain).all():
            raise ValueError("INVALID_PHYSICAL_EDGE_SUPPORT")
        pieces = []
        for a, b in zip(chain, chain[1:]):
            length = np.linalg.norm(b-a)
            if length > 0:
                t = np.linspace(0, 1, max(2, int(np.ceil(length/0.1))+1))
                pieces.append(a+t[:, None]*(b-a))
        if not pieces:
            raise ValueError("ZERO_LENGTH_PHYSICAL_EDGE_SUPPORT")
        return np.concatenate(pieces)

    def p95_to_segment(polyline, pred):
        q = samples(polyline)
        a, b = np.asarray(pred["a"][:2]), np.asarray(pred["b"][:2])
        d = b-a
        if d @ d < 1e-12:
            return float("inf")
        t = np.clip(((q-a) @ d)/(d @ d), 0, 1)
        return float(np.percentile(np.linalg.norm(q-(a+t[:, None]*d), axis=1), 95))

    pairs = sorted((p95_to_segment(edge["observed_support"], pred), gi, pi)
                   for gi, edge in enumerate(gt) for pi, pred in enumerate(predicted))
    assigned_gt, assigned_pred = {}, set()
    for dist, gi, pi in pairs:
        if gi not in assigned_gt and pi not in assigned_pred:
            assigned_gt[gi] = (pi, dist)
            assigned_pred.add(pi)
    rows = [{"edge_local_id": edge["edge_local_id"],
             "matched_predicted_index": assigned_gt[i][0] if i in assigned_gt else None,
             "observed_support_to_predicted_p95_m": assigned_gt[i][1] if i in assigned_gt else None,
             "within_15cm": assigned_gt[i][1] <= tolerance_m if i in assigned_gt else False}
            for i, edge in enumerate(gt)]
    full_physical_label = len(gt_all) >= 3 and len(gt) == len(gt_all)
    success = (all(row["within_15cm"] for row in rows) and len(predicted) == len(gt)) if full_physical_label else None
    return {"status": "FULL_EDGE_SUPPORT" if full_physical_label else "PARTIAL_EDGE_SUPPORT_ONLY",
            "success": success, "edges": rows,
            "predicted_observed_inner_edge_count": len(predicted)}


def run(data_root: Path, out_dir: Path, report_path: Path, manifest_path: Path | None,
        recognizer: Path | None):
    rois, annotation_status = scoring_rois(manifest_path)
    report = {"schema": "ship_perception.r0.oracle_local_geometry.1",
              "oracle_isolation": "EVALUATION_ONLY_NOT_PRODUCT_CONFIG",
              "annotation_status": annotation_status,
              "status": "NOT_MEASURABLE_NO_ADJUDICATED_ROI" if not rois else "RUNNING",
              "physical_edge_success_count": None,
              "physical_edge_success_total": None,
              "warning": "Rough ROI cannot establish true opening-side steel-edge accuracy without adjudicated observed edge support",
              "hatches": []}
    if not rois:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    if recognizer is None or not recognizer.is_file():
        raise FileNotFoundError("EXISTING_V15_RECOGNIZER_REQUIRED")
    config = read_json(Path("ship_perception/config/v15.json"))
    sources = {s["scan_id"]: s for s in read_json(DATASET)["scans"]}
    scenes = {s["scan_id"]: s for s in read_json(manifest_path)["scans"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in NORMAL:
        source = sources[sid]
        if source["split_role"] != "DEVELOPMENT":
            raise ValueError("SEALED_HOLDOUT_FORBIDDEN")
        pcd = data_root / source["pcd_path"]
        if hashlib.sha256(pcd.read_bytes()).hexdigest() != source["pcd_sha256"]:
            raise ValueError(f"PCD_HASH_MISMATCH:{sid}")
        points, _ = decode(pcd)
        for vessel_id, hatch in rois[sid]:
            # Find the adjudicated vessel that owns this hatch. A null vessel
            # polygon must fail; hatch ROI alone cannot exclude wharf/other ship.
            scene = scenes[sid]
            vessel = next(v for v in scene["vessels"] if v["vessel_local_id"] == vessel_id)
            vpoly = vessel.get("roi_raw_xy")
            hpoly = hatch.get("rough_polygon_raw_xy")
            if vpoly is None or hpoly is None:
                raise ValueError(f"ORACLE_ROI_MISSING:{sid}:{vessel_id}:{hatch['local_id']}")
            other = [h["rough_polygon_raw_xy"] for h in vessel["hatches"]
                     if h["local_id"] != hatch["local_id"] and h["rough_polygon_raw_xy"] is not None]
            cropped = crop_oracle_cloud(points, hpoly, vpoly, config["roi"]["support_search_m"])
            directory = out_dir / sid / hatch["local_id"]
            cache, model_path = directory / "oracle_input.sxyz", directory / "core_model.json"
            write_xyz_cache(cache, cropped)
            completed = subprocess.run([str(recognizer), "ship-frame", str(cache), str(model_path)],
                                       capture_output=True, text=True)
            if completed.returncode:
                model = {"frame_resolved": False, "deck": {"valid": False},
                         "hatches": [], "execution_failure": completed.stderr[-2000:]}
            else:
                model = read_json(model_path)
            local = diagnose(cropped, hpoly, vpoly, config, other)
            physical = score_physical_edges(hatch, model, config["evaluation"]["boundary_p95_m"])
            (directory / "profile_audit.json").write_text(json.dumps(local, allow_nan=False, indent=2), encoding="utf-8")
            report["hatches"].append({"scan_id": sid, "vessel_id": vessel_id,
                                      "hatch_id": hatch["local_id"],
                                      "crop_point_count": int(len(cropped)),
                                      "local_deck": local["local_deck"],
                                      "profile_observed_edge_count": local.get("observed_edge_count", 0),
                                      "core_deck_valid": model.get("deck", {}).get("valid", False),
                                      "core_hatch_count": len(model.get("hatches", [])),
                                      "core_observed_edge_count": sum(h.get("observed_edge_count", 0) for h in model.get("hatches", [])),
                                      "failure_layer": classify(local, model),
                                      "physical_edge_success": physical["success"],
                                      "physical_edge_scoring": physical,
                                      "artifacts": {"core_model": str(model_path),
                                                    "profile_audit": str(directory / "profile_audit.json")}})
    if len(report["hatches"]) != sum(EXPECTED_HATCHES.values()):
        raise ValueError("ORACLE_HATCH_COUNT_MISMATCH")
    scored = [h for h in report["hatches"] if h["physical_edge_success"] is not None]
    report["physical_edge_success_count"] = sum(h["physical_edge_success"] for h in scored) if scored else None
    report["physical_edge_success_total"] = len(scored) if scored else None
    report["status"] = "PHYSICAL_EDGE_SCORED" if len(scored) == len(report["hatches"]) else "DIAGNOSTIC_COMPLETE_PHYSICAL_EDGE_PARTLY_OR_UNSCORED"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--scene-manifest", type=Path)
    ap.add_argument("--recognizer", type=Path)
    ap.add_argument("--output-root", type=Path, default=Path("Ship-Unloader-Work/validation/r0_measurement/oracle"))
    ap.add_argument("--report", type=Path, default=Path("Ship-Unloader-Work/reports/oracle_local_geometry_report.json"))
    args = ap.parse_args()
    run(args.data_root, args.output_root, args.report, args.scene_manifest, args.recognizer)


if __name__ == "__main__":
    main()
