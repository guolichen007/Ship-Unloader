"""Run the six Development PCDs via manifest paths, then create a Chinese review CSV."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from .run import REPO, _atomic_json, run_file


NORMAL6 = {
    "08-01": "2026-08-01-05-09-32",
    "08-08": "2026-08-08-02-33-57",
    "4-16": "final_map_4_16_2d_005",
    "6-8": "final_map_6_8_2d_005",
    "7-16": "final_map_7_16_2d_005",
    "8-22": "final_map_8_22_cut",
}
REVIEW_COLUMNS = ("场景", "算法舱数", "舱编号", "舱数是否正确", "位置是否正确", "边界是否贴合",
                  "漏检", "误检", "错误合并", "错误拆分", "大约偏差米", "人工备注")
PERIMETER_COLUMNS = ("Proposal", "OpeningSeed", "PerimeterSegments", "SegmentDeckResolved",
                     "SegmentDeckAmbiguous", "ObservedProfileEdges", "Observed3DFaces",
                     "CompleteObserved", "Partial", "Unresolved", "Selected", "Confirmed",
                     "TouchesScanBoundary")


def run_batch(dataset_manifest, data_root, output_root, run_id, overrides=(), override_file=None):
    if not run_id or run_id in (".", "..") or any(c in run_id for c in "/\\:"):
        raise ValueError("INVALID_RUN_ID")
    directory = Path(output_root) / run_id
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("R1_BATCH_RUN_ALREADY_EXISTS:" + str(directory))
    manifest = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
    scans = {scan["scan_id"]: scan for scan in manifest["scans"]}
    if not set(NORMAL6.values()).issubset(scans):
        raise ValueError("NORMAL6_MISSING_FROM_MANIFEST")
    results = {}
    for scene, scan_id in NORMAL6.items():
        source = scans[scan_id]
        if source["split_role"] != "DEVELOPMENT":
            raise ValueError("NON_DEVELOPMENT_INPUT:" + scan_id)
        pcd = Path(data_root) / source["pcd_path"]
        if hashlib.sha256(pcd.read_bytes()).hexdigest() != source["pcd_sha256"]:
            raise ValueError("INPUT_SHA_MISMATCH:" + scan_id)
        try:
            result = run_file(pcd, output_root, run_id, scene_id=scene,
                              overrides=overrides, override_file=override_file)
        except Exception as error:
            failure = dict(scene_status="RUNTIME_ERROR", error_type=type(error).__name__,
                           reason=str(error), confirmed_hatch_count=None)
            _atomic_json(Path(output_root) / run_id / scene / "failure.json", failure)
            results[scene] = failure
            print("%s: RUNTIME_ERROR: %s" % (scene, error), flush=True)
            continue
        results[scene] = dict(scene_status=result["scene_status"],
                              confirmed_hatch_count=result["confirmed_hatch_count"],
                              **result["perimeter_summary"])
        print("%s: %s, %d" % (scene, result["scene_status"], result["confirmed_hatch_count"]), flush=True)
    _atomic_json(directory / "batch_summary.json", dict(schema="ship_perception.v15r.static_batch.2",
                                                      run_id=run_id, scenes=results))
    perimeter_target = directory / "perimeter_summary.csv"
    perimeter_temp = perimeter_target.with_name(perimeter_target.name + ".tmp")
    with perimeter_temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("场景",) + PERIMETER_COLUMNS)
        writer.writeheader()
        for scene, summary in results.items():
            writer.writerow({"场景": scene, **{key: summary.get(key, "")
                                              for key in PERIMETER_COLUMNS}})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(perimeter_temp, perimeter_target)
    target = directory / "人工复核.csv"
    temp = target.with_name(target.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for scene, summary in results.items():
            result_path = directory / scene / "result.json"
            hatches = json.loads(result_path.read_text(encoding="utf-8"))["hatches"] if result_path.exists() else []
            for hatch in hatches or [None]:
                writer.writerow({"场景": scene, "算法舱数": summary["confirmed_hatch_count"],
                                 "舱编号": hatch["hatch_id"] if hatch else ""})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, target)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path,
                        default=REPO / "ship_perception/datasets/v15_dataset_manifest.json")
    parser.add_argument("--data-root", type=Path, default=REPO / "Ship-Unloader-Data/legacy/hold_detector")
    parser.add_argument("--scene-set", choices=("normal6",), required=True)
    parser.add_argument("--output-root", type=Path, default=REPO / "Ship-Unloader-Work/r1_static")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--override-file", type=Path)
    args = parser.parse_args()
    run_batch(args.dataset_manifest, args.data_root, args.output_root, args.run_id,
              args.override, args.override_file)


if __name__ == "__main__":
    main()
