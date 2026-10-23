"""Capture /ship_scan/map and save exactly two review figures plus result data."""

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Dict
from typing import Optional

import numpy as np

# Also support python3 ship_perception/ros/ship_scan_once.py from any directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ship_perception.legacy.ros1.pointcloud2_reader import read_once
from ship_perception.legacy.ros1.ros_input_adapter import PointCloudFrame
from ship_perception.r1_static.corner_topology_review import review_points
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG
from ship_perception.r1_static.intensity_evidence import AUX_WEIGHT, annotate_result
from ship_perception.r1_static.pcd_showcase import prepare_geometry
from ship_perception.r1_static.run import (
    DEFAULT_CONFIG,
    _atomic_json,
    _git_sha,
    resolve_config,
)
from ship_perception.r1_static.static_showcase import render_showcase
from ship_perception.tools.stage_progress import StageProgress, stage


BASELINE_IMAGE = "01_geometry_only.png"
FUSION_IMAGE = "02_intensity_fusion.png"
RESULT_JSON = "03_results.json"
SUMMARY_TEXT = "04_input_summary.txt"


def _summary(frame: PointCloudFrame) -> Dict:
    values = (
        np.array([])
        if frame.intensity is None
        else frame.intensity[np.isfinite(frame.intensity)]
    )
    return dict(
        topic=frame.source_topic,
        frame_id=frame.frame_id,
        timestamp=frame.stamp,
        point_count=len(frame.xyz),
        source_point_count=frame.source_point_count,
        dropped_xyz_count=frame.source_point_count - len(frame.xyz),
        fields=list(frame.fields),
        intensity_available=bool(len(values)),
        intensity_field_present="intensity" in frame.fields,
        intensity_finite_count=int(len(values)),
        intensity_min=float(values.min()) if len(values) else None,
        intensity_p50=float(np.percentile(values, 50)) if len(values) else None,
        intensity_p95=float(np.percentile(values, 95)) if len(values) else None,
        input_sha256=frame.sha256,
    )


def _result(report: Dict, image: str, fusion_status: str) -> Dict:
    hatches = []
    for vessel in report["vessels"]:
        for row in vessel["rectangles"]:
            hatches.append(
                dict(
                    hatch_id=row["hatch_id"],
                    vessel_hypothesis_id=vessel["vessel_hypothesis_id"],
                    status=row["status"],
                    boundary_xy=row["polygon_after"],
                    display_boundary_xy=(row.get("height_review") or {}).get(
                        "polygon", row["polygon_after"]
                    ),
                    sides=row["sides"],
                    intensity_evidence=row.get("intensity_evidence"),
                )
            )
    return dict(
        hatch_count=len(hatches),
        hatches=hatches,
        image_path=image,
        intensity_fusion_status=fusion_status,
        intensity_used=any(
            (row["intensity_evidence"] or {}).get("intensity_used", False)
            for row in hatches
        ),
        diagnostics=report,
    )


def run_frame(
    frame: PointCloudFrame,
    output_dir: Path,
    save_snapshot: bool = False,
    disable_intensity_fusion: bool = False,
    progress: Optional[StageProgress] = None,
) -> Path:
    """Run shared XYZ providers, then independent baseline and fusion reviews."""
    with stage(progress, "LOAD_CONFIGURATION"):
        config, config_sha = resolve_config(DEFAULT_CONFIG)
        research_bytes = RESEARCH_CONFIG.read_bytes()
        research = json.loads(research_bytes)
        digest = frame.sha256
    b2, b4, counts = prepare_geometry(
        frame.xyz, "ship_scan", digest, config, research, progress=progress
    )
    with stage(progress, "GEOMETRY_BOUNDARY_AND_HEIGHT_REVIEW"):
        baseline, _, private = review_points(
            "ship_scan", frame.xyz, digest, b2, b4, config, research, height_rim=True
        )
    summary = _summary(frame)
    status = (
        "DISABLED"
        if disable_intensity_fusion
        else "UNAVAILABLE" if not summary["intensity_available"] else "AVAILABLE"
    )
    if status == "AVAILABLE":
        with stage(progress, "INTENSITY_FUSION_AND_HEIGHT_REVIEW"):
            fusion, _, _ = review_points(
                "ship_scan",
                frame.xyz,
                digest,
                b2,
                b4,
                config,
                research,
                height_rim=True,
                intensity=frame.intensity,
            )
            annotate_result(fusion, frame.xyz, frame.intensity, private, b2, b4, config)
        if not any(
            v["normalized_intensity_stats"]["status"] == "AVAILABLE"
            for v in fusion["vessels"]
        ):
            status = "UNAVAILABLE_NO_VESSEL_CONTRAST"
    else:
        with stage(progress, "INTENSITY_FALLBACK_" + status):
            fusion = copy.deepcopy(baseline)
    # Keep global H labels unique across vessels, retaining original diagnostic IDs.
    for report in (baseline, fusion):
        for index, row in enumerate(
            row for vessel in report["vessels"] for row in vessel["rectangles"]
        ):
            row["source_hatch_id"] = row["hatch_id"]
            row["hatch_id"] = "H{:03d}".format(index)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    final = output_dir / name
    # Publish only complete runs; the context manager removes failed staging dirs.
    with tempfile.TemporaryDirectory(
        prefix=".ship_hatch_", dir=str(output_dir)
    ) as staging:
        staged = Path(staging) / "result"
        staged.mkdir()
        with tempfile.TemporaryDirectory(prefix="ship_hatch_mpl_") as mpl_cache:
            previous = os.environ.get("MPLCONFIGDIR")
            os.environ["MPLCONFIGDIR"] = mpl_cache
            try:
                for report, filename, title, note in (
                    (baseline, BASELINE_IMAGE, "几何主算法", None),
                    (
                        fusion,
                        FUSION_IMAGE,
                        "融合强度辅助",
                        "强度辅助：{}；琥珀线为待复核开口候选".format(status),
                    ),
                ):
                    with stage(progress, "RENDER_" + Path(filename).stem):
                        render_showcase(
                            report,
                            frame.xyz,
                            private,
                            b2,
                            b4,
                            config,
                            staged,
                            height_only=True,
                            image_only=True,
                            image_name=filename,
                            title="原始 XYZ 高度图与船舱开口候选（{}）".format(title),
                            annotation=note,
                        )
            finally:
                if previous is None:
                    os.environ.pop("MPLCONFIGDIR", None)
                else:
                    os.environ["MPLCONFIGDIR"] = previous
        payload = dict(
            schema="ship_perception.legacy.ros1.hatch_once.1",
            **summary,
            config_sha256=config_sha,
            research_config_sha256=hashlib.sha256(research_bytes).hexdigest(),
            software_git_sha=_git_sha(),
            geometry_provider_counts=counts,
            intensity_aux_weight=AUX_WEIGHT,
            target_vessel_status="TARGET_VESSEL_UNRESOLVED",
            coordinate_frame="INPUT_FRAME_XY_METERS; UV_AXES_IN_DIAGNOSTICS",
            geometry_only=_result(baseline, str(final / BASELINE_IMAGE), "NOT_USED"),
            intensity_fusion=_result(fusion, str(final / FUSION_IMAGE), status),
        )
        with stage(progress, "SAVE_RESULT_JSON"):
            _atomic_json(staged / RESULT_JSON, payload)
        with (staged / SUMMARY_TEXT).open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            for key, value in summary.items():
                stream.write(
                    "{}={}\n".format(key.upper(), json.dumps(value, ensure_ascii=False))
                )
            stream.write("INTENSITY_FUSION_STATUS={}\n".format(status))
            stream.flush()
            os.fsync(stream.fileno())
        if save_snapshot:
            with stage(progress, "SAVE_INPUT_SNAPSHOT"):
                np.savez_compressed(
                    staged / "input_snapshot.npz",
                    xyz=frame.xyz,
                    intensity=(
                        np.array([]) if frame.intensity is None else frame.intensity
                    ),
                    metadata=json.dumps(summary, ensure_ascii=False),
                )
        with stage(progress, "PUBLISH_COMPLETE_OUTPUT"):
            staged.rename(final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/ship_scan/map")
    parser.add_argument(
        "--output_dir", "--output-dir", type=Path, default=Path("output")
    )
    parser.add_argument("--timeout_sec", "--timeout-sec", type=float, default=30.0)
    parser.add_argument("--save_snapshot", "--save-snapshot", action="store_true")
    parser.add_argument(
        "--disable_intensity_fusion", "--disable-intensity-fusion", action="store_true"
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout_sec) or args.timeout_sec <= 0:
        parser.error("--timeout_sec must be finite and positive")
    progress = StageProgress()
    try:
        print("Waiting for one PointCloud2 from {} ...".format(args.topic), flush=True)
        with progress.stage("CAPTURE_POINTCLOUD"):
            frame = read_once(args.topic, args.timeout_sec)
        print(
            "Received {} finite XYZ points; intensity field: {}".format(
                len(frame.xyz), frame.intensity is not None
            ),
            flush=True,
        )
        output = run_frame(
            frame,
            args.output_dir,
            args.save_snapshot,
            args.disable_intensity_fusion,
            progress=progress,
        )
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        progress.finish(False)
        return 1
    progress.finish(True)
    print("Output directory: {}".format(output))
    for filename in (BASELINE_IMAGE, FUSION_IMAGE, RESULT_JSON, SUMMARY_TEXT):
        print(output / filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
