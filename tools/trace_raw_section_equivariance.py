"""Offline capture/compare of the first Raw-profile difference under a rigid replay.

This diagnostic hooks the existing implementation without changing its result.
Its trace overhead must not be included in performance acceptance measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def capture(args):
    from ship_perception.perception.final_boundary import FinalBoundaryRefiner
    from ship_perception.perception.pipeline import _json_default
    from run_opening_ir_shadow import main as native_main

    args.output.mkdir(parents=True, exist_ok=True)
    mapping = (
        np.load(args.source_index_map, allow_pickle=False)
        if args.source_index_map
        else None
    )
    original = FinalBoundaryRefiner._search_raw_section
    records = []

    def traced(self, roi, axis, anchor, span, sign):
        result = original(self, roi, axis, anchor, span, sign)
        context = self.measurement_context
        if context is None:
            return result
        profile = context.side_measurements.profile(axis, anchor, span)
        points = profile.points
        if not len(points):
            return result
        # Exact membership of the same profile bounds, retaining all duplicates.
        box = np.r_[points[:, :2].min(0), points[:, :2].max(0)]
        selected = context.spatial.query_indexes(box)
        if len(selected) != len(points):
            raise ValueError("PROFILE_SOURCE_RECTANGLE_RECONSTRUCTION_MISMATCH")
        source_ids = context.raw_point_indexes[selected]
        if mapping is not None:
            source_ids = mapping[source_ids]
        source_ids = np.sort(source_ids).astype("<i8")
        probe = self.config["boundary"]["side_probe_m"]
        normal_coordinates = points[:, axis]
        low = (normal_coordinates >= anchor - probe) & (normal_coordinates < anchor)
        high = (normal_coordinates >= anchor) & (normal_coordinates <= anchor + probe)
        primary = getattr(context.local, "primary_cache", context.local)
        axes = np.asarray(primary.vessel.axes)
        uv = np.zeros(2)
        uv[axis] = anchor
        uv[1 - axis] = sum(span) / 2
        record = dict(
            axis=axis,
            sign=sign,
            anchor=anchor,
            span=list(span),
            axes=axes.tolist(),
            probe_input_xy=(uv @ axes).tolist(),
            source_row_count=len(source_ids),
            source_rows_sha256=digest(source_ids),
            profile_row_count=len(points),
            profile_xyz_sha256=digest(points),
            left_count=int(low.sum()),
            right_count=int(high.sum()),
            left_median_z=float(np.median(points[low, 2])) if low.any() else None,
            right_median_z=float(np.median(points[high, 2])) if high.any() else None,
            mode_nominations=profile.edge_modes(anchor, sign),
            result=result,
        )
        records.append(record)
        return result

    argv = sys.argv
    try:
        sys.argv = [
            "run_opening_ir_shadow",
            "--pcd",
            str(args.pcd),
            "--output-dir",
            str(args.output / "capture"),
            "--local-rim-review",
        ]
        with patch.object(FinalBoundaryRefiner, "_search_raw_section", traced):
            code = native_main()
    finally:
        sys.argv = argv
        (args.output / "raw_profile_trace.json").write_text(
            json.dumps(
                dict(
                    input_sha256=hashlib.sha256(args.pcd.read_bytes()).hexdigest(),
                    trace_scope="EXISTING_RAW_SECTION_DIAGNOSTIC_NOT_PERFORMANCE",
                    records=records,
                ),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
    return code


def compare(args):
    before = json.loads(args.before.read_bytes())
    after = json.loads(args.after.read_bytes())
    transform = json.loads(args.transform.read_bytes())
    rotation = np.asarray(transform["rotation_xy"])
    origin = np.asarray(transform["origin_xy"])
    shift = np.asarray(transform["translation_xyz"][:2])
    rows = []
    for a in after["records"]:
        inverse = (np.asarray(a["probe_input_xy"]) - origin - shift) @ rotation + origin
        options = [
            b
            for b in before["records"]
            if b["axis"] == a["axis"] and b["sign"] == a["sign"]
        ]
        if not options:
            continue
        b = min(
            options,
            key=lambda r: np.linalg.norm(inverse - np.asarray(r["probe_input_xy"])),
        )
        distance = float(np.linalg.norm(inverse - np.asarray(b["probe_input_xy"])))
        axes_delta = float(
            np.max(abs(np.asarray(a["axes"]) @ rotation - np.asarray(b["axes"])))
        )
        same_source = a["source_rows_sha256"] == b["source_rows_sha256"]
        same_split = (a["left_count"], a["right_count"]) == (
            b["left_count"],
            b["right_count"],
        )
        first = (
            "B1_AXIS"
            if axes_delta > 64 * np.finfo(float).eps
            else (
                "RAW_FINITE_ROW_SELECTION"
                if not same_source
                else (
                    "PHYSICAL_STATION_ANCHOR"
                    if distance
                    > 64 * np.finfo(float).eps * max(1.0, np.linalg.norm(inverse))
                    else (
                        "RAW_PROFILE_SPLIT"
                        if not same_split
                        else "NO_DIFFERENCE_IN_CAPTURED_PREFIX"
                    )
                )
            )
        )
        rows.append(
            dict(
                first_difference=first,
                inverse_axis_max_delta=axes_delta,
                inverse_probe_distance_m=distance,
                same_source_rows=same_source,
                same_profile_split=same_split,
                before=b,
                after=a,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                transform=transform,
                pairs=rows,
                limitation="NEAREST_SAME_AXIS_SIGN_STATIONS;UNMATCHED_OR_MOVED_STATIONS_REPORTED_NOT_ASSUMED_EQUIVALENT",
            ),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("CAPTURED_STATION_PAIRS=" + str(len(rows)))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("capture")
    run.add_argument("--pcd", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--source-index-map", type=Path)
    diff = commands.add_parser("compare")
    for name in ("before", "after", "transform", "output"):
        diff.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    return capture(args) if args.action == "capture" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
