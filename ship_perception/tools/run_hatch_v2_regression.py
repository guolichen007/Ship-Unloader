"""Private-data family regression; references are read by this tool only.

The manifest stays outside Git. Runtime never imports this module or sees
reference coordinates. A normal process exit is not a recognition pass.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np


def coverage(old, new, tolerance):
    """Signed convex-edge distances; test the entire old opening, not count."""
    points = np.asarray(old, dtype=float)
    polygon = np.asarray(new, dtype=float)
    edges = np.roll(polygon, -1, axis=0) - polygon
    area = np.cross(polygon, np.roll(polygon, -1, axis=0)).sum()
    orientation = 1 if area >= 0 else -1
    lengths = np.linalg.norm(edges, axis=1)
    if (lengths == 0).any():
        return False, float("inf")
    signed = (
        orientation
        * np.cross(edges[None, :, :], points[:, None, :] - polygon[None, :, :])
        / lengths
    )
    inward = max(0.0, -float(signed.min()))
    return inward <= tolerance, inward


def run_case(case, output):
    from ship_perception.io.pcd_source import read_pcd
    from ship_perception.perception.pipeline_v2 import run_perception
    from ship_perception.perception.pipeline import Timing, _json_default
    from ship_perception.perception.output import save_output
    from ship_perception.tools.stage_progress import StageProgress
    from ship_perception.r1_static.run import resolve_config

    start = time.perf_counter()
    progress = StageProgress()
    with progress.stage("READ_PCD"):
        frame = read_pcd(case["pcd"], case.get("frame_id", "pcd"))
    result = run_perception(
        frame, progress, berth_reference=case.get("berth_reference")
    )
    directory = save_output(result, frame, output, False, Timing(progress))
    config, _ = resolve_config()
    old_path = case.get("old_result")
    matches = []
    if old_path:
        old = json.loads(Path(old_path).read_text(encoding="utf-8"))
        for hatch in old["geometry_only"]["hatches"]:
            possibilities = []
            for current in result["geometry_only"]["hatches"]:
                covered, inward = coverage(
                    hatch["boundary_xy"],
                    current["boundary_xy"],
                    config["geometry"]["plane_inlier_m"],
                )
                possibilities.append((inward, current["hatch_id"], covered))
            best = min(possibilities) if possibilities else (None, None, False)
            matches.append(
                dict(
                    old_hatch_id=hatch["hatch_id"],
                    new_hatch_id=best[1],
                    max_inward_m=best[0],
                    old_opening_covered=best[2],
                )
            )
    old_pass = case.get("old_pass", False)
    monotonic = (
        "FAIL_OLD_PASS_UNDERCOVERED"
        if old_pass
        and (not matches or not all(m["old_opening_covered"] for m in matches))
        else (
            "COVERAGE_RETAINED_REVIEW_REQUIRED"
            if matches
            else "NO_ACCEPTED_REFERENCE_PENDING"
        )
    )
    selection = result["berth_selection"]
    profile = result["precision_profiler"]
    row = dict(
        scene=case["scene"],
        family=case.get("family", "UNCLASSIFIED"),
        input_sha256=frame.provenance["wire_sha256"],
        software_sha=result["software_git_sha"],
        source_identity=result["software_source_identity"],
        detected_vessel_count=sum(
            v["classification_status"] == "VESSEL_HYPOTHESIS"
            for v in result["scene_context"]["vessels"]
        ),
        vessel_hypothesis_count=len(result["scene_context"]["vessels"]),
        selected_berth_vessel=selection["active_vessel_id"],
        berth_status=selection["status"],
        ignored_vessels=selection["ignored_vessel_ids"],
        candidate_sources=sorted(
            {
                p["expert"]
                for v in result["per_vessel_status"]
                for p in v.get("proposal_registry", [])
            }
        ),
        proposal_count=profile.get("proposal_count", 0),
        physical_query_count=profile.get("physical_query_count", 0),
        precision_query_count=profile.get("precision_query_count", 0),
        final_hatch_count=result["geometry_only"]["hatch_count"],
        false_exterior_count="PENDING_INDEPENDENT_REVIEW",
        rejected_exterior_count=result["rejected_exterior_candidates"]["total_count"],
        elapsed_s=time.perf_counter() - start,
        performance_budget=result["performance_budget"],
        monotonic_regression=monotonic,
        old_reference_coverage=matches,
        recognition_status=(
            "UNRESOLVED_BERTH"
            if selection["active_vessel_id"] is None
            else "REVIEW_REQUIRED_NOT_CERTIFIED"
        ),
        steel_edge_certified=False,
        output_dir=str(directory),
    )
    (output / "case_summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    progress.finish(True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--timeout-sec", type=float, default=75.0)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.case_index is not None:
        run_case(cases[args.case_index], args.output_dir)
        return 0
    records = []
    for index, case in enumerate(cases):
        target = args.output_dir / case["scene"]
        target.mkdir(parents=True, exist_ok=True)
        with (target / "terminal.log").open("w", encoding="utf-8") as stream:
            try:
                code = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ship_perception.tools.run_hatch_v2_regression",
                        "--manifest",
                        str(args.manifest.resolve()),
                        "--output-dir",
                        str(target.resolve()),
                        "--case-index",
                        str(index),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_sec,
                ).returncode
                row = (
                    json.loads(
                        (target / "case_summary.json").read_text(encoding="utf-8")
                    )
                    if code == 0
                    else dict(
                        scene=case["scene"], status="PROCESS_FAILED", exit_code=code
                    )
                )
            except subprocess.TimeoutExpired:
                row = dict(
                    scene=case["scene"],
                    status="PROCESS_TIMEOUT",
                    elapsed_s=args.timeout_sec,
                )
        records.append(row)
        (args.output_dir / "matrix.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return (
        2
        if any(
            r.get("monotonic_regression", "").startswith("FAIL")
            or r.get("status") in ("PROCESS_FAILED", "PROCESS_TIMEOUT")
            for r in records
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
