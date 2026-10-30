"""Shared English CLI and the ROS-independent PCD entry point."""

import argparse
from pathlib import Path
import sys

from ship_perception.tools.stage_progress import StageProgress
from ship_perception.tools.cli_environment import configure_numerical_environment

configure_numerical_environment()


def parser(description):
    command = argparse.ArgumentParser(description=description)
    command.add_argument(
        "--output-dir", "--output_dir", type=Path, default=Path("output")
    )
    command.add_argument("--save-snapshot", "--save_snapshot", action="store_true")
    command.add_argument(
        "--berth-reference",
        type=Path,
        help="Surveyed site reference in the input frame; no ship priors",
    )
    command.add_argument(
        "--pipeline",
        choices=("v2", "v1-diagnostic"),
        default="v2",
        help="V2 production core or explicitly historical diagnostic comparison",
    )
    command.add_argument(
        "--no-legacy-evidence",
        action="store_true",
        help="Disable independent historical geometry comparison; local recall still runs",
    )
    command.add_argument("--mode", choices=("review-first", "bounded"), default="bounded",
                         help="Review first or bounded measurement; default changes only after Normal6 passes")
    command.add_argument("--operator-vessel-roi", type=Path,
                         help="Explicit operator vessel index scope; original input is retained")
    return command


def execute(frame, args, progress):
    from .pipeline import Timing

    if getattr(args, "pipeline", "v2") == "v1-diagnostic":
        from .pipeline import run_perception

        options = {}
    else:
        from .pipeline_v2 import run_perception

        options = dict(berth_reference=getattr(args, "berth_reference", None),
                       mode=getattr(args,"mode","bounded"),
                       operator_vessel_roi=getattr(args,"operator_vessel_roi",None))
    from .output import save_output

    from .review_runtime import ReviewRunMonitor
    from .pipeline import source_identity
    with ReviewRunMonitor(args.output_dir,dict(source_identity=source_identity(),
                          input_sha256=frame.provenance.get("wire_sha256",frame.sha256))) as monitor:
        if getattr(args,"pipeline","v2")=="v2":options['monitor']=monitor
        payload = run_perception(frame, progress, legacy_evidence=not args.no_legacy_evidence, **options)
        monitor.checkpoint(stage="RENDER_AND_SAVE",review_objects=payload['geometry_only']['hatches'])
        directory = save_output(payload, frame, args.output_dir, args.save_snapshot, Timing(progress))
    progress.finish(True)
    print(
        "HATCH_REVIEW_COUNT=" + str(payload["geometry_only"]["hatch_count"]), flush=True
    )
    print(
        "HATCH_COUNT_STATUS=" + payload["geometry_only"]["hatch_count_status"],
        flush=True,
    )
    print("INTENSITY_STATUS=" + payload["intensity_status"]["status"], flush=True)
    print(
        "UNRESOLVED_PROPOSAL_COUNT="
        + str(
            sum(
                v["unresolved_review_proposals"]["total_count"]
                for v in payload["per_vessel_status"]
            )
        ),
        flush=True,
    )
    print("OUTPUT_DIR=" + str(directory), flush=True)
    print("HATCH_CORNERS_JSON=" + str(directory / "05_hatch_corners.json"), flush=True)
    return 0


def failed(error, args, progress, stage):
    from .output import save_failure

    directory = save_failure(
        args.output_dir, error, progress.first_failed_stage or stage
    )
    progress.finish(False)
    print("ERROR=" + str(error), file=sys.stderr)
    print("FAILURE_DIAGNOSTICS=" + str(directory), file=sys.stderr)
    return 130 if isinstance(error, KeyboardInterrupt) else 1


def pcd_main():
    command = parser("Read one PCD and generate hatch review images and JSON")
    command.add_argument("--pcd", required=True, type=Path)
    command.add_argument("--frame-id", default="pcd")
    args = command.parse_args()
    progress = StageProgress()
    stage = "READ_PCD"
    try:
        from ship_perception.io.pcd_source import read_pcd

        with progress.stage(stage):
            frame = read_pcd(args.pcd, args.frame_id)
        stage = "PERCEPTION"
        return execute(frame, args, progress)
    except (Exception, KeyboardInterrupt) as error:
        return failed(error, args, progress, stage)


if __name__ == "__main__":
    raise SystemExit(pcd_main())
