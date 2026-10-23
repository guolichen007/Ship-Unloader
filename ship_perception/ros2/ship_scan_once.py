"""ROS2 capture adapter; all recognition is transport-neutral."""

from ship_perception.perception.cli import parser, execute, failed
from ship_perception.tools.stage_progress import StageProgress
from .pointcloud2_source import read_once


def main():
    command = parser("Capture one ROS2 PointCloud2 map and run hatch review")
    command.add_argument("--topic", default="/ship_scan/map")
    command.add_argument(
        "--qos", choices=("auto", "map", "sensor", "reliable"), default="auto"
    )
    command.add_argument("--timeout-sec", "--timeout_sec", type=float, default=30.0)
    args = command.parse_args()
    progress = StageProgress()
    stage = "CAPTURE_POINTCLOUD2"
    try:
        with progress.stage(stage):
            frame = read_once(args.topic, args.timeout_sec, args.qos)
        stage = "PERCEPTION"
        return execute(frame, args, progress)
    except (Exception, KeyboardInterrupt) as error:
        return failed(error, args, progress, stage)


if __name__ == "__main__":
    raise SystemExit(main())
