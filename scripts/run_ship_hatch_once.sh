#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
echo "[ENV] Checking ROS2 and numerical dependencies..."
if ! "$PYTHON_BIN" -s -c 'import rclpy, sensor_msgs.msg, numpy, scipy, matplotlib; from jsonschema import Draft202012Validator'; then
    echo "ERROR: Source /opt/ros/humble/setup.bash and install requirements from the ROS2 field guide." >&2
    exit 1
fi
cd -- "$REPO_ROOT"
echo "[ENV] READY transport=ROS2 distro=${ROS_DISTRO:-UNKNOWN}"
exec "$PYTHON_BIN" -s -m ship_perception.ros2.ship_scan_once "$@"
