#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python 3 is required (set PYTHON_BIN to override)." >&2
    exit 1
fi
# A custom interpreter must expose the sourced ROS 1 Python packages too.
if ! "$PYTHON_BIN" -c 'import rospy, sensor_msgs.msg, numpy, scipy, matplotlib; from jsonschema import Draft202012Validator' >/dev/null 2>&1; then
    echo "ERROR: Source ROS 1 Noetic and install requirements-ros.txt in a ROS-compatible Python environment." >&2
    echo "See docs/ROS_HATCH_ONCE.md for setup instructions." >&2
    exit 1
fi
cd -- "$REPO_ROOT"
exec "$PYTHON_BIN" -m ship_perception.ros.ship_scan_once "$@"
