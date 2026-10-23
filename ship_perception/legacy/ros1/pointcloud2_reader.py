"""Strict PointCloud2 byte decoder with lazy ROS 1 imports."""

import math
from typing import Any

import numpy as np

from .ros_input_adapter import PointCloudFrame


def decode_pointcloud2(message: Any, topic: str = "/ship_scan/map") -> PointCloudFrame:
    """Honor PointField offsets, endian, point stride and organized row padding."""
    width, height = int(message.width), int(message.height)
    step, row_step = int(message.point_step), int(message.row_step)
    if width <= 0 or height <= 0 or step <= 0 or row_step < width * step:
        raise ValueError("INVALID_POINTCLOUD2_LAYOUT")
    if len(message.data) != row_step * height:
        raise ValueError("POINTCLOUD2_DATA_SIZE_MISMATCH")
    formats = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    names, dtypes, offsets = [], [], []
    fields = tuple(field.name for field in message.fields)
    if len(set(fields)) != len(fields):
        raise ValueError("DUPLICATE_POINTCLOUD2_FIELD")
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError("POINTCLOUD2_MISSING_XYZ")
    for field in message.fields:
        if field.name not in ("x", "y", "z", "intensity"):
            continue
        if field.count != 1 or field.datatype not in formats:
            raise ValueError("UNSUPPORTED_SCALAR_FIELD:" + field.name)
        dtype = np.dtype(
            (">" if message.is_bigendian else "<") + formats[field.datatype]
        )
        if field.offset < 0 or field.offset + dtype.itemsize > step:
            raise ValueError("POINTCLOUD2_FIELD_OUT_OF_BOUNDS:" + field.name)
        names.append(field.name)
        dtypes.append(dtype)
        offsets.append(field.offset)
    dtype = np.dtype(dict(names=names, formats=dtypes, offsets=offsets, itemsize=step))
    records = np.ndarray(
        (height, width),
        dtype=dtype,
        buffer=bytes(message.data),
        strides=(row_step, step),
    )
    xyz = np.column_stack([records[name].reshape(-1) for name in ("x", "y", "z")])
    valid = np.isfinite(xyz).all(axis=1)
    intensity = (
        np.asarray(records["intensity"], dtype=float).reshape(-1)[valid]
        if "intensity" in fields
        else None
    )
    return PointCloudFrame(
        xyz[valid],
        intensity,
        message.header.frame_id,
        int(message.header.stamp.secs),
        int(message.header.stamp.nsecs),
        topic,
        fields,
        width * height,
    )


def read_once(topic: str, timeout_sec: float) -> PointCloudFrame:
    """Wait for exactly one complete ROS 1 message; never concatenate frames."""
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError("TIMEOUT_SEC_MUST_BE_POSITIVE")
    try:
        import rospy
        from sensor_msgs.msg import PointCloud2
    except ImportError as error:
        raise RuntimeError(
            "ROS 1 is unavailable. Source /opt/ros/noetic/setup.bash and your workspace."
        ) from error
    if not rospy.core.is_initialized():
        rospy.init_node(
            "ship_hatch_once",
            anonymous=True,
            disable_signals=True,
            disable_rostime=True,
        )
    advertised = dict(rospy.get_published_topics())
    resolved_topic = rospy.resolve_name(topic)
    if (
        resolved_topic in advertised
        and advertised[resolved_topic] != "sensor_msgs/PointCloud2"
    ):
        raise ValueError("TOPIC_TYPE_MUST_BE_SENSOR_MSGS_POINTCLOUD2:" + resolved_topic)
    try:
        message = rospy.wait_for_message(
            resolved_topic, PointCloud2, timeout=timeout_sec
        )
    except rospy.ROSException as error:
        raise RuntimeError(
            "No PointCloud2 received from {} within {} seconds: {}".format(
                resolved_topic, timeout_sec, error
            )
        ) from error
    return decode_pointcloud2(message, resolved_topic)
