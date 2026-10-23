"""Discover compatible QoS and consume exactly one complete ROS2 map message."""

import json
import math
import os
import time
from dataclasses import replace

from ship_perception.io.pointcloud2_codec import decode_pointcloud2


class CaptureError(RuntimeError):
    def __init__(self, code, diagnostics):
        super().__init__(code)
        self.code, self.diagnostics = code, diagnostics


def _qos_dict(qos):
    return dict(
        reliability=qos.reliability.name,
        durability=qos.durability.name,
        history=qos.history.name,
        depth=qos.depth,
    )


def read_once(
    topic="/ship_scan/map", timeout_sec=30.0, qos_mode="auto", max_points=12000000
):
    """Own a private context/node and clean them on success, timeout or Ctrl+C."""
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError("INVALID_CAPTURE_TIMEOUT")
    if qos_mode not in ("auto", "map", "sensor", "reliable"):
        raise ValueError("INVALID_QOS_MODE")
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from rclpy.qos import qos_check_compatible, QoSCompatibility
    from sensor_msgs.msg import PointCloud2

    context = Context()
    rclpy.init(args=[], context=context)
    node = Node("ship_hatch_capture", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    started = time.monotonic()
    deadline = started + timeout_sec
    subscription = None
    diagnostics = dict(
        transport="ROS2",
        ros_distro=os.getenv("ROS_DISTRO", "UNKNOWN"),
        rmw=rclpy.get_rmw_implementation_identifier(),
        topic=topic,
        ros_domain_id=os.getenv("ROS_DOMAIN_ID", "0"),
        ros_localhost_only=os.getenv("ROS_LOCALHOST_ONLY", "0"),
        message_type="sensor_msgs/msg/PointCloud2",
        qos_mode=qos_mode,
        publisher_count=0,
        offered_qos=[],
        selected_qos=None,
        messages_consumed=0,
        warnings=[],
    )
    try:
        endpoints = []
        while time.monotonic() < deadline:
            endpoints = node.get_publishers_info_by_topic(topic)
            if endpoints:
                break
            executor.spin_once(
                timeout_sec=min(0.1, max(0.0, deadline - time.monotonic()))
            )
        diagnostics["publisher_count"] = len(endpoints)
        diagnostics["offered_qos"] = [
            dict(
                node_name=e.node_name,
                node_namespace=e.node_namespace,
                topic_type=e.topic_type,
                endpoint_gid=list(e.endpoint_gid),
                **_qos_dict(e.qos_profile),
            )
            for e in endpoints
        ]
        if not endpoints:
            raise CaptureError("NO_PUBLISHER", diagnostics)
        if any(e.topic_type != "sensor_msgs/msg/PointCloud2" for e in endpoints):
            raise CaptureError("WRONG_TOPIC_TYPE", diagnostics)
        if len(endpoints) != 1:
            raise CaptureError("MULTIPLE_PUBLISHERS_AMBIGUOUS", diagnostics)
        offered = endpoints[0].qos_profile
        if qos_mode == "auto":
            if offered.reliability not in (
                ReliabilityPolicy.RELIABLE,
                ReliabilityPolicy.BEST_EFFORT,
            ) or offered.durability not in (
                DurabilityPolicy.VOLATILE,
                DurabilityPolicy.TRANSIENT_LOCAL,
            ):
                raise CaptureError("UNRESOLVED_PUBLISHER_QOS", diagnostics)
            reliability, durability = offered.reliability, offered.durability
        else:
            reliability = (
                ReliabilityPolicy.BEST_EFFORT
                if qos_mode == "sensor"
                else ReliabilityPolicy.RELIABLE
            )
            durability = (
                DurabilityPolicy.TRANSIENT_LOCAL
                if qos_mode == "map"
                else DurabilityPolicy.VOLATILE
            )
        selected = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=reliability,
            durability=durability,
        )
        diagnostics["selected_qos"] = _qos_dict(selected)
        compatibility, reason = qos_check_compatible(offered, selected)
        if compatibility == QoSCompatibility.ERROR:
            diagnostics["incompatibility_reason"] = reason
            raise CaptureError("INCOMPATIBLE_QOS", diagnostics)
        if offered.durability == DurabilityPolicy.VOLATILE:
            warning = "MAP_IS_VOLATILE: late subscriber cannot rely on historical delivery; waiting for next publish"
            diagnostics["warnings"].append(warning)
            print(warning, flush=True)
        received = []

        def callback(message):
            if not received:
                received.append(message)

        subscription = node.create_subscription(PointCloud2, topic, callback, selected)
        while not received and time.monotonic() < deadline:
            executor.spin_once(
                timeout_sec=min(0.1, max(0.0, deadline - time.monotonic()))
            )
        if not received:
            raise CaptureError("CAPTURE_TIMEOUT", diagnostics)
        node.destroy_subscription(subscription)
        subscription = None
        diagnostics.update(
            messages_consumed=1, capture_wall_time_sec=time.monotonic() - started
        )
        frame = decode_pointcloud2(received[0], topic=topic, max_points=max_points)
        provenance = frame.provenance
        provenance["runtime"] = diagnostics
        return replace(
            frame,
            provenance_json=json.dumps(provenance, sort_keys=True),
            copy_arrays=False,
        )
    finally:
        diagnostics["capture_wall_time_sec"] = time.monotonic() - started
        if subscription is not None:
            node.destroy_subscription(subscription)
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        context.try_shutdown()
