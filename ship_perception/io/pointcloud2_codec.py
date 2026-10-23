"""ROS-free PointCloud2 layout decoder; accepts ROS2-shaped message objects."""

import hashlib
import json

import numpy as np

from .frame import PointCloudFrame


def decode_pointcloud2(message, topic="/ship_scan/map", max_points=12000000):
    """Validate bounds before allocation; preserve finite XYZ/intensity alignment."""
    width, height = int(message.width), int(message.height)
    step, row_step = int(message.point_step), int(message.row_step)
    if width <= 0 or height <= 0 or step <= 0 or row_step < width * step:
        raise ValueError("INVALID_POINTCLOUD2_LAYOUT")
    if width * height > max_points:
        raise ValueError("POINTCLOUD2_POINT_LIMIT_EXCEEDED")
    wire = memoryview(message.data)
    if not wire.c_contiguous or wire.nbytes != row_step * height:
        raise ValueError("POINTCLOUD2_DATA_SIZE_MISMATCH")
    wire = wire.cast("B")
    formats = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    fields = tuple(str(f.name) for f in message.fields)
    if len(set(fields)) != len(fields):
        raise ValueError("DUPLICATE_POINTCLOUD2_FIELD")
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError("POINTCLOUD2_MISSING_XYZ")
    names, dtypes, offsets, layout = [], [], [], []
    intervals = []
    for f in message.fields:
        if f.datatype not in formats or int(f.count) <= 0:
            raise ValueError("UNSUPPORTED_POINTCLOUD2_FIELD:" + f.name)
        dtype = np.dtype((">" if message.is_bigendian else "<") + formats[f.datatype])
        start, end = int(f.offset), int(f.offset) + dtype.itemsize * int(f.count)
        if start < 0 or end > step:
            raise ValueError("POINTCLOUD2_FIELD_OUT_OF_BOUNDS:" + f.name)
        layout.append(
            dict(
                name=f.name, offset=start, datatype=int(f.datatype), count=int(f.count)
            )
        )
        if f.name not in ("x", "y", "z", "intensity"):
            continue
        if f.count != 1:
            raise ValueError("UNSUPPORTED_SCALAR_FIELD:" + f.name)
        if any(start < b and end > a for a, b in intervals):
            raise ValueError("OVERLAPPING_POINTCLOUD2_SCALARS")
        intervals.append((start, end))
        names.append(f.name)
        dtypes.append(dtype)
        offsets.append(start)
    dtype = np.dtype(dict(names=names, formats=dtypes, offsets=offsets, itemsize=step))
    records = np.ndarray(
        (height, width), dtype=dtype, buffer=wire, strides=(row_step, step)
    )
    xyz = np.column_stack([records[name].reshape(-1) for name in ("x", "y", "z")])
    valid = np.isfinite(xyz).all(axis=1)
    if not valid.any():
        raise ValueError("NO_FINITE_XYZ")
    if not valid.all():
        xyz = xyz[valid]
    intensity = None
    if "intensity" in fields:
        intensity = np.array(
            records["intensity"].reshape(-1)[valid], dtype="<f8", copy=True
        )
    descriptor = dict(
        width=width,
        height=height,
        point_step=step,
        row_step=row_step,
        is_bigendian=bool(message.is_bigendian),
        fields=layout,
    )
    layout_json = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    wire_sha = hashlib.sha256(wire).hexdigest()
    identity = dict(
        transport="ROS2",
        topic=topic,
        frame_id=str(message.header.frame_id),
        stamp_sec=int(message.header.stamp.sec),
        stamp_nanosec=int(message.header.stamp.nanosec),
        wire_sha256=wire_sha,
        layout_sha256=hashlib.sha256(layout_json.encode()).hexdigest(),
        wire_bytes=wire.nbytes,
        layout=descriptor,
        intensity_source_calibration="UNKNOWN",
        intensity_dtype=(
            dtype.fields["intensity"][0].str if "intensity" in fields else None
        ),
    )
    identity["raw_message_identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    return PointCloudFrame(
        xyz,
        intensity,
        identity["frame_id"],
        identity["stamp_sec"],
        identity["stamp_nanosec"],
        topic,
        fields,
        width * height,
        json.dumps(identity),
        copy_arrays=False,
    )
