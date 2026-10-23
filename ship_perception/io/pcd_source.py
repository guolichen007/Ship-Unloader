"""Offline PCD source with the same finite XYZ/intensity contract as ROS2."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .frame import PointCloudFrame


def read_pcd(path, frame_id="pcd", max_points=12000000):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    wire_sha = digest.hexdigest()
    with path.open("rb") as stream:
        header = {}
        for _ in range(64):
            line = stream.readline(4097)
            if not line or len(line) > 4096:
                raise ValueError("INVALID_PCD_HEADER")
            words = line.decode("ascii").strip().split()
            if not words or words[0].startswith("#"):
                continue
            if words[0] in header:
                raise ValueError("DUPLICATE_PCD_HEADER")
            header[words[0]] = words[1:]
            if words[0] == "DATA":
                break
        names = header["FIELDS"]
        sizes, counts = list(map(int, header["SIZE"])), list(
            map(int, header.get("COUNT", ["1"] * len(names)))
        )
        n, width, height = (
            int(header["POINTS"][0]),
            int(header["WIDTH"][0]),
            int(header["HEIGHT"][0]),
        )
        if n <= 0 or n != width * height or n > max_points:
            raise ValueError("PCD_POINT_LIMIT_OR_LAYOUT_INVALID")
        selected_names = [name for name in names if name != "_"]
        if len(set(selected_names)) != len(selected_names) or not {
            "x",
            "y",
            "z",
        }.issubset(names):
            raise ValueError("PCD_MISSING_OR_DUPLICATE_FIELD")
        if not len(names) == len(sizes) == len(counts) == len(header["TYPE"]):
            raise ValueError("PCD_FIELD_LAYOUT_INVALID")
        formats = {
            ("F", 4): "<f4",
            ("F", 8): "<f8",
            ("I", 1): "i1",
            ("U", 1): "u1",
            ("I", 2): "<i2",
            ("U", 2): "<u2",
            ("I", 4): "<i4",
            ("U", 4): "<u4",
        }
        dtype_fields = []
        for field_index, (name, size, count, kind) in enumerate(
            zip(names, sizes, counts, header["TYPE"])
        ):
            if (
                count <= 0
                or (kind, size) not in formats
                or (name in ("x", "y", "z", "intensity") and count != 1)
            ):
                raise ValueError("UNSUPPORTED_PCD_FIELD_LAYOUT")
            storage_name = "_padding%d" % field_index if name == "_" else name
            dtype_fields.append(
                (storage_name, formats[kind, size], (count,))
                if count > 1
                else (storage_name, formats[kind, size])
            )
        dtype = np.dtype(dtype_fields)
        if header["DATA"] == ["binary"]:
            records = np.fromfile(stream, dtype=dtype, count=n)
            tail = stream.read()
            if len(records) != n:
                raise ValueError("PCD_PAYLOAD_SIZE_MISMATCH")
            if tail:
                # Preserve the repository's immutable, SHA-bound PCL zero-tail exception.
                from ship_perception.tools.v15_pcd import decode

                decode(path)
        elif header["DATA"] == ["ascii"] and all(c == 1 for c in counts):
            table = np.loadtxt(stream, ndmin=2)
            if table.shape != (n, len(names)):
                raise ValueError("PCD_ASCII_LAYOUT_INVALID")
            records = np.empty(n, dtype=dtype)
            for index, name in enumerate(dtype.names):
                records[name] = table[:, index]
        else:
            raise ValueError("UNSUPPORTED_PCD_DATA_ENCODING")
    xyz = np.column_stack([records[k] for k in ("x", "y", "z")])
    finite = np.isfinite(xyz).all(axis=1)
    intensity = records["intensity"][finite] if "intensity" in names else None
    return PointCloudFrame(
        xyz[finite],
        intensity,
        frame_id,
        0,
        0,
        "",
        tuple(names),
        n,
        json.dumps(
            dict(
                transport="PCD",
                path=str(path),
                wire_sha256=wire_sha,
                pcd_header=header,
                intensity_source_calibration="UNKNOWN",
                intensity_dtype=(
                    dtype.fields["intensity"][0].str if "intensity" in names else None
                ),
            )
        ),
    )
