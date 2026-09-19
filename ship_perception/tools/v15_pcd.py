"""严格的开发侧 XYZ 解码；不解析标注，不应用 VIEWPOINT。"""
import hashlib
import struct
from pathlib import Path


class UnsupportedPCD(ValueError):
    pass


def decode(path):
    def fail(reason):
        raise UnsupportedPCD("UNSUPPORTED_PCD_FORMAT: " + reason)
    with Path(path).open("rb") as stream:
        header = {}
        for _ in range(64):
            line = stream.readline(4097)
            if not line or len(line) > 4096:
                fail("HEADER")
            try:
                words = line.decode("ascii").strip().split()
            except UnicodeDecodeError:
                fail("HEADER_ENCODING")
            if not words or words[0].startswith("#"):
                continue
            if words[0] in header:
                fail("DUPLICATE_HEADER")
            header[words[0]] = words[1:]
            if words[0] == "DATA":
                break
        if header.get("DATA") != ["binary"]:
            fail("DATA")
        try:
            fields, types = header["FIELDS"], header["TYPE"]
            sizes = list(map(int, header["SIZE"]))
            counts = list(map(int, header["COUNT"]))
            n = int(header["POINTS"][0])
            width, height = int(header["WIDTH"][0]), int(header["HEIGHT"][0])
        except (KeyError, ValueError, IndexError):
            fail("HEADER_VALUES")
        if not (len(fields) == len(types) == len(sizes) == len(counts)):
            fail("FIELD_LENGTH")
        if n <= 0 or width <= 0 or height <= 0 or width * height != n:
            fail("POINT_COUNT")
        offsets, stride = {}, 0
        for field, typ, size, count in zip(fields, types, sizes, counts):
            if count <= 0 or count > 16:
                fail("COUNT")
            if field in ("x", "y", "z"):
                if field in offsets or (typ, size, count) != ("F", 4, 1):
                    fail("XYZ_LAYOUT")
                offsets[field] = stride
            elif not ((field == "_" and (typ, size) == ("U", 1)) or
                      (field == "intensity" and (typ, size, count) == ("F", 4, 1))):
                fail("FIELD_TYPE")
            stride += size * count
        if set(offsets) != {"x", "y", "z"}:
            fail("MISSING_XYZ")
        header_bytes = stream.tell()
        payload = stream.read()
        trailing = payload[n * stride:]
        # Audited PCL mmap writer layout: allocation includes a 4096-byte header
        # reservation, while data starts immediately after the text header.
        padding = 4096-header_bytes
        if len(payload) == n*stride+padding and 0 < padding < 4096 and not any(trailing):
            header["AUDITED_ZERO_TAIL_BYTES"] = [str(padding)]
            payload = payload[:n*stride]
        if len(payload) != n * stride:
            fail("PAYLOAD_LENGTH")
    # numpy stays in the tool lane; core C++ and Ubuntu PCL do not depend on it.
    import numpy as np
    points = np.empty((n, 3), dtype="<f4")
    for axis, field in enumerate(("x", "y", "z")):
        points[:, axis] = np.ndarray((n,), dtype="<f4", buffer=payload,
                                    offset=offsets[field], strides=(stride,))
    if not np.isfinite(points).all():
        fail("NONFINITE_XYZ")
    points[points == 0] = 0  # canonical +0, including negative zero
    return points, header


def digest(points):
    import numpy as np
    p = np.array(points, dtype="<f4", copy=True)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("INVALID_XYZ")
    p[p == 0] = 0
    order = np.lexsort((p[:, 2], p[:, 1], p[:, 0]))
    return dict(point_count=len(p), finite_count=len(p),
                ordered_xyz_sha256=hashlib.sha256(p.tobytes()).hexdigest(),
                unordered_xyz_sha256=hashlib.sha256(p[order].tobytes()).hexdigest(),
                bbox_xyz=[p.min(axis=0).tolist(), p.max(axis=0).tolist()])


def write_cache(points, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as stream:
        stream.write(b"SXYZV15\0")
        stream.write(struct.pack("<Q", len(points)))
        stream.write(points.astype("<f4", copy=False).tobytes())
