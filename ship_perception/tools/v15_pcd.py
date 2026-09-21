"""严格的开发侧 XYZ 解码；不解析标注，不应用 VIEWPOINT。"""
import hashlib
import json
import io
import struct
from pathlib import Path


class UnsupportedPCD(ValueError):
    pass


def decode(path, *, padding_contracts=None):
    def fail(reason):
        raise UnsupportedPCD("UNSUPPORTED_PCD_FORMAT: " + reason)
    file_bytes=Path(path).read_bytes()
    with io.BytesIO(file_bytes) as stream:
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
        if trailing:
            # The writer signature alone is not authorization to ignore bytes.
            # Bind the exception to a previously audited immutable file identity.
            if padding_contracts is None:
                manifest = Path(__file__).resolve().parents[1]/"datasets/v15_dataset_manifest.json"
                padding_contracts = json.loads(manifest.read_text(encoding="utf-8")).get("pcd_padding_contracts", [])
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            authorized = any(
                item["expected_file_sha256"] == file_hash and
                item["expected_point_payload_bytes"] == n*stride and
                item["expected_trailing_zero_bytes"] == len(trailing)
                for item in padding_contracts)
            padding = 4096-header_bytes
            if not (authorized and len(trailing) == padding and 0 < padding < 4096 and not any(trailing)):
                fail("UNAUTHORIZED_TRAILING_BYTES")
            header["AUDITED_ZERO_TAIL_BYTES"] = [str(padding)]
            header["AUDITED_FILE_SHA256"] = [file_hash]
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
