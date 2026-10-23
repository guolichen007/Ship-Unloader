"""Immutable finite point arrays and auditable acquisition identity."""

from dataclasses import dataclass, field, InitVar
import hashlib
import json
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PointCloudFrame:
    xyz: np.ndarray
    intensity: Optional[np.ndarray]
    frame_id: str
    stamp_sec: int
    stamp_nanosec: int
    source_topic: str
    fields: Tuple[str, ...]
    source_point_count: int
    provenance_json: str = "{}"
    copy_arrays: InitVar[bool] = True
    _semantic_sha256: str = field(init=False, repr=False)

    def __post_init__(self, copy_arrays):
        source = np.asarray(self.xyz)
        dtype = (
            "<f4" if source.dtype.kind == "f" and source.dtype.itemsize <= 4 else "<f8"
        )
        xyz = np.array(source, dtype=dtype, order="C", copy=copy_arrays)
        if (
            xyz.ndim != 2
            or xyz.shape[1] != 3
            or not len(xyz)
            or not np.isfinite(xyz).all()
            or self.source_point_count < len(xyz)
        ):
            raise ValueError("INVALID_FINITE_XYZ_FRAME")
        if not 0 <= self.stamp_nanosec < 1000000000:
            raise ValueError("INVALID_TIMESTAMP_NANOSECONDS")
        if not copy_arrays and not xyz.flags.owndata:
            raise ValueError("FRAME_REQUIRES_OWNED_ARRAYS")
        identity = json.loads(self.provenance_json)
        if not isinstance(identity, dict):
            raise ValueError("INVALID_FRAME_PROVENANCE")
        object.__setattr__(
            self,
            "provenance_json",
            json.dumps(identity, sort_keys=True, allow_nan=False),
        )
        object.__setattr__(self, "fields", tuple(self.fields))
        xyz.setflags(write=False)
        object.__setattr__(self, "xyz", xyz)
        digest = hashlib.sha256(memoryview(xyz).cast("B"))
        if self.intensity is None:
            digest.update(b"intensity:none")
        else:
            values = np.array(self.intensity, dtype="<f8", order="C", copy=copy_arrays)
            if values.shape != (len(xyz),):
                raise ValueError("INTENSITY_XYZ_ALIGNMENT_ERROR")
            if not copy_arrays and not values.flags.owndata:
                raise ValueError("FRAME_REQUIRES_OWNED_ARRAYS")
            invalid = ~np.isfinite(values)
            if invalid.any():
                if not values.flags.writeable:
                    values = values.copy()
                values[invalid] = np.nan
            digest.update(memoryview(values).cast("B"))
            values.setflags(write=False)
            object.__setattr__(self, "intensity", values)
        object.__setattr__(self, "_semantic_sha256", digest.hexdigest())

    @property
    def sha256(self):
        return self._semantic_sha256

    @property
    def provenance(self):
        return json.loads(self.provenance_json)

    @property
    def stamp(self):
        return "{}.{:09d}".format(self.stamp_sec, self.stamp_nanosec)
