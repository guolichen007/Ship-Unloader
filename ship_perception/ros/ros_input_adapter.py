"""Validated, aligned XYZ/intensity frame shared with the offline pipeline."""

from dataclasses import dataclass
import hashlib
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PointCloudFrame:
    xyz: np.ndarray
    intensity: Optional[np.ndarray]
    frame_id: str
    stamp_secs: int
    stamp_nsecs: int
    source_topic: str
    fields: Tuple[str, ...]
    source_point_count: int

    def __post_init__(self) -> None:
        source = np.asarray(self.xyz)
        # Keep FLOAT32 arithmetic identical to the existing strict PCD chain;
        # promote only integer inputs and preserve incoming FLOAT64 precision.
        coordinate_dtype = (
            "<f4" if source.dtype.kind == "f" and source.dtype.itemsize <= 4 else "<f8"
        )
        xyz = np.array(source, dtype=coordinate_dtype, order="C", copy=True)
        if (
            xyz.ndim != 2
            or xyz.shape[1] != 3
            or not len(xyz)
            or not np.isfinite(xyz).all()
        ):
            raise ValueError("INVALID_FINITE_XYZ_FRAME")
        xyz.setflags(write=False)
        object.__setattr__(self, "xyz", xyz)
        if self.intensity is not None:
            values = np.array(self.intensity, dtype="<f8", copy=True)
            if values.shape != (len(xyz),):
                raise ValueError("INTENSITY_XYZ_ALIGNMENT_ERROR")
            # Nonfinite intensity is missing evidence, never a reason to drop XYZ.
            values[~np.isfinite(values)] = np.nan
            values.setflags(write=False)
            object.__setattr__(self, "intensity", values)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256(self.xyz.tobytes())
        digest.update(
            b"intensity:none" if self.intensity is None else self.intensity.tobytes()
        )
        return digest.hexdigest()

    @property
    def stamp(self) -> str:
        return "{}.{:09d}".format(self.stamp_secs, self.stamp_nsecs)
