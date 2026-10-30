"""One immutable review frame and explicit normal-form line comparisons."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ReviewFrame:
    vessel_id: str
    source_frame: str
    axis_matrix: tuple

    def __post_init__(self):
        axes = np.asarray(self.axis_matrix, dtype=float)
        numerical_tolerance = 64 * np.finfo(float).eps
        if (
            axes.shape != (2, 2)
            or not np.isfinite(axes).all()
            or not np.allclose(
                axes @ axes.T, np.eye(2), rtol=0, atol=numerical_tolerance
            )
        ):
            raise ValueError("FRAME_CONTRACT_VIOLATION_NON_ORTHONORMAL_REVIEW_AXES")

    def to_review(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, float) @ np.asarray(self.axis_matrix).T

    def to_input(self, uv: np.ndarray) -> np.ndarray:
        return np.asarray(uv, float) @ np.asarray(self.axis_matrix)

    def audit(self) -> dict:
        return dict(
            coordinate_frame="VESSEL_REVIEW_UV",
            source_frame=self.source_frame,
            axis_source="B1_VESSEL_WIDE_AXIS_ANCHOR_NOT_GROUND_TRUTH",
            axis_matrix=[list(row) for row in self.axis_matrix],
            vessel_id=self.vessel_id,
            transform_applied="p_review=p_input_xy @ axis_matrix.T",
            transform_count=1,
            rho_contract="dot(point_review,unit_normal_review)=rho_m",
            along_contract="dot(point_review,line_direction_review)=along_m",
        )


def unit_line(line: Mapping):
    normal = np.asarray(line["normal_uv"], float)
    rho = float(line.get("rho", line.get("final_rho")))
    length = float(np.linalg.norm(normal))
    if (
        normal.shape != (2,)
        or not np.isfinite(normal).all()
        or not np.isfinite(rho)
        or length == 0
    ):
        raise ValueError("FRAME_CONTRACT_VIOLATION_INVALID_LINE")
    return normal / length, rho / length


def line_points(line: Mapping) -> np.ndarray:
    normal, rho = unit_line(line)
    direction = np.array([-normal[1], normal[0]])
    return rho * normal + np.asarray(line["support_span"])[:, None] * direction


def line_displacements(
    previous: Mapping, candidate: Mapping, stations: np.ndarray
) -> np.ndarray:
    """Physical displacements along the previous normal at fixed stations.

    A scalar rho difference is valid only for equal normals. Reconstructing
    points first makes this operation commute with translations and rotations.
    """
    old_normal, old_rho = unit_line(previous)
    new_normal, new_rho = unit_line(candidate)
    denominator = float(new_normal @ old_normal)
    if abs(denominator) <= np.finfo(float).eps:
        raise ValueError("FRAME_CONTRACT_VIOLATION_ORTHOGONAL_SIDE_MODES")
    stations = np.asarray(stations, float)
    points = stations + (old_rho - stations @ old_normal)[:, None] * old_normal
    return (new_rho - points @ new_normal) / denominator


def line_coordinate_at(line: Mapping, station: np.ndarray, axis: int) -> float:
    normal, rho = unit_line(line)
    if abs(normal[axis]) <= np.finfo(float).eps:
        raise ValueError("FRAME_CONTRACT_VIOLATION_INVALID_SIDE_AXIS")
    other = 1 - axis
    return float((rho - normal[other] * station[other]) / normal[axis])
