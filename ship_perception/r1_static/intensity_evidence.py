"""Vessel-local reflectivity evidence; never a steel classifier or hatch seed."""

from typing import Dict, Optional, Tuple

import numpy as np

from .height_grid import R1HeightGrid


AUX_WEIGHT = 0.05


class IntensityEvidence:
    """Robust normalized median raster with measured two-sided edge support."""

    def __init__(
        self, xyz: np.ndarray, intensity: np.ndarray, axes: np.ndarray, cell_m: float
    ):
        valid = np.isfinite(intensity)
        self.grid = None
        self.stats = {"status": "UNAVAILABLE_NO_FINITE_INTENSITY"}
        if not valid.any():
            return
        values = intensity[valid]
        p5, p50, p95 = np.percentile(values, [5, 50, 95])
        self.stats = dict(
            status="AVAILABLE",
            p5=float(p5),
            p50=float(p50),
            p95=float(p95),
            finite_count=int(valid.sum()),
            intensity_mad=float(np.median(abs(values - p50))),
        )
        if p95 - p5 <= np.finfo(float).eps * max(1.0, abs(p95), abs(p5)):
            self.stats["status"] = "UNAVAILABLE_NO_ROBUST_CONTRAST"
            return
        normalized = np.clip((values - p5) / (p95 - p5), 0, 1)
        axial = xyz[valid, :2] @ axes.T
        self.grid = R1HeightGrid.from_points(
            np.column_stack((axial, normalized)), cell_m
        )
        self.stats.update(
            median_intensity=float(np.median(normalized)),
            normalized_mad=float(np.median(abs(normalized - np.median(normalized)))),
        )
        for axis, label in ((1, "grad_u"), (0, "grad_v")):
            delta = np.diff(self.grid.median, axis=axis) / cell_m
            finite = delta[np.isfinite(delta)]
            self.stats[label] = float(np.median(abs(finite))) if len(finite) else None

    def edge(self, axis: int, rho: float, span: Tuple[float, float]) -> Dict:
        """Compare paired occupied cells along an existing geometric side."""
        empty = dict(
            intensity_edge_score=0.0,
            intensity_support_fraction=0.0,
            edge_consistency=0.0,
            paired_cell_count=0,
            median_delta=0.0,
            intensity_used=False,
        )
        grid = self.grid
        if grid is None or span[1] <= span[0]:
            return empty
        along = np.arange(span[0] + grid.cell_m / 2, span[1], grid.cell_m)
        if not len(along):
            return empty
        samples = []
        for sign in (-1, 1):
            xy = np.zeros((len(along), 2))
            xy[:, axis] = rho + sign * grid.cell_m
            xy[:, 1 - axis] = along
            rows, columns = grid.cell_indices(xy)
            inside = (
                (rows >= 0)
                & (rows < grid.count.shape[0])
                & (columns >= 0)
                & (columns < grid.count.shape[1])
            )
            values = np.full(len(along), np.nan)
            values[inside] = grid.median[rows[inside], columns[inside]]
            samples.append(values)
        valid = np.isfinite(samples).all(axis=0)
        if valid.sum() < 3:
            return empty
        delta = (samples[1] - samples[0])[valid]
        support = float(valid.mean())
        signed = float(np.median(delta))
        # Sign agreement penalizes alternating cargo texture; missing cells
        # contribute no contrast and cannot be counted as supporting samples.
        consistency = float(max(np.mean(delta > 0), np.mean(delta < 0)))
        score = float(abs(signed) * consistency * support)
        return dict(
            intensity_edge_score=score,
            intensity_support_fraction=support,
            edge_consistency=consistency,
            paired_cell_count=int(valid.sum()),
            median_delta=signed,
            intensity_used=score > 0,
        )

    def corner(self, uv: np.ndarray, half_length_m: float) -> float:
        """Require local measured contrast in both directions near a corner."""
        support = [
            self.edge(
                axis,
                uv[axis],
                (uv[1 - axis] - half_length_m, uv[1 - axis] + half_length_m),
            )["intensity_edge_score"]
            for axis in (0, 1)
        ]
        return min(support)


def annotate_result(
    report: Dict,
    xyz: np.ndarray,
    intensity: Optional[np.ndarray],
    private: Dict,
    b2: Dict,
    b4: Dict,
    config: Dict,
) -> None:
    """Attach auxiliary side/corner confidence without changing evidence roles."""
    from .static_showcase import _showcase_axes

    for vessel in report["vessels"]:
        ident = vessel["vessel_hypothesis_id"]
        axes, _ = _showcase_axes(ident, b2, b4)
        indexes = private["vessel_point_indexes"][ident]
        scorer = (
            None
            if intensity is None
            else IntensityEvidence(
                xyz[indexes],
                intensity[indexes],
                axes,
                config["geometry"]["candidate_voxel_m"],
            )
        )
        vessel["normalized_intensity_stats"] = (
            scorer.stats if scorer else dict(status="UNAVAILABLE_OR_DISABLED")
        )
        for hatch in vessel["rectangles"]:
            review = hatch.get("height_review") or {}
            local_axes = np.asarray(review.get("axes", axes))
            if scorer is not None and not np.allclose(local_axes, axes):
                scorer = IntensityEvidence(
                    xyz[indexes],
                    intensity[indexes],
                    local_axes,
                    config["geometry"]["candidate_voxel_m"],
                )
                axes = local_axes
            box = (
                np.asarray(review.get("polygon", hatch["polygon_after"])) @ local_axes.T
            )
            low, high = box.min(axis=0), box.max(axis=0)
            sides = {}
            for name, axis, end in (
                ("U0", 0, 0),
                ("U1", 0, 1),
                ("V0", 1, 0),
                ("V1", 1, 1),
            ):
                rho = (low if end == 0 else high)[axis]
                sides[name] = (
                    scorer.edge(axis, rho, (low[1 - axis], high[1 - axis]))
                    if scorer
                    else dict(
                        intensity_edge_score=0.0,
                        intensity_used=False,
                        intensity_support_fraction=0.0,
                    )
                )
            half_length = config["boundary"]["profile_half_length_m"]
            corners = [scorer.corner(uv, half_length) if scorer else 0.0 for uv in box]
            score = float(
                np.mean([row["intensity_edge_score"] for row in sides.values()])
            )
            hatch["intensity_evidence"] = dict(
                sides=sides,
                intensity_corner_scores=corners,
                intensity_score=score,
                auxiliary_score_bonus=AUX_WEIGHT * score,
                intensity_used=any(row["intensity_used"] for row in sides.values()),
                role="AUXILIARY_REVIEW_ONLY_NOT_STEEL",
            )
