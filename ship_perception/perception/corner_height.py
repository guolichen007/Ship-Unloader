"""Observed local coaming/deck upper-surface Z for display and export only."""

import math
import numpy as np

from ship_perception.r1_static.local_deck import (
    _plane_from_three,
    _huber_plane,
    _anchor_height,
)


CORNER_SIDES = (("U0", "V0"), ("U1", "V0"), ("U1", "V1"), ("U0", "V1"))


class CornerHeightMeasurement:
    def __init__(self, measurement, config):
        self.measurement, self.config = measurement, config

    def measure(self, result):
        polygon = np.asarray(result["boundary_uv"])
        return [self._corner(polygon, index, result) for index in range(4)]

    def _corner(self, polygon, index, result):
        config = self.config
        corner = polygon[index]
        row = dict(
            z_m=None,
            z_status="UNOBSERVED",
            z_uncertainty_m=None,
            z_definition="LOCAL_COAMING_OR_DECK_UPPER_SURFACE",
            method="RAW_LOCAL_GRAVITY_COMPATIBLE_SURFACE_PLANE",
            raw_source_scope="SELECTED_VESSEL_RAW",
            point_index_sha256=self.measurement.local.vessel.point_index_sha256,
            support_raw_point_count=0,
            near_corner_raw_point_count=0,
            used_for_detection=False,
            steel_edge_certified=False,
        )
        if not all(
            result["sides"][side].get("independent_geometry_support", False)
            for side in CORNER_SIDES[index]
        ):
            row["reason"] = "ADJACENT_BOUNDARY_NOT_INDEPENDENTLY_SUPPORTED"
            return row
        search = config["roi"]["support_search_m"]
        points = self.measurement.spatial.query(
            [*(corner - search), *(corner + search)]
        )
        if not len(points):
            row["reason"] = "NO_LOCAL_RAW_RETURNS"
            return row
        # Keep the external corner quadrant of the measured opening, rather
        # than using cargo or the hold floor to assign an upper-surface Z.
        orientation = (
            1 if np.cross(polygon, np.roll(polygon, -1, axis=0)).sum() >= 0 else -1
        )
        outside = np.ones(len(points), dtype=bool)
        for edge_index in ((index - 1) % 4, index):
            start, end = polygon[edge_index], polygon[(edge_index + 1) % 4]
            inward_distance = (
                orientation
                * np.cross(end - start, points[:, :2] - start)
                / np.linalg.norm(end - start)
            )
            outside &= inward_distance <= 0
        points = points[outside]
        minimum = config["geometry"]["min_plane_points"]
        if len(points) < minimum:
            row["reason"] = "INSUFFICIENT_LOCAL_UPPER_SURFACE_SUPPORT"
            return row
        points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
        tolerance = config["geometry"]["plane_inlier_m"]
        angle = math.radians(config["geometry"]["normal_angle_deg"])
        near = (
            np.linalg.norm(points[:, :2] - corner, axis=1)
            <= config["boundary"]["corner_join_m"]
        )
        rng = np.random.default_rng(2026)
        hypotheses = []
        for _ in range(config["geometry"]["ransac_iterations"]):
            plane = _plane_from_three(
                points[rng.choice(len(points), 3, replace=False)], angle
            )
            if plane is None:
                continue
            mask = abs(points @ plane[0] + plane[1]) <= tolerance
            if (
                mask.sum() < minimum
                or mask.mean() < config["roi"]["min_enclosure_ratio"]
                or np.count_nonzero(mask & near) < config["boundary"]["line_min_points"]
            ):
                continue
            height = _anchor_height(plane, corner)
            if any(abs(height - old[0]) <= tolerance for old in hypotheses):
                continue
            refined = _huber_plane(points[mask], plane, tolerance, angle)
            if refined is None:
                continue
            residual = abs(points @ refined[0] + refined[1])
            inliers = residual <= tolerance
            if np.count_nonzero(inliers & near) < config["boundary"]["line_min_points"]:
                continue
            hypotheses.append(
                (_anchor_height(refined, corner), refined, inliers, residual)
            )
        if not hypotheses:
            row["reason"] = "NO_UNIQUE_OBSERVED_UPPER_SURFACE_AT_CORNER"
            return row
        if np.ptp([h[0] for h in hypotheses]) > tolerance:
            row["z_status"] = "AMBIGUOUS"
            row["reason"] = "MULTIPLE_LOCAL_SURFACE_HEIGHTS"
            return row
        height, plane, inliers, residual = max(hypotheses, key=lambda h: h[2].sum())
        xy_uncertainty = result.get("corners", [{}] * 4)[index].get(
            "uncertainty_m", 0.0
        )
        uncertainty = max(
            tolerance,
            float(np.percentile(residual[inliers], 95) / plane[0][2]),
            float(np.linalg.norm(plane[0][:2]) / plane[0][2] * xy_uncertainty),
        )
        row.update(
            z_m=float(height),
            z_status="OBSERVED_LOCAL_SURFACE_ESTIMATE",
            z_uncertainty_m=uncertainty,
            support_raw_point_count=int(inliers.sum()),
            near_corner_raw_point_count=int(np.count_nonzero(inliers & near)),
            local_plane_uvz=dict(normal=plane[0].tolist(), offset_m=float(plane[1])),
            reason="MEASURED_EXTERNAL_CORNER_SURFACE_WITH_NEARBY_RAW_SUPPORT",
        )
        return row
