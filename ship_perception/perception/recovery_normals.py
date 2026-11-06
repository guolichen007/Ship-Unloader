"""Shared physical 3-D neighbourhoods for the separate rim recovery path."""

import numpy as np
from scipy.spatial import cKDTree


def row_digest(indexes):
    import hashlib

    return hashlib.sha256(
        np.sort(indexes).astype("<i8", copy=False).tobytes()
    ).hexdigest()


class SpatialRawNormals:
    """Balance neighbourhood nominations; final eligibility uses original Raw."""

    def __init__(self, points, indexes, config):
        self.points, self.ids, self.config = points, indexes, config
        step = config["geometry"]["candidate_voxel_m"]
        scaled = (points - points.min(0)) / step
        cells = np.floor(
            scaled + 64 * np.finfo(float).eps * np.maximum(1.0, abs(scaled))
        ).astype(np.int64)
        order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
        _, at = np.unique(cells[order], axis=0, return_index=True)
        self.representatives = order[at]
        self.tree = cKDTree(points[self.representatives])
        self.order = np.argsort(indexes, kind="stable")
        self.computed = np.zeros(len(points), bool)
        self.valid = np.zeros(len(points), bool)
        self.normals = np.zeros_like(points)
        self.source_digest = row_digest(indexes[self.representatives])

    def positions(self, indexes):
        at = np.searchsorted(self.ids[self.order], indexes)
        if np.any(at >= len(self.ids)) or not np.array_equal(
            self.ids[self.order][at], indexes
        ):
            raise ValueError("RECOVERY_NORMAL_SOURCE_OUTSIDE_CURRENT_CELL_RAW")
        return self.order[at]

    def estimate(self, indexes):
        g = self.config["geometry"]
        pending = np.unique(np.asarray(indexes, dtype=np.int64))
        pending = pending[~self.computed[pending]]
        for start in range(0, len(pending), 8192):
            ids = pending[start : start + 8192]
            distance, near = self.tree.query(
                self.points[ids],
                k=g["normal_k"],
                distance_upper_bound=g["normal_radius_m"],
                workers=1,
            )
            live = np.isfinite(distance)
            count = live.sum(1)
            neighbors = self.points[self.representatives[np.where(live, near, 0)]]
            delta = neighbors - self.points[ids, None, :]
            mean = (delta * live[:, :, None]).sum(1) / np.maximum(count, 1)[:, None]
            centered = (delta - mean[:, None, :]) * live[:, :, None]
            covariance = (
                np.einsum("nki,nkj->nij", centered, centered)
                / np.maximum(count, 1)[:, None, None]
            )
            values, vectors = np.linalg.eigh(covariance)
            maximum = np.maximum(values[:, 2], np.finfo(float).tiny)
            self.valid[ids] = (
                (count >= g["normal_min_points"])
                & (values[:, 1] / maximum >= g["normal_min_secondary_ratio"])
                & ((values[:, 1] - values[:, 0]) / maximum >= g["planarity_min"])
            )
            self.normals[ids] = vectors[:, :, 0]
            self.computed[ids] = True
        return self.valid[indexes], self.normals[indexes]

    def face(self, points, indexes, cut, tangent_uv, inward_uv):
        g, b = self.config["geometry"], self.config["boundary"]
        nominated = np.flatnonzero(abs(points[:, 1] - cut) <= g["candidate_voxel_m"])
        empty = dict(
            qualified=False,
            method="PHYSICALLY_BALANCED_FULL_3D_RAW_NORMALS",
            point_indexes=[],
            raw_index_digest=row_digest(np.array([], dtype=np.int64)),
            normal_sample_count=0,
        )
        if not len(nominated):
            return empty
        live, normals = self.estimate(self.positions(indexes[nominated]))
        keep = (
            live
            & (abs(normals[:, 2]) <= np.sin(np.deg2rad(g["normal_angle_deg"])))
            & (
                abs(normals[:, :2] @ inward_uv)
                >= np.cos(np.deg2rad(g["normal_angle_deg"]))
            )
        )
        nominated, normals = nominated[keep], normals[keep]
        if len(nominated) < g["min_plane_points"]:
            return empty
        observed = points[nominated]
        normal = np.column_stack(
            (normals[:, :2] @ tangent_uv, normals[:, :2] @ inward_uv, normals[:, 2])
        )
        normal *= np.where(normal[:, 1] >= 0, 1.0, -1.0)[:, None]
        n = np.median(normal, axis=0)
        n /= np.linalg.norm(n)
        residual = abs((observed - np.median(observed, axis=0)) @ n)
        _, count = np.unique(
            np.floor(
                (observed[:, 2] - observed[:, 2].min()) / g["candidate_voxel_m"]
            ).astype(np.int64),
            return_counts=True,
        )
        layers = int(np.count_nonzero(count >= g["normal_min_points"]))
        vertical, along = float(np.ptp(observed[:, 2])), float(np.ptp(observed[:, 0]))
        qualified = bool(
            vertical >= b["face_vertical_span_min_m"]
            and along >= b["line_min_length_m"]
            and layers >= 2
            and np.median(residual) <= g["plane_inlier_m"]
            and np.quantile(residual, 0.95) <= b["profile_max_residual_m"]
        )
        ids = indexes[nominated]
        return dict(
            qualified=qualified,
            method=empty["method"],
            point_indexes=ids.tolist() if qualified else [],
            raw_index_digest=row_digest(ids),
            raw_count=len(ids),
            normal_sample_count=len(ids),
            normal_stz=n.tolist(),
            residual_mad_m=float(np.median(residual)),
            residual_p95_m=float(np.quantile(residual, 0.95)),
            vertical_span_m=vertical,
            along_span_m=along,
            independent_height_layers=layers,
            normal_neighborhood_voxel_m=g["candidate_voxel_m"],
            normal_reference_raw_digest=self.source_digest,
        )
