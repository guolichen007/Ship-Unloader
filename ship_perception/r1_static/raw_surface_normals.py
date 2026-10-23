"""Local 3D covariance qualifies a face; its physical steel role stays unknown."""

import numpy as np
from scipy.spatial import cKDTree


class LocalSurfaceNormals:
    """Cache isotropic k-neighbour estimates only for nominated raw-point ROIs."""

    def __init__(self, xyz, config):
        # Canonical order makes KD-tree tie handling independent of wire order.
        order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        self.xyz = np.asarray(xyz, dtype=float)[order]
        self.config = config
        self.tree = cKDTree(self.xyz)
        self.computed = np.zeros(len(xyz), bool)
        self.valid = np.zeros(len(xyz), bool)
        self.normals = np.zeros((len(xyz), 3))
        self.axis_orders = [np.argsort(self.xyz[:, a], kind="stable") for a in (0, 1)]
        self.axis_positions = [
            self.xyz[order, a] for a, order in enumerate(self.axis_orders)
        ]

    def _estimate(self, indexes):
        indexes = indexes[~self.computed[indexes]]
        geometry = self.config["geometry"]
        # This is a memory batch size, not a sampling or geometric threshold.
        for start in range(0, len(indexes), 8192):
            ids = indexes[start : start + 8192]
            distance, near = self.tree.query(
                self.xyz[ids],
                k=geometry["normal_k"],
                distance_upper_bound=geometry["normal_radius_m"],
                workers=1,
            )
            mask = np.isfinite(distance)
            count = mask.sum(axis=1)
            neighbours = self.xyz[np.where(mask, near, 0)]
            # Center nearby deltas, keeping a large world origin out of the
            # covariance arithmetic. Neighbours come from the full 3D cloud,
            # never the thin axial stripe used to nominate the candidate.
            delta = neighbours - self.xyz[ids, None, :]
            mean = (
                np.sum(delta * mask[:, :, None], axis=1) / np.maximum(count, 1)[:, None]
            )
            centered = (delta - mean[:, None, :]) * mask[:, :, None]
            covariance = np.einsum("nki,nkj->nij", centered, centered)
            covariance /= np.maximum(count, 1)[:, None, None]
            values, vectors = np.linalg.eigh(covariance)
            scale = np.maximum(values[:, 2], np.finfo(float).tiny)
            self.valid[ids] = (
                (count >= geometry["normal_min_points"])
                & (values[:, 1] / scale >= geometry["normal_min_secondary_ratio"])
                & ((values[:, 1] - values[:, 0]) / scale >= geometry["planarity_min"])
            )
            self.normals[ids] = vectors[:, :, 0]
            self.computed[ids] = True

    def face_evidence(self, axis, rho, interval):
        geometry, boundary = self.config["geometry"], self.config["boundary"]
        # The coarse nomination's existing half-cell uncertainty locates the
        # query. It does not restrict any point's 3D normal neighbourhood.
        band = geometry["coarse_voxel_m"] / 2
        positions = self.axis_positions[axis]
        lo = np.searchsorted(positions, rho - band, side="left")
        hi = np.searchsorted(positions, rho + band, side="right")
        ids = self.axis_orders[axis][lo:hi]
        along = self.xyz[ids, 1 - axis]
        ids = ids[(along >= interval[0]) & (along <= interval[1])]
        self._estimate(ids)
        angle = np.radians(geometry["normal_angle_deg"])
        live = ids[
            self.valid[ids]
            & (np.abs(self.normals[ids, 2]) <= np.sin(angle))
            & (np.abs(self.normals[ids, axis]) >= np.cos(angle))
        ]
        vertical_span = float(np.ptp(self.xyz[live, 2])) if len(live) else 0.0
        along_span = float(np.ptp(self.xyz[live, 1 - axis])) if len(live) else 0.0
        accepted = bool(
            len(live) >= geometry["min_plane_points"]
            and vertical_span >= boundary["face_vertical_span_min_m"]
            and along_span >= boundary["line_min_length_m"]
        )
        return dict(
            status="LOCAL_3D_NORMAL_FACE" if accepted else "UNVERIFIED_VERTICAL_FACE",
            method="ISOTROPIC_3D_LOCAL_COVARIANCE",
            candidate_point_count=int(len(ids)),
            valid_normal_point_count=int(np.count_nonzero(self.valid[ids])),
            vertical_normal_point_count=int(len(live)),
            vertical_span_m=vertical_span,
            along_span_m=along_span,
            normal_k=geometry["normal_k"],
            normal_radius_m=geometry["normal_radius_m"],
            physical_role="UNKNOWN",
        )
