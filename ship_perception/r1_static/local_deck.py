"""Candidate-local planar patch detection with connected normal-consistent support."""
import math

import numpy as np
from scipy import ndimage

from .height_grid import R1HeightGrid


def _plane_from_three(triangle, max_angle_rad):
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    length = float(np.linalg.norm(normal))
    if length < 1e-9:
        return None
    normal /= length
    if normal[2] < 0:
        normal = -normal
    if math.acos(float(np.clip(normal[2], -1, 1))) > max_angle_rad:
        return None
    return normal, -float(normal @ triangle[0])


def _huber_plane(points, initial, threshold, max_angle_rad):
    """Refine z=ax+by+c after support selection; never change plane ownership."""
    xy = points[:, :2].astype(float)
    z = points[:, 2].astype(float)
    design = np.column_stack((xy, np.ones(len(points))))
    normal, offset = initial
    beta = np.array((-normal[0] / normal[2], -normal[1] / normal[2], -offset / normal[2]))
    for _ in range(8):
        residual = z - design @ beta
        weights = np.minimum(1.0, threshold / np.maximum(np.abs(residual), 1e-12))
        beta = np.linalg.lstsq(design * np.sqrt(weights[:, None]), z * np.sqrt(weights), rcond=None)[0]
    refined = np.array((-beta[0], -beta[1], 1.0))
    refined /= np.linalg.norm(refined)
    if math.acos(float(np.clip(refined[2], -1, 1))) > max_angle_rad:
        return None
    return refined, -float(beta[2] * refined[2])


class PlanarRegionGrower:
    """Connected local patch from height, normal, and plane residual gates."""

    def __init__(self, config):
        self.config = config

    def grow(self, points, bbox_xy, plane):
        grid = R1HeightGrid.from_points(points, self.config["geometry"]["coarse_voxel_m"])
        xx, yy = grid.centers()
        x0, y0, x1, y1 = bbox_xy
        band = self.config["roi"]["support_search_m"]
        outer = (xx >= x0 - band) & (xx <= x1 + band) & (yy >= y0 - band) & (yy <= y1 + band)
        inside = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        ring = outer & ~inside & grid.occupancy
        normal, offset = plane
        predicted = -(normal[0] * xx + normal[1] * yy + offset) / normal[2]
        surface = np.where(grid.occupancy, grid.median, predicted)
        dy, dx = np.gradient(surface, grid.cell_m)
        local_norm = np.sqrt(dx * dx + dy * dy + 1)
        cosine = (-dx * normal[0] - dy * normal[1] + normal[2]) / local_norm
        angle_ok = cosine >= math.cos(math.radians(self.config["geometry"]["normal_angle_deg"]))
        residual = np.abs(grid.median * normal[2] + xx * normal[0] + yy * normal[1] + offset)
        gate = ring & angle_ok & (residual <= self.config["geometry"]["plane_inlier_m"])
        labels, number = ndimage.label(gate, structure=np.ones((3, 3), dtype=np.uint8))
        if not number:
            return np.empty((0, 3)), 0.0
        counts = np.bincount(labels[gate], minlength=number + 1)
        selected = labels == int(np.argmax(counts[1:]) + 1)
        row, col = grid.cell_indices(points[:, :2])
        support = selected[row, col] & (np.abs(points @ normal + offset) <= self.config["geometry"]["plane_inlier_m"])
        dispersion = float(np.median(np.arccos(np.clip(cosine[selected], -1, 1)))) if selected.any() else 0.0
        return points[support], dispersion


def estimate_local_deck(points, proposal, config):
    box = proposal.bbox_xy
    x0, y0, x1, y1 = box
    band = config["roi"]["support_search_m"]
    xy = points[:, :2]
    outer = (xy[:, 0] >= x0 - band) & (xy[:, 0] <= x1 + band) & (xy[:, 1] >= y0 - band) & (xy[:, 1] <= y1 + band)
    inner = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    ring_points = points[outer & ~inner].astype(float)
    result = dict(status="LOCAL_DECK_UNRESOLVED", normal_raw=None, offset=None,
                  ring_point_count=int(len(ring_points)), support_count=0, support_sector_count=0,
                  normal_dispersion_rad=None, residual_p50_m=None, residual_p95_m=None,
                  competing_plane_count=0, reason="INSUFFICIENT_RING_SUPPORT")
    if len(ring_points) < config["geometry"]["min_plane_points"]:
        return result, None, np.empty((0, 3))
    rng = np.random.default_rng(2026)
    sample = ring_points[rng.choice(len(ring_points), min(len(ring_points), 3000), replace=False)]
    threshold = config["geometry"]["plane_inlier_m"]
    max_angle = math.radians(config["geometry"]["normal_angle_deg"])
    hypotheses = []
    for _ in range(config["geometry"]["ransac_iterations"]):
        plane = _plane_from_three(sample[rng.choice(len(sample), 3, replace=False)], max_angle)
        if plane is None:
            continue
        normal, offset = plane
        support = np.abs(sample @ normal + offset) <= threshold
        count = int(support.sum())
        if count:
            hypotheses.append((count, plane))
    if not hypotheses:
        result["reason"] = "NO_LOCAL_PLANAR_PATCH"
        return result, None, np.empty((0, 3))
    hypotheses.sort(key=lambda row: -row[0])
    best_count, initial = hypotheses[0]
    competitors = []
    for count, (_, offset) in hypotheses[1:]:
        if count < best_count * 0.6:
            break
        if abs(offset - initial[1]) > 3 * threshold and all(abs(offset - old) > 3 * threshold for old in competitors):
            competitors.append(offset)
    result["competing_plane_count"] = len(competitors)
    support = ring_points[np.abs(ring_points @ initial[0] + initial[1]) <= threshold]
    refined = _huber_plane(support, initial, threshold, max_angle) if len(support) >= 3 else None
    if refined is None:
        result["reason"] = "DEGENERATE_PLANE_REFINE"
        return result, None, np.empty((0, 3))
    grown, dispersion = PlanarRegionGrower(config).grow(ring_points, box, refined)
    if len(grown) < config["geometry"]["min_plane_points"]:
        result["reason"] = "INSUFFICIENT_CONNECTED_PLANAR_SUPPORT"
        return result, None, grown
    final = _huber_plane(grown, refined, threshold, max_angle)
    if final is None:
        result["reason"] = "CONNECTED_PATCH_TILT_OUT_OF_RANGE"
        return result, None, grown
    normal, offset = final
    residual = np.abs(grown @ normal + offset)
    center = np.array(((x0 + x1) / 2, (y0 + y1) / 2))
    relative = grown[:, :2] - center
    sectors = np.floor((np.arctan2(relative[:, 1], relative[:, 0]) + np.pi) * 8 / (2 * np.pi)).astype(int) % 8
    sector_count = int(len(np.unique(sectors)))
    result.update(normal_raw=normal.tolist(), offset=float(offset), support_count=int(len(grown)),
                  support_sector_count=sector_count, normal_dispersion_rad=dispersion,
                  residual_p50_m=float(np.percentile(residual, 50)),
                  residual_p95_m=float(np.percentile(residual, 95)))
    if competitors:
        result.update(status="LOCAL_DECK_AMBIGUOUS", reason="COMPETING_PLANAR_PATCHES")
        return result, None, grown
    if sector_count < math.ceil(8 * config["roi"]["min_enclosure_ratio"]):
        result["reason"] = "INSUFFICIENT_SECTOR_SUPPORT"
        return result, None, grown
    if (not (np.min(grown[:, 0]) < center[0] < np.max(grown[:, 0])) or
            not (np.min(grown[:, 1]) < center[1] < np.max(grown[:, 1]))):
        result["reason"] = "ONE_SIDED_LOCAL_DECK_SUPPORT"
        return result, None, grown
    result.update(status="RESOLVED", reason="CONNECTED_LOCAL_PLANAR_PATCH")
    return result, final, grown
