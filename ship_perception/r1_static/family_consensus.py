"""Deterministic plane-family consensus for S4-R1 (product research chain).

Unlike the P1.5 forensic greedy clustering, this module is deterministic under
input-order shuffling, uses a strict-core (1x) + relaxed-attachment (2x)
two-level gate so a drifting cargo patch cannot seed its own "structure", and
classifies spatial roles from the real Height-node mask contour signed distance
(not the axis-aligned bounding box).
"""

import math
from collections import Counter

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .local_deck import _physical_components
from .opening_seed import _outer_contour
from .plane_consensus import normalize_normal


def spherical_mean(normals):
    normals = np.asarray(normals, dtype=float)
    if not len(normals):
        return np.array((0.0, 0.0, 1.0))
    first = normals[0]
    normals = normals * np.where(normals @ first >= 0, 1.0, -1.0)[:, None]
    mean = normals.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-12:
        return first
    return normalize_normal(mean / norm)


def deterministic_families(candidates, strict_angle_deg, strict_sep_m,
                           relaxed_angle_deg, relaxed_sep_m):
    """Strict-core + relaxed-attachment clustering, deterministic under shuffle.

    Phase 1 builds strict cores (1x gate) via a total-order greedy; Phase 2 lets
    a not-yet-stable candidate attach to an existing >=2-segment strict core at
    the relaxed (2x) gate. A final pass removes any member outside the relaxed
    gate of the robust representative (chain-bridging guard).
    """
    number = len(candidates)
    order = sorted(range(number), key=lambda i: (
        -candidates[i]["support_count"],
        candidates[i]["segment_id"],
        candidates[i]["candidate_index"],
    ))
    families = []
    family_of = [None] * number

    def representative(members):
        return (spherical_mean([candidates[index]["normal"] for index in members]),
                float(np.median([candidates[index]["d_centered"] for index in members])))

    def angle_to(normal, rep_normal):
        return math.degrees(math.acos(float(np.clip(float(normal @ rep_normal), -1.0, 1.0))))

    def distinct_segments(members):
        return {candidates[index]["segment_id"] for index in members}

    # Phase 1: strict cores.
    for index in order:
        normal = candidates[index]["normal"]
        d_center = candidates[index]["d_centered"]
        placed = False
        for family_index, family in enumerate(families):
            if (angle_to(normal, family["rep_normal"]) <= strict_angle_deg and
                    abs(d_center - family["rep_d_centered"]) <= strict_sep_m):
                family["members"].append(index)
                family["rep_normal"], family["rep_d_centered"] = representative(family["members"])
                family_of[index] = family_index
                placed = True
                break
        if not placed:
            family_of[index] = len(families)
            families.append(dict(members=[index], rep_normal=normal, rep_d_centered=d_center))

    # Phase 2: relaxed attachment, only to strict cores (>=2 distinct segments).
    for index in order:
        current = family_of[index]
        if current is not None and len(distinct_segments(families[current]["members"])) >= 2:
            continue
        for family_index, family in enumerate(families):
            if len(distinct_segments(family["members"])) < 2:
                continue
            if (angle_to(candidates[index]["normal"], family["rep_normal"]) <= relaxed_angle_deg and
                    abs(candidates[index]["d_centered"] - family["rep_d_centered"]) <= relaxed_sep_m):
                if current is not None and family_of[index] == current:
                    families[current]["members"].remove(index)
                    if families[current]["members"]:
                        families[current]["rep_normal"], families[current]["rep_d_centered"] = \
                            representative(families[current]["members"])
                family["members"].append(index)
                family["rep_normal"], family["rep_d_centered"] = representative(family["members"])
                family_of[index] = family_index
                break

    # Final: chain-bridging guard — drop members beyond the relaxed gate.
    for family in families:
        changed = True
        while changed:
            changed = False
            if not family["members"]:
                break
            rep_normal, rep_d_center = representative(family["members"])
            family["rep_normal"], family["rep_d_centered"] = rep_normal, rep_d_center
            far = [index for index in family["members"]
                   if angle_to(candidates[index]["normal"], rep_normal) > relaxed_angle_deg
                   or abs(candidates[index]["d_centered"] - rep_d_center) > relaxed_sep_m]
            if far:
                for index in far:
                    family["members"].remove(index)
                    family_of[index] = None
                changed = True

    return families, family_of


def node_mask_contours(node, grid):
    """Outer contours of every connected component of the real node mask."""
    from .model import SeedComponent
    labels, count = ndimage.label(node.mask, structure=np.ones((3, 3), dtype=bool))
    contours = []
    for component_id in range(1, count + 1):
        cells = np.argwhere(labels == component_id)
        if not len(cells):
            continue
        component = SeedComponent((grid.x0, grid.y0), grid.cell_m,
                                  tuple(map(tuple, cells.tolist())), node.bbox_xy)
        contour = _outer_contour(component)
        if contour:
            contours.append(np.asarray(contour, dtype=float))
    return contours


def _trace_contour(contour, step):
    parts = []
    for start, end in zip(contour, np.roll(contour, -1, axis=0)):
        count = max(2, int(np.ceil(np.linalg.norm(end - start) / step)) + 1)
        parts.append(np.linspace(start, end, count))
    return np.vstack(parts) if parts else np.empty((0, 2))


class NodeGeometry:
    """Signed-distance support relative to the real node mask contour."""

    def __init__(self, node, grid):
        self.mask = node.mask
        self.grid = grid
        contours = node_mask_contours(node, grid)
        if contours:
            self.sampled = np.vstack([_trace_contour(contour, grid.cell_m / 2)
                                      for contour in contours])
            self.tree = cKDTree(self.sampled)
        else:
            self.sampled = np.empty((0, 2))
            self.tree = None

    def signed_distance(self, points_xy):
        points_xy = np.asarray(points_xy, dtype=float)
        row, col = self.grid.cell_indices(points_xy)
        height, width = self.mask.shape
        valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        inside = np.zeros(len(points_xy), dtype=bool)
        inside[valid] = self.mask[row[valid], col[valid]]
        if self.tree is None or not len(points_xy):
            return np.where(inside, -np.inf, np.inf)
        distance = self.tree.query(points_xy)[0]
        return np.where(inside, -distance, distance)


def zone_counts(signed, inner_band, outer_band):
    """4 spatial zones from signed distance: deep interior / inner boundary /
    outer boundary / remote exterior."""
    interior_deep = int((signed < -inner_band).sum())
    boundary_inner = int(((signed >= -inner_band) & (signed < 0)).sum())
    boundary_outer = int(((signed >= 0) & (signed < outer_band)).sum())
    exterior_remote = int((signed >= outer_band).sum())
    return interior_deep, boundary_inner, boundary_outer, exterior_remote


def role_from_zones(interior_deep, boundary_inner, boundary_outer, exterior_remote,
                    median_width, narrow_threshold):
    total = interior_deep + boundary_inner + boundary_outer + exterior_remote
    if total == 0:
        return "UNKNOWN"
    inside_total = interior_deep + boundary_inner
    outside_total = boundary_outer + exterior_remote
    f_inside = inside_total / total
    f_outside = outside_total / total
    f_remote = exterior_remote / total
    f_near_outside = boundary_outer / total
    is_narrow = median_width <= narrow_threshold
    # Order matters: mostly-inside is cargo, a wide surface spanning the opening
    # boundary is overflow, far-outside is remote, and only a surface with no
    # meaningful interior support hugging the near-outside band is a deck.
    if f_inside >= 0.5:
        return "INTERIOR_SURFACE"
    if f_inside >= 0.10 and f_outside >= 0.10 and not is_narrow:
        return "ROLE_AMBIGUOUS_OVERFLOW"
    if f_remote >= 0.5:
        return "REMOTE_PLANAR_STRUCTURE"
    if f_near_outside >= 0.3:
        return "NARROW_BOUNDARY_STRIP" if is_narrow else "BROAD_PERIMETER_SUPPORT"
    return "ROLE_AMBIGUOUS"


def spatial_connectedness(member_ids, candidates, points, config):
    union = np.concatenate([candidates[index]["raw_ids"] for index in member_ids]) \
        if member_ids else np.empty(0, dtype=np.int64)
    union = np.unique(union)
    if not len(union):
        return 0.0
    components = _physical_components(points[union].astype(float),
                                      config["roi"]["support_connectivity_m"])
    if not components:
        return 0.0
    return float(max(len(component) for component in components) / len(union))


def assign_family_roles(families, candidates, points, node_geometries, config):
    """Attach a ROLE_HYPOTHESIS (signed-distance zones) to each family."""
    inner_band = config["roi"]["boundary_search_m"]
    outer_band = config["roi"]["support_search_m"]
    widths = []
    for family in families:
        if family["members"]:
            widths.append(float(np.median([candidates[index]["outward_width"]
                                          for index in family["members"]])))
    narrow_threshold = 0.5 * float(np.median(widths)) if widths else 0.0

    records = []
    for family_index, family in enumerate(families):
        members = family["members"]
        interior_deep = boundary_inner = boundary_outer = exterior_remote = 0
        for index in members:
            candidate = candidates[index]
            geometry = node_geometries.get(candidate["node_id"])
            if geometry is None or not len(candidate["raw_ids"]):
                continue
            signed = geometry.signed_distance(points[candidate["raw_ids"], :2])
            counts = zone_counts(signed, inner_band, outer_band)
            interior_deep += counts[0]
            boundary_inner += counts[1]
            boundary_outer += counts[2]
            exterior_remote += counts[3]
        median_width = float(np.median([candidates[index]["outward_width"] for index in members])) \
            if members else 0.0
        role = role_from_zones(interior_deep, boundary_inner, boundary_outer, exterior_remote,
                               median_width, narrow_threshold)
        segments = sorted({candidates[index]["segment_id"] for index in members})
        nodes = sorted({candidates[index]["node_id"] for index in members})
        normal_dispersion = max(
            (math.degrees(math.acos(float(np.clip(
                float(family["rep_normal"] @ candidates[index]["normal"]), -1.0, 1.0))))
             for index in members), default=0.0)
        separation_dispersion = max(
            (abs(candidates[index]["d_centered"] - family["rep_d_centered"]) for index in members),
            default=0.0)
        records.append(dict(
            family_id="f%03d" % family_index,
            member_candidate_indexes=[int(index) for index in members],
            unique_segment_votes=len(set(segments)),
            unique_node_votes=len(set(nodes)),
            segment_vote_ratio=float(len(set(segments)) /
                                     max(len({candidates[index]["segment_id"]
                                              for index in range(len(candidates))}), 1)),
            representative_normal=family["rep_normal"].tolist(),
            representative_offset=family["rep_d_centered"],
            normal_dispersion_deg=float(normal_dispersion),
            plane_separation_dispersion_m=float(separation_dispersion),
            support_point_count=int(sum(candidates[index]["support_count"] for index in members)),
            median_transverse_width=median_width,
            total_along_span=float(sum(candidates[index]["along_span"] for index in members)),
            spatial_connectedness=spatial_connectedness(members, candidates, points, config),
            support_interior_deep_fraction=float(interior_deep /
                                                 max(interior_deep + boundary_inner +
                                                     boundary_outer + exterior_remote, 1)),
            support_boundary_inner_fraction=float(boundary_inner /
                                                  max(interior_deep + boundary_inner +
                                                      boundary_outer + exterior_remote, 1)),
            support_boundary_outer_fraction=float(boundary_outer /
                                                  max(interior_deep + boundary_inner +
                                                      boundary_outer + exterior_remote, 1)),
            support_exterior_remote_fraction=float(exterior_remote /
                                                   max(interior_deep + boundary_inner +
                                                       boundary_outer + exterior_remote, 1)),
            role_hypothesis=role,
            node_ids=nodes,
            segment_ids=segments,
        ))
    records.sort(key=lambda row: (-row["unique_segment_votes"], -row["support_point_count"]))
    return records
