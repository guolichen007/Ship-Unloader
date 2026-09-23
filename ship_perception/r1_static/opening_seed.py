"""Observed opening seeds and their exterior-only perimeter search geometry."""
from collections import defaultdict
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .model import OpeningSeed, PerimeterSegment


_CROSS = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8)


def _outer_contour(component):
    """Trace the seed/exterior interface; enclosed holes never become sides."""
    cells = np.asarray(component.cells_rc, dtype=np.int64).reshape(-1, 2)
    if not len(cells):
        return ()
    low = cells.min(axis=0)
    high = cells.max(axis=0)
    mask = np.zeros(tuple(high - low + 5), dtype=bool)
    mask[cells[:, 0] - low[0] + 2, cells[:, 1] - low[1] + 2] = True
    unknown = ~mask
    border = np.zeros_like(mask)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    exterior = ndimage.binary_propagation(border & unknown, mask=unknown, structure=_CROSS)
    filled = ~exterior
    edges = set()
    for row, col in np.argwhere(filled):
        if not filled[row - 1, col]:
            edges.add(((col, row), (col + 1, row)))
        if not filled[row, col + 1]:
            edges.add(((col + 1, row), (col + 1, row + 1)))
        if not filled[row + 1, col]:
            edges.add(((col + 1, row + 1), (col, row + 1)))
        if not filled[row, col - 1]:
            edges.add(((col, row + 1), (col, row)))
    outgoing = defaultdict(list)
    for start, end in edges:
        outgoing[start].append(end)
    for ends in outgoing.values():
        ends.sort()
    loops = []
    while edges:
        start = min(edge[0] for edge in edges)
        current = start
        loop = []
        for _ in range(len(edges) + 1):
            options = [end for end in outgoing[current] if (current, end) in edges]
            if not options:
                break
            end = options[0]
            edges.remove((current, end))
            loop.append(current)
            current = end
            if current == start:
                if len(loop) >= 4:
                    loops.append(loop)
                break
    if not loops:
        return ()

    def signed_area(loop):
        xy = np.asarray(loop, dtype=float)
        return float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) -
                            xy[:, 1] * np.roll(xy[:, 0], -1))) / 2

    loop = max(loops, key=lambda item: abs(signed_area(item)))
    if signed_area(loop) < 0:
        loop = list(reversed(loop))
    origin_x, origin_y = component.origin_xy
    cell = component.cell_m
    return tuple((float(origin_x + (col + low[1] - 2) * cell),
                  float(origin_y + (row + low[0] - 2) * cell))
                 for col, row in loop)


def _rdp_open(points, tolerance):
    if len(points) <= 2:
        return points
    start, end = points[0], points[-1]
    direction = end - start
    length = float(np.linalg.norm(direction))
    distance = (np.linalg.norm(points - start, axis=1) if length < 1e-12 else
                np.abs(np.cross(direction, points - start)) / length)
    index = int(np.argmax(distance))
    if distance[index] <= tolerance:
        return points[[0, -1]]
    return np.vstack((_rdp_open(points[:index + 1], tolerance)[:-1],
                      _rdp_open(points[index:], tolerance)))


def _simplify_closed(contour, tolerance, minimum_length):
    xy = np.asarray(contour, dtype=float).reshape(-1, 2)
    if len(xy) < 4:
        return xy
    start = int(np.lexsort((xy[:, 1], xy[:, 0]))[0])
    xy = np.roll(xy, -start, axis=0)
    opposite = int(np.argmax(np.linalg.norm(xy - xy[0], axis=1)))
    if opposite == 0:
        return xy
    first = _rdp_open(xy[:opposite + 1], tolerance)
    second = _rdp_open(np.vstack((xy[opposite:], xy[0])), tolerance)
    vertices = np.vstack((first[:-1], second[:-1]))
    while len(vertices) > 3:
        lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
        shortest = int(np.argmin(lengths))
        if lengths[shortest] >= minimum_length:
            break
        vertices = np.delete(vertices, (shortest + 1) % len(vertices), axis=0)
    return vertices


class FovMap:
    """Exterior unknown cells from the observed XY grid, with sparse-gap closing."""

    def __init__(self, points, config):
        cell = config["geometry"]["coarse_voxel_m"]
        self.radius = config["roi"]["support_connectivity_m"]
        origin = np.min(points[:, :2], axis=0).astype(float)
        indices = np.floor((points[:, :2] - origin) / cell).astype(np.int64)
        cols, rows = indices[:, 0], indices[:, 1]
        occupied = np.zeros((int(rows.max()) + 1, int(cols.max()) + 1), dtype=bool)
        occupied[rows, cols] = True
        occupied = ndimage.binary_closing(occupied, structure=np.ones((3, 3), dtype=np.uint8))
        occupied = np.pad(occupied, 1, constant_values=False)
        unknown = ~occupied
        border = np.zeros_like(occupied)
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True
        exterior = ndimage.binary_propagation(border & unknown, mask=unknown, structure=_CROSS)
        row, col = np.nonzero(exterior)
        xy = np.column_stack((origin[0] + (col - 0.5) * cell,
                              origin[1] + (row - 0.5) * cell))
        self.tree = cKDTree(xy)

    def touches(self, contour):
        xy = np.asarray(contour, dtype=float).reshape(-1, 2)
        return bool(len(xy) and np.any(self.tree.query(xy)[0] <= self.radius))


def _boxes_overlap(first, second):
    return (min(first[2], second[2]) > max(first[0], second[0]) and
            min(first[3], second[3]) > max(first[1], second[1]))


def build_opening_seeds(proposals, points, config):
    """Fine components retain topology; coarse components add evidence only."""
    fov = FovMap(points, config)
    seeds = []
    for proposal in proposals:
        components = proposal.seed_components
        if not components:
            continue
        fine_size = min(component.cell_m for component in components)
        fine_ids = [index for index, component in enumerate(components)
                    if math.isclose(component.cell_m, fine_size)]
        coarse_ids = [index for index in range(len(components)) if index not in fine_ids]
        owned = set()
        for fine_id in fine_ids:
            child = components[fine_id]
            supports = [index for index in coarse_ids
                        if _boxes_overlap(child.bbox_xy, components[index].bbox_xy)]
            owned.update(supports)
            sources = (fine_id,) + tuple(supports)
            contour = _outer_contour(child)
            touches = fov.touches(contour)
            seeds.append(OpeningSeed("%s-s%03d" % (proposal.proposal_id, len(
                [seed for seed in seeds if seed.proposal_id == proposal.proposal_id])),
                                     proposal.proposal_id, child, sources, contour,
                                     tuple(sorted({components[index].cell_m for index in sources})),
                                     "PARTIAL_FOV" if touches else "FULLY_OBSERVED" if contour else "UNKNOWN_FOV",
                                     touches))
        for coarse_id in coarse_ids:
            if coarse_id in owned:
                continue
            child = components[coarse_id]
            contour = _outer_contour(child)
            touches = fov.touches(contour)
            seeds.append(OpeningSeed("%s-s%03d" % (proposal.proposal_id, len(
                [seed for seed in seeds if seed.proposal_id == proposal.proposal_id])),
                                     proposal.proposal_id, child, (coarse_id,), contour,
                                     (child.cell_m,),
                                     "PARTIAL_FOV" if touches else "FULLY_OBSERVED" if contour else "UNKNOWN_FOV",
                                     touches))
    return seeds, fov


def perimeter_segments(seed, fov, config):
    contour = _simplify_closed(seed.contour_xy, config["geometry"]["coarse_voxel_m"],
                               config["roi"]["min_deck_span_m"])
    segments = []
    if len(contour) < 3:
        return segments
    for index, (start, end) in enumerate(zip(contour, np.roll(contour, -1, axis=0))):
        length = float(np.linalg.norm(end - start))
        if length < config["boundary"]["line_min_length_m"]:
            continue
        tangent = (end - start) / length
        outward = np.array((tangent[1], -tangent[0]))
        touched = fov.touches(np.vstack((start, (start + end) / 2, end)))
        segments.append(PerimeterSegment("%s-g%03d" % (seed.seed_id, index),
                                         seed.seed_id, tuple(start), tuple(end),
                                         tuple(tangent), tuple(outward), length,
                                         "PARTIAL_FOV" if touched else "FULLY_OBSERVED"))
    return segments
