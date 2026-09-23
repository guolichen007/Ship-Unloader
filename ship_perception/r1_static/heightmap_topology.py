"""Research-only coarse heightmap region localization; never labels a steel edge."""
from dataclasses import dataclass, field
import math

import numpy as np
from scipy import ndimage

from .height_grid import R1HeightGrid
from .model import SeedComponent
from .opening_seed import _outer_contour


@dataclass
class RegionNode:
    node_id: str
    level_index: int
    level_m: float
    mask: np.ndarray
    area_m2: float
    bbox_xy: tuple
    fill_ratio: float
    dominant_height_m: float
    surrounding_height_m: float
    height_drop_m: float
    occupancy: float
    score: float
    side_height_rises_m: tuple
    side_rise_support_cells: tuple
    height_rise_side_count: int
    children: list = field(default_factory=list)
    parent: object = None
    persistence_levels: int = 1


def _height_levels(surface, valid, research):
    values = surface[valid]
    if not len(values):
        return []
    step = research["height_level_bin_m"]
    low, high = np.percentile(values, (1, 99))
    # An observed-data origin keeps height modes stable under raw Z shifts.
    origin = float(low)
    number = max(2, int(math.ceil((high - origin) / step)) + 1)
    bins = np.minimum(number - 1, np.maximum(0, np.floor((values - origin) / step).astype(int)))
    hist = np.bincount(bins, minlength=number).astype(float)
    peaks = []
    for index, value in enumerate(hist):
        left = hist[index - 1] if index else 0.0
        right = hist[index + 1] if index + 1 < number else 0.0
        if (value < research["height_level_min_fraction"] * len(values) or
                value <= left or value < right):
            continue
        height = origin + (index + .5) * step
        if peaks and height - peaks[-1][0] <= research["height_level_min_separation_m"]:
            if value > peaks[-1][1]:
                peaks[-1] = (height, value)
        else:
            peaks.append((height, value))
    # A remote high cluster generally belongs to a cabin, crane or wharf.
    # This is only an experimental vessel-height family, not target selection.
    for index in range(1, len(peaks)):
        if peaks[index][0] - peaks[index - 1][0] > research["separate_elevated_structure_gap_m"]:
            peaks = peaks[:index]
            break
    return [float(height) for height, _ in peaks]


def _vessel_support(valid):
    closed = ndimage.binary_closing(valid, structure=np.ones((3, 3), dtype=bool)) | valid
    labels, number = ndimage.label(closed, structure=np.ones((3, 3), dtype=bool))
    if not number:
        return np.zeros_like(valid)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _bridge(mask, cells):
    if cells <= 0:
        return mask
    width = 2 * cells + 1
    return mask | ndimage.binary_closing(mask, structure=np.ones((width, width), dtype=bool))


def _side_height_rises(grid, valid, box, config):
    """Measure coarse outside-minus-inside height on each box side, not steel."""
    x, y = grid.centers()
    x0, y0, x1, y1 = box
    width = config["roi"]["support_search_m"] / 2
    bands = (
        (((x >= x0 - width) & (x < x0) & (y >= y0) & (y < y1)),
         ((x >= x0) & (x < x0 + width) & (y >= y0) & (y < y1))),
        (((x >= x1) & (x < x1 + width) & (y >= y0) & (y < y1)),
         ((x >= x1 - width) & (x < x1) & (y >= y0) & (y < y1))),
        (((y >= y0 - width) & (y < y0) & (x >= x0) & (x < x1)),
         ((y >= y0) & (y < y0 + width) & (x >= x0) & (x < x1))),
        (((y >= y1) & (y < y1 + width) & (x >= x0) & (x < x1)),
         ((y >= y1 - width) & (y < y1) & (x >= x0) & (x < x1))),
    )
    rises, supports = [], []
    for outer, inner in bands:
        outer &= valid
        inner &= valid
        support = (int(outer.sum()), int(inner.sum()))
        supports.append(support)
        if min(support) < config["roi"]["min_deck_support_cells"]:
            rises.append(None)
        else:
            rises.append(float(np.median(grid.median[outer]) -
                                np.median(grid.median[inner])))
    return tuple(rises), tuple(supports)


def _nodes_at_level(grid, surface, valid, vessel, level_m, level_index, config, research):
    roi = config["roi"]
    low = (valid & vessel & (surface < level_m - roi["opening_drop_m"]) &
           (surface > level_m - roi["max_opening_depth_m"]))
    low = _bridge(low, research["gap_bridge_cells"]) & vessel
    labels, _ = ndimage.label(low, structure=np.ones((3, 3), dtype=bool))
    minimum_area = max(roi["opening_min_area_m2"],
                       research["minimum_region_vessel_area_fraction"] *
                       int(vessel.sum()) * grid.cell_m ** 2)
    nodes = []
    for label_id, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        region = labels == label_id
        cells = int(region.sum())
        area = cells * grid.cell_m ** 2
        if not minimum_area <= area <= roi["opening_max_area_m2"] or cells < roi["min_opening_cells"]:
            continue
        rr, cc = sl
        box = (grid.x0 + cc.start * grid.cell_m, grid.y0 + rr.start * grid.cell_m,
               grid.x0 + cc.stop * grid.cell_m, grid.y0 + rr.stop * grid.cell_m)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        if min(box[2] - box[0], box[3] - box[1]) < roi["min_deck_span_m"]:
            continue
        ring = (ndimage.binary_dilation(region,
                                        iterations=max(1, int(math.ceil(
                                            roi["support_search_m"] / grid.cell_m)))) &
                ~region & valid & vessel)
        if int(ring.sum()) < roi["min_deck_support_cells"]:
            continue
        dominant = float(np.median(grid.median[region & valid]))
        surrounding = float(np.median(grid.median[ring]))
        drop = surrounding - dominant
        if drop <= roi["opening_drop_m"]:
            continue
        fill = area / box_area
        occupancy = float(np.count_nonzero(region & valid) / cells)
        score = area * fill * fill * drop
        side_rises, side_supports = _side_height_rises(grid, valid & vessel, box, config)
        rise_count = sum(value is not None and value > roi["opening_drop_m"]
                         for value in side_rises)
        nodes.append(RegionNode("l%02d-c%04d" % (level_index, label_id), level_index,
                                level_m, region, area, box, float(fill), dominant,
                                surrounding, float(drop), occupancy, float(score),
                                side_rises, side_supports, rise_count))
    return nodes


def _link_levels(groups):
    for low, high in zip(groups, groups[1:]):
        for child in low:
            overlaps = [(int(np.count_nonzero(child.mask & parent.mask)), parent)
                        for parent in high]
            if not overlaps:
                continue
            count, parent = max(overlaps, key=lambda pair: (pair[0], pair[1].node_id))
            if count / max(int(child.mask.sum()), 1) >= .5:
                child.parent = parent
                parent.children.append(child)
    all_nodes = [node for group in groups for node in group]
    for node in all_nodes:
        node.persistence_levels = len({other.level_index for other in all_nodes
                                       if np.count_nonzero(node.mask & other.mask) /
                                       max(min(int(node.mask.sum()), int(other.mask.sum())), 1) >= .5})


def _choose(node, research):
    children = [child for child in node.children
                if child.area_m2 >= research["major_child_area_fraction"] * node.area_m2]
    children_choice = [_choose(child, research) for child in children]
    split_score = sum(score for _, score in children_choice)
    width = node.bbox_xy[2] - node.bbox_xy[0]
    height = node.bbox_xy[3] - node.bbox_xy[1]
    axis = 0 if width >= height else 1
    parent_span = width if axis == 0 else height
    centers = [(child.bbox_xy[axis] + child.bbox_xy[axis + 2]) / 2 for child in children]
    rectangular_subbasins = (len(children) >= 2 and
                             all(child.fill_ratio > node.fill_ratio for child in children) and
                             max(centers) - min(centers) >=
                             research["min_child_major_separation_fraction"] * parent_span)
    strongest_side_count = max((child.height_rise_side_count for child in children), default=0)
    supported_area = sum(child.area_m2 for child in children
                         if child.height_rise_side_count >= 1)
    if (node.height_rise_side_count == 0 and
            any(child.height_rise_side_count >= 1 and child.area_m2 >=
                research["minimum_single_child_support_fraction"] * node.area_m2
                for child in children)):
        supported = [(selection, score) for child, (selection, score) in
                     zip(children, children_choice)
                     if child.height_rise_side_count >= 1 and child.area_m2 >=
                     research["minimum_single_child_support_fraction"] * node.area_m2]
        return ([selected for selection, _ in supported for selected in selection],
                sum(score for _, score in supported))
    if (node.height_rise_side_count <= 1 and strongest_side_count >= 2 and
            supported_area >= research["minimum_split_support_fraction"] * node.area_m2):
        supported = [(selection, score) for child, (selection, score) in
                     zip(children, children_choice) if child.height_rise_side_count >= 1]
        return ([selected for selection, _ in supported for selected in selection],
                sum(score for _, score in supported))
    if children and (split_score > node.score or rectangular_subbasins):
        return [selected for choices, _ in children_choice for selected in choices], split_score
    return [node], node.score


def _fragment_overlap(a, b):
    x0, y0, x1, y1 = a.bbox_xy
    u0, v0, u1, v1 = b.bbox_xy
    intersection = max(0, min(x1, u1) - max(x0, u0)) * max(0, min(y1, v1) - max(y0, v0))
    return intersection / max(min((x1 - x0) * (y1 - y0), (u1 - u0) * (v1 - v0)), 1e-9)


def _combine_fragments(nodes, grid, valid, vessel, config, research):
    """Union overlapping same-level cargo fragments without changing basin count."""
    remaining, combined = list(nodes), []
    while remaining:
        group = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for other in list(remaining):
                if any(other.level_index == member.level_index and
                       _fragment_overlap(other, member) >=
                       research["fragment_box_overlap_min"] for member in group):
                    group.append(other)
                    remaining.remove(other)
                    changed = True
        if len(group) == 1:
            combined.append(group[0])
            continue
        mask = np.logical_or.reduce([item.mask for item in group])
        rows, cols = np.nonzero(mask)
        cell = grid.cell_m
        box = (grid.x0 + cols.min() * cell, grid.y0 + rows.min() * cell,
               grid.x0 + (cols.max() + 1) * cell, grid.y0 + (rows.max() + 1) * cell)
        area = int(mask.sum()) * cell ** 2
        fill = area / ((box[2] - box[0]) * (box[3] - box[1]))
        ring = (ndimage.binary_dilation(mask,
                                        iterations=max(1, int(math.ceil(
                                            config["roi"]["support_search_m"] / cell)))) &
                ~mask & valid & vessel)
        dominant = float(np.median(grid.median[mask & valid]))
        surrounding = float(np.median(grid.median[ring])) if ring.any() else dominant
        drop = surrounding - dominant
        rises, supports = _side_height_rises(grid, valid & vessel, box, config)
        rise_count = sum(value is not None and value > config["roi"]["opening_drop_m"]
                         for value in rises)
        combined.append(RegionNode("union(" + ",".join(item.node_id for item in group) + ")",
                                   group[0].level_index, group[0].level_m, mask,
                                   float(area), box, float(fill), dominant, surrounding,
                                   float(drop), float(np.count_nonzero(mask & valid) /
                                                      max(int(mask.sum()), 1)),
                                   float(area * fill * fill * max(drop, 0)),
                                   rises, supports, rise_count,
                                   persistence_levels=max(item.persistence_levels for item in group)))
    return combined


def _box_corners(node):
    x0, y0, x1, y1 = node.bbox_xy
    return [[float(x), float(y)] for x, y in
            ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def detect_regions(points, config, research):
    """Locate coarse height basins; selected regions are proposals, never observed steel."""
    points = np.asarray(points, dtype=np.float32)
    coarse = R1HeightGrid.from_points(points, config["geometry"]["coarse_voxel_m"])
    valid = coarse.count >= research["min_points_per_coarse_cell"]
    vessel = _vessel_support(valid)
    if not np.any(vessel):
        return dict(schema_version="ship_perception.v15r.heightmap_regions.1",
                    status="NO_VESSEL_SUPPORT", coordinate_frame="RAW_INPUT_FRAME",
                    region_count=0, regions=[], levels_m=[], level_hierarchy=[],
                    target_vessel_status="UNVERIFIED_RESEARCH_HEURISTIC",
                    limitation="HEIGHTMAP_REGION_BOXES_ARE_NOT_OBSERVED_STEEL_BOUNDARIES"), dict(
            coarse=coarse, valid=valid, vessel=vessel, masks=[])
    # Raw cell medians preserve transverse deck ridges. Morphology bridges
    # missing cells after thresholding, without blurring those height modes.
    surface = coarse.median
    levels = _height_levels(surface, valid & vessel, research)
    groups = [_nodes_at_level(coarse, surface, valid, vessel, level, index,
                              config, research) for index, level in enumerate(levels)]
    _link_levels(groups)
    roots = [node for group in groups for node in group if node.parent is None]
    chosen = [selected for root in roots for selected in _choose(root, research)[0]]
    if chosen:
        strongest = max(node.score for node in chosen)
        chosen = [node for node in chosen
                  if node.score >= strongest * research["minimum_relative_region_score"]]
    chosen = _combine_fragments(chosen, coarse, valid, vessel, config, research)
    chosen.sort(key=lambda node: ((node.bbox_xy[0] + node.bbox_xy[2]) / 2,
                                  (node.bbox_xy[1] + node.bbox_xy[3]) / 2))
    fine = R1HeightGrid.from_points(points, config["geometry"]["candidate_voxel_m"])
    fine_x, fine_y = fine.centers()
    coarse_row, coarse_col = coarse.cell_indices(np.column_stack((fine_x.ravel(), fine_y.ravel())))
    inside = ((coarse_row >= 0) & (coarse_row < coarse.count.shape[0]) &
              (coarse_col >= 0) & (coarse_col < coarse.count.shape[1]))
    records, masks = [], []
    for index, node in enumerate(chosen):
        fine_member = np.zeros(fine.count.size, dtype=bool)
        fine_member[inside] = node.mask[coarse_row[inside], coarse_col[inside]]
        fine_member = fine_member.reshape(fine.count.shape)
        fine_observed = fine_member & fine.occupancy
        fine_low = fine_observed & (fine.median < node.level_m - config["roi"]["opening_drop_m"])
        fine_support_ratio = float(fine_low.sum() / max(int(fine_observed.sum()), 1))
        labels, component_count = ndimage.label(node.mask,
                                                structure=np.ones((3, 3), dtype=bool))
        contour_parts = []
        for component_id in range(1, component_count + 1):
            cells = np.argwhere(labels == component_id)
            component = SeedComponent((coarse.x0, coarse.y0), coarse.cell_m,
                                      tuple(map(tuple, cells.tolist())), node.bbox_xy)
            contour_parts.append(_outer_contour(component))
        corners = _box_corners(node)
        contour = contour_parts[0] if len(contour_parts) == 1 else corners
        box_3d = [[float(x), float(y), float(z)] for z in
                  (node.dominant_height_m, node.surrounding_height_m) for x, y in corners]
        touches_crop = bool(np.any(node.mask[0]) or np.any(node.mask[-1]) or
                            np.any(node.mask[:, 0]) or np.any(node.mask[:, -1]))
        major_children = [child for child in node.children
                          if child.area_m2 >= research["major_child_area_fraction"] *
                          node.area_m2]
        alternatives = [dict(source_node=child.node_id, bbox_xy=list(child.bbox_xy),
                             area_m2=child.area_m2,
                             dominant_height_m=child.dominant_height_m)
                        for child in major_children]
        height_spread = (max(child.dominant_height_m for child in major_children) -
                         min(child.dominant_height_m for child in major_children)
                         if len(major_children) >= 2 else 0.0)
        records.append(dict(region_id="r%03d" % index,
                            status="HEIGHTMAP_REGION_CANDIDATE", steel_boundary_status="UNKNOWN",
                            region_evidence_type="OBSERVED_BEV_CONTOUR",
                            calibration_box_evidence_type="INFERRED_GEOMETRY",
                            source_node=node.node_id,
                            bbox_xy=list(node.bbox_xy), polygon_xy=contour,
                            polygon_semantics="COARSE_OUTER_CONTOUR" if len(contour_parts) == 1
                            else "AABB_OF_DISCONNECTED_HEIGHT_FRAGMENTS",
                            coarse_contour_parts_xy=contour_parts,
                            calibration_box_xy=corners, calibration_box_raw=box_3d,
                            area_m2=node.area_m2, dominant_height_m=node.dominant_height_m,
                            surrounding_height_m=node.surrounding_height_m,
                            height_drop_m=node.height_drop_m,
                            persistence_levels=node.persistence_levels,
                            occupancy=node.occupancy,
                            coarse_support_cells=int(node.mask.sum()),
                            fine_support_ratio=fine_support_ratio,
                            touches_true_crop_boundary=touches_crop,
                            topology_review="POSSIBLE_MERGE" if height_spread >
                            config["roi"]["opening_drop_m"] else "COARSE_TOPOLOGY_SELECTED",
                            alternate_subregions=alternatives,
                            side_height_rises_m=list(node.side_height_rises_m),
                            side_rise_support_cells=[list(row) for row in node.side_rise_support_cells],
                            height_rise_side_count=node.height_rise_side_count,
                            topology_score=node.score))
        masks.append(node.mask)
    hierarchy = [dict(node_id=node.node_id, level_m=node.level_m,
                      area_m2=node.area_m2, score=node.score,
                      dominant_height_m=node.dominant_height_m,
                      bbox_xy=list(node.bbox_xy),
                      side_height_rises_m=list(node.side_height_rises_m),
                      height_rise_side_count=node.height_rise_side_count,
                      child_ids=[child.node_id for child in node.children])
                 for group in groups for node in group]
    return (dict(schema_version="ship_perception.v15r.heightmap_regions.1",
                 status="REGION_CANDIDATES" if records else "NO_REGION_CANDIDATES",
                 coordinate_frame="RAW_INPUT_FRAME", region_count=len(records),
                 target_vessel_status="UNVERIFIED_RESEARCH_HEURISTIC",
                 grid=dict(x0=coarse.x0, y0=coarse.y0, cell_m=coarse.cell_m,
                           rows=int(coarse.count.shape[0]), cols=int(coarse.count.shape[1]),
                           valid_cells=int(valid.sum()), vessel_support_cells=int(vessel.sum())),
                 levels_m=levels, regions=records, level_hierarchy=hierarchy,
                 limitation="HEIGHTMAP_REGION_BOXES_ARE_NOT_OBSERVED_STEEL_BOUNDARIES"),
            dict(coarse=coarse, valid=valid, vessel=vessel, surface=surface, masks=masks))
