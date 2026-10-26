"""Vessel-local measurement caches and recall before global score suppression."""

from dataclasses import dataclass, field
import copy
import hashlib
import math

import numpy as np
from scipy import ndimage

from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.heightmap_topology import _side_height_rises
from ship_perception.r1_static.hatch_proposal import (
    _bev_provider,
    _raw3d_provider,
    _world_line,
)
from .context import BoundaryMeasurementContext, thaw


def cluster_levels(values, config, research):
    """Quantize observed levels, then separate anchored families without chaining."""
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if not len(values):
        return []
    step = research["height_level_bin_m"]
    origin = float(values[0])
    keys = np.floor((values - origin) / step).astype(int)
    groups = []
    for key in np.unique(keys):
        measured = values[keys == key]
        if (
            groups
            and np.median(measured) - groups[-1][0]
            < research["height_level_min_separation_m"]
        ):
            groups[-1][1].extend(measured.tolist())
        else:
            groups.append([float(np.median(measured)), measured.tolist()])
    return [float(np.median(v)) for _, v in groups]


@dataclass
class VesselLocalEvidence:
    vessel: object
    xyz: np.ndarray
    config: dict
    grids: dict = field(default_factory=dict)
    providers: object = None
    input_frame_providers: object = None
    provider_build_count: int = 0
    grid_build_counts: dict = field(default_factory=dict)
    measurement_context: object = None
    context_xyz: object = None
    context_grids: dict = field(default_factory=dict)

    @classmethod
    def build(cls, frame, vessel, config, measurement_context=None):
        points = frame.xyz[vessel.point_indexes]
        uvz = np.column_stack((points[:, :2] @ vessel.axes.T, points[:, 2]))
        uvz.setflags(write=False)
        if measurement_context is None:
            measurement_context = BoundaryMeasurementContext.build(vessel, [vessel])
        if measurement_context.vessel_id != vessel.vessel_id:
            raise ValueError("LOCAL_MEASUREMENT_CONTEXT_VESSEL_MISMATCH")
        if measurement_context.point_index_sha256 == vessel.point_index_sha256:
            context_uvz = uvz
        else:
            points = frame.xyz[measurement_context.point_indexes]
            context_uvz = np.column_stack((points[:, :2] @ vessel.axes.T, points[:, 2]))
            context_uvz.setflags(write=False)
        return cls(
            vessel,
            uvz,
            config,
            measurement_context=measurement_context,
            context_xyz=context_uvz,
        )

    def context_grid(self, name):
        """Independent measurement occupancy, never a replacement owned mask."""
        if self.context_xyz is self.xyz:
            return self.grid(name)
        if name not in self.context_grids:
            key = "coarse_voxel_m" if name == "coarse" else "candidate_voxel_m"
            self.context_grids[name] = R1HeightGrid.from_points(
                self.context_xyz, self.config["geometry"][key]
            )
        return self.context_grids[name]

    def grid(self, name):
        if name not in self.grids:
            key = "coarse_voxel_m" if name == "coarse" else "candidate_voxel_m"
            self.grids[name] = R1HeightGrid.from_points(
                self.xyz, self.config["geometry"][key]
            )
            self.grid_build_counts[name] = self.grid_build_counts.get(name, 0) + 1
        return self.grids[name]

    def structural_providers(self):
        if self.providers is None:
            local = thaw(self.vessel.public)
            local["local_axes"] = [[1, 0], [0, 1]]
            # Both providers work in exactly the frozen vessel UV frame.
            bev, grid = _bev_provider(
                self.xyz, local, self.config, fine_grid=self.grid("fine")
            )
            self.grids["fine"] = grid
            raw = _raw3d_provider(
                self.xyz, local, self.config, coarse_grid=self.grid("coarse")
            )
            self.providers = bev + raw
            self.provider_build_count += 1
            for row in self.providers:
                row["coordinate_frame"] = "VESSEL_LOCAL_UV"
                row["input_vessel_axes"] = self.vessel.axes.tolist()
                row["point_index_sha256"] = self.vessel.point_index_sha256
        return self.providers

    def side_candidates(self, box):
        from ship_perception.r1_static.r2h_b5_corner_forensics import _side_modes

        lines = [
            dict(
                normal_axis=p["rho_or_region"]["normal_axis"],
                rho=p["rho_or_region"]["rho"],
                along=p["support_interval"],
                provider_type=p["provider_type"],
                provider_id=p["provider_id"],
                source_vessel_id=self.vessel.vessel_id,
                source_support_ids=p["source_support_ids"],
                source_scope="SAME_VESSEL",
                coverage=p["coverage"],
            )
            for p in self.structural_providers()
        ]
        search = self.config["roi"]["support_search_m"]
        return {
            name: _side_modes(
                lines,
                axis,
                (box[1 - axis], box[3 - axis]),
                (box[position] - search, box[position] + search),
                self.config,
            )
            for name, axis, position in (
                ("U0", 0, 0),
                ("U1", 0, 2),
                ("V0", 1, 1),
                ("V1", 1, 3),
            )
        }

    def export_structural(self):
        """Transform cached measured lines for the independent legacy consumer."""
        if self.input_frame_providers is None:
            rows = copy.deepcopy(self.structural_providers())
            for row in rows:
                row["coordinate_frame"] = "RAW_INPUT_FRAME"
                row["local_axis_family"] = self.vessel.axes.tolist()
                line = row["rho_or_region"]
                row["source_geometry"]["endpoints_xy"] = _world_line(
                    self.vessel.axes,
                    line["normal_axis"],
                    line["rho"],
                    *row["support_interval"],
                )
            self.input_frame_providers = (
                [r for r in rows if r["provider_type"] == "BEV_RECTILINEAR_PROVIDER"],
                [r for r in rows if r["provider_type"] == "RAW3D_STRUCTURAL_PROVIDER"],
                self.grid("fine"),
            )
        return self.input_frame_providers


def local_height_recall(cache, research):
    """Discover complete and incomplete basins independently of global chosen."""
    grid = cache.grid("coarse")
    config = cache.config
    valid = grid.occupancy
    z = grid.median[valid]
    if not len(z):
        return [], dict(
            status="NO_OBSERVED_SUPPORT", before_filter_count=0, after_filter_count=0
        )
    # All occupied local height families participate, including those discarded
    # by the legacy elevated-cluster and strongest-relative-score heuristics.
    levels = cluster_levels(z, config, research)
    records, drops = [], {}
    seen = set()
    local_ring_levels = []

    def collect(levels_to_scan, source):
        for level in levels_to_scan:
            for polarity in ("LOW", "RAISED"):
                mask = valid & (
                    (grid.median < level - config["roi"]["opening_drop_m"])
                    if polarity == "LOW"
                    else (grid.median > level + config["roi"]["opening_drop_m"])
                )
                labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
                label_counts = np.bincount(labels.ravel())
                for label, slices in enumerate(ndimage.find_objects(labels), 1):
                    if slices is None:
                        continue
                    cells = int(label_counts[label])
                    area = cells * grid.cell_m**2
                    if (
                        cells < config["roi"]["min_opening_cells"]
                        or area < config["roi"]["opening_min_area_m2"]
                    ):
                        drops["BELOW_PHYSICAL_SUPPORT"] = (
                            drops.get("BELOW_PHYSICAL_SUPPORT", 0) + 1
                        )
                        continue
                    if area > config["roi"]["opening_max_area_m2"]:
                        drops["ABOVE_SEARCH_AREA"] = (
                            drops.get("ABOVE_SEARCH_AREA", 0) + 1
                        )
                        continue
                    rows, cols = slices
                    box = [
                        grid.x0 + cols.start * grid.cell_m,
                        grid.y0 + rows.start * grid.cell_m,
                        grid.x0 + cols.stop * grid.cell_m,
                        grid.y0 + rows.stop * grid.cell_m,
                    ]
                    key = (polarity, *box)
                    if key in seen:
                        continue
                    seen.add(key)
                    member = labels == label
                    ring = (
                        ndimage.binary_dilation(
                            member,
                            iterations=max(
                                1,
                                int(
                                    math.ceil(
                                        config["roi"]["support_search_m"] / grid.cell_m
                                    )
                                ),
                            ),
                        )
                        & ~member
                        & valid
                    )
                    surrounding = (
                        float(np.median(grid.median[ring])) if ring.any() else None
                    )
                    if surrounding is not None:
                        local_ring_levels.append(surrounding)
                    rises, supports = _side_height_rises(grid, valid, box, config)
                    records.append(
                        dict(
                            proposal_id="L{:05d}".format(len(records)),
                            vessel_hypothesis_id=cache.vessel.vessel_id,
                            bounds_axial=box,
                            level_m=float(level),
                            polarity=polarity,
                            proposal_source=source,
                            area_m2=area,
                            occupied_cells=cells,
                            ring_cells=int(ring.sum()),
                            surrounding_height_m=surrounding,
                            interior_height_m=float(np.median(grid.median[member])),
                            height_rise_side_count=sum(
                                v is not None and v >= config["roi"]["opening_drop_m"]
                                for v in rises
                            ),
                            side_height_rises_m=list(rises),
                            side_support_cells=[list(v) for v in supports],
                            touches_grid_boundary=bool(
                                rows.start == 0
                                or cols.start == 0
                                or rows.stop == valid.shape[0]
                                or cols.stop == valid.shape[1]
                            ),
                            geometry_role="HEIGHT_PROPOSAL_NOT_STEEL",
                        )
                    )

    collect(levels, "VESSEL_LOCAL_COMPLETE_LEVEL_FAMILIES")
    ring_levels = cluster_levels(local_ring_levels, config, research)
    additional = [
        v
        for v in ring_levels
        if all(abs(v - old) >= research["height_level_bin_m"] for old in levels)
    ]
    collect(additional, "LOCAL_RING_SHOULDER")
    return records, dict(
        status="LOCAL_RECALL",
        levels_m=levels,
        ring_levels_m=additional,
        before_filter_count=len(records),
        after_filter_count=len(records),
        global_relative_score_filter_applied=False,
        single_basin_allowed=True,
        partial_support_preserved=True,
        physically_rejected_counts=drops,
        candidate_digest=hashlib.sha256(repr(records).encode()).hexdigest(),
    )
