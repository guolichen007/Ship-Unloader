"""Exact raw-point indexing and once-per-section normal-coordinate profiles."""

from dataclasses import dataclass, field
import numpy as np


class RawSpatialIndex:
    """Sorted indices select original returns; no voxelization or point removal."""

    def __init__(self, xyz, counters=None):
        self.xyz = np.asarray(xyz)
        self.orders = tuple(
            np.argsort(self.xyz[:, axis], kind="stable") for axis in (0, 1)
        )
        self.coordinates = tuple(
            self.xyz[order, axis] for axis, order in enumerate(self.orders)
        )
        for array in self.orders + self.coordinates:
            array.setflags(write=False)
        self.counters = counters if counters is not None else {}

    def query_indexes(self, box):
        ranges = [
            np.searchsorted(self.coordinates[a], [box[a], box[a + 2]], side="left")
            for a in (0, 1)
        ]
        # Inclusive upper boundaries exactly match the historical raw queries.
        for a in (0, 1):
            ranges[a][1] = np.searchsorted(
                self.coordinates[a], box[a + 2], side="right"
            )
        axis = int(np.argmin([r[1] - r[0] for r in ranges]))
        low, high = ranges[axis]
        ids = self.orders[axis][low:high]
        other = self.xyz[ids, 1 - axis]
        ids = ids[(other >= box[1 - axis]) & (other <= box[3 - axis])]
        # Preserve input order, including tied coordinates and duplicate returns.
        ids = np.sort(ids)
        self.counters["spatial_query_count"] = (
            self.counters.get("spatial_query_count", 0) + 1
        )
        self.counters["spatial_candidate_points"] = self.counters.get(
            "spatial_candidate_points", 0
        ) + int(high - low)
        return ids

    def query(self, box):
        return self.xyz[self.query_indexes(box)]


@dataclass
class RawSectionProfile:
    points: np.ndarray
    axis: int
    span: tuple
    config: dict
    coordinates: np.ndarray = field(init=False)
    sorted_points: np.ndarray = field(init=False)

    def __post_init__(self):
        order = np.argsort(self.points[:, self.axis], kind="stable")
        self.sorted_points = self.points[order]
        self.coordinates = self.sorted_points[:, self.axis]

    def window(self, low, high, *, include_low=True, include_high=True):
        a = np.searchsorted(
            self.coordinates, low, side="left" if include_low else "right"
        )
        b = np.searchsorted(
            self.coordinates, high, side="right" if include_high else "left"
        )
        return self.sorted_points[a:b]

    def height(self, rho, sign):
        from .sections import robust_height_support

        width = self.config["boundary"]["side_probe_m"]
        return robust_height_support(
            self.window(rho - width, rho + width),
            self.axis,
            rho,
            self.span,
            sign,
            self.config,
        )

    def edge_modes(self, anchor, sign):
        """Find robust signed changes once; refine raw midpoints only at modes.

        Sliding windows are views of one sorted exact profile. They do not
        rebuild along-line pairing or rescan a vessel for every 5 cm station.
        """
        search = self.config["roi"]["boundary_search_m"]
        step = self.config["boundary"]["profile_step_m"]
        probe = self.config["boundary"]["side_probe_m"]
        minimum = self.config["boundary"]["profile_min_points"]
        changes = []
        for rho in np.arange(anchor - search, anchor + search + step / 2, step):
            low = self.window(rho - probe, rho, include_high=sign == -1)[:, 2]
            high = self.window(rho, rho + probe, include_low=sign == 1)[:, 2]
            inside, outside = (high, low) if sign == 1 else (low, high)
            if min(len(inside), len(outside)) < minimum:
                continue
            contrast = float(np.quantile(outside, 0.25) - np.quantile(inside, 0.75))
            if contrast >= self.config["roi"]["opening_drop_m"]:
                changes.append((float(rho), contrast))
        groups = []
        for item in changes:
            if groups and item[0] - groups[-1][-1][0] <= step * 1.5:
                groups[-1].append(item)
            else:
                groups.append([item])
        return [
            dict(
                peak_rho=max(g, key=lambda x: x[1])[0],
                center_rho=float(np.median([x[0] for x in g])),
                contrast_m=max(x[1] for x in g),
                observed_station_count=len(g),
            )
            for g in groups
        ]


class VesselMeasurementContext:
    """A single vessel's shared immutable raw coordinates and lazy caches."""

    def __init__(self, cache):
        self.local = cache
        self.raw_uvz = cache.context_xyz if cache.context_xyz is not None else cache.xyz
        self.raw_uvz.setflags(write=False)
        self.raw_point_indexes = (
            cache.measurement_context.point_indexes
            if cache.measurement_context is not None
            else cache.vessel.point_indexes
        )
        self.counters = dict(
            raw_profile_build_count=0,
            raw_profile_cache_hit=0,
            raw_edge_mode_count=0,
            continuous_refinement_count=0,
            normal_query_count=0,
            normal_points_evaluated=0,
            spatial_query_count=0,
        )
        self.spatial = RawSpatialIndex(self.raw_uvz, self.counters)
        self.height_profiles = {}
        self.raw_edge_modes = {}
        self.face_evidence = {}
        self.local_surface_normals = None

    @property
    def sorted_u_indexes(self):
        return self.spatial.orders[0]

    @property
    def sorted_v_indexes(self):
        return self.spatial.orders[1]

    def grid(self, name):
        return self.local.context_grid(name)
