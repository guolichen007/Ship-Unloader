"""Exact raw-point indexing and once-per-section normal-coordinate profiles."""

from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np
from ship_perception.r1_static.raw_surface_normals import LocalSurfaceNormals


class BudgetedLocalSurfaceNormals(LocalSurfaceNormals):
    """Same exact estimator with cooperative stops between memory batches."""

    def __init__(self, xyz, config, budget, counters):
        super().__init__(xyz, config)
        self.input_point_indexes = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        self.budget, self.counters = budget, counters

    def _estimate(self, indexes):
        pending = indexes[~self.computed[indexes]]
        for start in range(0, len(pending), 8192):
            if self.budget is not None:
                self.budget.check()
            ids = pending[start : start + 8192]
            super()._estimate(ids)
            self.counters["normal_points_evaluated"] = self.counters.get(
                "normal_points_evaluated", 0
            ) + len(ids)
            if self.budget is not None:
                self.budget.check()


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

    def view(self, low, high, span=None):
        """An exact sub-band of the existing stable sort, without another sort."""
        result = object.__new__(RawSectionProfile)
        result.points = self.window(low, high)
        span = self.span if span is None else tuple(span)
        if span != self.span:
            along = result.points[:, 1 - self.axis]
            result.points = result.points[(along >= span[0]) & (along <= span[1])]
        result.axis, result.span, result.config = self.axis, span, self.config
        result.sorted_points = result.points
        result.coordinates = result.points[:, self.axis]
        # Only contiguous normal-axis views share an index. An along filter
        # creates a new sequence, with its own exact order-statistics index.
        if span == self.span:
            result.quantile_owner = getattr(self, "quantile_owner", self)
            result.quantile_offset = getattr(self, "quantile_offset", 0) + int(
                np.searchsorted(self.coordinates, low, side="left")
            )
        return result

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
        stations = np.arange(anchor - search, anchor + search + step / 2, step)
        low_start = np.searchsorted(self.coordinates, stations - probe, side="left")
        low_stop = np.searchsorted(
            self.coordinates, stations, side="right" if sign == -1 else "left"
        )
        high_start = np.searchsorted(
            self.coordinates, stations, side="left" if sign == 1 else "right"
        )
        high_stop = np.searchsorted(self.coordinates, stations + probe, side="right")
        observed = np.minimum(low_stop - low_start, high_stop - high_start) >= minimum
        stations = stations[observed]
        starts = np.r_[low_start[observed], high_start[observed]]
        stops = np.r_[low_stop[observed], high_stop[observed]]
        quantiles = np.r_[
            np.full(len(stations), 0.25 if sign == 1 else 0.75),
            np.full(len(stations), 0.75 if sign == 1 else 0.25),
        ]
        values = self._window_quantiles(starts, stops, quantiles)
        low, high = np.split(values, 2)
        contrasts = low - high if sign == 1 else high - low
        changes = [
            (float(rho), float(contrast))
            for rho, contrast in zip(stations, contrasts)
            if contrast >= self.config["roi"]["opening_drop_m"]
        ]
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

    def _window_quantiles(self, starts, stops, quantiles):
        """Reuse exact order statistics without sorting each sliding window."""
        if not len(starts):
            return np.empty(0, dtype=float)
        owner = getattr(self, "quantile_owner", self)
        offset = getattr(self, "quantile_offset", 0)
        z = owner.sorted_points[:, 2]
        if (
            not np.issubdtype(z.dtype, np.floating)
            or not np.isfinite(z).all()
            or getattr(owner, "quantile_reserved_bytes", 0) > 64 * 1024 * 1024
        ):
            return np.array(
                [
                    np.quantile(z[a + offset : b + offset], q)
                    for a, b, q in zip(starts, stops, quantiles)
                ]
            )
        from .raw_window_quantiles import RawWindowQuantileIndex

        counters = getattr(owner, "quantile_counters", {})
        if not hasattr(owner, "quantile_index"):
            owner.quantile_index = RawWindowQuantileIndex(z)
            counters["raw_quantile_index_build_count"] = (
                counters.get("raw_quantile_index_build_count", 0) + 1
            )
        else:
            counters["raw_quantile_index_cache_hit"] = (
                counters.get("raw_quantile_index_cache_hit", 0) + 1
            )
        return owner.quantile_index.quantiles(
            starts + offset, stops + offset, quantiles
        )


class BoundarySideMeasurementCache:
    """Cache immutable spatial bands; every anchor still measures its own modes."""

    def __init__(self, spatial, config, counters, enabled=True):
        self.spatial, self.config, self.counters = spatial, config, counters
        self.enabled = enabled
        self.profiles = OrderedDict()
        self.quantile_profiles = OrderedDict()
        self.quantile_bytes = 0
        self.bytes = 0
        self.origin = spatial.xyz[:, :2].min(0)

    def profile(self, axis, anchor, span):
        band = (
            self.config["roi"]["boundary_search_m"]
            + self.config["boundary"]["side_probe_m"]
        )
        width = 2 * band
        tile = int(np.floor((anchor - self.origin[axis]) / width))
        low = self.origin[axis] + tile * width - band
        high = low + 2 * width
        # Overlapping along-line sections share a raw band as well. The exact
        # requested interval is filtered from its stable sort in view(); tile
        # dimensions partition storage only and never quantize a measurement.
        scale = 2 ** int(np.ceil(np.log2(max(width, span[1] - span[0]) / width)))
        along_width = width * scale
        along_tile = int(
            np.floor(((span[0] + span[1]) / 2 - self.origin[1 - axis]) / along_width)
        )
        along_low = self.origin[1 - axis] + (along_tile - 1) * along_width
        along_high = along_low + 3 * along_width
        key = (axis, tile, along_tile, scale)
        profile = self.profiles.get(key) if self.enabled else None
        if profile is None:
            box = [0.0] * 4
            box[axis], box[axis + 2] = low, high
            box[1 - axis], box[3 - axis] = along_low, along_high
            profile = RawSectionProfile(
                self.spatial.query(box), axis, (along_low, along_high), self.config
            )
            self.counters["raw_profile_build_count"] = (
                self.counters.get("raw_profile_build_count", 0) + 1
            )
            size = profile.points.nbytes + profile.sorted_points.nbytes
            while self.profiles and self.bytes + size > 64 * 1024 * 1024:
                _, old = self.profiles.popitem(last=False)
                self.bytes -= old.points.nbytes + old.sorted_points.nbytes
            if self.enabled and size <= 64 * 1024 * 1024:
                self.profiles[key] = profile
                self.bytes += size
        else:
            self.profiles.move_to_end(key)
            self.counters["side_profile_cache_hit"] = (
                self.counters.get("side_profile_cache_hit", 0) + 1
            )
            self.counters["raw_profile_cache_hit"] = (
                self.counters.get("raw_profile_cache_hit", 0) + 1
            )
        # Nearby anchors share the same along-filtered Raw sequence. Only
        # storage is bounded; every requested window still contains all Raw.
        span_key = (key, tuple(span))
        along = self.quantile_profiles.get(span_key) if self.enabled else None
        if along is None:
            along = profile.view(-np.inf, np.inf, span)
            along.quantile_counters = self.counters
            n = len(along.points)
            reserve = (
                2 * along.points.nbytes
                + 8 * n
                + 4 * (n + 1) * max(1, (n - 1).bit_length())
            )
            along.quantile_reserved_bytes = reserve
            owner = getattr(along, "quantile_owner", along)
            owner.quantile_counters = self.counters
            owner.quantile_reserved_bytes = reserve
            while (
                self.quantile_profiles
                and self.quantile_bytes + reserve > 64 * 1024 * 1024
            ):
                _, old = self.quantile_profiles.popitem(last=False)
                self.quantile_bytes -= old.quantile_reserved_bytes
            if self.enabled and reserve <= 64 * 1024 * 1024:
                self.quantile_profiles[span_key] = along
                self.quantile_bytes += reserve
        else:
            self.quantile_profiles.move_to_end(span_key)
        return along.view(anchor - band, anchor + band)


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
        self.side_measurements = BoundarySideMeasurementCache(
            self.spatial, cache.config, self.counters
        )
        from .raster_measurement_cache import RasterPhaseMeasurementCache

        self.rasters = RasterPhaseMeasurementCache(
            self.spatial, cache.config, self.counters
        )
        self.height_profiles = {}
        self.raw_edge_modes = {}
        self.face_evidence = {}
        self.investigated_sides = {}
        self.local_surface_normals = None
        from .local_deck_profiles import LocalDeckProfileCache

        self.local_deck_profiles = LocalDeckProfileCache(self, cache.config)

    @property
    def sorted_u_indexes(self):
        return self.spatial.orders[0]

    @property
    def sorted_v_indexes(self):
        return self.spatial.orders[1]

    def grid(self, name):
        return self.local.context_grid(name)
