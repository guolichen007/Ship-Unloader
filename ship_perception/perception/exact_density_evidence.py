"""V2-only exact count-ratio decisions; the frozen R1 provider stays intact."""

import math
from dataclasses import fields
from decimal import Decimal, localcontext
from functools import lru_cache

import numpy as np
from scipy import ndimage

from ship_perception.r1_static.hatch_proposal import (
    _measured_intervals,
    _record,
    _world_line,
)
from ship_perception.r1_static.raw_topview_evidence import CHANNELS, raster_channels
from .context import thaw
from .local_evidence import VesselLocalEvidence


@lru_cache(maxsize=4096)
def canonical_log1p_count(count):
    """Correctly rounded binary64 log of an integer count plus one.

    Decimal.ln is correctly rounded. Its adjacent decimal values bound the
    exact result; accept a binary64 value only when both interval endpoints
    round to the same value. This pure integer function caches no Raw data.
    """
    if count == 0:
        return 0.0
    for precision in (50, 100, 200, 400):
        with localcontext() as context:
            context.prec = precision
            value = Decimal(count + 1).ln()
            lower = float(context.next_minus(value))
            upper = float(context.next_plus(value))
            if lower == upper:
                return lower
    raise ValueError("DENSITY_LOG_ROUNDING_UNRESOLVED")


def exact_density_transition(left, right, minimum_points):
    """Reproduce the original binary64 log expression under a portable contract.

    An algebraic inclusive real-ratio test is not equivalent at count ties:
    the original program rounds each log before subtracting. Preserve that
    computation using certified log rounding, rather than a platform libm.
    Away from equality, int32 counts separate the ratio from two by far more
    than binary64 roundoff; only exact integer-ratio ties need logarithms.
    """
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    high, low = np.maximum(left, right), np.minimum(left, right)
    valid = (low > 0) & (high >= minimum_points)
    result = high + 1 > 2 * (low + 1)
    ties = valid & (high + 1 == 2 * (low + 1))
    threshold = canonical_log1p_count(1)
    for count in np.unique(low[ties]):
        count = int(count)
        delta = canonical_log1p_count(2 * count + 1) - canonical_log1p_count(count)
        result[ties & (low == count)] = delta >= threshold
    return result & valid


def exact_raster_modes(grid, config):
    """Keep frozen fragment/support semantics, changing only the density tie.

    The fragment loop mirrors R1's raster_modes because that frozen function
    has no channel injection API. No global function replacement is used.
    """
    channels = raster_channels(grid, config)
    for axis in (0, 1):
        left, right = (
            (grid.count[:, :-1], grid.count[:, 1:])
            if axis == 0
            else (grid.count[:-1, :], grid.count[1:, :])
        )
        channels[axis]["DENSITY_TRANSITION"] = exact_density_transition(
            left, right, config["geometry"]["outlier_min_points"]
        )
    minimum_cells = max(
        2, int(math.ceil(config["boundary"]["line_min_length_m"] / grid.cell_m))
    )
    modes = {0: [], 1: []}
    for normal_axis in (0, 1):
        shape = channels[normal_axis][CHANNELS[0]].shape
        for normal_index in range(shape[1 if normal_axis == 0 else 0]):
            row = {}
            for channel in CHANNELS:
                mask = channels[normal_axis][channel]
                values = (
                    mask[:, normal_index] if normal_axis == 0 else mask[normal_index, :]
                )
                row[channel] = np.flatnonzero(values)
            union = np.unique(np.concatenate(tuple(row.values())))
            if len(union) < minimum_cells:
                continue
            occupied = np.zeros(shape[0 if normal_axis == 0 else 1], dtype=bool)
            occupied[union] = True
            gap_cells = max(
                1, int(math.ceil(config["roi"]["support_search_m"] / grid.cell_m))
            )
            joined = ndimage.binary_closing(
                occupied, structure=np.ones(gap_cells), border_value=0
            )
            labels, count = ndimage.label(joined | occupied)
            origin_normal = grid.x0 if normal_axis == 0 else grid.y0
            origin_along = grid.y0 if normal_axis == 0 else grid.x0
            rho = origin_normal + (normal_index + 1) * grid.cell_m
            for label_id in range(1, count + 1):
                support = union[labels[union] == label_id]
                if len(support) < minimum_cells:
                    continue
                types = [
                    channel
                    for channel, ids in row.items()
                    if np.intersect1d(ids, support).size
                ]
                modes[normal_axis].append(
                    {
                        "rho": float(rho),
                        "along_values": (
                            origin_along + (support + 0.5) * grid.cell_m
                        ).tolist(),
                        "along_min": float(
                            origin_along + (support.min() + 0.5) * grid.cell_m
                        ),
                        "along_max": float(
                            origin_along + (support.max() + 0.5) * grid.cell_m
                        ),
                        "support_count": int(len(support)),
                        "evidence_types": types,
                        "channel_counts": {
                            channel: int(np.intersect1d(ids, support).size)
                            for channel, ids in row.items()
                        },
                        "cell_m": grid.cell_m,
                    }
                )
    return modes


def exact_bev_provider(points, vessel, config, *, fine_grid):
    """Export the same measured evidence schema from V2's supplied local grid."""
    axes = vessel["local_axes"]
    modes = exact_raster_modes(fine_grid, config)
    evidence = []
    for normal_axis in (0, 1):
        for mode in modes[normal_axis]:
            interval = [mode["along_min"], mode["along_max"]]
            length = max(interval[1] - interval[0], fine_grid.cell_m)
            evidence.append(
                _record(
                    vessel,
                    "BEV_RECTILINEAR_PROVIDER",
                    len(evidence),
                    "AXIAL_LINE_INTERVAL",
                    dict(
                        endpoints_xy=_world_line(
                            axes, normal_axis, mode["rho"], *interval
                        ),
                        channel_counts=mode["channel_counts"],
                        observed_intervals=_measured_intervals(
                            mode["along_values"],
                            config["boundary"]["max_support_gap_m"],
                            fine_grid.cell_m,
                        ),
                    ),
                    interval,
                    dict(normal_axis=normal_axis, rho=mode["rho"]),
                    min(1.0, mode["support_count"] * fine_grid.cell_m / length),
                    fine_grid.cell_m / 2,
                    mode["evidence_types"],
                )
            )
    return evidence, fine_grid


class ExactReviewEvidence(VesselLocalEvidence):
    """Exact local side view; historical B2/S4 topology retains its evidence."""

    @classmethod
    def from_frozen(cls, cache):
        result = cls(**{f.name: getattr(cache, f.name) for f in fields(cache)})
        result._frozen_cache = cache
        # Do not reuse B2's provider list as this view's completed result.
        result.providers = None
        return result

    def structural_providers(self):
        if self.providers is None:
            self.providers = self._frozen_cache.structural_providers()
        return self.providers

    def side_candidate_providers(self):
        primary = getattr(self, "primary_cache", self)
        if primary is not self:
            return primary.side_candidate_providers()
        if not hasattr(self, "_exact_side_providers"):
            frozen = self.structural_providers()
            vessel = thaw(self.vessel.public)
            vessel["local_axes"] = [[1, 0], [0, 1]]
            bev, _ = exact_bev_provider(
                self.xyz, vessel, self.config, fine_grid=self.grid("fine")
            )
            for row in bev:
                # Provider IDs are scoped to the evidence view. A new density
                # fragment must never impersonate a frozen B2/S4 fragment.
                for key in ("provider_id", "observation_id"):
                    row[key] = row[key].replace(
                        ":BEV_RECTILINEAR_PROVIDER:", ":V2_EXACT_DENSITY_BEV:"
                    )
                row["evidence_view"] = "V2_EXACT_DENSITY_LOCAL_SIDE"
                row["coordinate_frame"] = "VESSEL_LOCAL_UV"
                row["input_vessel_axes"] = self.vessel.axes.tolist()
                row["point_index_sha256"] = self.vessel.point_index_sha256
            self._exact_side_providers = bev + [
                r for r in frozen if r["provider_type"] == "RAW3D_STRUCTURAL_PROVIDER"
            ]
        return self._exact_side_providers

    def side_candidates(self, box):
        if not hasattr(self, "_exact_side_cache"):
            view = VesselLocalEvidence(
                **{f.name: getattr(self, f.name) for f in fields(self)}
            )
            view.providers = self.side_candidate_providers()
            self._exact_side_cache = view
        return self._exact_side_cache.side_candidates(box)
