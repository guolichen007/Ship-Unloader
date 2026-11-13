"""Measured vessel beam and soft layout families; never a coordinate setter."""

import copy
import numpy as np


def weighted_median(values, weights):
    values, weights = np.asarray(values), np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="stable")
    return float(
        values[order][
            np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2, side="left")
        ]
    )


class VesselLayoutModel:
    def __init__(self, cache):
        self.cache = cache
        self.width_family = None
        self.length_family = None
        self.centerline_family = None

    def local_beam(self, bounds):
        grid = self.cache.context_grid("coarse")
        u, v = grid.axis_centers()
        columns = np.flatnonzero((u >= bounds[0]) & (u <= bounds[2]))
        ports, stars = [], []
        for column in columns:
            rows = np.flatnonzero(grid.occupancy[:, column])
            if len(rows):
                ports.append(float(v[rows[0]]))
                stars.append(float(v[rows[-1]]))
        if not ports:
            return dict(status="UNKNOWN_LOCAL_BEAM")
        port, star = float(np.median(ports)), float(np.median(stars))
        beam = star - port
        if beam <= 0:
            return dict(status="UNKNOWN_LOCAL_BEAM")
        port_margin = bounds[1] - port
        star_margin = star - bounds[3]
        return dict(
            status="OBSERVED_SUPPORT_BEAM_NOT_CERTIFIED_HULL",
            local_vessel_beam_m=beam,
            port_margin_m=float(port_margin),
            starboard_margin_m=float(star_margin),
            hatch_to_beam_ratio=float((bounds[3] - bounds[1]) / beam),
            margin_symmetry=float(abs(port_margin - star_margin) / beam),
            measured_support_sections=len(ports),
            support_limits_m=[port, star],
        )

    def fit(self, measurements):
        # Input is physical queries, not thousands of duplicate expert votes.
        strong = [
            row for row in measurements if row.get("independent_support_sides", 0) >= 3
        ]
        if len(strong) < 2:
            return self
        weights = [max(1, row["independent_support_sides"]) for row in strong]
        for name, values in (
            ("width_family", [r["bounds"][3] - r["bounds"][1] for r in strong]),
            ("length_family", [r["bounds"][2] - r["bounds"][0] for r in strong]),
            (
                "centerline_family",
                [(r["bounds"][1] + r["bounds"][3]) / 2 for r in strong],
            ),
        ):
            median = weighted_median(values, weights)
            setattr(
                self,
                name,
                dict(
                    median_m=median,
                    mad_m=weighted_median(abs(np.asarray(values) - median), weights),
                    member_count=len(strong),
                ),
            )
        return self

    def review(self, bounds, orientation_angle_deg=0.0):
        original = tuple(bounds)
        width = bounds[3] - bounds[1]
        length = bounds[2] - bounds[0]
        tolerance = self.cache.config["boundary"]["corner_join_m"]
        diagnostics = dict(
            measured_width_m=float(width),
            measured_length_m=float(length),
            layout_median_width_m=None,
            layout_prior_used=False,
            boundary_modified=False,
            local_beam=self.local_beam(bounds),
            orientation_penalty=float(
                abs(orientation_angle_deg)
                / self.cache.config["boundary"]["merge_angle_deg"]
            ),
        )
        penalty = diagnostics["orientation_penalty"]
        beam = diagnostics["local_beam"]
        if "local_vessel_beam_m" in beam:
            measured_beam = beam["local_vessel_beam_m"]
            # Both observed deck margins contribute continuously. There is no
            # preferred width ratio and no coordinate or admission gate here.
            penalty += (
                max(0.0, beam["port_margin_m"]) + max(0.0, beam["starboard_margin_m"])
            ) / measured_beam + beam["margin_symmetry"]
        if self.width_family is not None:
            deviation = width - self.width_family["median_m"]
            width_score = abs(deviation) / max(tolerance, self.width_family["mad_m"])
            length_score = abs(length - self.length_family["median_m"]) / max(
                tolerance, self.length_family["mad_m"]
            )
            center = (bounds[1] + bounds[3]) / 2
            center_score = abs(center - self.centerline_family["median_m"]) / max(
                tolerance, self.centerline_family["mad_m"]
            )
            # Width is stronger than length; both affect ranking only. End
            # holds and single holds cannot be rejected by a family statistic.
            penalty += width_score + length_score / 4 + center_score
            diagnostics.update(
                layout_median_width_m=self.width_family["median_m"],
                width_deviation_m=float(deviation),
                layout_prior_used=True,
                width_family=copy.deepcopy(self.width_family),
                centerline_family=copy.deepcopy(self.centerline_family),
                anomaly="LAYOUT_WIDTH_OUTLIER" if width_score > 1 else None,
            )
        diagnostics["ranking_penalty"] = float(penalty)
        assert tuple(bounds) == original
        return diagnostics


def cross_deck_evidence(cache, left, right):
    if not hasattr(cache, "_v2_cross_decks"):
        cache._v2_cross_decks = {}
    key = (tuple(left), tuple(right))
    if key not in cache._v2_cross_decks:
        cache._v2_cross_decks[key] = _cross_deck_evidence(cache, left, right)
    return copy.deepcopy(cache._v2_cross_decks[key])


def _cross_deck_evidence(cache, left, right):
    """Measured intervening band and two physical faces, not a cargo-ridge vote."""
    low, high = left[2], right[0]
    if high <= low:
        return dict(status="NO_SEPARATE_CROSS_DECK", can_split=False)
    v0, v1 = max(left[1], right[1]), min(left[3], right[3])
    if v1 <= v0:
        return dict(status="NO_TRANSVERSE_OVERLAP", can_split=False)
    grid = cache.context_grid("fine")
    u, v = grid.centers()
    band = (u >= low) & (u <= high) & (v >= v0) & (v <= v1) & grid.occupancy
    providers = [
        p
        for p in cache.structural_providers()
        if p["rho_or_region"].get("normal_axis") == 0
        and low - cache.config["boundary"]["corner_join_m"]
        <= p["rho_or_region"]["rho"]
        <= high + cache.config["boundary"]["corner_join_m"]
        and p["support_interval"][1] >= v1
        and p["support_interval"][0] <= v0
    ]
    raw = [
        p for p in providers if "RAW3D_VERIFIED_VERTICAL_FACE" in p["evidence_types"]
    ]
    positions = sorted(p["rho_or_region"]["rho"] for p in raw)
    distinct = bool(
        positions
        and positions[-1] - positions[0] > cache.config["geometry"]["plane_inlier_m"]
    )
    enough = int(band.sum()) >= cache.config["roi"]["min_deck_support_cells"]
    probe = cache.config["roi"]["deck_patch_width_m"]
    strips = (
        (u >= low)
        & (u <= high)
        & (((v >= v0 - probe) & (v < v0)) | ((v > v1) & (v <= v1 + probe)))
        & grid.occupancy
    )
    band_z = grid.median[band]
    flat = bool(
        len(band_z)
        and np.median(abs(band_z - np.median(band_z)))
        <= cache.config["geometry"]["plane_inlier_m"]
    )
    deck_relation = bool(
        strips.any()
        and len(band_z)
        and abs(np.median(band_z) - np.median(grid.median[strips]))
        <= cache.config["roi"]["opening_drop_m"]
    )
    qualified = enough and distinct and flat and deck_relation
    return dict(
        status=(
            "MEASURED_CROSS_DECK_CANDIDATE"
            if qualified
            else "UNRESOLVED_CROSS_DECK_OR_CARGO_RIDGE"
        ),
        can_split=bool(qualified),
        deck_height_relation=deck_relation,
        flat_band_supported=flat,
        observed_band_cells=int(band.sum()),
        median_band_z_m=float(np.median(grid.median[band])) if band.any() else None,
        measured_face_provider_ids=[p["provider_id"] for p in raw],
        geometry_from_layout=False,
        steel_edge_certified=False,
    )
