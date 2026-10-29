"""Vessel-local line families with explicit observable and missing intervals.

These are topology witnesses and search nominations. They never rewrite a
measured rho, turn cargo height into steel, or enumerate rectangle products.
"""

from dataclasses import dataclass, asdict
import hashlib

import numpy as np

from .candidate_union import canonical
from .region_model import EvidenceSource, TopologySupport


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([float(start), float(end)])
    return tuple(tuple(i) for i in merged)


@dataclass(frozen=True)
class StructuralBoundaryFamily:
    family_id: str
    vessel_id: str
    normal_axis: int
    rho_consensus: float
    rho_modes: tuple
    observed_intervals: tuple
    segments: tuple
    provider_ids: tuple
    provider_types: tuple
    evidence_types: tuple
    source: EvidenceSource
    direction_consistency: float
    signed_side_consistency: str

    def __post_init__(self):
        if self.source.ownership != "ACTIVE_VESSEL_PRIMARY" or set(
            self.source.source_vessel_ids
        ) != {self.vessel_id}:
            raise ValueError("STRUCTURAL_FAMILY_REQUIRES_ACTIVE_VESSEL_PRIMARY")

    @property
    def observable_support_ratio(self):
        observable = sum(
            b - a
            for a, b, status in self.segments
            if status in ("OBSERVED_SUPPORTED", "OBSERVED_CONTRADICTED")
        )
        supported = sum(
            b - a for a, b, status in self.segments if status == "OBSERVED_SUPPORTED"
        )
        return supported / observable if observable else None

    def audit(self):
        intervals = self.observed_intervals
        return dict(
            asdict(self),
            observable_support_ratio=self.observable_support_ratio,
            span_distribution_m=[b - a for a, b in intervals],
            gap_distribution_m=[b[0] - a[1] for a, b in zip(intervals, intervals[1:])],
            coordinate_role="TOPOLOGY_FAMILY_NOT_MEASURED_STEEL",
            rho_copy_allowed=False,
            unobserved_is_negative_evidence=False,
            direction_consistency_role="PROVIDER_NOMINAL_AXES_NOT_RAW_LINE_FIT",
            source_point_index_role="RAW_FAMILY_BAND_INPUT_NOT_MEASURED_INLIERS",
        )

    def topology(self):
        paired = {"RAW3D_STRUCTURAL_PROVIDER", "BEV_RECTILINEAR_PROVIDER"}.issubset(
            self.provider_types
        )
        return TopologySupport(
            "STRUCTURAL_LINE_FAMILY",
            "SUPPORTING" if paired else "WEAK",
            self.provider_ids,
            self.source,
            self.audit(),
        )


def visibility_segments(along, cell_m, raw_counts, supported_intervals):
    """No provider support in a blank interval is unobserved, never contradicted.

    Raw occupancy alone cannot prove an expected coaming is absent. A measured
    continuous surface may later supply explicit contradictory evidence.
    """
    supported = np.zeros(len(along), dtype=bool)
    for start, end in supported_intervals:
        supported |= (along + cell_m / 2 >= start) & (along - cell_m / 2 <= end)
    result = []
    for value, count, has_support in zip(along, raw_counts, supported):
        if count == 0:
            state = "UNOBSERVED"
        elif has_support:
            state = "OBSERVED_SUPPORTED"
        else:
            state = "OCCLUDED_OR_SPARSE"
        start, end = float(value - cell_m / 2), float(value + cell_m / 2)
        if result and result[-1][2] == state:
            result[-1] = (result[-1][0], end, state)
        else:
            result.append((start, end, state))
    return tuple(result)


def build_structural_families(cache):
    primary = getattr(cache, "primary_cache", cache)
    providers = [
        p
        for p in primary.structural_providers()
        if p.get("geometry_type") == "AXIAL_LINE_INTERVAL"
        and p.get("vessel_hypothesis_id") == primary.vessel.vessel_id
    ]
    config = cache.config
    grid = primary.grid("fine")
    u, v = grid.axis_centers()
    frame = getattr(cache, "input_frame", None)
    transform = np.eye(4)
    transform[:2, :2] = primary.vessel.axes
    output = []
    for axis in (0, 1):
        rows = sorted(
            (p for p in providers if p["rho_or_region"].get("normal_axis") == axis),
            key=lambda p: (p["rho_or_region"]["rho"], p["provider_id"]),
        )
        groups = []
        for row in rows:
            # Anchor each rho cluster rather than allowing transitive drift.
            if (
                groups
                and row["rho_or_region"]["rho"] - groups[-1][0]["rho_or_region"]["rho"]
                <= config["boundary"]["corner_join_m"]
            ):
                groups[-1].append(row)
            else:
                groups.append([row])
        for group in groups:
            intervals = []
            for p in group:
                if p["provider_type"] == "RAW3D_STRUCTURAL_PROVIDER":
                    intervals.extend(
                        p.get("source_geometry", {}).get(
                            "observed_intervals", [p["support_interval"]]
                        )
                    )
            if not intervals:
                continue
            intervals = merge_intervals(intervals)
            rhos = tuple(float(p["rho_or_region"]["rho"]) for p in group)
            rho = float(np.median(rhos))
            along, normal = (v, u) if axis == 0 else (u, v)
            band = max(
                config["boundary"]["profile_step_m"],
                config["geometry"]["refine_voxel_m"],
            )
            near = abs(normal - rho) <= band
            counts = (
                grid.count[:, near].sum(1) if axis == 0 else grid.count[near, :].sum(0)
            )
            low, high = intervals[0][0], intervals[-1][1]
            in_span = (along + grid.cell_m / 2 >= low) & (
                along - grid.cell_m / 2 <= high
            )
            segments = visibility_segments(
                along[in_span], grid.cell_m, counts[in_span], intervals
            )
            xyz = primary.xyz
            mask = (
                (abs(xyz[:, axis] - rho) <= band)
                & (xyz[:, 1 - axis] >= low)
                & (xyz[:, 1 - axis] <= high)
            )
            raw_indexes = tuple(
                int(i) for i in primary.vessel.point_indexes[np.flatnonzero(mask)]
            )
            source = EvidenceSource(
                frame.frame_id if frame else "UNKNOWN",
                (
                    str(frame.provenance.get("sensor_id", "UNKNOWN"))
                    if frame
                    else "UNKNOWN"
                ),
                frame.stamp if frame else "UNKNOWN",
                raw_indexes,
                tuple(tuple(r) for r in transform),
                (primary.vessel.vessel_id,),
            )
            ids = tuple(sorted(p["provider_id"] for p in group))
            ident = (
                "SF"
                + hashlib.sha256(
                    canonical([primary.vessel.vessel_id, axis, ids]).encode()
                ).hexdigest()[:16]
            )
            output.append(
                StructuralBoundaryFamily(
                    ident,
                    primary.vessel.vessel_id,
                    axis,
                    rho,
                    rhos,
                    intervals,
                    segments,
                    ids,
                    tuple(sorted(set(p["provider_type"] for p in group))),
                    tuple(sorted(set(t for p in group for t in p["evidence_types"]))),
                    source,
                    1.0,
                    "UNRESOLVED_REQUIRES_LOCAL_RAW_ROLE",
                )
            )
    return tuple(output)


def attach_family_search(regions, families, config):
    """Expand local search without cropping prior or collapsing region identity."""
    associations = []
    for region in regions:
        pairs = []
        for pos in (1, 3):
            nearby = [
                f
                for f in families
                if f.normal_axis == 1
                and abs(f.rho_consensus - region.bounds[pos])
                <= config["roi"]["support_search_m"]
                and any(
                    b >= region.bounds[0] and a <= region.bounds[2]
                    for a, b in f.observed_intervals
                )
            ]
            pairs.append(
                min(
                    nearby,
                    key=lambda f: (
                        abs(f.rho_consensus - region.bounds[pos]),
                        f.family_id,
                    ),
                )
                if nearby
                else None
            )
        if all(pairs) and pairs[0].family_id != pairs[1].family_id:
            extent = [i for f in pairs for i in f.observed_intervals]
            envelope = (
                min(a for a, _ in extent),
                region.search_envelope[1],
                max(b for _, b in extent),
                region.search_envelope[3],
            )
            region.expand_search(envelope)
            region.topology_support += tuple(f.topology() for f in pairs)
            associations.append(
                dict(
                    region_id=region.region_id,
                    longitudinal_family_ids=[f.family_id for f in pairs],
                    relationship="SHARED_SEARCH_ONLY_IDENTITY_NOT_MERGED",
                    conservative_boundary_modified=False,
                    search_envelope=list(region.search_envelope),
                )
            )
    return associations
