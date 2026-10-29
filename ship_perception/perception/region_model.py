"""Persistent opening identity, topology support and independent side measures."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np


TOPOLOGY_STATES = frozenset(
    ("STRONG", "SUPPORTING", "WEAK", "CONTRADICTED", "UNOBSERVED")
)
BOUNDARY_STATES = frozenset(
    (
        "MEASURED_STRONG",
        "MEASURED_WEAK",
        "MULTIMODE",
        "OCCLUDED",
        "UNOBSERVED",
        "CONSERVATIVE_PRIOR",
    )
)
REGION_STATES = frozenset(
    ("REGION_STRONG", "REGION_PARTIAL", "REGION_AMBIGUOUS", "REGION_FOV_TRUNCATED")
)


@dataclass(frozen=True)
class EvidenceSource:
    frame_id: str
    sensor_id: str
    timestamp: str
    source_point_indexes: Optional[Tuple[int, ...]]
    vessel_frame_transform: tuple
    source_vessel_ids: tuple
    ownership: str = "ACTIVE_VESSEL_PRIMARY"


@dataclass(frozen=True)
class TopologySupport:
    source_type: str
    state: str
    provider_ids: tuple = ()
    source: Optional[EvidenceSource] = None
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.state not in TOPOLOGY_STATES:
            raise ValueError("INVALID_TOPOLOGY_SUPPORT_STATE")
        if self.source_type == "HEIGHT_TRANSITION" and self.state == "STRONG":
            raise ValueError("HEIGHT_ALONE_IS_NOT_STRONG_STRUCTURE")


@dataclass(frozen=True)
class BoundaryMeasurement:
    side_id: str
    state: str = "UNOBSERVED"
    rho: Optional[float] = None
    source_type: Optional[str] = None
    source: Optional[EvidenceSource] = None
    uncertainty_m: Optional[float] = None
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.state not in BOUNDARY_STATES:
            raise ValueError("INVALID_BOUNDARY_MEASUREMENT_STATE")
        if self.state.startswith("MEASURED") and self.source_type not in (
            "RAW_PROFILE_BREAK",
            "RAW_VERTICAL_FACE",
            "RAW_INTERSECTION",
            "MULTI_SECTION_CONSENSUS",
        ):
            raise ValueError("MEASURED_BOUNDARY_REQUIRES_RAW_MEASUREMENT")


@dataclass
class OpeningRegion:
    region_id: str
    vessel_id: str
    conservative_bounds: tuple
    search_envelope: tuple
    topology_support: tuple
    source_proposal_ids: tuple = ()
    component_bounds: tuple = ()
    status: str = "REGION_AMBIGUOUS"
    boundary_measurements: dict = field(default_factory=dict)
    precision_status: str = "UNRESOLVED"
    precision_attempts: list = field(default_factory=list)
    relationship: str = "INDEPENDENT_OPENING_CANDIDATE"

    def __post_init__(self):
        if self.status not in REGION_STATES:
            raise ValueError("INVALID_OPENING_REGION_STATE")
        for box in (self.conservative_bounds, self.search_envelope):
            if (
                len(box) != 4
                or not np.isfinite(box).all()
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                raise ValueError("DEGENERATE_OPENING_REGION")
        if not self.topology_support or not any(
            s.state in ("STRONG", "SUPPORTING", "WEAK")
            and s.source_type != "VESSEL_INTERIOR_CONTEXT"
            for s in self.topology_support
        ):
            raise ValueError("OPENING_REGION_REQUIRES_INDEPENDENT_TOPOLOGY")
        for support in self.topology_support:
            src = support.source
            if (
                src
                and src.ownership == "ACTIVE_VESSEL_PRIMARY"
                and set(src.source_vessel_ids) != {self.vessel_id}
            ):
                raise ValueError("OTHER_VESSEL_CANNOT_OWN_REGION")
        a, b = self.conservative_bounds, self.search_envelope
        if any(b[i] > a[i] for i in (0, 1)) or any(b[i] < a[i] for i in (2, 3)):
            raise ValueError("SEARCH_ENVELOPE_MUST_CONTAIN_CONSERVATIVE_BOUNDS")
        if not self.boundary_measurements:
            source = next((s.source for s in self.topology_support if s.source), None)
            self.boundary_measurements = {
                name: BoundaryMeasurement(
                    name, "CONSERVATIVE_PRIOR", a[pos], source=source
                )
                for name, pos in (("U0", 0), ("U1", 2), ("V0", 1), ("V1", 3))
            }

    @property
    def bounds(self):
        return self.conservative_bounds

    def expand_search(self, bounds):
        self.search_envelope = tuple(
            np.r_[
                np.minimum(self.search_envelope[:2], bounds[:2]),
                np.maximum(self.search_envelope[2:], bounds[2:]),
            ]
        )

    def record_precision(self, result):
        """A failed or weak final measurement never removes or shrinks a region."""
        self.precision_attempts.append(
            dict(
                status=result.get("status", "UNRESOLVED"),
                error_code=result.get("error_code"),
            )
        )
        for name, side in result.get("sides", {}).items():
            if name not in self.boundary_measurements:
                continue
            # The raw refiner retains its physical acceptance and move guards.
            # An unresolved result is evidence, not permission to overwrite prior.
            if not side.get("independent_geometry_support"):
                continue
            current = self.boundary_measurements[name]
            rho = side.get("rho_after_m")
            if rho is None or not np.isfinite(rho):
                continue
            detail = dict(side)
            # A guard evaluated around another candidate's cropped anchor
            # cannot authorize movement of this region's conservative prior.
            inward = side.get("side_max_endpoint_inward_move_m", 0.0)
            if inward > 0 and side.get("rho_before_m") != current.rho:
                detail["movement_decision"] = "KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY"
                detail["region_prior_mismatch"] = True
                self.boundary_measurements[name] = BoundaryMeasurement(
                    name,
                    "CONSERVATIVE_PRIOR",
                    current.rho,
                    source=current.source,
                    detail=detail,
                )
                continue
            detail["movement_decision"] = "RAW_MEASUREMENT_AVAILABLE_PRIOR_PRESERVED"
            self.boundary_measurements[name] = BoundaryMeasurement(
                name,
                "MEASURED_WEAK",
                float(rho),
                "MULTI_SECTION_CONSENSUS",
                current.source,
                side.get("boundary_uncertainty_m"),
                detail,
            )
            self.precision_status = "PARTIAL_MEASUREMENT_AVAILABLE"

    def audit(self):
        return dict(
            region_id=self.region_id,
            vessel_id=self.vessel_id,
            conservative_bounds=list(self.conservative_bounds),
            search_envelope=list(self.search_envelope),
            search_bounds_axial=list(self.search_envelope),
            region_status=self.status,
            topology_status=self.status,
            geometry_status=self.precision_status,
            steel_edge_benchmark_status="NOT_CERTIFIED",
            operator_confirmation="PENDING",
            topology_support=[asdict(s) for s in self.topology_support],
            boundary_measurements={
                k: asdict(v) for k, v in self.boundary_measurements.items()
            },
            source_proposal_ids=list(self.source_proposal_ids),
            component_bounds=[list(b) for b in self.component_bounds],
            geometry_role="CONSERVATIVE_REGION_NOT_MEASURED_STEEL",
            relationship=self.relationship,
            precision_attempts=list(self.precision_attempts),
        )
