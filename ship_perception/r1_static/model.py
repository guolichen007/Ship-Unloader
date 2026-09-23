"""Small raw-coordinate data types. No labels or historical model inputs."""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SeedComponent:
    """One original connected seed, including its own grid coordinates."""
    origin_xy: Tuple[float, float]
    cell_m: float
    cells_rc: Tuple[Tuple[int, int], ...]
    bbox_xy: Tuple[float, float, float, float]

    def record(self):
        return dict(origin_xy=list(self.origin_xy), cell_m=self.cell_m,
                    cells_rc=[list(cell) for cell in self.cells_rc],
                    bbox_xy=list(self.bbox_xy), cell_count=len(self.cells_rc))


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    bbox_xy: Tuple[float, float, float, float]
    evidence_cells: int
    mean_drop_m: float
    scales_m: Tuple[float, ...]
    seed_components: Tuple[SeedComponent, ...] = ()

    def record(self):
        return dict(proposal_id=self.proposal_id, bbox_xy=list(self.bbox_xy),
                    evidence_cells=self.evidence_cells, mean_drop_m=self.mean_drop_m,
                    scales_m=list(self.scales_m), source="STRUCTURAL_GEOMETRY_V1",
                    seed_components=[component.record() for component in self.seed_components])


@dataclass(frozen=True)
class OpeningSeed:
    """Observed lower-return search region, not a measured steel boundary."""
    seed_id: str
    proposal_id: str
    component: SeedComponent
    source_component_ids: Tuple[int, ...]
    contour_xy: Tuple[Tuple[float, float], ...]
    scale_support: Tuple[float, ...]
    fov_status: str
    touches_scan_boundary: bool

    @property
    def bbox_xy(self):
        return self.component.bbox_xy

    @property
    def evidence_cells(self):
        return len(self.component.cells_rc)

    def record(self):
        return dict(seed_id=self.seed_id, proposal_id=self.proposal_id,
                    source_components=list(self.source_component_ids),
                    cell_m=self.component.cell_m, bbox_xy=list(self.bbox_xy),
                    scale_support=list(self.scale_support), evidence_cells=self.evidence_cells,
                    contour_points=[list(point) for point in self.contour_xy],
                    fov_status=self.fov_status,
                    touches_scan_boundary=self.touches_scan_boundary,
                    source="LOWER_RETURN_STRUCTURAL_SEED_NOT_STEEL_BOUNDARY")


@dataclass(frozen=True)
class PerimeterSegment:
    segment_id: str
    opening_seed_id: str
    rough_start: Tuple[float, float]
    rough_end: Tuple[float, float]
    tangent: Tuple[float, float]
    outward_normal: Tuple[float, float]
    length_m: float
    fov_status: str

    def record(self):
        return dict(segment_id=self.segment_id, opening_seed_id=self.opening_seed_id,
                    rough_start=list(self.rough_start), rough_end=list(self.rough_end),
                    tangent=list(self.tangent), outward_normal=list(self.outward_normal),
                    length_m=self.length_m, fov_status=self.fov_status)


@dataclass(frozen=True)
class Opening:
    opening_id: str
    proposal_id: str
    bbox_xy: Tuple[float, float, float, float]
    support_cells: int
    area_m2: float
    mean_drop_m: float

    def record(self):
        return dict(opening_id=self.opening_id, proposal_id=self.proposal_id,
                    bbox_xy=list(self.bbox_xy), support_cells=self.support_cells,
                    area_m2=self.area_m2, mean_drop_m=self.mean_drop_m)
