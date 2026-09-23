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
