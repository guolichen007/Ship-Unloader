"""Index-preserving selected-vessel crop and explicit exterior attribution."""

from dataclasses import dataclass
import numpy as np
from ship_perception.r1_static.height_grid import R1HeightGrid
from .measurement import RawSpatialIndex
from .validation import ExteriorTopology


@dataclass(frozen=True)
class SelectedVesselRawContext:
    vessel: object
    measurement: object
    raw_point_indexes: np.ndarray
    exterior_point_indexes: np.ndarray
    other_vessel_indexes: np.ndarray
    site_reference: object

    @classmethod
    def build(cls, frame, scene, selection, config):
        ident = selection.get("active_vessel_id")
        if ident is None:
            raise ValueError("ACTIVE_BERTH_VESSEL_UNRESOLVED")
        vessel = next(v for v in scene.vessels if v.vessel_id == ident)
        measurement = scene.boundary_context(ident)
        owned = measurement.point_indexes
        xy = frame.xyz[owned, :2]
        margin = config["roi"]["support_search_m"]
        box = [*xy.min(0) - margin, *xy.max(0) + margin]
        nearby = RawSpatialIndex(frame.xyz).query_indexes(box)
        others = [
            v.point_indexes
            for v in scene.vessels
            if v.vessel_id != ident
            and v.public["classification_status"] == "VESSEL_HYPOTHESIS"
        ]
        other = (
            np.sort(np.concatenate(others)) if others else np.array([], dtype=np.int64)
        )
        # A different known core never supplies raw precision points. Unknown
        # context stays explicitly unknown; proximity is not ownership proof.
        outside = nearby[~np.isin(nearby, owned) & ~np.isin(nearby, other)]
        return cls(
            vessel, measurement, owned, outside, other, selection.get("reference")
        )

    def exterior(self, frame, config):
        indexes = np.unique(np.r_[self.raw_point_indexes, self.exterior_point_indexes])
        points = frame.xyz[indexes]
        uvz = np.column_stack((points[:, :2] @ self.vessel.axes.T, points[:, 2]))
        grid = R1HeightGrid.from_points(uvz, config["geometry"]["coarse_voxel_m"])
        owned = np.zeros(grid.count.shape, dtype=bool)
        core = np.isin(indexes, self.vessel.point_indexes)
        rows, cols = grid.cell_indices(uvz[core, :2])
        owned[rows, cols] = True
        unknown = grid.occupancy & ~owned
        reference = self.site_reference
        if reference is not None:
            d = (points[:, :2] - np.asarray(reference["origin"])) @ np.asarray(
                reference["normal"]
            )
            confirmed = d < -config["boundary"]["corner_join_m"]
            rows, cols = grid.cell_indices(uvz[confirmed, :2])
            unknown[rows, cols] = False
        return ExteriorTopology(
            grid,
            owned,
            config,
            unresolved_context=unknown,
            boundary_seeds_confirmed=False,
        )

    def audit(self):
        return dict(
            selected_vessel_id=self.vessel.vessel_id,
            raw_point_count=len(self.raw_point_indexes),
            exterior_context_point_count=len(self.exterior_point_indexes),
            other_vessel_points_excluded=len(self.other_vessel_indexes),
            raw_point_indexes_preserved=True,
            raw_points_downsampled=False,
            core_role="CONFIRMED_VESSEL_CORE",
            context_role="UNKNOWN_CONTEXT",
            exterior_seed_role="CONFIRMED_EXTERIOR_ONLY",
        )
