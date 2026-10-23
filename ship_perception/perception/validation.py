"""Measured exterior connectivity, vessel containment and same-vessel deck rings."""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.hatch_proposal import _side_evidence


@dataclass
class ExteriorTopology:
    grid: object
    owned: np.ndarray
    config: dict
    components: dict = field(default_factory=dict)
    level_evaluations: int = 0

    @classmethod
    def build(cls, frame, vessel, config):
        uvz = np.column_stack((frame.xyz[:, :2] @ vessel.axes.T, frame.xyz[:, 2]))
        grid = R1HeightGrid.from_points(uvz, config["geometry"]["coarse_voxel_m"])
        rows, cols = grid.cell_indices(uvz[vessel.point_indexes, :2])
        owned = np.zeros(grid.count.shape, dtype=bool)
        owned[rows, cols] = True
        return cls(grid, owned, config)

    def at_level(self, level):
        key = float(level)
        if key not in self.components:
            self.level_evaluations += 1
            observed = self.grid.occupancy
            low = observed & (
                self.grid.median < level - self.config["roi"]["opening_drop_m"]
            )
            # Missing cells never become flood-fill paths, including tiny gaps.
            labels, count = ndimage.label(low, structure=np.ones((3, 3), bool))
            boundary = np.zeros(low.shape, dtype=bool)
            boundary[[0, -1], :] = True
            boundary[:, [0, -1]] = True
            exterior = low & (~self.owned | boundary)
            external_ids = set(np.unique(labels[exterior]).tolist()) - {0}
            # Cache eviction changes recomputation only, never the measured
            # level or classification. Keep label rasters within 64 MiB.
            capacity = max(
                1, (64 * 1024 * 1024) // max(1, labels.nbytes + boundary.nbytes)
            )
            while len(self.components) >= capacity:
                self.components.pop(next(iter(self.components)))
            self.components[key] = (labels, external_ids, boundary)
        return self.components[key]

    def review(self, box, level):
        if level is None:
            return dict(
                status="UNRESOLVED_EXTERIOR_CONNECTIVITY", exterior_low_connected=None
            )
        labels, external_ids, boundary = self.at_level(level)
        u, v = self.grid.centers()
        inset = self.config["boundary"]["side_probe_m"]
        inside = (
            (u >= box[0] + inset)
            & (u <= box[2] - inset)
            & (v >= box[1] + inset)
            & (v <= box[3] - inset)
        )
        selected = labels[inside & (labels > 0)]
        if not len(selected):
            return dict(
                status="NO_LOWER_INTERIOR_COMPONENT", exterior_low_connected=False
            )
        ids, counts = np.unique(selected, return_counts=True)
        component = int(ids[np.argmax(counts)])
        member = labels == component
        outside = int(np.count_nonzero(member & ~self.owned))
        total = int(member.sum())
        overlap = float(
            np.count_nonzero(inside & member)
            / max(1, np.count_nonzero(inside & self.grid.occupancy))
        )
        connected = component in external_ids
        touches = bool(np.any(member & boundary))
        if (
            connected
            and outside / max(1, total) >= self.config["roi"]["min_enclosure_ratio"]
            and overlap >= self.config["roi"]["min_enclosure_ratio"]
        ):
            status = "REJECT_EXTERIOR_LOW_REGION"
        elif connected:
            status = "UNRESOLVED_EXTERIOR_CONNECTIVITY"
        else:
            status = "ENCLOSED_OBSERVED_LOW_COMPONENT"
        return dict(
            status=status,
            exterior_low_connected=connected,
            component_id=component,
            component_cells=total,
            external_owned_complement_cells=outside,
            candidate_component_overlap=overlap,
            touches_scene_grid=touches,
            missing_cells_traversed=False,
        )


def prepare_containment(grid):
    observed = grid.occupancy
    return (
        ndimage.binary_fill_holes(observed),
        ndimage.distance_transform_edt(~observed) * grid.cell_m,
    )


def vessel_containment(grid, box, config, prepared=None):
    """Containment is ownership context; filled cells never become observations."""
    observed = grid.occupancy
    envelope, distance = prepare_containment(grid) if prepared is None else prepared
    corners = np.asarray(
        [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
    )
    probes = np.vstack((corners, corners.mean(axis=0)))
    rows, cols = grid.cell_indices(probes)
    valid = (
        (rows >= 0)
        & (cols >= 0)
        & (rows < observed.shape[0])
        & (cols < observed.shape[1])
    )
    supported = np.zeros(5, bool)
    enclosed = np.zeros(5, bool)
    distances = [None] * 5
    tolerance = max(config["boundary"]["corner_join_m"], np.sqrt(2) * grid.cell_m)
    for index in np.flatnonzero(valid):
        r, c = rows[index], cols[index]
        distances[index] = float(distance[r, c])
        supported[index] = distance[r, c] <= tolerance
        enclosed[index] = envelope[r, c]
    status = (
        "CONTAINED"
        if supported.all()
        else (
            "PARTIAL_BOUNDARY_AMBIGUOUS"
            if enclosed[-1] or supported[-1]
            else "REJECT_OUTSIDE_VESSEL"
        )
    )
    return dict(
        status=status,
        centroid_inside_support=bool(supported[-1]),
        centroid_inside_envelope=bool(enclosed[-1]),
        corner_support=supported[:4].tolist(),
        distance_to_observed_support_m=distances,
        inferred_envelope_is_measurement=False,
    )


def deck_ring(cache, candidate):
    lines = cache.structural_providers()
    seed = dict(seed_type="HEIGHT_REGION", anchor_axis=None)
    sides = _side_evidence(candidate["bounds_axial"], lines, cache.config, seed)
    for side, rise, support in zip(
        ("U0", "U1", "V0", "V1"),
        candidate.get("side_height_rises_m", [None] * 4),
        candidate.get("side_support_cells", [[0, 0]] * 4),
    ):
        row = sides[side]
        provider_rows = [
            p for p in lines if p["provider_id"] in row["supporting_provider_ids"]
        ]
        kinds = {p["provider_type"] for p in provider_rows}
        row.update(
            same_vessel_support=bool(provider_rows),
            source_vessel_ids=sorted(
                set(p["vessel_hypothesis_id"] for p in provider_rows)
            ),
            raw3d_support="RAW3D_STRUCTURAL_PROVIDER" in kinds,
            bev_support="BEV_RECTILINEAR_PROVIDER" in kinds,
            vertical_face_support=any(
                "RAW3D_VERIFIED_VERTICAL_FACE" in p["evidence_types"]
                for p in provider_rows
            ),
            signed_height_rise_m=rise,
            occupied_support_cells=support,
            positive_height_transition=rise is not None
            and rise >= cache.config["roi"]["opening_drop_m"],
        )
    strong = sum(r["state"] == "MEASURED_STRONG" for r in sides.values())
    dual = sum(r["raw3d_support"] and r["bev_support"] for r in sides.values())
    return dict(
        schema="ship_perception.deck_ring.1",
        sides=sides,
        measured_strong_side_count=strong,
        dual_provider_side_count=dual,
        positive_height_side_count=sum(
            r["positive_height_transition"] for r in sides.values()
        ),
        steel_role_status="UNVERIFIED",
    )
