"""Raw-recomputed frozen topology, separate from final steel measurements.

This compatibility provider is evaluated before admission to the V2 runtime.
It intentionally preserves unresolved B5 regions. The expensive historical
prepare/review chain is not installed in the region-local production path.
"""

from dataclasses import dataclass, asdict
import hashlib
from typing import Optional, Tuple

import numpy as np

from ship_perception.r1_static.pcd_showcase import prepare_geometry
from ship_perception.r1_static.corner_topology_review import review_points
from ship_perception.r1_static.boundary_topology import _observations
from .legacy_expert import verify_legacy_sources
from .candidate_union import canonical
from .context import freeze, thaw


FROZEN_SOURCE_SHA = "51410e3f4f932fcbdcc7fc88b8f9979c60c2b954"


@dataclass(frozen=True)
class LegacyTopologyStatus:
    region_status: str
    geometry_status: str = "UNRESOLVED"
    steel_edge_benchmark_status: str = "NOT_CERTIFIED"
    operator_confirmation: str = "PENDING"


@dataclass(frozen=True)
class LegacySideEvidence:
    side_id: str
    rho_before: float
    rho_proposed: float
    rho_after: float
    max_inward_move: float
    candidate_modes: tuple
    support_before: object
    support_after: object
    frame_id: str
    sensor_id: str
    measurement_time: str
    vessel_frame_transform: tuple
    inward_move_proposed_m: float
    source_point_indexes: Optional[Tuple[int, ...]] = None
    vessel_frame_transform_direction: str = "INPUT_FRAME_TO_B1_VESSEL_UVZ"
    rho_frame: str = "RAW_RECOMPUTED_LEGACY_R2G_UV"
    status: str = "CONSERVATIVE_PRIOR"
    movement_decision: str = "KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY"

    def audit(self):
        return {name: thaw(value) for name, value in vars(self).items()}


@dataclass(frozen=True)
class LegacySeparatorEvidence:
    vessel_id: str
    diagnostic: object
    topology_status: str = "LEGACY_SEPARATOR_REVIEW_NOT_STEEL"


@dataclass(frozen=True)
class LegacyOpeningRegion:
    region_id: str
    vessel_id: str
    bounds: tuple
    axes: tuple
    source_hatch_id: str
    source_proposal_id: object
    inherited_status: str
    status: LegacyTopologyStatus
    sides: tuple
    profiles: tuple
    ownership: object

    def audit(self):
        uv = np.asarray(
            [
                [self.bounds[0], self.bounds[1]],
                [self.bounds[2], self.bounds[1]],
                [self.bounds[2], self.bounds[3]],
                [self.bounds[0], self.bounds[3]],
            ]
        )
        return dict(
            region_id=self.region_id,
            vessel_id=self.vessel_id,
            bounds_axial=list(self.bounds),
            vessel_axes=[list(a) for a in self.axes],
            conservative_boundary_xy=(uv @ np.asarray(self.axes)).tolist(),
            source_hatch_id=self.source_hatch_id,
            source_proposal_id=self.source_proposal_id,
            inherited_status=self.inherited_status,
            **asdict(self.status),
            topology_status="LEGACY_OPENING_REGION_CANDIDATE",
            geometry_role="CONSERVATIVE_REGION_NOT_FINAL_STEEL",
            candidate_available=True,
            sides={s.side_id: s.audit() for s in self.sides},
            profile_observations=thaw(self.profiles),
            local_ownership=thaw(self.ownership),
            corners=[
                dict(
                    x_m=float(p[0]),
                    y_m=float(p[1]),
                    z_observed=None,
                    z_reference_estimate=None,
                    z_reference_source=None,
                    z_uncertainty=None,
                )
                for p in uv @ np.asarray(self.axes)
            ],
        )


@dataclass(frozen=True)
class LegacyTopologyBatch:
    regions: tuple
    separators: tuple
    counts: object
    report: object

    def audit(self):
        return dict(
            schema="ship_perception.frozen_legacy_topology.1",
            source_sha=FROZEN_SOURCE_SHA,
            usage="HISTORICAL_TOPOLOGY_AUDIT_NOT_STEEL_GOLDEN",
            runtime_admission="PENDING_REAL_SCENE_RECALL_GATE",
            wall_contact_hard_gate=False,
            regions=[r.audit() for r in self.regions],
            separators=[
                dict(
                    vessel_id=s.vessel_id,
                    diagnostic=thaw(s.diagnostic),
                    topology_status=s.topology_status,
                )
                for s in self.separators
            ],
            counts=thaw(self.counts),
            legacy_review=thaw(self.report),
        )


class FrozenLegacyTopologyProvider:
    """Replay frozen prepare/review semantics without accepting final steel.

    No expected count, scene name, prior file or annotations are accepted.
    Full-scene replay is an offline audit. Runtime remains region-local until
    this compatibility result passes the historical coverage gate.
    """

    def recompute(self, frame, context, config, research, progress=None):
        context.validate_frame(frame)
        verify_legacy_sources()
        evidence = {}
        b2, b4, counts = prepare_geometry(
            frame.xyz,
            "current_raw_input",
            frame.sha256,
            config,
            research,
            progress,
            scene_context=context,
            evidence_sink=evidence,
        )
        report, _, _ = review_points(
            "current_raw_input",
            frame.xyz,
            frame.sha256,
            b2,
            b4,
            config,
            research,
            scene_context=context,
            height_rim=False,
        )
        return self.collect(frame, context, b4, report, evidence, counts, config)

    @staticmethod
    def collect(frame, context, b4, report, evidence, counts, config):
        """Keep every reviewed region, including empty-corner UNRESOLVED rows.

        A frozen side's proposed movement remains evidence, not an applied
        steel correction. The original Raw-recomputed R2G boundary is the
        conservative prior. No side or cargo-height gate deletes the region.
        """
        context.validate_frame(frame)
        originals = {
            (r["vessel_hypothesis_id"], r["hatch_id"]): r for r in b4["rectangles"]
        }
        profiles, _ = _observations(
            evidence.get("structural", {}).get("segment_resolutions", [])
        )
        regions, separators = [], []
        for vessel in report["vessels"]:
            vessel_id = vessel["vessel_hypothesis_id"]
            owned = next(v for v in context.vessels if v.vessel_id == vessel_id)
            ownership = dict(
                owned.identity(),
                point_selection=vessel.get("point_selection"),
                source_point_indexes_status="LEGACY_INLIER_INDEX_TRACE_UNAVAILABLE",
            )
            from ship_perception.r1_static.height_grid import R1HeightGrid

            grid = R1HeightGrid.from_points(
                frame.xyz[owned.point_indexes], config["geometry"]["coarse_voxel_m"]
            )
            vessel_transform = np.eye(4)
            vessel_transform[:2, :2] = owned.axes
            separators.append(
                LegacySeparatorEvidence(
                    vessel_id,
                    freeze(
                        dict(
                            shared_separator=vessel.get("shared_separator"),
                            u_boundary_chain=vessel.get("u_boundary_chain"),
                            section_rho_consensus=vessel.get("section_rho_consensus"),
                            selection_blocker=vessel.get("selection_blocker"),
                        )
                    ),
                )
            )
            for row in vessel["rectangles"]:
                original = originals[(vessel_id, row["hatch_id"])]
                axes = np.asarray(original["axes"], dtype=float)
                prior = np.asarray(original["polygon_before"]) @ axes.T
                bounds = tuple(np.r_[prior.min(0), prior.max(0)])
                ident = (
                    "LT"
                    + hashlib.sha256(
                        canonical([vessel_id, bounds, row["hatch_id"]]).encode()
                    ).hexdigest()[:16]
                )
                sides = []
                for name, position, sign in (
                    ("U0", 0, 1),
                    ("U1", 2, -1),
                    ("V0", 1, 1),
                    ("V1", 3, -1),
                ):
                    old = original["sides"][name]
                    proposed = (
                        row.get("side_stage_arbitration", {})
                        .get(name, {})
                        .get("final_rho", old["rho_after"])
                    )
                    sides.append(
                        LegacySideEvidence(
                            name,
                            bounds[position],
                            float(proposed),
                            bounds[position],
                            0.0,
                            tuple(freeze(m) for m in old.get("candidate_modes", [])),
                            None,
                            None,
                            frame.frame_id,
                            str(frame.provenance.get("sensor_id", "UNKNOWN")),
                            frame.stamp,
                            tuple(tuple(a) for a in vessel_transform),
                            max(0.0, sign * (proposed - bounds[position])),
                        )
                    )
                # Profiles are restricted to the region and its B1-owned cells.
                # The scoped observations are evidence points, not Raw inliers.
                local_profiles = []
                for p in profiles:
                    xy = np.asarray(p["xy"])
                    uv = xy @ axes.T
                    if not np.all((uv >= bounds[:2]) & (uv <= bounds[2:])):
                        continue
                    rr, cc = grid.cell_indices(xy.reshape(1, 2))
                    if (
                        0 <= rr[0] < grid.count.shape[0]
                        and 0 <= cc[0] < grid.count.shape[1]
                        and grid.occupancy[rr[0], cc[0]]
                    ):
                        local_profiles.append(
                            freeze(dict(p, source_vessel_id=vessel_id))
                        )
                regions.append(
                    LegacyOpeningRegion(
                        ident,
                        vessel_id,
                        bounds,
                        tuple(tuple(a) for a in axes),
                        row["hatch_id"],
                        row.get("hatch_hypothesis_id"),
                        row["status"],
                        LegacyTopologyStatus(
                            "REGION_PARTIAL"
                            if row["status"].startswith("PARTIAL")
                            else "REGION_AMBIGUOUS"
                        ),
                        tuple(sides),
                        tuple(local_profiles),
                        freeze(ownership),
                    )
                )
        return LegacyTopologyBatch(
            tuple(regions), tuple(separators), freeze(counts), freeze(report)
        )
