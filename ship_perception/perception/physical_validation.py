"""One physical decision layer for all experts; missing evidence is not reject."""

import numpy as np
from ship_perception.r1_static.hatch_proposal import _opening_evidence
from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.heightmap_topology import _side_height_rises
from .validation import deck_ring, vessel_containment, prepare_containment
from .experts import line_families, line_height, role_families


def cropped_grid(grid, box, margin):
    low = np.floor(
        (np.asarray(box[:2]) - margin - [grid.x0, grid.y0]) / grid.cell_m
    ).astype(int)
    high = (
        np.ceil(
            (np.asarray(box[2:]) + margin - [grid.x0, grid.y0]) / grid.cell_m
        ).astype(int)
        + 1
    )
    c0, r0 = np.maximum(low, 0)
    c1, r1 = np.minimum(high, grid.count.shape[::-1])
    if c1 <= c0 or r1 <= r0:
        return grid
    return R1HeightGrid(
        grid.x0 + c0 * grid.cell_m,
        grid.y0 + r0 * grid.cell_m,
        grid.cell_m,
        *(
            getattr(grid, k)[r0:r1, c0:c1]
            for k in ("count", "q10", "median", "q90", "minimum", "maximum")
        ),
    )


def resolve_physical_modes(proposal, cache):
    box = proposal.bounds
    families = role_families(cache)
    config = cache.config
    result = []
    for axis, position, sign in ((0, 0, 1), (0, 2, -1), (1, 1, 1), (1, 3, -1)):
        span = (box[1 - axis], box[3 - axis])
        length = span[1] - span[0]
        eligible = []
        for f in families[axis]:
            if (
                not f["raw3d"]
                or not f["bev"]
                or abs(f["rho"] - box[position]) > config["roi"]["support_search_m"]
            ):
                continue
            overlap = max(
                0, min(span[1], f["interval"][1]) - max(span[0], f["interval"][0])
            )
            if (
                overlap / max(config["geometry"]["plane_inlier_m"], length)
                < config["roi"]["min_enclosure_ratio"]
            ):
                continue
            if sign not in f["supported_signs"]:
                continue
            eligible.append(f)
        nearest = (
            min(eligible, key=lambda f: abs(f["rho"] - box[position]))
            if eligible
            else None
        )
        result.append(
            nearest["provider_ids"] + (("SIGNED_RAW_ROLE", sign),) if nearest else None
        )
    return tuple(result)


class UnifiedPhysicalValidation:
    def __init__(self, cache, exterior, selection=None):
        self.cache = cache
        self.exterior = exterior
        self.selection = selection or {}
        self.containment = prepare_containment(cache.context_grid("coarse"))
        self.decisions = []

    def search_face_priority(self, box):
        """Index already measured nominal faces; this score never admits a side."""
        cache, config = self.cache, self.cache.config
        if not hasattr(cache, "structural_providers"):
            return 0
        if not hasattr(cache, "_v21_normal_priority_index"):
            cache._v21_normal_priority_index = {}
            for axis, rows in line_families(cache).items():
                rows = sorted(
                    (f for f in rows if f["raw3d"] and f["bev"] and f["normal_face"]),
                    key=lambda f: f["rho"],
                )
                cache._v21_normal_priority_index[axis] = (
                    rows,
                    np.array([f["rho"] for f in rows]),
                )
        count = 0
        search = config["roi"]["support_search_m"]
        for axis, pos, sign in [(0, 0, 1), (0, 2, -1), (1, 1, 1), (1, 3, -1)]:
            rows, positions = cache._v21_normal_priority_index[axis]
            lo = np.searchsorted(positions, box[pos] - search, side="left")
            hi = np.searchsorted(positions, box[pos] + search, side="right")
            span = (box[1 - axis], box[3 - axis])
            count += any(
                (f["rho"] - (box[axis] + box[axis + 2]) / 2) * sign < 0
                and max(
                    0.0, min(span[1], f["interval"][1]) - max(span[0], f["interval"][0])
                )
                >= config["roi"]["min_enclosure_ratio"] * (span[1] - span[0])
                for f in rows[lo:hi]
            )
        return count

    def evaluate(self, query):
        config = self.cache.config
        box = list(query.bounds)
        row = dict(
            query_id=query.query_id,
            proposal_id=query.query_id,
            vessel_hypothesis_id=query.vessel_id,
            bounds_axial=box,
            candidate_sources=query.sources,
            source_proposal_ids=[p.proposal_id for p in query.proposals],
            skipped_verified_face_count=min(
                (
                    p.evidence()["skipped_verified_face_count"]
                    for p in query.proposals
                    if "skipped_verified_face_count" in p.evidence()
                ),
                default=0,
            ),
            internal_transverse_face_coverage=min(
                (
                    p.evidence()["internal_transverse_face_coverage"]
                    for p in query.proposals
                    if "internal_transverse_face_coverage" in p.evidence()
                ),
                default=0.0,
            ),
            state="PROPOSED",
            precision_allowed=False,
            raw_edge_probe_allowed=True,
            legacy_post_b2=getattr(
                getattr(self.cache, "_v21_adapter", None), "evidence", {}
            ).get(query.query_id, {}),
        )

        def decide(state, reason):
            row.update(state=state, reason=reason)
            if state == "REJECTED":
                row["raw_edge_probe_allowed"] = False
            self.decisions.append(row)
            return row

        if not row["legacy_post_b2"] and "LEGACY" in query.sources:
            row["legacy_post_b2"] = dict(
                support_side_count=max(
                    (
                        p.evidence().get("measured_strong_side_count", 0)
                        for p in query.proposals
                        if p.expert == "LEGACY"
                    ),
                    default=0,
                ),
                derived_search_evidence=False,
            )

        if query.vessel_id != self.cache.vessel.vessel_id:
            return decide("REJECTED", "CONFIRMED_OTHER_VESSEL")
        area = (box[2] - box[0]) * (box[3] - box[1])
        if box[2] <= box[0] or box[3] <= box[1]:
            return decide("REJECTED", "DEGENERATE_GEOMETRY")
        if (
            not config["roi"]["opening_min_area_m2"]
            <= area
            <= config["roi"]["opening_max_area_m2"]
        ):
            return decide("REJECTED", "INVALID_PHYSICAL_SIZE")
        grid = self.cache.context_grid("fine")
        local = cropped_grid(
            grid,
            box,
            config["roi"]["support_search_m"] / 2
            + config["boundary"]["side_probe_m"]
            + grid.cell_m,
        )
        opening = _opening_evidence(local, box, config)
        coarse = self.cache.context_grid("coarse")
        rises, counts = _side_height_rises(coarse, coarse.occupancy, box, config)
        row.update(
            opening_evidence=opening,
            side_height_rises_m=list(rises),
            side_support_cells=[list(x) for x in counts],
        )
        row["containment"] = vessel_containment(coarse, box, config, self.containment)
        row["deck_ring"] = deck_ring(self.cache, row)
        ring = row["deck_ring"]
        row["independent_support_sides"] = ring["measured_strong_side_count"]
        row["search_normal_face_side_count"] = self.search_face_priority(box)
        row["search_normal_face_role"] = "NOMINAL_PRIORITY_ONLY_NOT_PHYSICAL_ACCEPTANCE"
        levels = [x for x in opening["outside_side_median_z"] if x is not None]
        row["exterior"] = self.exterior.review(
            box, float(np.median(levels)) if levels else None
        )
        reference = self.selection.get("reference")
        if reference is not None:
            xy = (
                np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                @ self.cache.vessel.axes
            )
            if (xy - np.array(reference["origin"])) @ np.array(
                reference["normal"]
            ) < -config["boundary"]["corner_join_m"]:
                return decide("REJECTED", "CONFIRMED_OUTSIDE_SELECTED_VESSEL")
        exterior = row["exterior"]
        coaming = sum(s["vertical_face_support"] for s in ring["sides"].values())
        if (
            exterior["status"] == "REJECT_EXTERIOR_LOW_REGION"
            and ring["measured_strong_side_count"] < 3
            and coaming < 3
        ):
            return decide("REJECTED", "STRONG_EXTERIOR_WATER_CONNECTION")
        if row["containment"]["status"] == "REJECT_OUTSIDE_VESSEL":
            # Unknown B1 attribution is never upgraded to confirmed outside.
            return decide("UNRESOLVED", "VESSEL_BOUNDARY_OBSERVATION_INCOMPLETE")
        if ring["measured_strong_side_count"] < 3:
            row["topology_disposition"] = "KEEP_REGION_BOUNDARY_MEASUREMENT_UNRESOLVED"
            return decide("UNRESOLVED", "INSUFFICIENT_INDEPENDENT_ENCLOSURE_SUPPORT")
        positive = ring["positive_height_side_count"]
        if positive < 2 and coaming < 3:
            # Missing coarse height/normal evidence is not a reason to block
            # raw investigation of an independently supported enclosure.
            # It stays partial; the unchanged raw support/role guards decide
            # whether any final review geometry can be emitted.
            row["precision_allowed"] = True
            return decide(
                "PARTIAL", "LOADED_INTERIOR_OR_EDGE_ROLE_REQUIRES_RAW_INVESTIGATION"
            )
        row["precision_allowed"] = True
        if (
            row["containment"]["status"] != "CONTAINED"
            or exterior["status"] != "ENCLOSED_OBSERVED_LOW_COMPONENT"
            or opening["status"] != "OPENING_LOWER"
        ):
            return decide("PARTIAL", "INCOMPLETE_OWNERSHIP_OR_LOADED_INTERIOR")
        return decide("VALIDATED", "OBSERVED_ENCLOSURE_READY_FOR_RAW_PRECISION")
