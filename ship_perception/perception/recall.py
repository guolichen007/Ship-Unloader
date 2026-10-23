"""Bounded physical association and measured separator faces, before precision."""

import copy

import numpy as np
from scipy import ndimage

from ship_perception.r1_static.hatch_proposal import (
    _association_seeds,
    _evaluate_seed,
    _opening_evidence,
)
from ship_perception.r1_static.heightmap_topology import _side_height_rises
from ship_perception.r1_static.corner_topology_review import _separator_split
from .context import thaw
from .local_evidence import local_height_recall
from .validation import deck_ring, vessel_containment, prepare_containment


def box_overlap(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    return overlap / max(
        1e-9, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    )


def make_local_b2(cache, providers):
    vessel = thaw(cache.vessel.public)
    vessel["local_axes"] = [[1, 0], [0, 1]]
    return dict(
        scene=dict(vessel_hypotheses=[vessel]),
        per_vessel=[
            dict(
                vessel_hypothesis_id=cache.vessel.vessel_id,
                vessel_classification_status=vessel["classification_status"],
                provider_evidence=providers,
                hatch_hypotheses=[],
            )
        ],
    )


def propose_openings(cache, research):
    heights, height_audit = local_height_recall(cache, research)
    providers = cache.structural_providers()
    evidence = list(providers)
    for row in heights:
        evidence.append(
            dict(
                geometry_type="REGION_XY",
                provider_id=row["proposal_id"],
                source_geometry=dict(axial_bounds=row["bounds_axial"]),
            )
        )
    low, high = cache.xyz[:, :2].min(0), cache.xyz[:, :2].max(0)
    seeds = _association_seeds(evidence, [*low, *high], cache.config)
    vessel = thaw(cache.vessel.public)
    vessel["local_axes"] = [[1, 0], [0, 1]]
    candidates, rejected = [], []
    grid = cache.grid("coarse")
    containment = prepare_containment(grid)
    for seed in seeds:
        box = seed["bounds_axial"]
        if (box[2] - box[0]) * (box[3] - box[1]) > cache.config["roi"][
            "opening_max_area_m2"
        ]:
            rejected.append(dict(reason="ABOVE_PHYSICAL_SEARCH_AREA", bounds_axial=box))
            continue
        scored = _evaluate_seed(
            seed, providers, cache.grid("fine"), vessel, cache.config, ()
        )
        if scored is None:
            rejected.append(
                dict(reason="UNRESOLVED_SUB_RESOLUTION_REGION", bounds_axial=box)
            )
            continue
        if scored["measured_strong_side_count"] < 3:
            rejected.append(
                dict(reason="INSUFFICIENT_ENCLOSURE_SIDES", bounds_axial=box)
            )
            continue
        rises, support = _side_height_rises(grid, grid.occupancy, box, cache.config)
        row = dict(
            proposal_id="P{:05d}".format(len(candidates)),
            vessel_hypothesis_id=cache.vessel.vessel_id,
            bounds_axial=list(box),
            seed_type=seed["seed_type"],
            source_provider_ids=seed["source_provider_ids"],
            side_height_rises_m=list(rises),
            side_support_cells=[list(v) for v in support],
            opening_evidence=scored["opening_evidence"],
            role_conflicts=scored["role_conflicts"],
            height_rise_side_count=sum(
                v is not None and v >= cache.config["roi"]["opening_drop_m"]
                for v in rises
            ),
            geometry_role="REVIEW_PROPOSAL_NOT_STEEL",
        )
        row["deck_ring"] = deck_ring(cache, row)
        row["containment"] = vessel_containment(grid, box, cache.config, containment)
        if row["containment"]["status"] == "REJECT_OUTSIDE_VESSEL":
            rejected.append(dict(reason="REJECT_OUTSIDE_VESSEL", bounds_axial=box))
            continue
        # A narrow raised strip surrounded by lower returns is a competing
        # structural role. Keep the evidence, never number it as an opening.
        if scored["opening_evidence"].get("raised_strip_with_lower_regions_both_sides"):
            rejected.append(
                dict(reason="RAISED_STRIP_ROLE_AMBIGUOUS", bounds_axial=box)
            )
            continue
        candidates.append(row)
    return candidates, dict(
        height=height_audit,
        height_candidates=heights,
        association_count=len(seeds),
        rejected_candidates=rejected,
        provider_count=len(providers),
        accepted_for_validation=len(candidates),
    )


def validate_and_select(cache, candidates, exterior):
    """Conservative enclosures survive; large envelopes cannot swallow observed cells."""
    grid = cache.grid("fine")
    config = cache.config
    eligible, rejected = [], []
    for row in candidates:
        box = row["bounds_axial"]
        opening = row["opening_evidence"]
        # Only low interior cells participate in exterior-low rejection;
        # a raised cargo surface is not the surrounding low deck ring.
        level = opening.get("inside_median_z")
        outside_values = [
            v for v in opening.get("outside_side_median_z", []) if v is not None
        ]
        ring_level = float(np.median(outside_values)) if outside_values else None
        row["exterior"] = exterior.review(box, ring_level)
        raised = opening["status"] == "RAISED_INTERIOR_OR_CARGO"
        if row["exterior"]["status"] == "REJECT_EXTERIOR_LOW_REGION" and not raised:
            rejected.append(
                dict(
                    proposal_id=row["proposal_id"],
                    reason="REJECT_EXTERIOR_LOW_REGION",
                    bounds_axial=box,
                    exterior=row["exterior"],
                )
            )
            continue
        ring = row["deck_ring"]
        u, v = grid.centers()
        inset = config["boundary"]["side_probe_m"]
        interior = (
            (u > box[0] + inset)
            & (u < box[2] - inset)
            & (v > box[1] + inset)
            & (v < box[3] - inset)
            & grid.occupancy
        )
        lower = (
            interior & (grid.median < ring_level - config["roi"]["opening_drop_m"])
            if ring_level is not None
            else np.zeros(interior.shape, bool)
        )
        labels, number = ndimage.label(lower, structure=np.ones((3, 3), bool))
        counts = np.bincount(labels.ravel(), minlength=number + 1)[1:]
        largest = int(max(counts, default=0))
        row["observed_lower_interior"] = dict(
            cells=largest,
            area_m2=largest * grid.cell_m**2,
            reference_z_m=ring_level,
            height_does_not_certify_steel=True,
        )
        if (
            largest < config["roi"]["min_opening_cells"]
            or largest * grid.cell_m**2 < config["roi"]["opening_min_area_m2"]
        ):
            rejected.append(
                dict(
                    proposal_id=row["proposal_id"],
                    reason="UNRESOLVED_NO_OBSERVED_OPENING_INTERIOR",
                    bounds_axial=box,
                    review_only=True,
                )
            )
            continue
        if ring["positive_height_side_count"] < 2 and not raised:
            rejected.append(
                dict(
                    proposal_id=row["proposal_id"],
                    reason="INSUFFICIENT_SAME_VESSEL_DECK_RING",
                    bounds_axial=box,
                )
            )
            continue
        row["validation_status"] = (
            "PARTIAL_BOUNDARY_AMBIGUOUS"
            if row["containment"]["status"] != "CONTAINED"
            or row["exterior"]["status"] == "UNRESOLVED_EXTERIOR_CONNECTIVITY"
            else "OPENING_TOPOLOGY_REVIEW"
        )
        eligible.append(row)

    # The shortlist uses independently measured enclosure strength before
    # coarse completeness. Area never chooses a final physical side.
    def quality(row):
        box = row["bounds_axial"]
        ring = row["deck_ring"]
        return (
            ring["positive_height_side_count"],
            ring["measured_strong_side_count"],
            ring["dual_provider_side_count"],
            min(s["coverage"] for s in ring["sides"].values()),
        )

    eligible.sort(key=quality, reverse=True)
    selected, suppressed = [], []
    for row in eligible:
        # An overlapping coarse envelope must not erase a smaller observed
        # basin. Suppress only the same four search stations here; overlapping
        # physical alternatives compete after raw precision.
        prior = next(
            (
                r
                for r in selected
                if np.max(
                    abs(np.array(row["bounds_axial"]) - np.array(r["bounds_axial"]))
                )
                <= config["boundary"]["corner_join_m"]
            ),
            None,
        )
        if prior:
            suppressed.append(
                dict(
                    proposal_id=row["proposal_id"],
                    kept=prior["proposal_id"],
                    reason="NEAR_DUPLICATE_FOUR_SEARCH_STATIONS",
                )
            )
        else:
            selected.append(row)
    b2 = make_local_b2(cache, cache.structural_providers())
    vessel = b2["scene"]["vessel_hypotheses"][0]
    split_rows = []
    for row in selected:
        pending = [row]
        while pending:
            item = pending.pop(0)
            boxes, separator = _separator_split(item, b2, vessel, grid, config)
            if (
                isinstance(separator, list)
                and separator[1] - separator[0]
                <= config["boundary"]["beam_max_width_m"]
            ):
                for index, box in enumerate(boxes):
                    child = copy.deepcopy(item)
                    child.update(
                        bounds_axial=box,
                        proposal_id=item["proposal_id"] + ":" + str(index),
                        measured_separator_faces=separator,
                        topology_status="SEPARATE_MEASURED_SEPARATOR_FACES",
                    )
                    pending.append(child)
            else:
                item["topology_status"] = (
                    "AMBIGUOUS_TOPOLOGY"
                    if separator is not None
                    else "INDEPENDENT_OPENING_GROUP"
                )
                if separator is not None:
                    item["topology_ambiguity"] = separator
                split_rows.append(item)
    split_rows.sort(key=lambda r: (r["bounds_axial"][0], r["bounds_axial"][1]))
    for row in split_rows:
        coarse = cache.grid("coarse")
        rises, support = _side_height_rises(
            coarse, coarse.occupancy, row["bounds_axial"], config
        )
        row.update(
            side_height_rises_m=list(rises),
            side_support_cells=[list(v) for v in support],
        )
        row["deck_ring"] = deck_ring(cache, row)
        row["opening_evidence"] = _opening_evidence(grid, row["bounds_axial"], config)
    return split_rows, dict(
        rejected=rejected, suppressed=suppressed, independent_groups=len(split_rows)
    )
