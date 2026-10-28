"""Opening components nominate search regions; they never create steel edges."""

from dataclasses import dataclass
import hashlib
import numpy as np

from .candidate_union import canonical
from .sections import physical_sections


def intersection(first, second):
    return float(
        np.prod(
            np.maximum(
                0.0,
                np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]),
            )
        )
    )


def area(bounds):
    return float((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))


@dataclass(frozen=True)
class OpeningRegionSeed:
    region_id: str
    vessel_id: str
    bounds: tuple
    source_proposal_ids: tuple
    component_bounds: tuple
    status: str = "REGION_DETECTED"

    def audit(self):
        return dict(
            region_id=self.region_id,
            vessel_id=self.vessel_id,
            search_bounds_axial=list(self.bounds),
            source_proposal_ids=list(self.source_proposal_ids),
            component_bounds=[list(b) for b in self.component_bounds],
            region_status=self.status,
            boundary_status="BOUNDARY_UNRESOLVED",
            geometry_role="SEARCH_REGION_NOT_FINAL_BOUNDARY",
        )


class OpeningRegionCompiler:
    def __init__(self, config):
        self.config = config
        self.secondary = []
        self.compound = []

    def compile(self, vessel_id, proposals, structural_support=None):
        """Merge nested height levels and widthwise lobes, retain longitudinal ambiguity.

        A connected low component can contain several opening regions. It is
        retained as a compound search rather than erasing its two children.
        Components alone never assert that those children are separate hatches.
        """
        support = self.config["roi"]["min_enclosure_ratio"]
        proposals = tuple(proposals)
        minimum = self.config["boundary"]["profile_min_sections"]
        candidates = []
        for proposal in proposals:
            row = proposal.evidence()
            if proposal.expert != "HEIGHT_BASIN":
                continue
            box = proposal.bounds
            capacities = [
                max(
                    0,
                    len(
                        physical_sections(
                            (box[a], box[a + 2]),
                            self.config["geometry"]["refine_voxel_m"],
                            self.config,
                        )
                    )
                    - 1,
                )
                for a in (0, 1)
            ]
            closed = not row.get("touches_grid_boundary", False)
            outside, inside = row.get("surrounding_height_m"), row.get(
                "interior_height_m"
            )
            local_drop = None if outside is None or inside is None else outside - inside
            structural = (
                structural_support(box)
                if structural_support is not None
                else row.get("search_independent_support_sides", 0)
            )
            if (
                min(capacities) < minimum
                or not closed
                or local_drop is None
                or local_drop < self.config["roi"]["opening_drop_m"]
                or max(row.get("height_rise_side_count", 0), structural) < 3
            ):
                self.secondary.append(
                    dict(
                        proposal_id=proposal.proposal_id,
                        bounds_axial=list(box),
                        reason="REGION_EVIDENCE_INCOMPLETE_NOT_REJECTED",
                    )
                )
                continue
            candidates.append(proposal)
        groups = []
        for proposal in sorted(
            candidates, key=lambda p: (area(p.bounds), p.proposal_id)
        ):
            box = np.asarray(proposal.bounds)
            matching = [
                g
                for g in groups
                if any(
                    intersection(box, b)
                    / max(min(area(box), area(b)), np.finfo(float).tiny)
                    >= support
                    for b in g["components"]
                )
            ]
            if not matching:
                groups.append(
                    dict(
                        bounds=box, ids=[proposal.proposal_id], components=[tuple(box)]
                    )
                )
                continue
            # Floor lobes across the beam may join inside the same hold. A
            # longitudinal merger remains ambiguous until a physical divider
            # or a complete enclosure establishes the hatch relationship.
            same_longitudinal = all(
                max(
                    0.0,
                    min(g["bounds"][2], h["bounds"][2])
                    - max(g["bounds"][0], h["bounds"][0]),
                )
                >= support
                * min(g["bounds"][2] - g["bounds"][0], h["bounds"][2] - h["bounds"][0])
                for g in matching
                for h in matching
            )
            if len(matching) > 1 and not same_longitudinal:
                self.compound.append(
                    dict(
                        proposal_id=proposal.proposal_id,
                        bounds_axial=list(box),
                        component_source_ids=[i for g in matching for i in g["ids"]],
                        status="COMPOUND_REGION_RELATIONSHIP_UNRESOLVED",
                    )
                )
                continue
            group = matching[0]
            for other in matching[1:]:
                group["ids"] += other["ids"]
                group["components"] += other["components"]
                groups = [g for g in groups if g is not other]
            group["ids"].append(proposal.proposal_id)
            group["components"].append(tuple(box))
            all_boxes = np.asarray(group["components"])
            group["bounds"] = np.r_[all_boxes[:, :2].min(0), all_boxes[:, 2:].max(0)]
        # Adjacent floor lobes can share one pre-existing Legacy opening
        # region. This coalesces search only; it does not prove one hatch.
        legacy = [
            p
            for p in proposals
            if p.expert == "LEGACY"
            and p.evidence().get("opening_evidence", {}).get("status")
            == "OPENING_LOWER"
            and p.evidence().get("measured_strong_side_count", 0) >= 3
        ]
        changed = True
        while changed:
            changed = False
            for left, right in zip(
                sorted(groups, key=lambda g: g["bounds"][0]),
                sorted(groups, key=lambda g: g["bounds"][0])[1:],
            ):
                a, b = left["bounds"], right["bounds"]
                gap = max(0.0, b[0] - a[2])
                if gap > self.config["geometry"]["coarse_voxel_m"]:
                    continue
                if not any(
                    intersection(p.bounds, a) >= support * area(a)
                    and intersection(p.bounds, b) >= support * area(b)
                    for p in legacy
                ):
                    continue
                left["ids"] += right["ids"]
                left["components"] += right["components"]
                joined = np.asarray(left["components"])
                left["bounds"] = np.r_[joined[:, :2].min(0), joined[:, 2:].max(0)]
                groups = [g for g in groups if g is not right]
                changed = True
                break
        output = []
        for group in groups:
            bounds = tuple(group["bounds"])
            ident = (
                "R"
                + hashlib.sha256(
                    canonical([vessel_id, bounds, sorted(group["ids"])]).encode()
                ).hexdigest()[:16]
            )
            output.append(
                OpeningRegionSeed(
                    ident,
                    vessel_id,
                    bounds,
                    tuple(sorted(group["ids"])),
                    tuple(group["components"]),
                )
            )
        return tuple(sorted(output, key=lambda r: r.bounds))

    def audit(self):
        return dict(
            secondary_components=self.secondary,
            compound_components=self.compound,
            search_only=True,
            fixed_count_used=False,
        )


def build_regions(cache, proposals, core_bounds):
    from .validation import deck_ring

    compiler = OpeningRegionCompiler(cache.config)
    regions = list(
        compiler.compile(
            cache.vessel.vessel_id,
            proposals,
            lambda box: deck_ring(cache, dict(bounds_axial=box))[
                "measured_strong_side_count"
            ],
        )
    )
    raised = [
        p
        for p in proposals
        if p.expert == "LEGACY"
        and p.evidence().get("opening_evidence", {}).get("status")
        == "RAISED_INTERIOR_OR_CARGO"
    ]
    visible_area = sum(intersection(r.bounds, core_bounds) for r in regions)
    incomplete = (
        visible_area < area(core_bounds) * cache.config["roi"]["min_enclosure_ratio"]
    )
    if not regions or (incomplete and raised):
        ident = (
            "W"
            + hashlib.sha256(
                canonical(
                    [cache.vessel.vessel_id, list(core_bounds), "UNRESOLVED_STRUCTURE"]
                ).encode()
            ).hexdigest()[:16]
        )
        regions.append(
            OpeningRegionSeed(
                ident,
                cache.vessel.vessel_id,
                tuple(core_bounds),
                tuple(p.proposal_id for p in raised),
                (),
                (
                    "STRUCTURAL_ONLY_FALLBACK"
                    if not regions
                    else "STRUCTURAL_FALLBACK_UNRESOLVED_LOADED_CONTEXT"
                ),
            )
        )
    return tuple(regions), dict(
        compiler.audit(),
        opening_region_count=sum(r.status == "REGION_DETECTED" for r in regions),
        structural_fallback_count=sum(r.status != "REGION_DETECTED" for r in regions),
        visible_basin_search_area_m2=visible_area,
        core_search_area_m2=area(core_bounds),
        observed_raised_legacy_proposals=len(raised),
    )


def associate_region(query, row, regions):
    overlaps = [
        (
            intersection(query.bounds, r.bounds)
            / max(min(area(query.bounds), area(r.bounds)), np.finfo(float).tiny),
            r,
        )
        for r in regions
        if r.status == "REGION_DETECTED"
    ]
    matched = [(overlap, region) for overlap, region in overlaps if overlap > 0.0]
    chosen = (
        max(
            matched,
            key=lambda pair: (pair[0], -area(pair[1].bounds), pair[1].region_id),
        )[1]
        if matched
        else next((r for r in regions if r.status != "REGION_DETECTED"), None)
    )
    row["opening_region_ids"] = [region.region_id for overlap, region in matched]
    row["opening_region_id"] = (
        chosen.region_id
        if chosen is not None
        else "UNRESOLVED_STRUCTURE:" + query.vessel_id
    )
    row["physical_group_id"] = row["opening_region_id"]
    return row
