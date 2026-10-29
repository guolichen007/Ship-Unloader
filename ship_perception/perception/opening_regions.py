"""Opening components nominate search regions; they never create steel edges."""

from dataclasses import dataclass
import hashlib
import numpy as np

from .candidate_union import canonical
from .region_model import OpeningRegion, TopologySupport, EvidenceSource


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
        candidates = []
        for proposal in proposals:
            row = proposal.evidence()
            if proposal.expert != "HEIGHT_BASIN":
                continue
            box = proposal.bounds
            outside, inside = row.get("surrounding_height_m"), row.get(
                "interior_height_m"
            )
            local_drop = None if outside is None or inside is None else outside - inside
            structural = (
                structural_support(box)
                if structural_support is not None
                else row.get("search_independent_support_sides", 0)
            )
            if not (
                structural
                or row.get("height_rise_side_count", 0)
                or (
                    local_drop is not None
                    and abs(local_drop) >= self.config["roi"]["opening_drop_m"]
                )
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


def build_regions(cache, proposals, core_bounds, research=None):
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
    # Work domains remain search nominations. Actual opening identities carry
    # topology support independently of whether any steel side was measured.
    source_frame = getattr(cache, "input_frame", None)
    transform = np.eye(4)
    transform[:2, :2] = cache.vessel.axes
    source = EvidenceSource(
        source_frame.frame_id if source_frame else "UNKNOWN",
        (
            str(source_frame.provenance.get("sensor_id", "UNKNOWN"))
            if source_frame
            else "UNKNOWN"
        ),
        source_frame.stamp if source_frame else "UNKNOWN",
        None,
        tuple(tuple(row) for row in transform),
        (cache.vessel.vessel_id,),
    )
    formal = []
    by_id = {p.proposal_id: p for p in proposals}
    for seed in regions:
        if seed.status != "REGION_DETECTED":
            continue
        box = np.asarray(seed.bounds)
        margin = cache.config["roi"]["support_search_m"]
        envelope = tuple(np.r_[box[:2] - margin, box[2:] + margin])
        formal.append(
            OpeningRegion(
                seed.region_id,
                seed.vessel_id,
                seed.bounds,
                envelope,
                (
                    TopologySupport(
                        "HEIGHT_TRANSITION",
                        "SUPPORTING",
                        seed.source_proposal_ids,
                        source,
                    ),
                ),
                seed.source_proposal_ids,
                seed.component_bounds,
                (
                    "REGION_FOV_TRUNCATED"
                    if any(
                        by_id[pid].evidence().get("touches_grid_boundary", False)
                        for pid in seed.source_proposal_ids
                        if pid in by_id
                    )
                    else "REGION_PARTIAL"
                ),
            )
        )
    # The current V2 compiler owns identities. S4 contributes only owned
    # observations and graph relations, never legacy final rectangles.
    if hasattr(cache, "xyz"):
        if research is None:
            import json
            from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG

            research = json.loads(RESEARCH_CONFIG.read_bytes())
        from .s4_topology_bridge import S4TopologyBridge

        bridge = S4TopologyBridge(
            cache,
            research,
            budget=getattr(getattr(cache, "v2_measurement", None), "budget", None),
        )
        formal = bridge.compile_regions(formal)
        cache.s4_topology_bridge = bridge.audit()
    cache.opening_identities = tuple(formal)
    from .structural_family import build_structural_families, attach_family_search

    cache.structural_boundary_families = build_structural_families(cache)
    cache.structural_family_associations = attach_family_search(
        formal, cache.structural_boundary_families, cache.config
    )
    return tuple(formal) + tuple(
        r for r in regions if r.status != "REGION_DETECTED"
    ), dict(
        compiler.audit(),
        opening_region_count=len(formal),
        stable_s4_consumed=hasattr(cache, "s4_topology_bridge"),
        structural_fallback_count=sum(r.status != "REGION_DETECTED" for r in regions),
        visible_basin_search_area_m2=visible_area,
        core_search_area_m2=area(core_bounds),
        observed_raised_legacy_proposals=len(raised),
    )


def associate_region(query, row, regions):
    declared = {
        p.evidence().get("opening_region_id")
        for p in query.proposals
        if p.evidence().get("opening_region_id")
    }
    owned = [
        r for r in regions if r.region_id in declared and r.vessel_id == query.vessel_id
    ]
    overlaps = [
        (
            intersection(query.bounds, r.bounds)
            / max(min(area(query.bounds), area(r.bounds)), np.finfo(float).tiny),
            r,
        )
        for r in regions
        if r.vessel_id == query.vessel_id
        and (isinstance(r, OpeningRegion) or r.status == "REGION_DETECTED")
    ]
    matched = [(overlap, region) for overlap, region in overlaps if overlap > 0.0]
    chosen = (owned[0] if len(owned) == 1 else None) or (
        max(
            matched,
            key=lambda pair: (pair[0], -area(pair[1].bounds), pair[1].region_id),
        )[1]
        if matched
        else next(
            (
                r
                for r in regions
                if r.vessel_id == query.vessel_id
                and not isinstance(r, OpeningRegion)
                and r.status != "REGION_DETECTED"
            ),
            None,
        )
    )
    row["opening_search_region_ids"] = [region.region_id for overlap, region in matched]
    row["opening_region_ids"] = (
        [r.region_id for r in owned] if owned else row["opening_search_region_ids"]
    )
    row["region_identity_nomination"] = len(owned) == 1
    row["opening_region_match_basis"] = (
        "EXPLICIT_CURRENT_COMPILER_ID" if owned else "SPATIAL_SEARCH_ASSOCIATION_ONLY"
    )
    row["opening_region_id"] = (
        chosen.region_id
        if chosen is not None
        else "UNRESOLVED_STRUCTURE:" + query.vessel_id
    )
    row["physical_group_id"] = row["opening_region_id"]
    return row
