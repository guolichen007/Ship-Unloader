"""Visit distinct physical regions before competing nearby hypotheses."""

from collections import deque
from dataclasses import replace
import numpy as np

from .recall import box_overlap
from .candidate_union import canonical
from .proposal_compiler import evidence_lineage
from .sections import physical_sections


class PrecisionScheduler:
    def __init__(self, ready, config):
        self.config = config
        self.regions = []
        self.coverage = []
        for query, row in sorted(ready, key=self.rank):
            separator = canonical(
                row.get("legacy_post_b2", {}).get("measured_separator_faces")
            )
            region = next(
                (
                    r
                    for r in self.regions
                    if (
                        row.get("opening_region_id")
                        and r.get("opening_region_id") == row["opening_region_id"]
                    )
                    or (
                        not row.get("opening_region_id")
                        and r["topology"] == separator
                        and box_overlap(r["bounds"], query.bounds)
                        >= config["roi"]["min_enclosure_ratio"]
                        and sum(
                            abs(a - b) <= config["roi"]["support_search_m"]
                            for a, b in zip(r["bounds"], query.bounds)
                        )
                        >= 3
                        and all(
                            abs(
                                (r["bounds"][axis] + r["bounds"][axis + 2])
                                - (query.bounds[axis] + query.bounds[axis + 2])
                            )
                            / 2
                            <= config["roi"]["support_search_m"]
                            for axis in (0, 1)
                        )
                    )
                ),
                None,
            )
            if region is None:
                region = dict(
                    bounds=query.bounds,
                    topology=separator,
                    queue=deque(),
                    deferred=deque(),
                    opening_region_id=row.get("opening_region_id"),
                )
                self.regions.append(region)
            region["queue"].append((query, row))
        self.explanations = []

    def rank(self, item):
        query, row = item
        post = row.get("legacy_post_b2", {})
        ring = row.get("deck_ring", {})
        coaming = sum(
            s.get("vertical_face_support", False)
            for s in ring.get("sides", {}).values()
        )
        # A thin structural strip can have many measured faces yet cannot
        # supply the existing final section count at its nominal bounds.
        # Measure viable enclosures first; narrow hypotheses remain queued
        # and may expand through the unchanged Raw/structural search later.
        box = query.bounds
        nominal_capacity = sum(
            2
            * (
                len(
                    physical_sections(
                        (box[1 - axis], box[3 - axis]),
                        self.config["geometry"]["refine_voxel_m"],
                        self.config,
                    )
                )
                - 1
                >= self.config["boundary"]["profile_min_sections"]
            )
            for axis in (0, 1)
        )
        return (
            -int(nominal_capacity >= 3),
            -int(max(coaming, row.get("search_normal_face_side_count", 0)) >= 3),
            -int(coaming >= 3),
            row.get("internal_transverse_face_coverage", 0.0),
            row.get("skipped_verified_face_count", 0),
            -int(self.has_post_structure(row)),
            -coaming,
            -int(
                post.get("derived_search_evidence", False)
                and post.get("support_side_count", 0) >= 3
            ),
            -int(post.get("support_side_count", 0) >= 3),
            -ring.get("positive_height_side_count", 0),
            -row.get("independent_support_sides", 0),
            row.get("layout", {}).get("ranking_penalty", 0),
            query.query_id,
        )

    def __iter__(self):
        # Each region gets one visit per round. There is no fixed hatch count
        # or top-K: every unexplained region remains in the pending queues.
        while any(r["queue"] or r["deferred"] for r in self.regions):
            capacity_tier = min(
                self.rank((r["deferred"] or r["queue"])[0])[0]
                for r in self.regions
                if r["queue"] or r["deferred"]
            )
            for region in self.regions:
                if region["deferred"]:
                    query, row = region["deferred"].popleft()
                    row["scheduler_region_bounds"] = list(region["bounds"])
                    yield query, row
                    continue
                if (
                    region["queue"]
                    and self.rank(region["queue"][0])[0] == capacity_tier
                ):
                    queue = region["queue"]
                    first_tier = self.rank(queue[0])[0]
                    # Independent proposal paths must get a turn within the
                    # same region as well. Repeated structural variants must
                    # not starve a post-B2 hypothesis of that enclosure.
                    alternative = next(
                        (
                            i
                            for i, item in enumerate(queue)
                            if self.rank(item)[0] == first_tier
                            and self.evidence_family(item[1])
                            != region.get("last_family")
                        ),
                        0,
                    )
                    queue.rotate(-alternative)
                    query, row = queue.popleft()
                    queue.rotate(alternative)
                    region["last_family"] = self.evidence_family(row)
                    row["scheduler_region_bounds"] = list(region["bounds"])
                    row["scheduler_evidence_family"] = region["last_family"]
                    yield query, row

    def defer_performance(self, query, row):
        region = next(
            r
            for r in self.regions
            if tuple(row["scheduler_region_bounds"]) == tuple(r["bounds"])
        )
        row["precision_disposition"] = "DEFERRED_PERFORMANCE"
        region["deferred"].append((query, row))

    def pending_region_count(self, exclude_bounds=None):
        return sum(
            bool(r["queue"] or r["deferred"])
            and (exclude_bounds is None or tuple(r["bounds"]) != tuple(exclude_bounds))
            for r in self.regions
        )

    @staticmethod
    def has_post_structure(row):
        post = row.get("legacy_post_b2", {})
        return (
            post.get("local_physical_candidate_available", False)
            and post.get("support_side_count", 0) >= 3
            and sum(post.get("side_evidence_availability", {}).values()) >= 3
        )

    @staticmethod
    def evidence_family(row):
        if PrecisionScheduler.has_post_structure(row):
            return "POST_B2_WITH_LOCAL_STRUCTURE"
        if any(
            s.get("vertical_face_support", False)
            for s in row.get("deck_ring", {}).get("sides", {}).values()
        ):
            return "MEASURED_STRUCTURAL_CLOSURE"
        return "OTHER_LOCAL_EVIDENCE"

    def alternate(self, query, row):
        seen = {tuple(query.bounds)}
        alternatives = []
        for proposal in query.proposals:
            if proposal.bounds not in seen:
                seen.add(proposal.bounds)
                alternatives.append((replace(query, bounds=proposal.bounds), dict(row)))
        if alternatives:
            region = next(
                r
                for r in self.regions
                if tuple(row["scheduler_region_bounds"]) == tuple(r["bounds"])
            )
            for alternate in alternatives:
                alternate[1]["alternate_hypothesis"] = True
                region["queue"].append(alternate)

    def explained(self, query, row):
        tolerance = self.config["boundary"]["corner_join_m"]
        return any(
            vessel == query.vessel_id
            and np.max(abs(np.asarray(query.bounds) - np.asarray(bounds))) <= tolerance
            and topology
            == canonical(row.get("legacy_post_b2", {}).get("measured_separator_faces"))
            and lineage == tuple(sorted({evidence_lineage(p) for p in query.proposals}))
            for vessel, bounds, topology, lineage in self.coverage
        )

    def record(self, query, row, result):
        uv = np.asarray(result["boundary_uv"])
        self.coverage.append(
            (
                query.vessel_id,
                (*uv.min(0), *uv.max(0)),
                canonical(
                    row.get("legacy_post_b2", {}).get("measured_separator_faces")
                ),
                tuple(sorted({evidence_lineage(p) for p in query.proposals})),
            )
        )
        self.explanations.append(
            dict(
                query_id=query.query_id,
                outcome=result["status"],
                coverage_status="PARTIAL_REVIEW_REGION_EXPLAINED_NOT_STEEL_CERTIFIED",
            )
        )

    def update_layout(self, layout):
        for region in self.regions:
            for query, row in region["queue"]:
                row["layout"] = layout.review(query.bounds)
            region["queue"] = deque(sorted(region["queue"], key=self.rank))
