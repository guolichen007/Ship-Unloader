"""Visit distinct physical regions before competing nearby hypotheses."""

from collections import deque
from dataclasses import replace
import numpy as np

from .recall import box_overlap
from .candidate_union import canonical
from .proposal_compiler import evidence_lineage


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
                    if r["topology"] == separator
                    and box_overlap(r["bounds"], query.bounds)
                    >= config["roi"]["min_enclosure_ratio"]
                ),
                None,
            )
            if region is None:
                region = dict(bounds=query.bounds, topology=separator, queue=deque())
                self.regions.append(region)
            region["queue"].append((query, row))
        self.explanations = []

    @staticmethod
    def rank(item):
        query, row = item
        post = row.get("legacy_post_b2", {})
        ring = row.get("deck_ring", {})
        return (
            -int(
                post.get("derived_search_evidence", False)
                and post.get("support_side_count", 0) >= 3
            ),
            -int(post.get("support_side_count", 0) >= 3),
            -ring.get("positive_height_side_count", 0),
            -row.get("independent_support_sides", 0),
            -sum(
                s.get("vertical_face_support", False)
                for s in ring.get("sides", {}).values()
            ),
            row.get("layout", {}).get("ranking_penalty", 0),
            query.query_id,
        )

    def __iter__(self):
        # Each region gets one visit per round. There is no fixed hatch count
        # or top-K: every unexplained region remains in the pending queues.
        while any(r["queue"] for r in self.regions):
            for region in self.regions:
                if region["queue"]:
                    query, row = region["queue"].popleft()
                    row["scheduler_region_bounds"] = list(region["bounds"])
                    yield query, row

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
