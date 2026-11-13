"""Compile nominal evidence before any signed Raw role measurement."""

from dataclasses import dataclass
import hashlib

import numpy as np

from .candidate_union import PhysicalBoundaryQuery, canonical


def proposal_topology(proposal):
    row = proposal.evidence()
    return canonical(row.get("measured_separator_faces", row.get("topology_ambiguity")))


def evidence_lineage(proposal):
    row = proposal.evidence()
    return canonical(row.get("physical_modes", sorted(proposal.lineage)))


@dataclass(frozen=True)
class CoarsePhysicalCandidateGroup:
    group_id: str
    vessel_id: str
    bounds: tuple
    proposals: tuple
    topology: str
    provider_lineage: str

    @property
    def search_envelope(self):
        boxes = np.asarray([p.bounds for p in self.proposals])
        return tuple(np.r_[boxes[:, :2].min(0), boxes[:, 2:].max(0)])

    def hypotheses(self):
        # A failed representative does not destroy physically different members.
        seen = set()
        for proposal in self.proposals:
            if proposal.bounds not in seen:
                seen.add(proposal.bounds)
                yield proposal

    def query(self, proposal=None):
        return PhysicalBoundaryQuery(
            self.group_id,
            self.vessel_id,
            proposal.bounds if proposal is not None else self.bounds,
            self.proposals,
            (),
        )

    def audit(self):
        return dict(
            group_id=self.group_id,
            vessel_id=self.vessel_id,
            bounds_axial=list(self.bounds),
            source_proposal_ids=[p.proposal_id for p in self.proposals],
            candidate_sources=sorted({p.expert for p in self.proposals}),
            member_bounds=[list(p.bounds) for p in self.proposals],
            union_search_envelope=list(self.search_envelope),
            provider_lineages=sorted({evidence_lineage(p) for p in self.proposals}),
            separator_topology=self.topology,
        )


class ProposalCompiler:
    def __init__(self, config, lineage_resolver=None):
        self.config = config
        self.lineage_resolver = lineage_resolver
        self.exact_duplicate_count = 0

    def compile(self, proposals):
        exact = {}
        objects = []
        self.exact_duplicate_count = 0
        for proposal in proposals:
            topology = proposal_topology(proposal)
            lineage = (
                self.lineage_resolver(proposal)
                if self.lineage_resolver
                else evidence_lineage(proposal)
            )
            key = (proposal.vessel_id, proposal.bounds, lineage, topology)
            if key in exact:
                exact[key]["members"].append(proposal)
                self.exact_duplicate_count += 1
                continue
            item = dict(
                vessel=proposal.vessel_id,
                bounds=proposal.bounds,
                lineage=lineage,
                topology=topology,
                members=[proposal],
            )
            exact[key] = item
            objects.append(item)
        groups = []
        buckets = {}
        join = self.config["boundary"]["corner_join_m"]
        numerical = self.config["geometry"]["plane_inlier_m"]
        for item in objects:
            key = (item["vessel"], item["lineage"], item["topology"])
            bucket = int(np.floor(item["bounds"][0] / join))
            nearby = [
                g
                for b in (bucket - 1, bucket, bucket + 1)
                for g in buckets.get((key, b), ())
            ]
            # Unknown lineage only permits numerical duplicates. Complete
            # provider agreement permits corner-join-sized nominal offsets.
            tolerance = join if item["lineage"] not in ("[]", "null") else numerical
            group = next(
                (
                    g
                    for g in nearby
                    if np.max(abs(np.asarray(g["bounds"]) - item["bounds"]))
                    <= tolerance
                ),
                None,
            )
            if group is None:
                group = dict(item, members=[])
                groups.append(group)
                buckets.setdefault((key, bucket), []).append(group)
            group["members"].extend(item["members"])
        result = []
        for group in groups:
            members = tuple(
                sorted(
                    group["members"],
                    key=lambda p: (
                        p.expert != "LEGACY",
                        -p.evidence().get("measured_strong_side_count", 0),
                        p.proposal_id,
                    ),
                )
            )
            ident = canonical(
                [group["vessel"], group["bounds"], group["lineage"], group["topology"]]
            )
            result.append(
                CoarsePhysicalCandidateGroup(
                    "C" + hashlib.sha256(ident.encode()).hexdigest()[:16],
                    group["vessel"],
                    members[0].bounds,
                    members,
                    group["topology"],
                    group["lineage"],
                )
            )
        return tuple(result)
