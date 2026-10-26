"""Immutable proposals and conservative, physical-query deduplication.

An expert owns its evidence, never the lifetime of another expert's proposal.
Union is lossless; a query retains every contributing immutable proposal.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np


def canonical(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=convert
    )


@dataclass(frozen=True)
class CandidateProposal:
    proposal_id: str
    vessel_id: str
    expert: str
    bounds: tuple
    evidence_json: str
    lineage: tuple = ()

    @classmethod
    def from_evidence(cls, expert, vessel_id, row, index=0):
        box = tuple(float(v) for v in row["bounds_axial"])
        if len(box) != 4 or not np.isfinite(box).all():
            raise ValueError("INVALID_PROPOSAL_BOUNDS")
        evidence = json.loads(canonical(row))
        ident = row.get("proposal_id", row.get("hatch_hypothesis_id", str(index)))
        return cls(
            expert + ":" + str(ident),
            vessel_id,
            expert,
            box,
            canonical(evidence),
            tuple(
                row.get("source_provider_ids", row.get("supporting_provider_ids", ()))
            ),
        )

    def evidence(self):
        return json.loads(self.evidence_json)


@dataclass(frozen=True)
class PhysicalBoundaryQuery:
    query_id: str
    vessel_id: str
    bounds: tuple
    proposals: tuple
    mode_signature: tuple

    @property
    def sources(self):
        return sorted({p.expert for p in self.proposals})


def candidate_union(batches):
    """Keep even identical proposals: provenance is not geometric suppression."""
    return tuple(p for batch in batches for p in batch)


def physical_queries(proposals, config, mode_resolver=None):
    """Aggregate the same four physical modes, never containment/area alone.

    Without four known modes only numerical near-duplicates are coalesced.
    Distinct separator hypotheses and vessels always remain separate.
    """
    groups = []
    complete_index = {}
    partial_index = {}
    tolerance = config["geometry"]["plane_inlier_m"]
    for proposal in proposals:
        row = proposal.evidence()
        topology = canonical(
            row.get("measured_separator_faces", row.get("topology_ambiguity"))
        )
        modes = tuple(mode_resolver(proposal)) if mode_resolver else ()
        complete = len(modes) == 4 and all(m is not None for m in modes)
        key = (proposal.vessel_id, topology, modes)
        group = complete_index.get(key) if complete else None
        bucket = int(np.floor(proposal.bounds[0] / tolerance))
        if not complete:
            nearby = []
            for b in (bucket - 1, bucket, bucket + 1):
                nearby.extend(partial_index.get((proposal.vessel_id, topology, b), ()))
            matches = [
                g
                for g in nearby
                if np.max(abs(np.asarray(g["bounds"]) - proposal.bounds)) <= tolerance
            ]
            group = min(matches, key=lambda g: g["index"]) if matches else None
        if group is None:
            group = dict(
                vessel=proposal.vessel_id,
                bounds=proposal.bounds,
                modes=modes,
                topology=topology,
                complete=complete,
                members=[],
            )
            group["index"] = len(groups)
            groups.append(group)
            if complete:
                complete_index[key] = group
            else:
                partial_index.setdefault(
                    (proposal.vessel_id, topology, bucket), []
                ).append(group)
        group["members"].append(proposal)
    result = []
    for g in groups:
        identity = canonical([g["vessel"], g["bounds"], g["modes"], g["topology"]])
        result.append(
            PhysicalBoundaryQuery(
                "Q" + hashlib.sha256(identity.encode()).hexdigest()[:16],
                g["vessel"],
                g["bounds"],
                tuple(g["members"]),
                g["modes"],
            )
        )
    return tuple(result)
