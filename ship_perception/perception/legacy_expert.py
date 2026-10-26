"""Frozen historical B2 proposal expert; no B4/B5 or final precision here."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from ship_perception.r1_static.hatch_proposal import propose_for_vessel
from .candidate_union import CandidateProposal, canonical
from .context import thaw


@dataclass(frozen=True)
class LegacyProposalBatch:
    raw_output_json: str
    output_sha256: str
    proposals: tuple

    def raw_output(self):
        return json.loads(self.raw_output_json)


def verify_legacy_sources():
    package = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (package / "config/legacy_proposal_freeze.json").read_text(encoding="utf-8")
    )
    for relative, expected in manifest["files"].items():
        value = (package / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError("LEGACY_EXPERT_BEHAVIOR_CHANGED:" + relative)
    return manifest["base_sha"]


class LegacyProposalExpert:
    name = "LEGACY"

    def propose(self, cache, research):
        # The historical function and providers are called verbatim. Transport
        # coordinates are explicitly frozen vessel UV, not a second B1 run.
        vessel = thaw(cache.vessel.public)
        vessel["local_axes"] = [[1, 0], [0, 1]]
        providers = cache.structural_providers()
        historical = propose_for_vessel(
            cache.xyz,
            vessel,
            cache.config,
            research,
            structural_override=(
                [
                    p
                    for p in providers
                    if p["provider_type"] == "BEV_RECTILINEAR_PROVIDER"
                ],
                [
                    p
                    for p in providers
                    if p["provider_type"] == "RAW3D_STRUCTURAL_PROVIDER"
                ],
                cache.grid("fine"),
            ),
        )
        raw = canonical(historical)
        proposals = tuple(
            CandidateProposal.from_evidence(self.name, cache.vessel.vessel_id, row, i)
            for i, row in enumerate(historical["hatch_hypotheses"])
            if row.get("bounds_axial") is not None
        )
        return LegacyProposalBatch(
            raw, hashlib.sha256(raw.encode()).hexdigest(), proposals
        )
