"""Frozen historical B2 proposal expert; no B4/B5 or final precision here."""

from dataclasses import dataclass
import ast
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


def _b2_symbol_hash(package, specification):
    """Freeze B2 adapter symbols without freezing unrelated V2 functions."""
    relative, qualified = specification.split(":", 1)
    source = (Path(package) / relative).read_text(encoding="utf-8")
    parent = ast.parse(source)
    parts = qualified.split(".")
    fields = parts[-1] == "@fields"
    for name in parts[:-1] if fields else parts:
        parent = next(n for n in parent.body if getattr(n, "name", None) == name)
    nodes = (
        [
            *parent.decorator_list,
            *[n for n in parent.body if isinstance(n, (ast.Assign, ast.AnnAssign))],
        ]
        if fields
        else [parent]
    )
    fragments = []
    lines = source.splitlines()
    for node in nodes:
        start = min(
            [node.lineno] + [d.lineno for d in getattr(node, "decorator_list", ())]
        )
        fragments.append("\n".join(lines[start - 1 : node.end_lineno]))
    return hashlib.sha256("\n".join(fragments).encode("utf-8")).hexdigest()


def verify_legacy_b2_core(package=None):
    """Protect the audited B2 dependency closure, using the original hashes."""
    package = (
        Path(package) if package is not None else Path(__file__).resolve().parents[1]
    )
    manifest = json.loads(
        (package / "config/legacy_b2_core_freeze.json").read_text(encoding="utf-8")
    )
    for relative, expected in manifest["files"].items():
        value = (package / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError("LEGACY_EXPERT_BEHAVIOR_CHANGED:" + relative)
    for specification, expected in manifest.get("adapter_symbols", {}).items():
        if _b2_symbol_hash(package, specification) != expected:
            raise ValueError("LEGACY_EXPERT_BEHAVIOR_CHANGED:" + specification)
    return dict(
        status="PASS",
        contract="LEGACY_B2_CORE_FROZEN",
        base_sha=manifest["base_sha"],
        checked_files=len(manifest["files"]),
        checked_adapter_symbols=len(manifest.get("adapter_symbols", {})),
    )


def verify_legacy_sources():
    """Compatibility SHA view of the B2 contract, not full-chain certification."""
    return verify_legacy_b2_core()["base_sha"]


def legacy_full_chain_reference(package=None):
    """Keep every original full-chain hash; explicitly report authorized drift."""
    package = (
        Path(package) if package is not None else Path(__file__).resolve().parents[1]
    )
    manifest = json.loads(
        (package / "config/legacy_proposal_freeze.json").read_text(encoding="utf-8")
    )
    authorized = {
        "r1_static/" + name + ".py"
        for name in (
            "perimeter_deck",
            "boundary_refinement",
            "boundary_topology",
            "family_reference",
            "rectangle_fusion",
        )
    }
    changed = []
    for relative, expected in manifest["files"].items():
        current = hashlib.sha256(
            (package / relative).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        if current != expected:
            changed.append(
                dict(
                    path=relative,
                    reference_sha256=expected,
                    current_sha256=current,
                    authorized_s4=relative in authorized,
                )
            )
    return dict(
        contract="LEGACY_FULL_CHAIN_REFERENCE",
        base_sha=manifest["base_sha"],
        status=(
            "REFERENCE_UNCHANGED"
            if not changed
            else (
                "REFERENCE_CHANGED_EXPECTED"
                if all(r["authorized_s4"] for r in changed)
                else "REFERENCE_CHANGED_UNEXPECTED"
            )
        ),
        classification=(
            "UNCHANGED"
            if not changed
            else (
                "CHANGED_EXPECTED_S4_ONLY"
                if all(r["authorized_s4"] for r in changed)
                else "CHANGED_OUTSIDE_APPROVED_S4"
            )
        ),
        full_chain_frozen_pass=False,
        changed_files=changed,
    )


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
