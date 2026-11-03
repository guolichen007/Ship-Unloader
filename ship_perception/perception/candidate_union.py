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


@dataclass(frozen=True)
class OpeningObservation:
    """A frozen current observation, never an automatically confirmed opening."""

    observation_id: str
    vessel_id: str
    kind: str
    source_id: str
    evidence_json: str
    member_support_cells: tuple = ()
    features_json: str = "{}"

    @classmethod
    def from_evidence(
        cls, kind, vessel_id, source_id, evidence, members=(), **features
    ):
        frozen = canonical(evidence)
        identity = canonical([vessel_id, kind, source_id, frozen])
        cells = tuple(sorted(set(tuple(int(v) for v in cell) for cell in members)))
        return cls(
            "O" + hashlib.sha256(identity.encode()).hexdigest()[:20],
            vessel_id,
            kind,
            str(source_id),
            frozen,
            cells,
            canonical(features),
        )

    def audit(self):
        return dict(
            observation_id=self.observation_id,
            vessel_id=self.vessel_id,
            kind=self.kind,
            source_id=self.source_id,
            evidence=json.loads(self.evidence_json),
            member_support_cells=[list(c) for c in self.member_support_cells],
            features=json.loads(self.features_json),
        )


@dataclass(frozen=True)
class PhysicalOpeningHypothesis:
    """One persistent physical interpretation with all alternative witnesses.

    This is a shadow IR object. It has no Hatch conversion and no selector.
    Conservative bounds are not a boundary measurement or steel certificate.
    """

    hypothesis_id: str
    vessel_id: str
    observation_ids: tuple
    height_component_lineage: tuple
    member_support_cells: tuple
    interior_evidence_json: str
    terminal_family_ids: tuple
    longitudinal_family_ids: tuple
    separator_relations: tuple
    shared_boundary_relations: tuple
    conservative_bounds: tuple
    source_proposal_ids: tuple
    source_region_ids: tuple
    status: str

    def audit(self):
        return dict(
            hypothesis_id=self.hypothesis_id,
            vessel_id=self.vessel_id,
            observation_ids=list(self.observation_ids),
            height_component_lineage=list(self.height_component_lineage),
            member_support_cells=[list(c) for c in self.member_support_cells],
            interior_evidence=json.loads(self.interior_evidence_json),
            terminal_family_ids=list(self.terminal_family_ids),
            longitudinal_family_ids=list(self.longitudinal_family_ids),
            separator_relations=list(self.separator_relations),
            shared_boundary_relations=list(self.shared_boundary_relations),
            conservative_bounds=list(self.conservative_bounds),
            source_proposal_ids=list(self.source_proposal_ids),
            source_region_ids=list(self.source_region_ids),
            status=self.status,
        )


@dataclass(frozen=True)
class OpeningHypothesisIR:
    """Lossless union of source proposals, observations, and canonical basins.

    Experts cannot consume another expert's lifetime. Unassigned context
    evidence remains an explicit IR observation with an empty association;
    it is never turned into an opening just to make an accounting gate pass.
    """

    source_proposals: tuple
    observations: tuple
    hypotheses: tuple
    observation_mapping_json: str
    relations_json: str
    lineage_json: str

    @classmethod
    def compile(cls, proposals, observations, min_support_cells, support_ratio):
        proposals = tuple(proposals)
        observations = tuple(sorted(observations, key=lambda o: o.observation_id))
        if len({o.observation_id for o in observations}) != len(observations):
            raise ValueError("DUPLICATE_IR_OBSERVATION_ID")
        features = {o.observation_id: json.loads(o.features_json) for o in observations}
        by_id = {o.observation_id: o for o in observations}
        members = {
            o.observation_id: frozenset(o.member_support_cells) for o in observations
        }
        basins = [
            o
            for o in observations
            if o.kind == "HEIGHT_LEVEL_COMPONENT"
            and features[o.observation_id].get("origin") == "LOCAL_HEIGHT_RECALL"
        ]
        if any(not members[o.observation_id] for o in basins):
            raise ValueError("HEIGHT_COMPONENT_MEMBERSHIP_NOT_RECOVERED")
        node_ids = {
            (o.vessel_id, o.source_id): o.observation_id
            for o in observations
            if o.kind == "HEIGHT_LEVEL_COMPONENT"
            and features[o.observation_id].get("origin")
            == "S4_HIERARCHY_AFTER_PARTITION"
        }
        separators = [
            o
            for o in observations
            if o.kind == "SEPARATOR_FACE"
            and features[o.observation_id].get("qualified")
        ]
        local = [o for o in observations if o.kind == "LOCAL_ENCLOSURE_WITNESS"]
        local_by_height = {}
        for o in local:
            local_by_height.setdefault(
                (o.vessel_id, features[o.observation_id]["height_source_id"]), []
            ).append(o)

        def has_support(a, b):
            common = len(a & b)
            return bool(
                b and common >= min_support_cells and common >= len(b) * support_ratio
            )

        def observed_intersection(observation_id, basin_cells):
            f = features[observation_id]
            if "local_grid_overlap" in f:
                return sum(
                    weight
                    for r, c, weight in f["local_grid_overlap"]
                    if (r, c) in basin_cells
                )
            return float(len(members[observation_id] & basin_cells))

        def observation_supported(observation_id, basin_cells):
            common = observed_intersection(observation_id, basin_cells)
            area = features[observation_id].get(
                "support_area_cell_units", len(members[observation_id])
            )
            return bool(
                area and common >= min_support_cells and common >= area * support_ratio
            )

        owned_nodes = {}
        for basin in basins:
            owned_nodes[basin.observation_id] = frozenset(
                name
                for (vessel, name), ident in node_ids.items()
                if vessel == basin.vessel_id
                and observation_supported(ident, members[basin.observation_id])
            )

        def crosses_separator(a, b=None):
            a_nodes = owned_nodes[a]
            b_nodes = owned_nodes[b] if b is not None else a_nodes
            return tuple(
                o.observation_id
                for o in separators
                if o.vessel_id == by_id[a].vessel_id
                and (
                    features[o.observation_id]["node_ids"][0] in a_nodes
                    and features[o.observation_id]["node_ids"][1] in b_nodes
                    or features[o.observation_id]["node_ids"][1] in a_nodes
                    and features[o.observation_id]["node_ids"][0] in b_nodes
                )
            )

        compound = {
            o.observation_id: crosses_separator(o.observation_id) for o in basins
        }

        def compatible(a, b):
            fa, fb = features[a.observation_id], features[b.observation_id]
            if a.vessel_id != b.vessel_id or fa["polarity"] != fb["polarity"]:
                return False
            if fa["grid_id"] != fb["grid_id"] or crosses_separator(
                a.observation_id, b.observation_id
            ):
                return False
            # Current contradictory interior or terminal observations remain
            # alternatives; no score is allowed to overwrite either witness.
            la = local_by_height.get((a.vessel_id, a.source_id), [])
            lb = local_by_height.get((b.vessel_id, b.source_id), [])
            for x in la:
                fx = features[x.observation_id]
                for y in lb:
                    fy = features[y.observation_id]
                    roles = {fx.get("interior_role"), fy.get("interior_role")}
                    if roles == {"LOWER_INTERIOR", "RAISED_OR_CARGO_INTERIOR"}:
                        return False
                    ex, ey = fx.get("terminal_bands"), fy.get("terminal_bands")
                    if (
                        fx.get("independent_terminal_pair")
                        and fy.get("independent_terminal_pair")
                        and ex
                        and ey
                    ):
                        if any(
                            max(u[0], v[0]) > min(u[1], v[1]) for u, v in zip(ex, ey)
                        ):
                            return False
            return True

        # Inclusion is persistence of the actual connected component as its
        # threshold changes. Bbox overlap, area, strong-side counts and final
        # review eligibility play no part in this equivalence relation.
        groups = [{o.observation_id} for o in basins]
        edges = []
        for i, a in enumerate(basins):
            for b in basins[i + 1 :]:
                ma, mb = members[a.observation_id], members[b.observation_id]
                if (
                    not (ma <= mb or mb <= ma)
                    or compound[a.observation_id]
                    or compound[b.observation_id]
                ):
                    continue
                if compatible(a, b):
                    edges.append((a.observation_id, b.observation_id))
        for a, b in sorted(edges):
            ga, gb = next(g for g in groups if a in g), next(
                g for g in groups if b in g
            )
            if ga is gb:
                continue
            if all(compatible(by_id[x], by_id[y]) for x in ga for y in gb):
                ga.update(gb)
                groups.remove(gb)

        mapping = {o.observation_id: set() for o in observations}
        hypothesis_data, lineage = {}, []
        for group in sorted(groups, key=lambda g: sorted(g)):
            originals = [by_id[i] for i in sorted(group)]
            vessel = originals[0].vessel_id
            sources = sorted(o.source_id for o in originals)
            ident = (
                "PH"
                + hashlib.sha256(canonical([vessel, sources]).encode()).hexdigest()[:20]
            )
            cells = frozenset().union(*(members[i] for i in group))
            boxes = [features[i]["bounds"] for i in group]
            separators_for_group = sorted({r for i in group for r in compound[i]})
            hypothesis_data[ident] = dict(
                vessel=vessel,
                height_sources=sources,
                cells=cells,
                observations=set(group),
                boxes=boxes,
                local_interpretations=[],
                terminals=set(),
                longitudinal=set(),
                source_regions=set(),
                separators=set(separators_for_group),
                shared=set(),
                status=(
                    "COMPOUND_INTERPRETATION"
                    if separators_for_group
                    else "CANONICAL_BASIN_INTERPRETATION"
                ),
            )
            for i in group:
                mapping[i].add(ident)
            lineage.append(
                dict(
                    lineage_id=ident,
                    kind="PERSISTENT_BASIN_LINEAGE",
                    vessel_id=vessel,
                    polarity=features[originals[0].observation_id]["polarity"],
                    height_observation_ids=sorted(group),
                    source_component_ids=sources,
                    levels_m=sorted({features[i]["level_m"] for i in group}),
                    member_count=len(cells),
                    compound_interpretation=bool(separators_for_group),
                )
            )

        # Complete local explanations attach to their original physical basin,
        # including entries which the old production compiler called compound
        # or corroboration. Neither legacy 'qualified' nor >=3 sides deletes it.
        by_height = {
            (o.vessel_id, o.source_id): mapping[o.observation_id] for o in basins
        }
        for observation in local:
            f = features[observation.observation_id]
            targets = by_height.get(
                (observation.vessel_id, f["height_source_id"]), set()
            )
            mapping[observation.observation_id].update(targets)
            for ident in targets:
                h = hypothesis_data[ident]
                h["observations"].add(observation.observation_id)
                h["local_interpretations"].append(
                    dict(
                        observation_id=observation.observation_id,
                        complete_enclosure=bool(f.get("complete_enclosure")),
                        bounds=f.get("enclosure_bounds"),
                        interior_role=f.get("interior_role"),
                        legacy_qualified=json.loads(observation.evidence_json).get(
                            "qualified"
                        ),
                    )
                )
                if f.get("complete_enclosure"):
                    h["boxes"].append(f["enclosure_bounds"])

        # All S4 fragments and node slices remain observations. Multiple
        # associations are permitted and never spawn another Opening model.
        for o in observations:
            if o.kind in (
                "HEIGHT_LEVEL_COMPONENT",
                "S4_REGION_FRAGMENT",
            ) and o.observation_id not in {b.observation_id for b in basins}:
                for ident, h in hypothesis_data.items():
                    common = observed_intersection(o.observation_id, h["cells"])
                    area = features[o.observation_id].get(
                        "support_area_cell_units", len(members[o.observation_id])
                    )
                    if (
                        o.vessel_id == h["vessel"]
                        and common >= min_support_cells
                        and common >= min(len(h["cells"]), area) * support_ratio
                    ):
                        mapping[o.observation_id].add(ident)
                for ident in mapping[o.observation_id]:
                    hypothesis_data[ident]["observations"].add(o.observation_id)
                    if o.kind == "S4_REGION_FRAGMENT":
                        hypothesis_data[ident]["source_regions"].add(o.source_id)

        # Provenance references propagate evidence associations, not identity.
        # The payload and all references are retained even if no opening owns
        # this current context evidence.
        source_index = {}
        for o in observations:
            source_index.setdefault((o.vessel_id, o.source_id), []).append(
                o.observation_id
            )
        changed = True
        while changed:
            before = sum(len(v) for v in mapping.values())
            for o in observations:
                f = features[o.observation_id]
                for source_id in f.get("association_source_ids", []):
                    for source_observation in source_index.get(
                        (o.vessel_id, source_id), []
                    ):
                        mapping[o.observation_id].update(mapping[source_observation])
                for target_source in f.get("evidence_provider_ids", []):
                    for target in source_index.get((o.vessel_id, target_source), []):
                        mapping[target].update(mapping[o.observation_id])
            changed = before != sum(len(v) for v in mapping.values())

        relations = []
        for o in observations:
            targets = sorted(mapping[o.observation_id])
            for ident in targets:
                hypothesis_data[ident]["observations"].add(o.observation_id)
            if o.kind == "S4_FAMILY":
                role = features[o.observation_id].get("axis_role")
                for ident in targets:
                    h = hypothesis_data[ident]
                    if role == "TRANSVERSE":
                        h["terminals"].add(o.source_id)
                    elif role == "LONGITUDINAL":
                        h["longitudinal"].add(o.source_id)
                if len(targets) > 1:
                    rid = (
                        "R"
                        + hashlib.sha256(
                            canonical([o.observation_id, targets]).encode()
                        ).hexdigest()[:20]
                    )
                    relations.append(
                        dict(
                            relation_id=rid,
                            relationship="SHARED_BOUNDARY_EVIDENCE",
                            observation_ids=[o.observation_id],
                            hypothesis_ids=targets,
                            exclusive_opening_ownership=False,
                        )
                    )
                    for ident in targets:
                        hypothesis_data[ident]["shared"].add(rid)
            if o.kind == "SEPARATOR_FACE":
                f = features[o.observation_id]
                for interpretation in (
                    "COMPOUND_INTERPRETATION",
                    "SPLIT_INTERPRETATION",
                ):
                    rid = (
                        "R"
                        + hashlib.sha256(
                            canonical(
                                [o.observation_id, interpretation, targets]
                            ).encode()
                        ).hexdigest()[:20]
                    )
                    relations.append(
                        dict(
                            relation_id=rid,
                            relationship="AMBIGUOUS",
                            interpretation=interpretation,
                            observation_ids=[o.observation_id],
                            hypothesis_ids=targets,
                            qualified=bool(f.get("qualified")),
                            parent_deleted=False,
                            selection_executed=False,
                        )
                    )
                    for ident in targets:
                        hypothesis_data[ident]["separators"].add(rid)

        hypotheses = []
        for ident, h in sorted(hypothesis_data.items()):
            boxes = np.asarray(h["boxes"], float)
            bounds = tuple(np.r_[boxes[:, :2].min(0), boxes[:, 2:].max(0)])
            source_proposals = tuple(
                sorted(
                    p.proposal_id
                    for p in proposals
                    if p.vessel_id == h["vessel"]
                    and p.evidence().get("proposal_id") in h["height_sources"]
                )
            )
            hypotheses.append(
                PhysicalOpeningHypothesis(
                    ident,
                    h["vessel"],
                    tuple(sorted(h["observations"])),
                    tuple(h["height_sources"]),
                    tuple(sorted(h["cells"])),
                    canonical(
                        dict(
                            local_enclosure_interpretations=h["local_interpretations"],
                            member_role="CURRENT_OBSERVED_COMPONENT_PERSISTENCE_NOT_CERTIFIED_STEEL",
                        )
                    ),
                    tuple(sorted(h["terminals"])),
                    tuple(sorted(h["longitudinal"])),
                    tuple(sorted(h["separators"])),
                    tuple(sorted(h["shared"])),
                    bounds,
                    source_proposals,
                    tuple(sorted(h["source_regions"])),
                    h["status"],
                )
            )
        associations = [
            dict(
                observation_id=o.observation_id,
                hypothesis_ids=sorted(mapping[o.observation_id]),
                status=(
                    "ATTACHED_OBSERVATION"
                    if mapping[o.observation_id]
                    else "UNASSIGNED_CONTEXT_OBSERVATION_RETAINED_IN_IR"
                ),
                source_evidence_sha256=hashlib.sha256(
                    o.evidence_json.encode()
                ).hexdigest(),
            )
            for o in observations
        ]
        return cls(
            proposals,
            observations,
            tuple(hypotheses),
            canonical(associations),
            canonical(sorted(relations, key=lambda r: r["relation_id"])),
            canonical(lineage),
        )

    def audit(self):
        return dict(
            schema="ship_perception.opening_hypothesis_ir.1",
            mode="SHADOW_ONLY",
            source_proposals=[
                dict(
                    proposal_id=p.proposal_id,
                    vessel_id=p.vessel_id,
                    expert=p.expert,
                    bounds=list(p.bounds),
                    evidence=p.evidence(),
                    lineage=list(p.lineage),
                )
                for p in self.source_proposals
            ],
            observations=[o.audit() for o in self.observations],
            hypotheses=[h.audit() for h in self.hypotheses],
            height_lineages=json.loads(self.lineage_json),
            production_selection_executed=False,
            production_geometry_changed=False,
        )
