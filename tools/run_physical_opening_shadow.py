"""Audit physical opening evidence from one current R3 snapshot.

This tool is deliberately separate from native selection and geometry.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from run_opening_ir_shadow import atomic, cell_intersections, cell_overlap
from shadow_terminal_evidence import by_local_interior


BOUND = "BOUND_CURRENT_OPENING"
UNRESOLVED = "UNRESOLVED_ASSOCIATION"
CONFLICT = "CONFLICT_PHYSICAL_OWNERSHIP"


def lattice_contract(snapshot, bundle):
    provenance = snapshot["provenance"]
    s4 = bundle["s4"]
    frame = s4.get("review_frame") or {}
    native, local = provenance["s4_grid"], provenance["local_grid"]
    same_frame = (
        provenance["vessel_id"] == bundle["vessel_id"] == frame.get("vessel_id")
        and frame.get("coordinate_frame") == "VESSEL_REVIEW_UV"
        and provenance["b1_axes"] == bundle["b1_axes"] == frame.get("axis_matrix")
        and frame.get("transform_count") == 1
        and provenance["selected_primary_index_sha256"]
        == bundle["selected_primary_index_sha256"]
        == s4.get("selected_primary_point_index_sha256")
        and native == bundle["s4_grid"]
        and local == bundle["local_grid"]
        and all(native[k] == s4["observation_grid"][k] for k in ("x0", "y0", "cell_m"))
        and all(
            math.isfinite(g[k]) for g in (native, local) for k in ("x0", "y0", "cell_m")
        )
        and all(g["cell_m"] > 0 and min(g["shape"]) > 0 for g in (native, local))
    )
    return dict(
        verified=same_frame,
        coordinate_frame="VESSEL_REVIEW_UV",
        index_contract="row=V, column=U; floor bins; positive rectangle overlap",
        native_s4=native,
        canonical=local,
        origin_delta_uv_m=[native[k] - local[k] for k in ("x0", "y0")],
        same_resolution=native["cell_m"] == local["cell_m"],
    )


def cell_centers(cells, grid):
    rc = np.asarray(sorted(cells), dtype=float).reshape(-1, 2)
    return (rc[:, ::-1] + 0.5) * grid["cell_m"] + [grid["x0"], grid["y0"]]


def cell_set(cells):
    return {tuple(cell) for cell in cells}


def weighted_membership(native_cells, native, canonical):
    """Keep all overlap mass and use only uniquely dominant targets as members."""
    links = defaultdict(list)
    native_area = native["cell_m"] ** 2
    target_area = canonical["cell_m"] ** 2
    for source, target, area in cell_intersections(native_cells, native, canonical):
        links[source].append((target, area))
    records, members = [], set()
    for source, targets in sorted(links.items()):
        best = max(area for _, area in targets)
        # One ULP of cell area is a numeric tie, regardless of vessel geometry.
        tied = [
            target for target, area in targets
            if math.isclose(area, best, rel_tol=0, abs_tol=np.spacing(native_area))
        ]
        if len(tied) == 1:
            members.add(tied[0])
        records.extend(
            dict(
                source_cell_rc=source,
                target_cell_rc=target,
                overlap_area_m2=area,
                source_fraction=area / native_area,
                target_fraction=area / target_area,
            )
            for target, area in targets
        )
    return members, records


def grid_difference(canonical, structure):
    """Measure both differences in the same mapped lattice, retaining positions."""
    common = canonical & structure
    if not common:
        return None
    rc = np.asarray(sorted(canonical | structure))
    origin = rc.min(axis=0)
    shape = tuple(rc.max(axis=0) - origin + 1)

    def difference(first, second):
        diff = sorted(first - second)
        mask = np.zeros(shape, dtype=bool)
        mask[tuple((np.asarray(sorted(second)) - origin).T)] = True
        distances = ndimage.distance_transform_cdt(~mask, metric="chessboard")
        positions = np.asarray(diff, dtype=int).reshape(-1, 2) - origin
        mask.fill(False)
        mask[tuple(positions.T)] = True
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
        return dict(
            cells_rc=diff,
            distance_cells=dict(
                sorted(Counter(map(int, distances[tuple(positions.T)])).items())
            ),
            component_sizes=sorted(
                np.bincount(labels.ravel())[1:].tolist(), reverse=True
            ),
            component_count=count,
        )

    return dict(
        common_cells_rc=sorted(common),
        canonical_only=difference(canonical, structure),
        structure_only=difference(structure, canonical),
    )


def signed_support(points, line):
    return points @ np.asarray(line["normal_uv"]) - line["rho"]


def wall_check(core, points, line):
    """Check a physical side using native centers, including the wall uncertainty."""
    distance = signed_support(core, line)
    uncertainty = line["rho_uncertainty"]
    positive = bool(max(distance) > uncertainty)
    negative = bool(min(distance) < -uncertainty)
    side = int(positive) - int(negative)
    margins = side * signed_support(points, line)
    return dict(
        family_id=line["family_id"],
        normal_uv=line["normal_uv"],
        rho_m=line["rho"],
        uncertainty_m=uncertainty,
        side=side,
        minimum_signed_margin_m=float(min(margins)),
        crosses=bool(not side or min(margins) < -uncertainty),
    )


def axial_split(a, b):
    return (a[2] <= b[0] or b[2] <= a[0]) and max(a[1], b[1]) < min(a[3], b[3])


def local_scopes(hypothesis, observations, providers, measured=None):
    scopes = []
    for witness in hypothesis["interior_evidence"]["local_enclosure_interpretations"]:
        if (
            not witness["complete_enclosure"]
            or witness["interior_role"] != "LOWER_INTERIOR"
        ):
            continue
        observation = observations[witness["observation_id"]]
        entry = observation["evidence"]["before_identity_eligibility"]
        if (
            entry["containment"]["status"] != "CONTAINED"
            or entry["exterior"]["status"] != "ENCLOSED_OBSERVED_LOW_COMPONENT"
            or not entry.get("member_support_cells_rc")
            or not all(
                side["raw3d_coverage"] > 0 and side["bev_coverage"] > 0
                for side in entry["sides"].values()
            )
        ):
            continue
        ends = [
            {
                provider
                for provider in entry["sides"][side]["verified_face_provider_ids"]
                if provider in providers
                and providers[provider]["vessel_hypothesis_id"]
                == hypothesis["vessel_id"]
            }
            for side in ("U0", "U1")
        ]
        membership = frozenset(map(tuple, entry["member_support_cells_rc"]))
        matches = (measured or {}).get(membership, [])
        evidence, raw_pair = matches[0] if len(matches) == 1 else ([], False)
        independent = bool(all(ends) and not ends[0].intersection(ends[1]))
        if raw_pair and not independent:
            ends = [
                {row["raw_original_index_digest"] for row in evidence if row["side"] == side}
                for side in ("U0", "U1")
            ]
        scopes.append(
            dict(
                route="HEIGHT_BASIN",
                observation_id=witness["observation_id"],
                hypothesis_id=hypothesis["hypothesis_id"],
                bounds=witness["bounds"],
                cells=entry["member_support_cells_rc"],
                end_evidence=[sorted(end) for end in ends],
                independent_ends=independent or raw_pair,
                terminal_evidence=[
                    dict(row, hypothesis_id=hypothesis["hypothesis_id"])
                    for row in evidence
                ],
                independent_raw_terminal_pair=raw_pair,
            )
        )
    return scopes


def structural_scopes(observations, vessel_id, families, bundle, lattice):
    node_cells = {
        row["node_id"]: cell_set(row["cells"]) for row in bundle["s4_nodes_after"]
    }
    native, local = lattice["native_s4"], lattice["canonical"]
    for observation in observations.values():
        if observation["kind"] != "S4_REGION_FRAGMENT":
            continue
        region = observation["evidence"]
        if observation["vessel_id"] != vessel_id or region["vessel_id"] != vessel_id:
            continue
        identity = region["identity_evidence"]
        opposite = set(identity.get("opposite_terminal_family_ids", []))
        ends = [
            opposite.intersection(
                region["side_topology_support"].get(side, {}).get("provider_ids", [])
            )
            for side in ("U0", "U1")
        ]
        raw = [
            {
                source
                for family_id in end
                if family_id in families
                and families[family_id]["vessel_id"] == vessel_id
                for source in families[family_id]["source_observation_ids"]
            }
            for end in ends
        ]
        cells = observation["features"].get("local_grid_overlap", [])
        if cells:
            nodes = identity["member_height_nodes"]
            native_cells = cell_set(observation["member_support_cells"])
            lineage = set().union(*(node_cells.get(node, set()) for node in nodes))
            verified = (
                lattice["verified"]
                and identity["source"] == "CURRENT_S4_TOPOLOGY"
                and bool(native_cells)
                and bool(nodes)
                and all(node in node_cells for node in nodes)
                and native_cells == cell_set(identity["member_support_cells_rc"])
                and native_cells <= lineage
                and all(
                    identity["member_support_grid"][k] == native[k]
                    for k in ("x0", "y0", "cell_m")
                )
                and all(
                    0 <= r < native["shape"][0] and 0 <= c < native["shape"][1]
                    for r, c in native_cells
                )
                and cells == cell_overlap(sorted(native_cells), native, local)
            )
            members, links = weighted_membership(sorted(native_cells), native, local)
            yield dict(
                route="STRUCTURE_TOPOLOGY",
                observation_id=observation["observation_id"],
                region_id=region["region_id"],
                bounds=region["conservative_bounds"],
                vessel_id=vessel_id,
                cells=[cell[:2] for cell in cells],
                dominant_target_cells_audit=sorted(members),
                weighted_overlap=links,
                end_families=[sorted(end) for end in ends],
                end_evidence=[sorted(end) for end in raw],
                independent_ends=bool(all(raw) and not raw[0].intersection(raw[1])),
                node_ids=nodes,
                native_cells=sorted(native_cells),
                lineage_verified=verified,
                independent_enclosure=identity["current_independent_enclosure_scope"],
                source_role=region["identity_role"],
            )


def association(scope, members, own, competitors, separators, families, lattice):
    structure = cell_set(scope["cells"])
    diff = grid_difference(members, structure)
    if diff is None:
        return None
    result = dict(scope_id=scope["observation_id"], **diff)
    if "weighted_overlap" in scope:
        weights = scope["weighted_overlap"]
        result["mapped_area_m2"] = math.fsum(row["overlap_area_m2"] for row in weights)
        result["common_area_m2"] = math.fsum(
            row["overlap_area_m2"] for row in weights if row["target_cell_rc"] in members
        )
    common = cell_set(diff["common_cells_rc"])
    canonical_grid, native_grid = lattice["canonical"], lattice["native_s4"]
    core = cell_centers(common, canonical_grid)
    points = np.concatenate(
        (
            cell_centers(members, canonical_grid),
            cell_centers(scope["native_cells"], native_grid),
        )
    )
    edges = separators.get(scope["observation_id"], [])
    lines = {
        f["family_id"]: f
        for edge in edges
        for f in edge["lines_by_scope"][scope["observation_id"]]
    }
    terminals = {
        family_id: families[family_id]
        for end in scope["end_families"]
        for family_id in end
    }
    walls = [wall_check(core, points, line) for line in {**lines, **terminals}.values()]
    result["wall_checks"] = walls
    result["separator_checks"] = [wall for wall in walls if wall["family_id"] in lines]
    result["separator_family_ids"] = sorted(lines)
    extra = structure - members
    competing = sorted(
        other_id for other_id, interior in competitors.items() if extra & interior
    )
    if any(w["crosses"] for w in walls) or competing:
        status, reason = CONFLICT, "WRONG_SIDE_OF_WALL_OR_COMPETING_INTERIOR"
    elif not scope["lineage_verified"]:
        status, reason = UNRESOLVED, "CURRENT_S4_LINEAGE_OR_LATTICE_UNVERIFIED"
    elif not extra:
        status, reason = BOUND, "OBSERVED_MEMBER_SUBSET"
    elif (
        lattice["same_resolution"]
        and scope["independent_ends"]
        and lines
        and any(cell_set(w["cells"]) == members for w in own)
        and all(
            max(diff[key]["distance_cells"], default=0) <= 1
            for key in ("canonical_only", "structure_only")
        )
    ):
        status, reason = BOUND, "BIDIRECTIONAL_BOUNDARY_ALIAS_WITH_SIGNED_WALLS"
    else:
        status, reason = (
            UNRESOLVED,
            "DISTINCT_SUPPORT_OR_INSUFFICIENT_PHYSICAL_OWNERSHIP",
        )
    result.update(status=status, reason=reason, competing_interior_scope_ids=competing)
    return result


def separator_lines(witness, families):
    family = families.get(witness.get("family_id"))
    if family is not None and family["axis_role"] == "TRANSVERSE":
        return [family]
    if witness["kind"] == "CURRENT_RAW3D_BEV_TWO_FACE_SEPARATOR":
        return [
            dict(
                family_id="TWO_FACE:" + str(rho),
                normal_uv=[1, 0],
                rho=rho,
                rho_uncertainty=0,
            )
            for rho in sorted(witness["separator_faces"])
        ]
    return []


def separator_evidence(observations, scopes, families, grid, vessel):
    # Bind to the current S4 nodes and verify physical side using unmapped cells.
    pairs = [
        (
            a,
            b,
            cell_centers(a["native_cells"], grid),
            cell_centers(b["native_cells"], grid),
        )
        for a, b in combinations(scopes, 2)
        if a.get("lineage_verified")
        and b.get("lineage_verified")
        and axial_split(a["bounds"], b["bounds"])
    ]
    evidence = []
    for o in observations.values():
        if o["kind"] != "SEPARATOR_FACE" or not o["features"].get("qualified"):
            continue
        relation = o["evidence"]["relation"]
        if (
            o["vessel_id"] != vessel
            or relation["relation_decision"] != "SPLIT"
            or not relation["ownership"]["same_active_vessel"]
        ):
            continue
        nodes = set(o["features"]["node_ids"])
        witness = o["evidence"]["witness"]
        lines = separator_lines(witness, families)
        if not lines:
            continue
        for a, b, a_uv, b_uv in pairs:
            if not nodes.intersection(a["node_ids"]) or not nodes.intersection(
                b["node_ids"]
            ):
                continue
            a_id, b_id = a["observation_id"], b["observation_id"]
            a_check, b_check = wall_check(a_uv, a_uv, lines[0]), wall_check(
                b_uv, b_uv, lines[-1]
            )
            if (
                a_check["crosses"]
                or b_check["crosses"]
                or a_check["side"] == b_check["side"]
            ):
                continue
            if len(lines) == 2 and (a_check["side"], b_check["side"]) != (-1, 1):
                # Order the pair by its observed side, then check both face positions.
                a_check, b_check = wall_check(a_uv, a_uv, lines[-1]), wall_check(
                    b_uv, b_uv, lines[0]
                )
                if (
                    a_check["crosses"]
                    or b_check["crosses"]
                    or (a_check["side"], b_check["side"]) != (1, -1)
                ):
                    continue
            by_id = {line["family_id"]: line for line in lines}
            evidence.append(
                dict(
                    observation_id=o["observation_id"],
                    scope_ids=[a_id, b_id],
                    witness=witness,
                    lines_by_scope={
                        a_id: [by_id[a_check["family_id"]]],
                        b_id: [by_id[b_check["family_id"]]],
                    },
                    signed_checks={a_id: a_check, b_id: b_check},
                )
            )
    return evidence


def separator_support_audit(separators, families, bundle):
    groups = defaultdict(set)
    for edge in separators:
        witness = edge["witness"]
        key = witness.get("family_id") or "TWO_FACE:" + str(
            sorted(witness["separator_faces"])
        )
        groups[key].add(edge["observation_id"])
    raw = {row["observation_id"]: row for row in bundle["s4"]["raw_observations"]}
    return [
        dict(
            structure_key=key,
            reference_ids=sorted(refs),
            reference_count=len(refs),
            source_observation_ids=sorted(sources),
            source_rows=[
                dict(
                    observation_id=i,
                    segment_id=raw[i]["segment_id"],
                    source_index=raw[i]["source_index"],
                    evidence_type=raw[i]["evidence_type"],
                )
                for i in sorted(sources)
                if i in raw
            ],
            selected_primary_index_sha256=bundle["selected_primary_index_sha256"],
            raw_point_sets="NOT_EXPORTED_BY_S4",
            independent_sections="UNVERIFIED",
        )
        for key, refs in sorted(groups.items())
        for sources in [set(families.get(key, {}).get("source_observation_ids", []))]
    ]


def audit(snapshot, bundle, raw_xyz=None):
    observations = {o["observation_id"]: o for o in snapshot["observations"]}
    providers = {p["provider_id"]: p for p in bundle["providers"]}
    vessel = snapshot["provenance"]["vessel_id"]
    lattice = lattice_contract(snapshot, bundle)
    families = {
        o["evidence"]["family_id"]: o["evidence"]
        for o in observations.values()
        if o["kind"] == "S4_FAMILY" and o["vessel_id"] == vessel
    }
    scopes = list(structural_scopes(observations, vessel, families, bundle, lattice))
    hypotheses = snapshot["hypotheses"]
    raw_uv = (
        raw_xyz[:, :2] @ np.asarray(bundle["b1_axes"]).T
        if raw_xyz is not None
        else None
    )
    measured = by_local_interior(bundle, vessel, raw_uv)
    for h in hypotheses:
        if h["vessel_id"] == vessel:
            scopes.extend(local_scopes(h, observations, providers, measured))

    separators = separator_evidence(
        observations, scopes, families, lattice["native_s4"], vessel
    )
    edges_by_scope = defaultdict(list)
    for edge in separators:
        for scope_id in edge["scope_ids"]:
            edges_by_scope[scope_id].append(edge)
    grounded = [
        s
        for s in scopes
        if (s["independent_ends"] or s["observation_id"] in edges_by_scope)
        and s.get("lineage_verified", True)
        and s.get("source_role")
        not in ("COMPOUND_SEARCH_GROUP", "WEAK_UNASSIGNED_REGION")
    ]
    grounded_ids = {s["observation_id"] for s in grounded}
    for scope in scopes:
        scope["review_role"] = (
            "PHYSICAL_OPENING_HYPOTHESIS"
            if scope["observation_id"] in grounded_ids
            else "SEARCH_DOMAIN"
        )

    def separated_members(a, b):
        ac = [c for _, c in a["cells"]]
        bc = [c for _, c in b["cells"]]
        return bool(ac and bc and (max(ac) < min(bc) or max(bc) < min(ac)))

    independent_pairs = [
        (a, b)
        for a, b in combinations(grounded, 2)
        if max(a["bounds"][1], b["bounds"][1]) < min(a["bounds"][3], b["bounds"][3])
        and separated_members(a, b)
    ]
    records = []
    for h in hypotheses:
        members = set(map(tuple, h["member_support_cells"]))
        attached = [s for s in scopes if s["observation_id"] in h["observation_ids"]]
        own = [s for s in attached if s.get("hypothesis_id") == h["hypothesis_id"]]
        associations = []
        if h["vessel_id"] == vessel and h["status"] != "COMPOUND_INTERPRETATION":
            for scope in attached:
                if "node_ids" not in scope:
                    continue
                scope_id = scope["observation_id"]
                rivals = {
                    other["observation_id"]: cell_set(other["cells"])
                    for other in grounded
                    if "node_ids" in other
                    and other["independent_ends"]
                    and other["observation_id"] != scope_id
                    and any(
                        other["observation_id"] in e["scope_ids"]
                        for e in edges_by_scope[scope_id]
                    )
                }
                match = association(
                    scope, members, own, rivals, edges_by_scope, families, lattice
                )
                if match is not None:
                    associations.append(match)
        bound = {
            row["scope_id"] for row in associations if row["status"] == BOUND
        } & grounded_ids
        compound = h["status"] == "COMPOUND_INTERPRETATION" or any(
            cell_set(a["cells"]) <= members and cell_set(b["cells"]) <= members
            for a, b in independent_pairs
        )
        if compound:
            role, reason = (
                "COMPOUND_INTERPRETATION",
                "MULTIPLE_PHYSICAL_SCOPES_OR_IR_COMPOUND",
            )
        elif own or bound:
            role, reason = (
                "PHYSICAL_OPENING_HYPOTHESIS",
                "CURRENT_INTERIOR_OR_STRUCTURE_ROUTE",
            )
        else:
            role, reason = "SEARCH_DOMAIN", "NO_DIRECT_PHYSICAL_SCOPE_BINDING"
        records.append(
            dict(
                hypothesis_id=h["hypothesis_id"],
                role=role,
                reason=reason,
                observation_ids=h["observation_ids"],
                basin_witness_ids=[s["observation_id"] for s in own],
                structure_scope_ids=sorted(bound),
                associated_scope_ids=[s["observation_id"] for s in attached],
                structure_associations=associations,
                bounds=h["conservative_bounds"],
                member_cells=h["member_support_cells"],
            )
        )
    return dict(
        input_sha256=snapshot["provenance"]["input_sha256"],
        source_identity=snapshot["provenance"]["source_identity"],
        selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        terminal_evidence_sha256=hashlib.sha256(
            Path(__file__).with_name("shadow_terminal_evidence.py").read_bytes()
        ).hexdigest(),
        observed_components=[
            o["observation_id"]
            for o in observations.values()
            if o["kind"] == "HEIGHT_LEVEL_COMPONENT"
        ],
        counts=dict(Counter(row["role"] for row in records)),
        records=records,
        scopes=scopes,
        grid_contract=lattice,
        separator_evidence=separators,
        separator_support_audit=separator_support_audit(separators, families, bundle),
        grounded_scope_ids=[s["observation_id"] for s in grounded],
    )


def structure_adjacency(scopes, separator_pairs):
    """Require a signed separator for actual neighbors and preserve the chain."""
    by_id = {scope["observation_id"]: scope for scope in scopes}
    owners = {key: cell_set(scope["cells"]) for key, scope in by_id.items()}
    extents = {}
    for key, cells in owners.items():
        if cells:
            rows, cols = zip(*cells)
            extents[key] = min(cols), max(cols), min(rows), max(rows)
    ordered = sorted(extents, key=lambda key: (extents[key][0], key))
    blocked, graph = set(), []
    for i, a in enumerate(ordered):
        a0, a1, av0, av1 = extents[a]
        next_scope = None
        for b in ordered[i + 1:]:
            if by_id[a].get("vessel_id") != by_id[b].get("vessel_id"):
                continue
            b0, _, bv0, bv1 = extents[b]
            if max(av0, bv0) > min(av1, bv1):
                continue
            pair = frozenset((a, b))
            if owners[a] & owners[b]:
                blocked.update(pair)
                graph.append(dict(nodes=[a, b], relation="COMPETING_INTERIOR"))
            elif a1 < b0 and next_scope is None:
                next_scope = b
                relation = "ADJACENT_DISTINCT" if pair in separator_pairs else "AMBIGUOUS"
                if relation == "AMBIGUOUS":
                    blocked.update(pair)
                graph.append(dict(nodes=[a, b], relation=relation))
    return blocked, graph


def select(review):
    """Only resolve relationships whose support has a unique owner."""
    scopes = {s["observation_id"]: s for s in review["scopes"]}
    grounded = set(review["grounded_scope_ids"])
    for record in review["records"]:
        record["selected"] = False
        record["review_state"] = (
            "UNRESOLVED_OPENING"
            if record["role"] == "PHYSICAL_OPENING_HYPOTHESIS"
            else record["role"]
        )
        record["qualified_scope_ids"] = sorted(
            grounded.intersection(
                record["basin_witness_ids"] + record["structure_scope_ids"]
            )
        )
        record["selection_reason"] = (
            "NOT_AN_INDEPENDENT_OPENING"
            if record["role"] != "PHYSICAL_OPENING_HYPOTHESIS"
            else "AMBIGUOUS_NO_BOUND_TERMINAL_PAIR_OR_SEPARATOR"
        )
    candidates = [
        r for r in review["records"] if r["role"] == "PHYSICAL_OPENING_HYPOTHESIS"
    ]
    grid = review["grid_contract"]["canonical"]
    witnesses = {}
    review["canonical_separator_evidence"] = []
    for record in candidates:
        cells = cell_set(record["member_cells"])
        if any(
            cell_set(scopes[i]["cells"]) == cells for i in record["basin_witness_ids"]
        ):
            witnesses[record["hypothesis_id"]] = cell_centers(cells, grid)
        record["separator_witness_ids"] = []
    # A compound or fragment may corroborate a partition without lending its
    # terminals or its footprint to either enclosed interior.
    for a, b in combinations(candidates, 2):
        if a["hypothesis_id"] not in witnesses or b["hypothesis_id"] not in witnesses:
            continue
        a_matches = {m["scope_id"]: m for m in a["structure_associations"]}
        b_matches = {m["scope_id"]: m for m in b["structure_associations"]}
        for edge in review["separator_evidence"]:
            for a_id, b_id in (edge["scope_ids"], edge["scope_ids"][::-1]):
                if a_id not in a_matches or b_id not in b_matches:
                    continue
                signed = []
                for record, scope_id, matches in (
                    (a, a_id, a_matches),
                    (b, b_id, b_matches),
                ):
                    (line,) = edge["lines_by_scope"][scope_id]
                    core = cell_centers(matches[scope_id]["common_cells_rc"], grid)
                    signed.append(
                        wall_check(core, witnesses[record["hypothesis_id"]], line)
                    )
                if all(
                    not check["crosses"]
                    and check["side"] == edge["signed_checks"][scope_id]["side"]
                    for check, scope_id in zip(signed, (a_id, b_id))
                ):
                    review["canonical_separator_evidence"].append(
                        dict(
                            separator_id=edge["observation_id"],
                            hypothesis_ids=[a["hypothesis_id"], b["hypothesis_id"]],
                            scope_ids=[a_id, b_id],
                            signed_checks=signed,
                        )
                    )
                    for record in (a, b):
                        record["separator_witness_ids"].append(edge["observation_id"])
    for candidate in candidates:
        support = set(candidate["qualified_scope_ids"])
        if not support and not candidate["separator_witness_ids"]:
            continue
        owned = {
            token
            for scope in support
            for side in scopes[scope]["end_evidence"]
            for token in side
        }
        cells = set(map(tuple, candidate["member_cells"]))
        conflicts = []
        for other in candidates:
            if other is candidate or not (
                other["qualified_scope_ids"] or other["separator_witness_ids"]
            ):
                continue
            other_support = set(other["qualified_scope_ids"])
            other_raw = {
                token
                for scope in other_support
                for side in scopes[scope]["end_evidence"]
                for token in side
            }
            shared_separator = any(
                support.intersection(edge["scope_ids"])
                and other_support.intersection(edge["scope_ids"])
                for edge in review["separator_evidence"]
            )
            # A qualified relation establishes adjacency, but only a raw
            # token explicitly attributed to that boundary could be shared.
            shared_raw = owned.intersection(other_raw)
            shared_raw -= {
                token
                for edge in review["separator_evidence"]
                if support.intersection(edge["scope_ids"])
                and other_support.intersection(edge["scope_ids"])
                for group in review["separator_support_audit"]
                if group["structure_key"] == edge["witness"].get("family_id")
                for token in group["source_observation_ids"]
            }
            shared_cells = cells.intersection(map(tuple, other["member_cells"]))
            if support.intersection(other_support) or shared_raw or shared_cells:
                conflicts.append(
                    dict(
                        hypothesis_id=other["hypothesis_id"],
                        shared_raw=sorted(shared_raw),
                        shared_cell_count=len(shared_cells),
                        qualified_separator=shared_separator,
                    )
                )
        candidate["conflicts"] = conflicts
        candidate["selected"] = not conflicts
        candidate["review_state"] = (
            "COMPETING_INTERIOR_REVIEW" if conflicts else "SELECTED_OPENING"
        )
        candidate["selection_reason"] = (
            "AMBIGUOUS_OVERLAPPING_OR_UNALLOCATED_SUPPORT"
            if conflicts
            else "INDEPENDENT_PHYSICAL_SUPPORT"
        )
    selected = [r for r in candidates if r["selected"]]
    review["substructure_relations"] = []
    for parent in selected:
        parent["separator_witness_ids"] = sorted(set(parent["separator_witness_ids"]))
        if parent["hypothesis_id"] not in witnesses:
            continue
        for match in parent["structure_associations"]:
            scope = scopes[match["scope_id"]]
            if (
                scope["lineage_verified"]
                and max(match["canonical_only"]["distance_cells"], default=0) > 1
                and max(match["structure_only"]["distance_cells"], default=0) <= 1
                and match["separator_checks"]
                and not any(w["crosses"] for w in match["separator_checks"])
                and not match["competing_interior_scope_ids"]
            ):
                review["substructure_relations"].append(
                    dict(
                        scope_id=match["scope_id"],
                        parent_hypothesis_id=parent["hypothesis_id"],
                        relationship="SUBSTRUCTURE_OF_OPENING",
                        transfer_geometry=False,
                    )
                )

    # Keep an independently enclosed S4 scope available when height recall has
    # no equivalent canonical interpretation. Its native footprint is retained.
    occupied = [cell_set(r["member_cells"]) for r in selected]
    subordinate = {r["scope_id"] for r in review["substructure_relations"]}
    partitioned = {
        frozenset(edge["scope_ids"]) for edge in review["separator_evidence"]
    }
    structure = [
        scope
        for scope_id, scope in scopes.items()
        if scope_id in grounded
        and scope_id not in subordinate
        and scope.get("lineage_verified")
        and scope["independent_ends"]
        and any(scope_id in pair for pair in partitioned)
        and not any(cell_set(scope["cells"]) & cells for cells in occupied)
    ]
    blocked, graph = structure_adjacency(structure, partitioned)
    review["ownership_graph"] = graph + [
        dict(nodes=[r["hypothesis_id"], other["hypothesis_id"]], relation="COMPETING_INTERIOR")
        for r in candidates
        for other in r.get("conflicts", [])
        if r["hypothesis_id"] < other["hypothesis_id"]
    ]
    review["selected_structure_scope_ids"] = [
        s["observation_id"] for s in structure if s["observation_id"] not in blocked
    ]
    review["unresolved_openings"] = [
        dict(hypothesis_id=r["hypothesis_id"], reason=r["selection_reason"])
        for r in candidates if not r["selected"]
    ] + [
        dict(scope_id=s["observation_id"], reason="UNRESOLVED_STRUCTURE_OWNERSHIP")
        for s in structure if s["observation_id"] in blocked
    ]
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pcd", type=Path, help="same input for face-section source audit")
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    snapshot = json.loads(
        (args.snapshot_dir / "opening_hypothesis_ir.json").read_bytes()
    )
    bundle = json.loads(
        (args.snapshot_dir / "recorded_observation_bundle.json").read_bytes()
    )
    if (
        snapshot["provenance"]["input_sha256"] != bundle["input_sha256"]
        or bundle["errors"]
    ):
        raise ValueError("INVALID_SHADOW_EVIDENCE")
    raw_xyz = None
    if args.pcd is not None:
        from ship_perception.io.pcd_source import read_pcd

        frame = read_pcd(args.pcd)
        if frame.provenance["wire_sha256"] != bundle["input_sha256"]:
            raise ValueError("SHADOW_RAW_INPUT_IDENTITY_MISMATCH")
        raw_xyz = frame.xyz
    review = audit(snapshot, bundle, raw_xyz)
    atomic(args.output, select(review) if args.select else review)


if __name__ == "__main__":
    main()
