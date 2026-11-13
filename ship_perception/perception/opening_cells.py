"""Same-vessel opening ownership for the opt-in review path."""

from collections import defaultdict
import hashlib

import numpy as np

from .constrained_rectangle import separator_constraints
from .opening_identity_eligibility import owned_enclosure_bounds


def _within(inner, outer, margin):
    return all(
        outer[axis] - margin <= inner[axis]
        and inner[axis + 2] <= outer[axis + 2] + margin
        for axis in (0, 1)
    )


def _has_terminal_pair(region):
    evidence = region.identity_evidence
    return bool(evidence.get("opposite_terminal_family_ids")) or (
        evidence.get("source") == "CURRENT_BOUNDED_LOCAL_EVIDENCE_VECTOR"
        and evidence.get("qualified") is True
    )


def _member_subset(child, parent):
    first, second = child.get("member_support_grid"), parent.get("member_support_grid")
    rows = child.get("member_support_cells_rc")
    parent_rows = parent.get("member_support_cells_rc")
    if not first or not second or not rows or not parent_rows:
        return None
    rc = np.asarray(rows, dtype="<i8")
    uv = (rc[:, ::-1] + 0.5) * first["cell_m"] + [first["x0"], first["y0"]]
    mapped = np.floor((uv - [second["x0"], second["y0"]]) / second["cell_m"]).astype(int)[:, ::-1]
    target = set(map(tuple, parent_rows))
    if not all(tuple(row) in target for row in mapped):
        return None
    return dict(grid=first, count=len(rc),
                grid_member_sha256=hashlib.sha256(rc.tobytes()).hexdigest())


def _shared_end(child, owner):
    return sorted({
        provider
        for end in ("U0", "U1")
        for provider in (
            set(child.side_topology_support[end].provider_ids)
            & set(owner.side_topology_support[end].provider_ids)
        )
    })


def _bounded_box(box, guards):
    result = list(box)
    for guard in guards:
        if guard["bound"] == "lower_u":
            result[0] = max(result[0], guard["value"])
        elif guard["bound"] == "upper_u":
            result[2] = min(result[2], guard["value"])
    return result if result[0] < result[2] and result[1] < result[3] else None


def _crosses_neighbor(bounds, owner, regions, margin):
    for other in regions:
        if other.vessel_id != owner.vessel_id or other.region_id == owner.region_id:
            continue
        a, b = owner.conservative_bounds, other.conservative_bounds
        if a[2] <= b[0] and bounds[2] > b[0] + margin:
            return True
        if b[2] <= a[0] and bounds[0] < b[2] - margin:
            return True
    return False


def _parent_links(child, anchors, regions, local, guards, cell_m):
    for owner in anchors:
        if child.vessel_id != owner.vessel_id:
            continue
        own_bounds = owner.conservative_bounds
        sub_bounds = child.conservative_bounds
        if sub_bounds[0] < own_bounds[0] - cell_m or sub_bounds[2] > own_bounds[2] + cell_m:
            continue
        contained = _member_subset(child.identity_evidence, owner.identity_evidence)
        if _within(sub_bounds, own_bounds, cell_m) and contained:
            yield owner, dict(relation="PART_OF", reason="OBSERVED_INTERIOR_SUBSET_WITHOUT_INDEPENDENT_TERMINAL_PAIR",
                              enclosure_id=None, shared_terminal_ids=_shared_end(child, owner),
                              member_support=contained, bounds_uv=list(own_bounds))
            continue
        shared = _shared_end(child, owner)
        if not shared:
            continue
        for nomination_id in sorted(owner.identity_evidence.get("local_enclosure_corroboration", ())):
            entry = local.get((owner.vessel_id, nomination_id))
            if not entry:
                continue
            bounds = _bounded_box(owned_enclosure_bounds(entry),
                                  guards[owner.region_id] + guards[child.region_id])
            subset = _member_subset(child.identity_evidence, entry)
            if (bounds and subset and _member_subset(owner.identity_evidence, entry)
                    and _within(sub_bounds, bounds, cell_m) and _within(own_bounds, bounds, cell_m)
                    and not _crosses_neighbor(bounds, owner, regions, cell_m)):
                yield owner, dict(relation="PART_OF", reason="SHARED_END_AND_CURRENT_WHOLE_ENCLOSURE",
                                  enclosure_id=nomination_id, shared_terminal_ids=shared,
                                  member_support=subset, bounds_uv=bounds)
                break


def _adjacency(regions, bridge):
    owners = defaultdict(set)
    for region in regions:
        for node in region.identity_evidence.get("member_height_nodes", ()):
            owners[node].add(region.region_id)
    witnesses = defaultdict(set)
    for relation in bridge.get("topology_relations", ()):
        a, b = relation["node_ids"]
        families = {r["family_id"] for r in relation.get("separator_evidence", ())
                    if r.get("qualified") and r.get("family_id")}
        if families:
            for first in owners[a]:
                for second in owners[b]:
                    if first != second:
                        witnesses[frozenset((first, second))].update(families)
    links = []
    ordered = sorted(regions, key=lambda r: (r.vessel_id, r.conservative_bounds[0], r.region_id))
    for a, b in zip(ordered, ordered[1:]):
        if a.vessel_id != b.vessel_id or a.conservative_bounds[2] > b.conservative_bounds[0]:
            continue
        families = witnesses[frozenset((a.region_id, b.region_id))]
        if families:
            links.append(dict(first=a.region_id, second=b.region_id, relation="ADJACENT_SEPARATED",
                              separator_family_ids=sorted(families)))
    return links


def opening_cells(regions, bridge, local_evidence, cell_m):
    """Assign weak fragments only to a unique supported owner in the same frame."""
    anchors = [r for r in regions if _has_terminal_pair(r)]
    local = {(row["nomination"]["vessel_hypothesis_id"], row["nomination"]["proposal_id"]): row
             for row in local_evidence if row["qualified"]}
    guards = {r.region_id: separator_constraints(r, bridge) for r in regions}
    assigned = defaultdict(list)
    removed = set()
    links = []
    for child in regions:
        if _has_terminal_pair(child):
            continue
        parents = list(_parent_links(child, anchors, regions, local, guards, cell_m))
        if len(parents) != 1:
            if parents:
                links.append(dict(first=child.region_id, competing_owners=sorted(r.region_id for r, _ in parents),
                                  relation="UNRESOLVED_COMPETING"))
            continue
        owner, evidence = parents[0]
        link = dict(first=child.region_id, second=owner.region_id, **evidence)
        assigned[owner.region_id].append(link)
        removed.add(child.region_id)
        links.append(link)
    # An expanded cell must have a single, unambiguous enclosure. Preserve
    # competing fragments as review candidates when the whole domain differs.
    for owner_id, children in list(assigned.items()):
        alternatives = {tuple(row["bounds_uv"]) for row in children if row["enclosure_id"]}
        if len(alternatives) > 1:
            for row in children:
                removed.remove(row["first"])
                row.update(relation="UNRESOLVED_COMPETING", reason="COMPETING_WHOLE_ENCLOSURES")
            del assigned[owner_id]
    retained = [r for r in regions if r.region_id not in removed]
    links.extend(_adjacency(retained, bridge))
    cells = []
    for region in retained:
        children = assigned[region.region_id]
        alternatives = {tuple(row["bounds_uv"]) for row in children if row["enclosure_id"]}
        bounds = next(iter(alternatives)) if alternatives else region.conservative_bounds
        cells.append(dict(
            cell_id="OC:" + region.region_id, vessel_id=region.vessel_id,
            primary_region_id=region.region_id,
            candidate_ids=[region.region_id] + sorted(row["first"] for row in children),
            source_proposal_ids=sorted(set(region.source_proposal_ids)
                                       | {row["enclosure_id"] for row in children if row["enclosure_id"]}),
            bounds_uv=list(bounds), expanded_review_only=bool(alternatives),
            separator_constraints=guards[region.region_id],
            adjacent_separator_family_ids=sorted({family for row in links
                if row["relation"] == "ADJACENT_SEPARATED" and region.region_id in (row["first"], row["second"])
                for family in row["separator_family_ids"]}),
            identity_state="REVIEW_PARTIAL",
        ))
    return cells, links


def coarse_review(cell, axes, measurement, state):
    """Report an orthogonal estimate with no invented measured sides or heights."""
    bounds = _bounded_box(cell["bounds_uv"], cell["separator_constraints"])
    if bounds is None:
        return None
    u0, v0, u1, v1 = bounds
    uv = np.asarray([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    xy = uv @ axes
    # Raw IDs here describe occupancy in the estimate, not steel edge support.
    indexes = measurement.spatial.query_indexes(bounds)
    source = np.sort(measurement.raw_point_indexes[indexes]).astype("<i8", copy=False)
    return dict(
        cell_id=cell["cell_id"], opening_identity=cell["primary_region_id"],
        vessel_hypothesis_id=cell["vessel_id"], geometry_role="OPENING_GEOMETRIC_REVIEW",
        strict_rectangle_state=state, steel_certified=False, certification_status="NOT_CERTIFIED",
        boundary_xy=xy.tolist(), boundary_uv=uv.tolist(),
        corners_xyz=[[float(x), float(y), None] for x, y in xy],
        side_states={name: "INFERRED_UNMEASURED" for name in ("U0", "U1", "V0", "V1")},
        raw_occupancy_finite_index_count=len(source),
        raw_occupancy_finite_index_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        editable=True,
    )
