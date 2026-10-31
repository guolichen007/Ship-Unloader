"""Compile existing, bounded nomination evidence before opening admission.

No hatch search, line product, prior polygon or expected count lives here.
Height supplies a work domain. Existing same-vessel providers supply separate
Raw3D and BEV witnesses; S4 remains an independent source, not a prerequisite.
"""

import hashlib
import numpy as np
from scipy import ndimage

from .candidate_union import canonical
from .local_evidence import local_height_recall
from .region_model import OpeningRegion, TopologySupport
from .validation import vessel_containment
from .opening_identity_eligibility import compile_opening_roles, owned_enclosure_bounds

SIDES = (("U0", 0, 0, -1), ("U1", 0, 2, 1), ("V0", 1, 1, -1), ("V1", 1, 3, 1))


def side_evidence_vector(cache, nomination):
    """One traversal per local side; competing bands are never averaged."""
    config = cache.config
    providers = {p["provider_id"]: p for p in cache.structural_providers()}
    candidates = cache.side_candidates(nomination["bounds_axial"])
    sides = {}
    for name, axis, pos, sign in SIDES:
        rows = sorted(candidates[name], key=lambda r: r["rho"])
        groups = []
        for row in rows:
            if (
                min(row["primary_raw3d"], row["primary_bev"])
                < config["roi"]["min_enclosure_ratio"]
            ):
                continue
            if (
                groups
                and row["rho"] - groups[-1][-1]["rho"]
                <= config["boundary"]["corner_join_m"]
            ):
                groups[-1].append(row)
            else:
                groups.append([row])
        quality = [
            max(min(r["primary_raw3d"], r["primary_bev"]) for r in g) for g in groups
        ]
        winners = (
            [i for i, q in enumerate(quality) if q == max(quality)] if quality else []
        )
        chosen = groups[winners[0]] if len(winners) == 1 else []
        ids = sorted({pid for r in chosen for pid in r["provider_ids"]})
        sides[name] = dict(
            normal_uv=np.eye(2)[axis].tolist(),
            coordinate_frame="VESSEL_REVIEW_UV",
            tangent_uv=np.eye(2)[1 - axis].tolist(),
            parameter_contract="PROVIDER_AXIS_NORMAL_FORM_IN_B1",
            selected_rho=(
                float(
                    max(r["rho"] for r in chosen)
                    if sign < 0
                    else min(r["rho"] for r in chosen)
                )
                if chosen
                else None
            ),
            conservative_rho=(
                float(
                    min(r["rho"] for r in chosen)
                    if sign < 0
                    else max(r["rho"] for r in chosen)
                )
                if chosen
                else None
            ),
            selected_band_rhos=[r["rho"] for r in chosen],
            raw3d_coverage=max((r["primary_raw3d"] for r in chosen), default=0.0),
            bev_coverage=max((r["primary_bev"] for r in chosen), default=0.0),
            provider_ids=ids,
            verified_face_provider_ids=[
                pid
                for pid in ids
                if "RAW3D_VERIFIED_VERTICAL_FACE" in providers[pid]["evidence_types"]
            ],
            observed_intervals=[
                list(providers[pid]["support_interval"]) for pid in ids
            ],
            competing_bands=[[r["rho"] for r in g] for g in groups],
            state="LOCAL_ENCLOSURE_WITNESS" if chosen else "MULTIMODE_OR_UNOBSERVED",
            measured_steel=False,
        )
    return sides


def enclosure_admission(nomination, sides, config):
    complete = all(sides[n]["selected_rho"] is not None for n, *_ in SIDES)
    if not complete:
        return False, "LOCAL_ORTHOGONAL_ENCLOSURE_INCOMPLETE"
    box = [sides[n]["selected_rho"] for n in ("U0", "V0", "U1", "V1")]
    if min(box[2] - box[0], box[3] - box[1]) < config["roi"]["min_deck_span_m"]:
        return False, "SAME_SIDE_OR_NARROW_STRUCTURE_NOT_AN_ENCLOSED_INTERIOR"
    # A raised surface is only a nomination. Two independently observed end
    # faces plus the opposing longitudinal witnesses establish its enclosure;
    # sand height itself neither confirms nor vetoes an opening.
    end_faces = [set(sides[n]["verified_face_provider_ids"]) for n in ("U0", "U1")]
    distinct_ends = all(end_faces) and not end_faces[0].intersection(end_faces[1])
    if not distinct_ends:
        if nomination["polarity"] == "RAISED":
            return False, "RAISED_STRUCTURE_WITHOUT_INDEPENDENT_END_ENCLOSURE"
        rises = nomination.get("side_height_rises_m", [])
        if len(rises) < 2 or not all(
            v is not None and v >= config["roi"]["opening_drop_m"] for v in rises[:2]
        ):
            return False, "NO_OPPOSITE_END_OPENING_ROLE_FROM_RAW_OR_HEIGHT"
    return True, "CURRENT_HEIGHT_NOMINATION_AND_ORTHOGONAL_RAW_BEV_ENCLOSURE"


def compile_local_evidence(cache, research):
    rows = getattr(cache, "_v2_height_recall", None)
    rows = rows[0] if rows is not None else local_height_recall(cache, research)[0]
    grid = cache.grid("coarse")
    xx, yy = grid.centers()
    labels_cache = {}
    output = []
    for row in rows:
        sides = side_evidence_vector(cache, row)
        qualified, reason = enclosure_admission(row, sides, cache.config)
        exterior = getattr(cache, "s4_exterior", None)
        exterior_review = (
            exterior.review(row["bounds_axial"], row["level_m"])
            if exterior is not None
            else dict(status="UNOBSERVED")
        )
        containment = vessel_containment(
            getattr(cache, "primary_cache", cache).grid("coarse"),
            row["bounds_axial"],
            cache.config,
        )
        if (
            exterior_review["status"] == "REJECT_EXTERIOR_LOW_REGION"
            or containment["status"] == "REJECT_OUTSIDE_VESSEL"
        ):
            qualified, reason = False, "CONFIRMED_EXTERIOR_OR_OUTSIDE_SELECTED_VESSEL"
        entry = dict(
            nomination=dict(row),
            sides=sides,
            qualified=qualified,
            reason=reason,
            exterior=exterior_review,
            containment=containment,
            compilation_stage="BEFORE_OPENING_IDENTITY",
            four_line_cartesian_enumeration_count=0,
            b2_lineage_required=False,
            s4_strong_required=False,
            measured_strong_count_required=False,
        )
        if qualified:
            from ship_perception.r1_static.opening_semantics import _interior_evidence

            measured_box = [sides[n]["selected_rho"] for n in ("U0", "V0", "U1", "V1")]
            entry["current_interior_corner_evidence"] = _interior_evidence(
                cache.context_grid("fine"), measured_box, cache.config
            )
            paired_ends = all(
                sides[n]["verified_face_provider_ids"] for n in ("U0", "U1")
            )
            if (
                not paired_ends
                and entry["current_interior_corner_evidence"].get(
                    "corner_support_count", 0
                )
                < 3
            ):
                entry.update(
                    qualified=False,
                    reason="NO_INDEPENDENT_END_ENCLOSURE_OR_THREE_SUPPORTED_CORNERS",
                )
            key = (row["level_m"], row["polarity"])
            if key not in labels_cache:
                delta = cache.config["roi"]["opening_drop_m"]
                mask = grid.occupancy & (
                    (grid.median < row["level_m"] - delta)
                    if row["polarity"] == "LOW"
                    else (grid.median > row["level_m"] + delta)
                )
                labels_cache[key] = ndimage.label(
                    mask, structure=np.ones((3, 3), bool)
                )[0]
            labels = labels_cache[key]
            b = row["bounds_axial"]
            inside = (xx >= b[0]) & (xx < b[2]) & (yy >= b[1]) & (yy < b[3])
            ids, counts = np.unique(labels[inside & (labels > 0)], return_counts=True)
            matching = [
                int(i)
                for i, count in zip(ids, counts)
                if count == row["occupied_cells"]
                and np.count_nonzero(labels == i) == count
            ]
            member = np.isin(labels, matching) & inside
            entry["member_support_cells_rc"] = np.argwhere(member).tolist()
            entry["member_support_grid"] = dict(
                x0=grid.x0, y0=grid.y0, cell_m=grid.cell_m
            )
            entry["member_support_role"] = (
                "FROZEN_OBSERVED_NOMINATION_CONTEXT_NOT_PRIMARY_RELABEL"
            )
            if not member.any():
                entry.update(
                    qualified=False,
                    reason="ORIGINAL_NOMINATION_MEMBER_SUPPORT_NOT_RECOVERED",
                )
        output.append(entry)
    return compile_opening_roles(cache, output)


def _overlap(a, b):
    return float(
        np.prod(np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    )


def resolve_local_identities(regions, evidence, cache, source):
    """Preserve supported identities; reconcile local witnesses and children."""
    config = cache.config
    limit = config["roi"]["min_enclosure_ratio"]
    result = list(regions)
    for region in result:
        b = region.conservative_bounds
        if (
            region.identity_role.startswith("CONFIRMED")
            and min(b[2] - b[0], b[3] - b[1]) < config["roi"]["min_deck_span_m"]
        ):
            region.identity_role = "WEAK_UNASSIGNED_REGION"
            region.identity_evidence["independent_extent_gate"] = (
                "NARROW_STRUCTURE_WITHOUT_COMPLETE_OPENING_EXTENT"
            )
    opening_owners = [r for r in regions if r.identity_role.startswith("CONFIRMED")]
    for region in result:
        if not region.identity_role.startswith("CONFIRMED"):
            continue
        lateral = any(
            region.side_topology_support[n].state in ("SUPPORTING", "STRONG")
            for n in ("V0", "V1")
        )
        local = cache.side_candidates(region.conservative_bounds)
        paired_lateral = all(
            any(min(m["primary_raw3d"], m["primary_bev"]) >= limit for m in local[n])
            for n in ("V0", "V1")
        )
        families = getattr(cache, "s4_topology_bridge", {}).get("side_families", [])
        terminal_ids = region.identity_evidence.get("opposite_terminal_family_ids", [])
        terminal_families = [f for f in families if f["family_id"] in terminal_ids]
        outside_terminal_scope = False
        b = region.conservative_bounds
        corners = np.asarray([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
        if terminal_families:
            extensions = []
            for f in terminal_families:
                support = sorted(f["support_span"])
                along = corners @ np.asarray(f["direction_uv"])
                extensions.append(
                    max(0.0, support[0] - along.min(), along.max() - support[1])
                )
            outside_terminal_scope = min(extensions) > config["roi"]["support_search_m"]
        if not lateral and not paired_lateral and outside_terminal_scope:
            region.identity_role = "WEAK_UNASSIGNED_REGION"
            region.identity_evidence["identity_admission"] = (
                "TERMINALS_WITHOUT_LONGITUDINAL_ENCLOSURE_NOT_AN_OPENING"
            )
            region.identity_evidence["structural_child_role"] = (
                "UNRESOLVED_CHILD_STRUCTURE"
            )
    eligible = [e for e in evidence if e["qualified"]]
    # Higher observed support first prevents tiny internal height slices from
    # becoming another identity. A measured separator keeps already distinct
    # S4 identities; no bbox-overlap rule merges those identities here.
    eligible.sort(
        key=lambda e: (
            -e["nomination"]["occupied_cells"],
            e["nomination"]["proposal_id"],
        )
    )
    admitted = []
    for e in eligible:
        box = e["nomination"]["bounds_axial"]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if e["nomination"]["polarity"] == "RAISED" and not any(
            _overlap(box, r.conservative_bounds) >= area * limit
            and r.identity_evidence.get("opposite_terminal_family_ids")
            for r in opening_owners
        ):
            e["identity_reason"] = (
                "RAISED_SURFACE_WITHOUT_PREEXISTING_OPENING_OWNERSHIP"
            )
            continue
        existing = [
            r
            for r in result
            if r.identity_role.startswith("CONFIRMED")
            and _overlap(box, r.conservative_bounds) >= area * limit
        ]
        if existing:
            for r in existing:
                r.identity_evidence.setdefault(
                    "local_enclosure_corroboration", []
                ).append(e["nomination"]["proposal_id"])
            continue
        same = next(
            (
                a
                for a in admitted
                if _overlap(box, a.conservative_bounds) >= area * (1 - limit)
            ),
            None,
        )
        if same is not None:
            same.identity_evidence.setdefault("child_structure_evidence", []).append(
                dict(
                    nomination_id=e["nomination"]["proposal_id"],
                    role="BOUNDARY_EVIDENCE_OF_PARENT",
                )
            )
            continue
        ident = (
            "LER"
            + hashlib.sha256(
                canonical(
                    [cache.vessel.vessel_id, e["nomination"]["proposal_id"]]
                ).encode()
            ).hexdigest()[:16]
        )
        margin = config["roi"]["support_search_m"]
        envelope = tuple(
            np.r_[np.asarray(box[:2]) - margin, np.asarray(box[2:]) + margin]
        )
        enclosure_box = owned_enclosure_bounds(e)
        if (
            min(
                enclosure_box[2] - enclosure_box[0], enclosure_box[3] - enclosure_box[1]
            )
            < config["roi"]["min_deck_span_m"]
        ):
            e["identity_reason"] = (
                "NO_INDEPENDENT_EXTENT_AFTER_CURRENT_PARTITION_OWNERSHIP"
            )
            continue
        r = OpeningRegion(
            ident,
            cache.vessel.vessel_id,
            tuple(enclosure_box),
            envelope,
            (
                TopologySupport(
                    "BOUNDED_LOCAL_ENCLOSURE",
                    "SUPPORTING",
                    tuple(
                        pid for s in e["sides"].values() for pid in s["provider_ids"]
                    ),
                    source,
                ),
            ),
            (e["nomination"]["proposal_id"],),
            (tuple(box),),
            "REGION_PARTIAL",
        )
        r.identity_role = "CONFIRMED_INDEPENDENT_OPENING"
        r.identity_evidence = dict(source="CURRENT_BOUNDED_LOCAL_EVIDENCE_VECTOR", **e)
        cells = np.asarray(e["member_support_cells_rc"], dtype=int)
        grid = e["member_support_grid"]
        points = (cells[:, ::-1] + 0.5) * grid["cell_m"] + [grid["x0"], grid["y0"]]
        owned = np.all(
            (points >= enclosure_box[:2]) & (points <= enclosure_box[2:]), axis=1
        )
        r.identity_evidence.update(
            member_support_cells_rc=cells[owned].tolist(),
            unassigned_nomination_member_support_cells_rc=cells[~owned].tolist(),
            support_freeze_stage="AFTER_CURRENT_ENCLOSURE_OWNERSHIP_BEFORE_REVIEW_ASSEMBLY",
            bounds_source="CONSERVATIVE_EXTENT_OF_CURRENT_ORTHOGONAL_PROVIDER_WITNESS_BANDS_NOT_SEARCH_ENVELOPE",
        )
        for name, s in e["sides"].items():
            r.side_topology_support[name] = TopologySupport(
                "RAW3D_BEV_ENCLOSURE_WITNESS",
                "SUPPORTING",
                tuple(s["provider_ids"]),
                source,
            )
        result.append(r)
        admitted.append(r)
    return result


def replay_review_constraints(region, config):
    output = {}
    evidence = region.identity_evidence
    for name, axis, pos, sign in SIDES:
        side = evidence["sides"][name]
        ownership = evidence.get("side_ownership_constraints", {}).get(name)
        values = side["selected_band_rhos"]
        intervals = side["observed_intervals"]
        low = min(i[0] for i in intervals)
        high = max(i[1] for i in intervals)
        span = [low, high] if axis == 0 else [-high, -low]
        output[name] = [
            dict(
                mode_id=region.region_id
                + ":"
                + name
                + ":CONSERVATIVE_WITNESS_BAND"
                + (
                    ":OWNED:" + ownership["raw_point_index_sha256"][:12]
                    if ownership
                    else ""
                ),
                source="TOPOLOGY_SUPPORTED",
                state="CONSERVATIVE_PRIOR",
                rho=(
                    region.conservative_bounds[pos]
                    if ownership
                    else side["conservative_rho"]
                ),
                normal_uv=np.eye(2)[axis].tolist(),
                support_span=span,
                uncertainty_m=max(
                    config["geometry"]["plane_inlier_m"], max(values) - min(values)
                ),
                physical_role="CURRENT_INDEPENDENT_ENCLOSURE_WITNESS_NOT_MEASURED_STEEL",
                physical_role_improved=False,
                independent_raw_sections=0,
                topology_support_ids=side["provider_ids"],
                raw_support_ids=[],
                candidate_band_rhos=values,
                unconstrained_witness_rho=side["conservative_rho"],
                side_ownership_constraint=ownership,
                uncertainty_kind="CONSERVATIVE_STRUCTURAL_ALIAS_BAND_NOT_RAW_FIT_PRECISION",
            )
        ]
    return output
