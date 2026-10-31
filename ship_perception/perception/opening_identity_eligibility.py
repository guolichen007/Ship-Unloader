"""Opening role and unresolved internal partition, before main review admission.

A closed wall arrangement is not an opening role. All thresholds here are
existing physical parameters; unknown context remains diagnostic, not deleted.
"""

import hashlib
import numpy as np
from .context import thaw
from .current_separator import _separator_split
from .measurement import RawSpatialIndex
from .sections import physical_sections, robust_height_support


def opening_role_eligibility(entry, config):
    nomination = entry["nomination"]
    if nomination["polarity"] == "LOW":
        rises = nomination.get("side_height_rises_m", [])
        drop = config["roi"]["opening_drop_m"]
        orthogonal = len(rises) == 4 and all(
            any(v is not None and v >= drop for v in rises[a:b])
            for a, b in ((0, 2), (2, 4))
        )
        if not orthogonal:
            return False, "NO_ORTHOGONAL_POSITIVE_OPENING_ROLE"
    interior = entry.get("current_interior_corner_evidence", {})
    if interior.get("status") in (
        "HEIGHT_AMBIGUOUS",
        "INSUFFICIENT_INTERIOR_OR_RING_SUPPORT",
    ):
        return False, "NO_INDEPENDENT_OPENING_INTERIOR_ROLE"
    return True, "CURRENT_OPENING_ROLE_AND_LOCAL_ENCLOSURE"


def internal_partition_review(entry, cache):
    # A raised material contour can be an internal cargo ridge. This weaker
    # topology test never splits cargo or certifies a cross-deck plane.
    if (
        entry.get("current_interior_corner_evidence", {}).get("status")
        != "LOWER_INTERIOR"
    ):
        return None
    primary = getattr(cache, "primary_cache", cache)
    config = primary.config
    box = entry["nomination"]["bounds_axial"]
    vessel = thaw(primary.vessel.public)
    vessel["local_axes"] = [[1, 0], [0, 1]]
    b2 = dict(
        scene=dict(vessel_hypotheses=[vessel]),
        per_vessel=[
            dict(
                vessel_hypothesis_id=primary.vessel.vessel_id,
                provider_evidence=primary.structural_providers(),
            )
        ],
    )
    parts, faces = _separator_split(
        dict(bounds_axial=box), b2, vessel, primary.grid("fine"), config
    )
    if not isinstance(faces, list) or len(parts) != 2:
        return None
    probe = config["boundary"]["side_probe_m"]
    query = [faces[0] - probe, box[1], faces[1] + probe, box[3]]
    measurement = getattr(cache, "v2_measurement", None)
    if measurement is not None:
        ids = measurement.spatial.query_indexes(query)
        source_ids = measurement.raw_point_indexes[ids]
        owned = np.sort(np.asarray(primary.vessel.point_indexes))
        order = np.searchsorted(owned, source_ids)
        keep = order < len(owned)
        keep[keep] &= owned[order[keep]] == source_ids[keep]
        points = measurement.raw_uvz[ids[keep]]
        original_ids = source_ids[keep]
    else:
        spatial = getattr(primary, "_identity_primary_spatial", None)
        if spatial is None:
            spatial = RawSpatialIndex(primary.xyz)
            primary._identity_primary_spatial = spatial
        ids = spatial.query_indexes(query)
        points = primary.xyz[ids]
        original_ids = primary.vessel.point_indexes[ids]
    edges = physical_sections(
        (box[1], box[3]), config["geometry"]["refine_voxel_m"], config
    )
    samples = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        samples.append(
            dict(
                span_m=[lo, hi],
                left=robust_height_support(points, 0, faces[0], (lo, hi), -1, config),
                right=robust_height_support(points, 0, faces[1], (lo, hi), 1, config),
            )
        )
    observed = [
        s
        for s in samples
        if s["left"]["status"] == "MEASURED" and s["right"]["status"] == "MEASURED"
    ]
    positive = [
        s
        for s in observed
        if min(s["left"]["signed_contrast_m"], s["right"]["signed_contrast_m"])
        >= config["roi"]["opening_drop_m"]
    ]
    compound = (
        len(observed) >= config["boundary"]["profile_min_sections"]
        and len(positive) >= len(observed) * config["roi"]["min_enclosure_ratio"]
    )
    digest = hashlib.sha256(
        np.asarray(sorted(original_ids), dtype="<i8").tobytes()
    ).hexdigest()
    return dict(
        status="COMPOUND_SEARCH_GROUP" if compound else "UNRESOLVED_INTERNAL_BAND",
        compound_search_group=bool(compound),
        separator_faces=list(faces),
        parent_bounds=list(box),
        nomination_id=entry["nomination"]["proposal_id"],
        observed_sections=len(observed),
        positive_opposite_role_sections=len(positive),
        raw_samples=samples,
        source="CURRENT_PRIMARY_RAW_AND_RAW3D_BEV_INTERNAL_TOPOLOGY",
        raw_point_index_sha256=digest,
        raw_point_count=len(original_ids),
        source_point_indexes=original_ids.tolist(),
        measured_cross_deck_certified=False,
        steel_edge_certified=False,
        coordinate_frame="VESSEL_REVIEW_UV",
    )


def compile_opening_roles(cache, entries):
    config = cache.config
    barriers = []
    for entry in entries:
        if not entry["qualified"]:
            continue
        valid, reason = opening_role_eligibility(entry, config)
        if not valid:
            entry.update(qualified=False, reason=reason)
            continue
        partition = internal_partition_review(entry, cache)
        if partition:
            entry["internal_partition_review"] = partition
        if partition and partition["compound_search_group"]:
            entry.update(
                qualified=False,
                reason="INTERNAL_SEPARATOR_ROLE_COMPOUND_NOT_ONE_KNOWN_HATCH",
                identity_role_hint="COMPOUND_SEARCH_GROUP",
            )
            barriers.append(partition)
    for entry in entries:
        if not entry["qualified"]:
            continue
        b = entry["nomination"]["bounds_axial"]
        constraints = {}
        for barrier in barriers:
            parent = barrier["parent_bounds"]
            overlap = np.prod(
                np.maximum(
                    0.0, np.minimum(b[2:], parent[2:]) - np.maximum(b[:2], parent[:2])
                )
            )
            area = (b[2] - b[0]) * (b[3] - b[1])
            if overlap < area * config["roi"]["min_enclosure_ratio"]:
                continue
            left, right = barrier["separator_faces"]
            middle = (left + right) / 2
            if b[0] < left and b[2] <= right:
                bound = constraints.get("U1", {}).get("upper_u", float("inf"))
                constraints["U1"] = dict(
                    upper_u=min(bound, left),
                    source=barrier["source"],
                    source_nomination_id=barrier["nomination_id"],
                    separator_faces=[left, right],
                    raw_point_index_sha256=barrier["raw_point_index_sha256"],
                    steel_certified=False,
                )
            elif b[0] >= left and b[2] > right:
                bound = constraints.get("U0", {}).get("lower_u", -float("inf"))
                constraints["U0"] = dict(
                    lower_u=max(bound, right),
                    source=barrier["source"],
                    source_nomination_id=barrier["nomination_id"],
                    separator_faces=[left, right],
                    raw_point_index_sha256=barrier["raw_point_index_sha256"],
                    steel_certified=False,
                )
            elif b[0] >= left and b[2] <= right:
                entry.update(
                    qualified=False,
                    reason="LOCAL_STRUCTURE_INSIDE_CURRENT_SEPARATOR_BAND",
                    identity_role_hint="WEAK_UNASSIGNED_REGION",
                )
            elif b[0] < left and b[2] > right:
                entry.update(
                    qualified=False,
                    reason="NOMINATION_CROSSES_CURRENT_INTERNAL_PARTITION",
                    identity_role_hint="COMPOUND_SEARCH_GROUP",
                )
        if constraints:
            entry["side_ownership_constraints"] = constraints
    return entries


def owned_enclosure_bounds(entry):
    box = [entry["sides"][n]["conservative_rho"] for n in ("U0", "V0", "U1", "V1")]
    ownership = entry.get("side_ownership_constraints", {})
    if "U0" in ownership:
        box[0] = max(box[0], ownership["U0"]["lower_u"])
    if "U1" in ownership:
        box[2] = min(box[2], ownership["U1"]["upper_u"])
    return box
