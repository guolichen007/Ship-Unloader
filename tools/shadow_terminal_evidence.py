"""Attach existing measured terminals to exactly matching Shadow interiors."""

import hashlib
from collections import defaultdict
from itertools import product

import numpy as np


def fingerprint(indexes):
    return hashlib.sha256(np.asarray(sorted(indexes), dtype="<i8").tobytes()).hexdigest()


def profile_sections(mode, inward):
    raw = mode.get("raw_profile_sections", [])
    for proof in mode["opening_assignment"]["sections"]:
        if not proof.get("supported"):
            continue
        section = raw[proof["section_index"]]["source_pair_provenance"]
        if section["normal_axis"] != 0 or section["inward_sign"] != inward:
            continue
        contrast = section["outside_surface_median_z"] - section["inside_surface_median_z"]
        indexes = set(section["source_pair_point_indexes"])
        if contrast > 0 and indexes:
            yield tuple(proof["span"]), indexes, contrast


def face_sections(mode, raw_uv, config):
    if raw_uv is None or mode["opening_assignment"].get("cuts_positive_opening_material"):
        return
    indexes = np.asarray(sorted(set(mode.get("source_point_indexes", []))), dtype=int)
    if not len(indexes) or indexes[0] < 0 or indexes[-1] >= len(raw_uv):
        return
    for proof in mode["opening_assignment"]["sections"]:
        lo, hi = proof["span"]
        in_span = indexes[(raw_uv[indexes, 1] >= lo) & (raw_uv[indexes, 1] <= hi)]
        if (
            len(in_span) >= config["boundary"]["line_min_points"]
            and proof["inside_owned_member_points"] >= config["boundary"]["profile_min_points"]
        ):
            yield (lo, hi), set(in_span.tolist()), None


def terminals(hatch, raw_uv, config):
    sides = hatch.get("raw_evidence_geometry", hatch)["sides"]
    minimum = config["boundary"]["profile_min_sections"]
    for name, inward in (("U0", 1), ("U1", -1)):
        for mode in sides[name]["candidate_modes"]:
            if (
                mode["source"] != "MEASURED_RAW"
                or not mode.get("independent_opening_role_gain")
                or mode.get("opening_assignment_state") != "ASSIGNED_CURRENT_OPENING"
                or mode["normal_uv"][0] <= 0
            ):
                continue
            profile = bool(mode.get("raw_profile_sections"))
            sections = (
                profile_sections(mode, inward) if profile else face_sections(mode, raw_uv, config)
            )
            records, owned, spans = [], set(), set()
            for span, indexes, contrast in sections:
                if span in spans or not indexes.difference(owned):
                    continue
                spans.add(span)
                owned.update(indexes)
                records.append(
                    dict(span_m=span, index_digest=fingerprint(indexes), index_count=len(indexes),
                         signed_height_contrast_m=contrast)
                )
            if len(records) < minimum:
                continue
            yield name, owned, dict(
                source_kind="RAW_PAIRED_PROFILE" if profile else "RAW3D_FACE",
                side=name, inward_sign=inward,
                source_observation_ids=mode.get("raw_support_ids", []),
                source_mode_id=mode["mode_id"],
                raw_original_index_digest=fingerprint(owned),
                raw_index_semantic="ORIGINAL_FINITE_INPUT_FRAME_ROW",
                section_index_digests=records, independent_section_count=len(records),
                uv_support_span=mode["support_span"],
                fitted_line=dict(normal_uv=mode["normal_uv"], rho_m=mode["rho"]),
                deck_context_provenance=mode.get("physical_role_gain_basis"),
                opening_side_assignment=mode["opening_assignment_state"],
                member_support_sha256=mode["opening_assignment"].get("frozen_member_support_sha256"),
                measurement_uncertainty_m=mode["uncertainty_m"],
                evidence_is_direct_raw=True, steel_certified=False,
            )


def by_local_interior(bundle, vessel, raw_uv):
    geometry = bundle.get("native_geometry") or {}
    grid = bundle["local_grid"]
    regions = {r["region_id"]: r for r in geometry.get("opening_regions", [])}
    by_members = defaultdict(list)
    for hatch in geometry.get("hatches", []):
        region = regions.get(hatch["opening_identity"])
        if region is None or region["vessel_id"] != vessel:
            continue
        identity = region["identity_evidence"]
        membership = identity.get("member_support_cells_rc")
        source_grid = identity.get("member_support_grid", {})
        if not membership or any(source_grid.get(k) != grid[k] for k in ("x0", "y0", "cell_m")):
            continue
        sides = defaultdict(list)
        for side, indexes, row in terminals(hatch, raw_uv, bundle["config"]):
            row.update(vessel_id=vessel, source_opening_identity=region["region_id"])
            sides[side].append((indexes, row))
        evidence = [row for group in sides.values() for _, row in group]
        independent = any(
            a.isdisjoint(b) and left["member_support_sha256"] == right["member_support_sha256"]
            for (a, left), (b, right) in product(sides["U0"], sides["U1"])
            if left["member_support_sha256"]
        )
        by_members[frozenset(map(tuple, membership))].append((evidence, independent))
    return by_members
