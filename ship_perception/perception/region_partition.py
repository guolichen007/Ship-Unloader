"""Attribute connected height members across current measured separator faces.

This refines existing height-node ownership. It does not enumerate rectangles
or run a legacy detector. Original S4 observations are never modified.
"""

import copy
import hashlib
from types import SimpleNamespace
import numpy as np

from .candidate_union import canonical
from .context import thaw
from .current_separator import _initial_seed_groups, _separator_split


def partition_current_nodes(nodes, xy, families, cache, legacy_rows):
    config = cache.config
    vessel = thaw(cache.vessel.public)
    vessel["local_axes"] = [[1, 0], [0, 1]]
    b2 = dict(
        scene=dict(vessel_hypotheses=[vessel]),
        per_vessel=[
            dict(
                vessel_hypothesis_id=cache.vessel.vessel_id,
                provider_evidence=cache.structural_providers(),
                hatch_hypotheses=legacy_rows,
            )
        ],
    )
    output = dict(nodes)
    audit = []
    for seed, members in _initial_seed_groups(legacy_rows, config):
        box = seed["bounds_axial"]
        boxes, separator = _separator_split(
            seed, b2, vessel, cache.grid("fine"), config
        )
        if not isinstance(separator, list) or len(boxes) != 2:
            continue
        scope = np.all((xy >= box[:2]) & (xy <= box[2:]), axis=-1)
        # Positive current Raw3D+BEV ends constrain this body corridor. Search
        # margins are not substituted for unobserved terminal positions.
        ends = [
            f
            for f in families
            if f["axis_role"] == "TRANSVERSE"
            and "OBSERVED_3D_FACE" in f["evidence_types"]
            and sum(
                max(0.0, min(hi, box[3]) - max(lo, box[1]))
                for lo, hi in f["observed_intervals"]
            )
            >= (box[3] - box[1]) * config["roi"]["min_enclosure_ratio"]
        ]
        seed_id = hashlib.sha256(
            canonical(
                [cache.vessel.vessel_id, seed["hatch_hypothesis_id"], separator]
            ).encode()
        ).hexdigest()[:16]
        for ident, node in sorted(nodes.items()):
            if ident not in output or not node.mask.any():
                continue
            associated = np.count_nonzero(node.mask & scope)
            if associated < config["roi"]["min_opening_cells"] or associated < (
                node.mask.sum() * config["roi"]["min_enclosure_ratio"]
            ):
                continue
            own_ends = ends
            # Families are expressed in the current S4 basis; construct their
            # actual observed points before expressing terminal limits in B1.
            end_u = [
                float(
                    (
                        f["rho"] * np.asarray(f["normal_uv"])
                        + np.mean(f["support_span"]) * np.asarray(f["direction_uv"])
                    )[0]
                )
                for f in own_ends
            ]
            if not end_u or min(end_u) >= separator[0] or max(end_u) <= separator[1]:
                continue
            corridor = (
                node.mask
                & (xy[:, :, 1] >= box[1])
                & (xy[:, :, 1] <= box[3])
                & (xy[:, :, 0] >= min(end_u) - config["boundary"]["corner_join_m"])
                & (xy[:, :, 0] <= max(end_u) + config["boundary"]["corner_join_m"])
            )
            masks = [
                corridor & (xy[:, :, 0] <= separator[0]),
                corridor & (xy[:, :, 0] >= separator[1]),
            ]
            usable = [
                m.sum() >= config["roi"]["min_opening_cells"]
                and m.sum() * cache.grid("coarse").cell_m ** 2
                >= config["roi"]["opening_min_area_m2"]
                for m in masks
            ]
            if not any(usable):
                continue
            del output[ident]
            entry = dict(
                source_height_node=ident,
                source_proposal_ids=[m["hatch_hypothesis_id"] for m in members],
                separator_source="CURRENT_RAW3D_BEV_TWO_FACE_SEPARATOR",
                separator_faces=list(separator),
                source_mask_cell_count=int(node.mask.sum()),
                unassigned_cells=int(node.mask.sum() - sum(m.sum() for m in masks)),
                current_enclosure_bounds=list(box),
                terminal_s4_family_ids=[f["family_id"] for f in own_ends],
                children=[],
            )
            for side, mask in enumerate(masks):
                if not usable[side]:
                    continue
                child_id = ident + ":P" + seed_id + ":" + str(side)
                attrs = copy.copy(vars(node))
                points = xy[mask]
                attrs.update(
                    node_id=child_id,
                    mask=mask,
                    bbox_xy=tuple(np.r_[points.min(0), points.max(0)]),
                    source_height_node=ident,
                    opening_domain=dict(
                        domain_id=seed_id,
                        side=side,
                        separator_faces=list(separator),
                        source="CURRENT_RAW3D_BEV_TWO_FACE_SEPARATOR",
                    ),
                )
                output[child_id] = SimpleNamespace(**attrs)
                entry["children"].append(
                    dict(node_id=child_id, side=side, support_cells=int(mask.sum()))
                )
            audit.append(entry)
    return output, audit
