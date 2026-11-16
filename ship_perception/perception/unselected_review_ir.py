"""Preserve local Raw opening evidence when offline berth selection is unresolved.

These are IR search nominations, never main review frames or control geometry.
The existing local evidence/identity mechanisms are reused without new gates.
"""

import hashlib
import numpy as np

from ship_perception.perception.exact_density_evidence import ExactReviewEvidence
from ship_perception.perception.local_evidence import VesselLocalEvidence
from ship_perception.perception.local_enclosure_replay import (
    compile_local_evidence,
    resolve_local_identities,
)


def unselected_opening_ir(frame, scene, config, research, checkpoint=None):
    records = {}
    input_hash = frame.provenance.get("wire_sha256", frame.sha256)
    minimum = (
        research["min_points_per_coarse_cell"] * config["roi"]["min_deck_support_cells"]
    )
    for vessel in scene.vessels:
        if checkpoint:
            checkpoint(vessel.vessel_id)
        result = dict(
            status="UNRESOLVED_VESSEL_SELECTION",
            vessel_id=vessel.vessel_id,
            input_sha256=input_hash,
            vessel_point_index_sha256=vessel.point_index_sha256,
            main_review_visible=False,
            steel_certified=False,
            control_eligible=False,
            candidates=[],
            nominations=[],
        )
        records[vessel.vessel_id] = result
        if len(vessel.point_indexes) < minimum:
            result["reason"] = "INSUFFICIENT_RAW_SUPPORT_FOR_EXISTING_PROVIDERS"
            continue
        try:
            # Own immutable B1 indexes only. A different body cannot lend points
            # merely because the selected berth is unknown.
            cache = ExactReviewEvidence.from_frozen(
                VesselLocalEvidence.build(frame, vessel, config)
            )
            cache.primary_cache = cache
            entries = compile_local_evidence(cache, research)
            regions = resolve_local_identities([], entries, cache, None)
            result["nominations"] = [
                dict(
                    nomination=e["nomination"],
                    local_enclosure_supported=e["qualified"],
                    reason=e["reason"],
                    identity_state="UNRESOLVED_VESSEL_SELECTION",
                    main_review_visible=False,
                    steel_certified=False,
                    control_eligible=False,
                )
                for e in entries
            ]
            grid = cache.grid("coarse")
            raw_rows, raw_cols = grid.cell_indices(cache.xyz[:, :2])
            raw_keys = raw_rows * grid.count.shape[1] + raw_cols
            for region in regions:
                evidence = region.identity_evidence
                cells = np.asarray(
                    evidence["member_support_cells_rc"], dtype=int
                ).reshape(-1, 2)
                keys = cells[:, 0] * grid.count.shape[1] + cells[:, 1]
                indexes = vessel.point_indexes[np.isin(raw_keys, keys)]
                if not len(indexes):
                    result.setdefault("unobserved_region_ids", []).append(
                        region.region_id
                    )
                    continue
                digest = hashlib.sha256(
                    np.asarray(indexes, dtype="<i8").tobytes()
                ).hexdigest()
                identity = hashlib.sha256(
                    (input_hash + vessel.point_index_sha256 + digest).encode()
                ).hexdigest()
                box = region.conservative_bounds
                polygon = (
                    np.asarray(
                        [
                            [box[0], box[1]],
                            [box[2], box[1]],
                            [box[2], box[3]],
                            [box[0], box[3]],
                        ]
                    )
                    @ vessel.axes
                )
                result["candidates"].append(
                    dict(
                        candidate_id="UNSELECTED_RAW_IR:" + identity,
                        vessel_id=vessel.vessel_id,
                        input_sha256=input_hash,
                        local_region_id=region.region_id,
                        identity_state="UNRESOLVED_VESSEL_SELECTION",
                        geometry_role="IR_SEARCH_NOMINATION_NOT_CALIBRATED_FRAME",
                        frame_id=frame.frame_id,
                        bounds_coordinate_frame="B1_VESSEL_UV",
                        nomination_coordinate_frame="RAW_INPUT_XY",
                        local_axes=vessel.axes.tolist(),
                        bounds_uv=list(box),
                        nomination_boundary_xy=polygon.tolist(),
                        source_point_indexes=indexes.tolist(),
                        raw_point_index_sha256=digest,
                        raw_point_count=len(indexes),
                        raw_support_role="OBSERVED_INTERIOR_MEMBERS_NOT_STEEL_INLIERS",
                        local_enclosure_evidence=evidence,
                        main_review_visible=False,
                        steel_certified=False,
                        control_eligible=False,
                    )
                )
        except (ValueError, KeyError, FloatingPointError) as error:
            result.update(status="LOCAL_RAW_IR_FAILED", reason=str(error))
    return records
