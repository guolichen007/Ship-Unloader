"""Explicit operator vessel scope, never hatch geometry or an automatic berth."""

import hashlib
import json
from pathlib import Path
import numpy as np


def select_operator_vessel(frame, context, path):
    raw = Path(path).read_bytes()
    request = json.loads(raw)
    allowed = {
        "schema",
        "frame_id",
        "source_input_sha256",
        "point_indexes_file",
        "point_indexes_sha256",
        "provenance",
    }
    if (
        set(request) - allowed
        or request.get("schema") != "ship_perception.operator_vessel_roi.1"
    ):
        raise ValueError("INVALID_OPERATOR_VESSEL_SELECTION")
    if (
        request["frame_id"] != frame.frame_id
        or request["source_input_sha256"] != frame.provenance["wire_sha256"]
    ):
        raise ValueError("OPERATOR_ROI_INPUT_IDENTITY_MISMATCH")
    data = Path(request["point_indexes_file"])
    content = data.read_bytes()
    if hashlib.sha256(content).hexdigest() != request["point_indexes_sha256"]:
        raise ValueError("OPERATOR_ROI_INDEX_FILE_HASH_MISMATCH")
    indexes = np.load(data, allow_pickle=False)
    if (
        indexes.ndim != 1
        or indexes.dtype.kind not in "iu"
        or not len(indexes)
        or indexes.min() < 0
        or indexes.max() >= len(frame.xyz)
    ):
        raise ValueError("INVALID_OPERATOR_ROI_INDEXES")
    candidates = [
        v
        for v in context.vessels
        if v.public["classification_status"] == "VESSEL_HYPOTHESIS"
        and np.isin(v.point_indexes, indexes).any()
    ]
    active = candidates[0].vessel_id if len(candidates) == 1 else None
    return dict(
        status=(
            "OPERATOR_SELECTED_VESSEL_NOT_SITE_CERTIFIED"
            if active
            else "OPERATOR_VESSEL_SELECTION_AMBIGUOUS"
        ),
        active_vessel_id=active,
        ignored_vessel_ids=[
            v.vessel_id for v in context.vessels if active and v.vessel_id != active
        ],
        reference=None,
        operator_roi_sha256=hashlib.sha256(raw).hexdigest(),
        candidate_vessel_ids=[v.vessel_id for v in candidates],
        original_input_preserved=True,
        raw_points_filtered_by_roi=False,
        site_reference_certified=False,
    )
