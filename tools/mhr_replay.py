"""Verify saved Raw evidence against the original PCD without using scorer truth."""

import argparse
import json
from pathlib import Path

import numpy as np

from ship_perception.manual_review.contract import RAW_ROWS, digest
from ship_perception.manual_review.frame_session import FrameSession
from ship_perception.manual_review.matcher import match
from ship_perception.manual_review.side_candidates import CONFIG, raw_set


def replay(frame, evidence):
    request, result = evidence["request"], evidence["result"]
    identity = ("wire_sha256", "semantic_sha256", "frame_id")
    if any(request[k] != frame.manifest[k] for k in identity):
        raise ValueError("REPLAY_FRAME_MISMATCH")
    if result.get("raw_rows") != RAW_ROWS or result.get("config") != CONFIG.audit():
        raise ValueError("REPLAY_CONFIG_OR_INDEX_CONTRACT_MISMATCH")
    n = len(frame.frame.xyz)
    count = 0
    for side in result["sides"]:
        for mode in side["modes"]:
            if (
                digest(mode["train"]) != mode["train_sha256"]
                or digest(mode["heldout"]) != mode["heldout_sha256"]
            ):
                raise ValueError("SECTION_DIGEST_MISMATCH")
            train, heldout = raw_set(mode["train"]), raw_set(mode["heldout"])
            if train & heldout or any(
                type(i) is not int or not 0 <= i < n for i in train | heldout
            ):
                raise ValueError("RAW_ROWS_INVALID")
            count += len(train | heldout)
        if (
            "joint_heldout" in side
            and digest(side["joint_heldout"]) != side["joint_heldout_sha256"]
        ):
            raise ValueError("JOINT_SECTION_DIGEST_MISMATCH")
    recalculated = match(frame, np.array(request["client_xy"]), request)
    comparable = ("match_state", "reason", "geometry", "sides", "separators")
    if any(digest(result[k]) != digest(recalculated[k]) for k in comparable):
        raise ValueError("REPLAY_MEASUREMENT_DIFFERENCE")
    return {
        "status": "PASS",
        "raw_rows": count,
        "wire_sha256": request["wire_sha256"],
        "execution_state": recalculated["execution_state"],
        "match_state": recalculated["match_state"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcd", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_bytes())
    frame = FrameSession(args.pcd, frame_id=evidence["request"]["frame_id"])
    print(json.dumps(replay(frame, evidence)))


if __name__ == "__main__":
    main()
