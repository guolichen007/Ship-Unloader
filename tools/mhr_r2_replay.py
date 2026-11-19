"""Recompute R2 private Raw evidence from a PCD and an archived explicit draft."""

import argparse
import json
from pathlib import Path

import numpy as np

from ship_perception.manual_review.contract import IDENTITY, digest, selection
from ship_perception.manual_review.frame_session import FrameSession
from ship_perception.manual_review.guided import guided_match
from ship_perception.manual_review.workbench import SELECT, compact_evidence
from ship_perception.manual_review.workbench_auto import load_auto


def verify(frame, record, targets):
    incoming, saved = record["request"], record["result"]
    for k in ("wire_sha256", "semantic_sha256", "frame_id"):
        if incoming[k] != frame.manifest[k]:
            raise ValueError("REPLAY_FRAME_MISMATCH")
    request = {**incoming, **{k: frame.manifest[k] for k in IDENTITY}}
    xy = selection(
        {k: request[k] for k in set(IDENTITY) | SELECT},
        frame.manifest,
        frame.projection,
        targets,
    )
    if saved.get("execution_state") in ("TIMEOUT", "CANCELLED", "BUDGET_EXCEEDED"):
        return dict(status="SKIPPED_TIMING_OR_BUDGET", reason=saved["reason"])
    if saved.get("evidence_schema") != "mhr_raw_observation_refs.2":
        raise ValueError("REPLAY_SCHEMA")
    raw_count = 0
    for side in saved["sides"]:
        rows = side["observations"]
        for row in rows:
            for key in ("face", "cap", "interior", "competing_interior"):
                ids = row[key]
                if any(
                    type(i) is not int or i < 0 or i >= len(frame.frame.xyz)
                    for i in ids
                ):
                    raise ValueError("RAW_INDEX_RANGE")
                raw_count += len(ids)
        for mode in side["modes"]:
            if digest([rows[i] for i in mode["train_refs"]]) != mode["train_sha256"]:
                raise ValueError("TRAIN_REF_DIGEST")
    for selected, side in zip(saved["selected_sides"], saved["sides"]):
        mode = next(
            m for m in side["modes"] if m["train_sha256"] == selected["train_sha256"]
        )
        train = set(mode["train_refs"])
        hold = set(selected["heldout_refs"])
        rows = side["observations"]
        if (
            train & hold
            or any(rows[i]["slab"] % 2 != 0 for i in train)
            or any(rows[i]["slab"] % 2 != 1 for i in hold)
        ):
            raise ValueError("TRAIN_HELDOUT_OVERLAP")
        if (
            digest([rows[i] for i in selected["heldout_refs"]])
            != selected["joint_heldout_sha256"]
        ):
            raise ValueError("HELDOUT_REF_DIGEST")
    actual = compact_evidence(guided_match(frame, xy, request, targets))
    for key in (
        "execution_state",
        "match_state",
        "reason",
        "sides",
        "selected_sides",
        "geometry",
        "coverage_audit",
        "separator_evidence",
    ):
        if digest(saved.get(key)) != digest(actual.get(key)):
            raise ValueError("REPLAY_MISMATCH_" + key)
    return dict(
        status="PASS", match_state=actual["match_state"], raw_index_references=raw_count
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pcd", type=Path)
    p.add_argument("evidence", type=Path)
    p.add_argument("--import-run", type=Path)
    p.add_argument("--cell-m", type=float, default=0.1)
    args = p.parse_args()
    record = json.loads(args.evidence.read_bytes())
    frame = FrameSession(args.pcd, args.cell_m, record["request"]["frame_id"])
    bundle = (
        load_auto(args.import_run, frame)
        if args.import_run
        else dict(targets={}, bundle_digest=None)
    )
    if bundle["bundle_digest"] != record["request"]["bundle_digest"]:
        raise ValueError("REPLAY_AUTO_BUNDLE_MISMATCH")
    print(json.dumps(verify(frame, record, bundle["targets"])))


if __name__ == "__main__":
    main()
