"""Offline semantic availability gate, independent of steel-edge accuracy."""

import argparse
import json
from pathlib import Path


def check(result, golden):
    expected = next(
        (
            s
            for s in golden["scenes"]
            if s["input_sha256"] == result["input"]["semantic_sha256"]
        ),
        None,
    )
    if expected is None:
        return dict(status="DATA_PENDING", steel_edge_certified=False)
    compiled = {g["group_id"] for g in result.get("compiled_candidate_registry", [])}
    current = {
        e["query_id"]: e
        for v in result["per_vessel_status"]
        for e in v.get("legacy_post_b2_evidence", [])
        if e["query_id"] in compiled
    }
    records, used = [], set()
    for candidate in expected["candidates"]:
        b2 = candidate["b2_proposal_id"]
        if not b2:
            records.append(
                dict(
                    candidate_id=candidate["candidate_id"],
                    status="FAIL",
                    reason="HISTORICAL_B2_LINEAGE_UNASSIGNED_PATH_NOT_PROVEN",
                )
            )
            continue
        eligible = [
            e
            for e in current.values()
            if "LEGACY:" + b2 in e["source_proposal_ids"] and e["query_id"] not in used
        ]
        findings = []
        for evidence in eligible:
            sides = evidence["side_evidence_availability"]
            failures = []
            if evidence["separator_topology"] != candidate["separator_topology"]:
                failures.append("SEPARATOR_TOPOLOGY_NOT_PRESERVED")
            if evidence["support_side_count"] < candidate["support_side_count"]:
                failures.append("SIDE_SUPPORT_AVAILABILITY_LOST")
            if any(
                (old["raw3d"] or old["bev"]) and not sides.get(name, False)
                for name, old in candidate["side_evidence_availability"].items()
            ):
                failures.append("SIDE_EVIDENCE_AVAILABILITY_LOST")
            if candidate["profile_observations_available"] and not evidence.get(
                "profile_observations"
            ):
                failures.append("LOCAL_PROFILE_EVIDENCE_NOT_RECOVERED")
            if not evidence["local_physical_candidate_available"]:
                failures.append("PHYSICAL_CANDIDATE_UNAVAILABLE")
            findings.append((len(failures), evidence["query_id"], failures))
        best = (
            min(findings)
            if findings
            else (1, None, ["DISTINCT_POST_B2_PHYSICAL_CANDIDATE_UNAVAILABLE"])
        )
        if best[1] is not None:
            used.add(best[1])
        records.append(
            dict(
                candidate_id=candidate["candidate_id"],
                b2_proposal_id=b2,
                compiled_group_id=best[1],
                status="PASS" if best[0] == 0 else "FAIL",
                failures=best[2],
            )
        )
    return dict(
        status=(
            "PASS"
            if records and all(r["status"] == "PASS" for r in records)
            else "FAIL"
        ),
        usage="POST_B2_PATH_AVAILABILITY_NOT_STEEL_GOLDEN",
        input_sha256=result["input"]["semantic_sha256"],
        software_sha=result["software_git_sha"],
        expected_candidate_count=expected["candidate_count"],
        candidate_checks=records,
        steel_edge_certified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = check(
        json.loads(args.result.read_bytes()), json.loads(args.golden.read_bytes())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
