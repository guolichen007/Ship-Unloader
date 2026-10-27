"""Offline candidate lifecycle accounting. Never imported by the detector.

Missing instrumentation is reported as unavailable, never converted to a zero
measurement count. Spatial matches to old objects are associations, not identity.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def identity(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def topology(row):
    return row.get("measured_separator_faces", row.get("topology_ambiguity"))


def supported_sides(result, minimum):
    supported = []
    for name, side in result.get("sides", {}).items():
        physical = min(
            (p.get("physical_section_count", 0) for p in side.get("phase_runs", [])),
            default=0,
        )
        if physical <= 0 or not side.get("independent_geometry_support", False):
            continue
        if min(
            side.get(key, 0)
            for key in (
                "raw_subvoxel_section_count",
                "positive_height_section_count",
                "support_section_count",
            )
        ) >= min(minimum, physical):
            supported.append(name)
    return supported


def funnel(result, minimum_sections):
    proposals = [
        dict(p, vessel_id=v["vessel_hypothesis_id"])
        for v in result["per_vessel_status"]
        for p in v.get("proposal_registry", [])
    ]
    queries = result.get("physical_candidate_registry", [])
    attempts = {a["query_id"]: a for a in result.get("precision_attempts", [])}
    finals = {h["proposal_id"]: h for h in result["geometry_only"]["hatches"]}
    suppressed = {
        p["proposal_id"]: p
        for v in result["per_vessel_status"]
        for p in v.get("precision_suppressed_candidates", {}).get("samples", [])
    }
    suppression_complete = all(
        v.get("precision_suppressed_candidates", {}).get("omitted_count", 0) == 0
        for v in result["per_vessel_status"]
    )
    compiled = result.get("compiled_candidate_registry", [])
    compile_by_query = {g["group_id"]: g for g in compiled}
    query_by_id = {}
    for query in queries:
        for pid in query.get("source_proposal_ids", []):
            query_by_id.setdefault(pid, []).append(query)
    exact = Counter(
        identity(
            [
                p["vessel_id"],
                p["bounds_axial"],
                sorted(
                    p["evidence"].get(
                        "source_provider_ids",
                        p["evidence"].get("supporting_provider_ids", []),
                    )
                ),
                topology(p["evidence"]),
            ]
        )
        for p in proposals
    )
    trace = []
    for proposal in proposals:
        pid = proposal["proposal_id"]
        for query in query_by_id.get(pid, [None]):
            group = compile_by_query.get(query.get("query_id")) if query else None
            qid = query.get("query_id") if query else None
            attempt = attempts.get(qid)
            final = finals.get(qid)
            probe = query.get("raw_probe") if query else None
            if final:
                stage, reason = None, None
                outcome = final["status"]
            elif query is None:
                stage, reason = (
                    "PHYSICAL_QUERY_COMPILATION",
                    "QUERY_NOT_REACHED_BEFORE_BUDGET_STOP",
                )
                outcome = "UNRESOLVED"
            elif query.get("state") == "PROPOSED":
                stage, reason = "PHYSICAL_VALIDATION", query.get(
                    "reason", "VALIDATION_NOT_REACHED"
                )
                outcome = "UNRESOLVED"
            elif query.get("state") == "REJECTED" or not query.get(
                "raw_edge_probe_allowed", query.get("precision_allowed", False)
            ):
                stage, reason = "PHYSICAL_VALIDATION", query.get("reason")
                outcome = query.get("state", "UNRESOLVED")
            elif probe and not probe.get("full_precision_allowed", False):
                stage, reason = "RAW_EDGE_PROBE", probe.get(
                    "reason", "RAW_PROBE_INSUFFICIENT"
                )
                outcome = "UNRESOLVED"
            elif (
                query.get("precision_disposition")
                == "EXPLAINED_BY_MEASURED_REVIEW_REGION"
            ):
                stage, reason = "FINAL_SELECTION", query["precision_disposition"]
                outcome = "EXPLAINED"
            elif not attempt and query.get("raw_edge_probe_allowed") and not probe:
                stage, reason = "RAW_EDGE_PROBE", "RAW_PROBE_BUDGET_DEFERRED"
                outcome = "UNRESOLVED"
            elif not attempt:
                stage, reason = (
                    "FULL_PRECISION",
                    "PRECISION_BUDGET_DEFERRED_OR_INTERRUPTED",
                )
                outcome = "UNRESOLVED"
            elif qid in suppressed:
                stage, reason = "FINAL_SELECTION", suppressed[qid]["reason"]
                outcome = "SUPPRESSED"
            else:
                measured = attempt["result"]
                sides = supported_sides(measured, minimum_sections)
                stage = "FULL_PRECISION"
                reason = measured.get("error_code") or (
                    "INSUFFICIENT_FINAL_PHYSICAL_SIDES"
                    if len(sides) < 3
                    else (
                        "FINAL_SELECTION_NOT_FULLY_RECORDED"
                        if not suppression_complete
                        else "FINAL_ADMISSION_UNRESOLVED"
                    )
                )
                outcome = "UNRESOLVED"
            trace.append(
                dict(
                    proposal_id=pid,
                    legacy_proposal_id=pid if proposal["expert"] == "LEGACY" else None,
                    expert=proposal["expert"],
                    vessel_id=proposal["vessel_id"],
                    original_bounds=proposal["bounds_axial"],
                    compiled_group_id=group["group_id"] if group else qid,
                    compiled_group_status=(
                        "RECORDED" if group else "NOT_IMPLEMENTED_IN_BASELINE"
                    ),
                    physical_query_id=qid,
                    query_bounds=query.get("bounds_axial") if query else None,
                    query_changed_bounds=bool(
                        query and proposal["bounds_axial"] != query.get("bounds_axial")
                    ),
                    validation_state=query.get("state") if query else None,
                    validation_reason=query.get("reason") if query else None,
                    raw_probe=probe,
                    full_precision_attempted=attempt is not None,
                    final_support_sides=(
                        supported_sides(attempt["result"], minimum_sections)
                        if attempt
                        else []
                    ),
                    first_blocking_stage=stage,
                    first_blocking_reason=reason,
                    outcome=outcome,
                )
            )
    probe_rows = [q["raw_probe"] for q in queries if q.get("raw_probe")]
    raw = dict(
        instrumented=bool(probe_rows), attempted=len(probe_rows) if probe_rows else None
    )
    for count in (4, 3, 2):
        raw[str(count) + "-side"] = (
            sum(p["raw_probe_support_side_count"] == count for p in probe_rows)
            if probe_rows
            else None
        )
    raw["conflict"] = (
        sum(p.get("raw_probe_conflict_side_count", 0) > 0 for p in probe_rows)
        if probe_rows
        else None
    )
    raw["insufficient"] = (
        sum(not p.get("full_precision_allowed", False) for p in probe_rows)
        if probe_rows
        else None
    )
    attempt_support = [
        len(supported_sides(a["result"], minimum_sections))
        for a in result.get("precision_attempts", [])
    ]
    return dict(
        schema="ship_perception.forensic_funnel.1",
        usage="OFFLINE_LIFECYCLE_NOT_STEEL_GOLDEN",
        input_sha=result["input"]["semantic_sha256"],
        software_sha=result["software_git_sha"],
        source_identity=result["software_source_identity"],
        proposal_count_by_expert=dict(Counter(p["expert"] for p in proposals)),
        proposal_count=len(proposals),
        exact_duplicate_count=sum(n - 1 for n in exact.values()),
        compiled_group_count=len(compiled) if compiled else None,
        compiler_instrumented=bool(compiled),
        physical_query_count=len(queries),
        validation=dict(
            **{
                state: sum(q.get("state") == state for q in queries)
                for state in (
                    "VALIDATED",
                    "PARTIAL",
                    "UNRESOLVED",
                    "REJECTED",
                    "PROPOSED",
                )
            },
            reason_histogram=dict(
                Counter(q.get("reason", "UNRECORDED") for q in queries)
            ),
        ),
        raw_probe=raw,
        instrumentation_limits=dict(
            raw_probe_absent_in_baseline=not bool(probe_rows),
            compiler_absent_in_baseline=not bool(compiled),
            suppression_registry_complete=suppression_complete,
            interrupted_precision_attempts_not_recorded=bool(
                result.get("pipeline_health", {}).get("isolated_failures")
            )
            and not result.get(
                "precision_attempt_registry_includes_interrupted", False
            ),
        ),
        full_precision=dict(
            attempted=len(result.get("precision_attempts", [])),
            accepted=sum(s >= 4 for s in attempt_support),
            partial=sum(s == 3 for s in attempt_support),
            unresolved=sum(s < 3 for s in attempt_support),
            budget_deferred=len(
                {
                    t["physical_query_id"]
                    for t in trace
                    if t["first_blocking_reason"]
                    == "PRECISION_BUDGET_DEFERRED_OR_INTERRUPTED"
                }
            ),
        ),
        final=dict(
            hatch_count=len(finals),
            unresolved_count=len(
                {
                    t["physical_query_id"] or t["proposal_id"]
                    for t in trace
                    if t["outcome"] == "UNRESOLVED"
                }
            ),
        ),
        first_blocker_histogram=dict(
            Counter(
                (t["first_blocking_stage"] or "FINAL")
                + ":"
                + (t["first_blocking_reason"] or "ADMITTED")
                for t in trace
            )
        ),
        performance=result.get("precision_profiler", {}),
        stage_elapsed=result.get("stage_elapsed", {}),
        proposal_trace=trace,
    )


def markdown(report):
    lines = [
        "# Candidate failure funnel",
        "",
        "Offline diagnostics; historical rectangles are not steel-edge Golden.",
        "",
        "| scene | proposals | exact duplicates | groups | physical queries | precision | final |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scene in report["scenes"]:
        f = scene["funnel"]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                scene["scene"],
                f["proposal_count"],
                f["exact_duplicate_count"],
                f["compiled_group_count"],
                f["physical_query_count"],
                f["full_precision"]["attempted"],
                f["final"]["hatch_count"],
            )
        )
    for scene in report["scenes"]:
        f = scene["funnel"]
        lines.extend(
            [
                "",
                "## " + scene["scene"],
                "",
                "```json",
                json.dumps(f["first_blocker_histogram"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-min-sections", type=int, required=True)
    args = parser.parse_args()
    scenes = []
    for row in json.loads(args.matrix.read_bytes()):
        if not row.get("output_dir"):
            continue
        path = Path(row["output_dir"]) / "03_results.json"
        result = json.loads(path.read_bytes())
        scenes.append(
            dict(
                scene=row["scene"],
                result_path=str(path),
                result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                funnel=funnel(result, args.profile_min_sections),
            )
        )
    report = dict(scenes=scenes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "forensic_funnel.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "forensic_funnel.md").write_text(
        markdown(report), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
