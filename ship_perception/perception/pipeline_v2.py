"""Selected-vessel multi-expert union, one physical validation and precision."""

from dataclasses import replace
import hashlib
import json
import time
from pathlib import Path
import numpy as np

from ship_perception.r1_static.run import resolve_config
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG
from .pipeline import (
    Timing,
    software_identity,
    source_identity,
    numerical_environment,
    bounded,
    precision_suppression,
)
from .context import SceneEvidenceContext, thaw
from .local_evidence import VesselLocalEvidence
from .measurement import VesselMeasurementContext
from .berth import load_berth_reference, select_berth_vessel
from .selected_context import SelectedVesselRawContext
from .legacy_expert import LegacyProposalExpert, verify_legacy_sources
from .candidate_union import candidate_union, physical_queries, PhysicalBoundaryQuery
from .proposal_compiler import ProposalCompiler, evidence_lineage
from .legacy_post_b2 import LegacyPostB2EvidenceAdapter
from .raw_boundary_probe import RawBoundaryProbe
from .precision_scheduler import PrecisionScheduler
from .corner_height import CornerHeightMeasurement
from .experts import (
    HeightBasinExpert,
    StructuralLineExpert,
    LocalEnclosureExpert,
    LoadedHoldExpert,
    LargeSandHoldExpert,
    LayoutRecoveryExpert,
    MultiHoldLayoutExpert,
    line_families,
    role_families,
)
from .physical_validation import UnifiedPhysicalValidation, resolve_physical_modes
from .layout import VesselLayoutModel, cross_deck_evidence
from .final_boundary import FinalBoundaryRefiner, polygon
from .intensity import admission_stats
from .budget import PerformanceBudget, PerformanceBudgetExceeded
from .precision_diagnostics import PrecisionDiagnostics


def _deferred(row, axes, reason=None):
    bounds = row["bounds_axial"]
    uv = polygon(bounds)
    return dict(
        row,
        status="UNRESOLVED",
        reason=reason or row.get("reason", "PRECISION_NOT_RUN"),
        proposal_id=row.get("query_id", row.get("proposal_id")),
        geometry_role="UNREFINED_REVIEW_PROPOSAL_NOT_STEEL",
        proposal_boundary_uv=uv.tolist(),
        proposal_boundary_xy=(uv @ axes).tolist(),
        precision_admitted=False,
    )


def _canonical_queries(proposals, cache):
    from .candidate_union import canonical

    families = line_families(cache)

    def nominal_lineage(proposal):
        modes = []
        box = proposal.bounds
        for axis, position in ((0, 0), (0, 2), (1, 1), (1, 3)):
            span = (box[1 - axis], box[3 - axis])
            nearby = [
                f
                for f in families[axis]
                if f["raw3d"]
                and f["bev"]
                and abs(f["rho"] - box[position])
                <= cache.config["boundary"]["corner_join_m"]
                and max(
                    0, min(span[1], f["interval"][1]) - max(span[0], f["interval"][0])
                )
                >= (span[1] - span[0]) * cache.config["roi"]["min_enclosure_ratio"]
            ]
            if not nearby:
                return evidence_lineage(proposal)
            modes.append(
                min(nearby, key=lambda f: abs(f["rho"] - box[position]))["provider_ids"]
            )
        return canonical(modes)

    compiler = ProposalCompiler(cache.config, nominal_lineage)
    groups = compiler.compile(proposals)
    cache._v21_compiled_registry = [g.audit() for g in groups]
    cache._v21_exact_duplicates = compiler.exact_duplicate_count
    legacy = getattr(cache, "_v21_post_b2_queries", ())
    queries = [g.query() for g in groups]
    unmatched = []
    for query in legacy:
        evidence = cache._v21_adapter.evidence[query.query_id]
        match = next(
            (
                q
                for q in queries
                if q.vessel_id == query.vessel_id
                and q.bounds == query.bounds
                and evidence["measured_separator_faces"] is None
            ),
            None,
        )
        if match is None:
            unmatched.append(query)
        else:
            cache._v21_adapter.evidence[match.query_id] = dict(
                evidence,
                query_id=match.query_id,
                source_proposal_ids=[p.proposal_id for p in match.proposals],
            )
    cache._v21_compiled_registry.extend(
        dict(
            group_id=q.query_id,
            vessel_id=q.vessel_id,
            bounds_axial=list(q.bounds),
            source_proposal_ids=[p.proposal_id for p in q.proposals],
            candidate_sources=q.sources,
            member_bounds=[list(p.bounds) for p in q.proposals],
            union_search_envelope=cache._v21_adapter.evidence[q.query_id][
                "original_group_envelope"
            ],
            separator_topology=cache._v21_adapter.evidence[q.query_id][
                "separator_topology"
            ],
        )
        for q in unmatched
    )
    return tuple(queries) + tuple(unmatched)


def final_support_sides(result, config):
    """Count applied, independently supported sides, never empty sections.

    A span shorter than one physical section has no support. In particular,
    len([])-1 must not become a negative admission requirement. A guarded or
    conflicting line remains unresolved even if its search found raw points.
    """
    supported = []
    for name, side in result.get("sides", {}).items():
        runs = side.get("phase_runs", [])
        physical = min((r.get("physical_section_count", 0) for r in runs), default=0)
        if physical <= 0 or not side.get("independent_geometry_support", False):
            continue
        required = min(config["boundary"]["profile_min_sections"], physical)
        if (
            min(
                side.get("raw_subvoxel_section_count", 0),
                side.get("positive_height_section_count", 0),
                side.get("support_section_count", 0),
            )
            >= required
        ):
            supported.append(name)
    return supported


def run_perception(frame, progress=None, legacy_evidence=True, berth_reference=None):
    timing = Timing(progress)
    # Reserve the documented five seconds for rendering outside this core.
    budget = PerformanceBudget(total_s=55.0)
    config, config_sha = resolve_config()
    from .runtime_patch import git_source_state

    _, runtime_patch = git_source_state(Path(__file__).resolve().parents[2])
    research_bytes = RESEARCH_CONFIG.read_bytes()
    research = json.loads(research_bytes)
    if (
        len(frame.xyz) > config["geometry"]["max_input_points"]
        or np.max(np.ptp(frame.xyz, axis=0)) > config["geometry"]["max_local_extent_m"]
    ):
        raise ValueError("INPUT_POINT_OR_EXTENT_LIMIT_EXCEEDED")
    with timing.stage("SCENE_PREPROCESS"), budget.stage("SCENE_PREPROCESS"):
        context = SceneEvidenceContext.build(
            frame, config, research, config_sha, software_identity()
        )
        context.validate_frame(frame)
    path = (
        berth_reference
        or Path(__file__).resolve().parents[1] / "config/site/berth_reference.yaml"
    )
    reference = (
        load_berth_reference(path, frame.frame_id)
        if not isinstance(path, dict)
        else path
    )
    with timing.stage("BERTH_SIDE_VESSEL_SELECTION"), budget.stage(
        "BERTH_SIDE_VESSEL_SELECTION"
    ):
        selection = select_berth_vessel(frame, context, config, reference)
    hatches = []
    rejected = []
    failures = []
    by_vessel = []
    profiler = {}
    legacy_batches = []
    query_rows = []
    selected_audit = None
    precision_attempts = []
    precision_diagnostics = PrecisionDiagnostics()
    compiled_registry = []
    for vessel in context.vessels:
        by_vessel.append(
            dict(
                vessel_hypothesis_id=vessel.vessel_id,
                classification_status=vessel.public["classification_status"],
                status=(
                    "IGNORED_NON_BERTH_VESSEL"
                    if selection["active_vessel_id"] is not None
                    and vessel.vessel_id != selection["active_vessel_id"]
                    else selection["status"]
                ),
                hatches=[],
                unresolved_candidates=bounded([]),
                unresolved_review_proposals=bounded([]),
                independent_groups=[],
                aggregate_summary=dict(group_count=0, hatch_count=0),
            )
        )
    active = selection["active_vessel_id"]
    if active is not None:
        vessel = next(v for v in context.vessels if v.vessel_id == active)
        item = next(v for v in by_vessel if v["vessel_hypothesis_id"] == active)
        deferred = []
        ready = []
        cache = None
        validation = None
        layout = None
        measurement = None
        adapter = None
        proposals_seen = []
        expert_elapsed = {}
        queries = ()
        try:
            selected = SelectedVesselRawContext.build(frame, context, selection, config)
            selected_audit = selected.audit()
            base_cache = VesselLocalEvidence.build(
                frame, vessel, config, selected.measurement
            )
            measurement_vessel = replace(
                vessel,
                point_indexes=selected.measurement.point_indexes,
                point_index_sha256=selected.measurement.point_index_sha256,
            )
            cache = VesselLocalEvidence(
                measurement_vessel,
                base_cache.context_xyz,
                config,
                measurement_context=selected.measurement,
                context_xyz=base_cache.context_xyz,
            )
            if base_cache.context_xyz is base_cache.xyz:
                cache = base_cache
            measurement = VesselMeasurementContext(cache)
            measurement.budget = budget
            cache.v2_measurement = measurement
            with timing.stage("LOCAL_RECALL"), budget.stage("LOCAL_RECALL"):
                batches = []
                adapter = LegacyPostB2EvidenceAdapter(cache)
                cache._v21_adapter = adapter
                if legacy_evidence:
                    verify_legacy_sources()
                    expert_started = time.perf_counter()
                    batch = LegacyProposalExpert().propose(base_cache, research)
                    batches.append(batch.proposals)
                    proposals_seen.extend(batch.proposals)
                    expert_elapsed["LEGACY"] = time.perf_counter() - expert_started
                    legacy_batches.append(
                        dict(
                            vessel_id=active,
                            proposal_sha256=batch.output_sha256,
                            proposal_count=len(batch.proposals),
                            frozen_source_base_sha="51410e3f4f932fcbdcc7fc88b8f9979c60c2b954",
                            raw_proposal_evidence=batch.raw_output(),
                        )
                    )
                # Providers retain their immutable B1-core provenance. Extra
                # selected-vessel returns remain available to height and raw
                # precision without building another eager normal tree.
                cache.providers = base_cache.structural_providers()
                cache.provider_build_count = base_cache.provider_build_count
                if legacy_evidence:
                    cache._v21_post_b2_queries = adapter.adapt(batch, budget)
                for expert in (
                    HeightBasinExpert(),
                    StructuralLineExpert(),
                    LocalEnclosureExpert(),
                    LoadedHoldExpert(),
                    LargeSandHoldExpert(),
                ):
                    budget.check()
                    expert_started = time.perf_counter()
                    batch = expert.propose(cache, research, budget)
                    batches.append(batch)
                    proposals_seen.extend(batch)
                    expert_elapsed[expert.name] = time.perf_counter() - expert_started
                union = candidate_union(batches)
                item["expert_elapsed_s"] = expert_elapsed
                item["candidate_union"] = dict(
                    proposal_count=len(union),
                    expert_counts={
                        p.expert: sum(q.expert == p.expert for q in union)
                        for p in union
                    },
                    lossless_union=True,
                )
            with timing.stage("UNIFIED_PHYSICAL_VALIDATION"), budget.stage(
                "UNIFIED_PHYSICAL_VALIDATION"
            ):
                queries = _canonical_queries(union, cache)
                item["candidate_union"]["physical_query_count"] = len(queries)
                exterior = selected.exterior(frame, config)
                validation = UnifiedPhysicalValidation(cache, exterior, selection)
                for query in queries:
                    budget.check()
                    row = validation.evaluate(query)
                    query_rows.append(row)
                    if row["state"] == "REJECTED":
                        rejected.append(row)
                    elif row.get("raw_edge_probe_allowed", row["precision_allowed"]):
                        ready.append((query, row))
                    else:
                        deferred.append(_deferred(row, vessel.axes))
                layout = VesselLayoutModel(cache)
                # Recovery searches reuse real structural observations; a gap
                # without physical edges is a diagnostic, never a new hatch.
                validated = [
                    dict(
                        bounds=q.bounds,
                        independent_support_sides=r.get("independent_support_sides", 0),
                    )
                    for q, r in ready
                ]
                recovery, gaps = LayoutRecoveryExpert().search(
                    cache, validated, research, budget
                )
                multi, multi_gaps = MultiHoldLayoutExpert().search(
                    cache, validated, research, budget
                )
                recovery += multi
                proposals_seen.extend(recovery)
                item["layout_gap_searches"] = gaps
                item["multi_hold_searches"] = multi_gaps
                if recovery:
                    combined = _canonical_queries(union + recovery, cache)
                    previous = {
                        (
                            q.mode_signature
                            if all(m is not None for m in q.mode_signature)
                            and q.mode_signature
                            else q.query_id
                        )
                        for q in queries
                    }
                    for query in combined:
                        key = (
                            query.mode_signature
                            if all(m is not None for m in query.mode_signature)
                            and query.mode_signature
                            else query.query_id
                        )
                        if key in previous:
                            existing = next(
                                (
                                    r
                                    for r in query_rows
                                    if r["query_id"] == query.query_id
                                ),
                                None,
                            )
                            if existing is not None:
                                existing.update(
                                    candidate_sources=query.sources,
                                    source_proposal_ids=[
                                        p.proposal_id for p in query.proposals
                                    ],
                                )
                                ready = [
                                    (query if q.query_id == query.query_id else q, r)
                                    for q, r in ready
                                ]
                            continue
                        row = validation.evaluate(query)
                        query_rows.append(row)
                        if row.get("raw_edge_probe_allowed", row["precision_allowed"]):
                            ready.append((query, row))
                        elif row["state"] == "REJECTED":
                            rejected.append(row)
                        else:
                            deferred.append(_deferred(row, vessel.axes))
                for query, row in ready:
                    row["layout"] = layout.review(query.bounds)
                ready.sort(
                    key=lambda qr: (
                        -sum(
                            s["vertical_face_support"]
                            for s in qr[1]["deck_ring"]["sides"].values()
                        ),
                        -qr[1]["deck_ring"]["positive_height_side_count"],
                        -qr[1]["independent_support_sides"],
                        qr[1]["layout"]["ranking_penalty"],
                        qr[0].query_id,
                    )
                )
                item["cross_deck_evidence"] = [
                    cross_deck_evidence(cache, a[0].bounds, b[0].bounds)
                    for a, b in zip(
                        sorted(ready, key=lambda qr: qr[0].bounds[0]),
                        sorted(ready, key=lambda qr: qr[0].bounds[0])[1:],
                    )
                ]
            refiner = getattr(measurement, "refiner", None) or FinalBoundaryRefiner(
                measurement.raw_uvz, [], config, measurement
            )
            measurement.refiner = refiner
            probe = RawBoundaryProbe(measurement, refiner)
            scheduler = PrecisionScheduler(ready, config)
            for region in scheduler.regions:
                for scheduled_query, scheduled_row in region["queue"]:
                    scheduled_row["scheduler_region_bounds"] = list(region["bounds"])
                    precision_diagnostics.schedule(scheduled_query, scheduled_row)
            with timing.stage("FINAL_PRECISION"), budget.stage(
                "FINAL_PRECISION", hard=True
            ):
                measured_family = []
                for position, (query, row) in enumerate(scheduler):
                    trace = precision_diagnostics.schedule(query, row)
                    with trace.visit():
                        try:
                            budget.check()
                            if scheduler.explained(query, row):
                                row["precision_disposition"] = (
                                    "EXPLAINED_BY_MEASURED_REVIEW_REGION"
                                )
                                continue
                            if row.get("alternate_hypothesis"):
                                original_row = row
                                row = validation.evaluate(query)
                                row.update(
                                    alternate_hypothesis=True,
                                    scheduler_region_bounds=original_row[
                                        "scheduler_region_bounds"
                                    ],
                                )
                                query_rows.append(row)
                                if row["state"] == "REJECTED":
                                    continue
                            row["legacy_post_b2"] = adapter.evidence.get(
                                query.query_id, row.get("legacy_post_b2", {})
                            )
                            side_candidates = cache.side_candidates(query.bounds)
                            refiner.observations = []
                            refiner.observation_key = ""
                            raw_probe = probe.evaluate(query, row, side_candidates)
                            if not raw_probe[
                                "full_precision_allowed"
                            ] and refiner.can_investigate_sides(
                                query, row, side_candidates
                            ):
                                row["raw_probe_before_shared_measurements"] = dict(
                                    elapsed_s=raw_probe["elapsed_s"],
                                    sides={
                                        name: side["status"]
                                        for name, side in raw_probe["sides"].items()
                                    },
                                )
                                refiner.investigate_sides(query, row, side_candidates)
                                # RawProbe only consumes completed independent Raw
                                # measurements. It still computes no new normals,
                                # raster or corners, and applies the same admission.
                                raw_probe = probe.evaluate(query, row, side_candidates)
                            row["raw_probe"] = raw_probe
                            row["full_precision_allowed"] = raw_probe[
                                "full_precision_allowed"
                            ]
                            if not raw_probe["full_precision_allowed"]:
                                deferred.append(
                                    _deferred(row, vessel.axes, raw_probe["reason"])
                                )
                                if not row.get("alternate_hypothesis"):
                                    scheduler.alternate(query, row)
                                continue
                            profile_started = time.perf_counter()
                            try:
                                observations = (
                                    adapter.measure_profiles(
                                        query, raw_probe, measurement
                                    )
                                    if query.query_id in adapter.evidence
                                    else []
                                )
                            finally:
                                row["local_profile_elapsed_s"] = (
                                    time.perf_counter() - profile_started
                                )
                            proposal = dict(
                                row,
                                measured_side_candidates=side_candidates,
                                profile_observations=observations,
                                shared_physical_measurements=bool(
                                    row.get("raw_face_investigation")
                                ),
                            )
                            # All nominal inputs remain available for directional
                            # movement guards even after physical-mode dedup.
                            boxes = np.array([p.bounds for p in query.proposals])
                            proposal["bounds_before_separator"] = [
                                *boxes[:, :2].min(0),
                                *boxes[:, 2:].max(0),
                            ]
                            attempt = dict(
                                query_id=query.query_id,
                                bounds_axial=list(query.bounds),
                                status="INTERRUPTED",
                                result=dict(
                                    status="UNRESOLVED",
                                    error_code="FULL_PRECISION_BUDGET_INTERRUPTED",
                                    sides={},
                                ),
                            )
                            precision_attempts.append(attempt)
                            trace.row["full_precision_started"] = True
                            result = refiner.refine(proposal, vessel.axes)
                            attempt.update(status="COMPLETED", result=result)
                            trace.row["full_precision_completed"] = True
                        except PerformanceBudgetExceeded:
                            deferred.extend(
                                _deferred(r, vessel.axes, "PERFORMANCE_BUDGET_EXCEEDED")
                                for _, r in ready
                                if r.get("raw_probe") is None
                            )
                            raise
                        if not result.get("boundary_xy"):
                            deferred.append(
                                _deferred(row, vessel.axes, "RAW_PRECISION_UNRESOLVED")
                            )
                            if not row.get("alternate_hypothesis"):
                                scheduler.alternate(query, row)
                            continue
                        supported_sides = final_support_sides(result, config)
                        support = len(supported_sides)
                        result.update(
                            proposal_id=query.query_id,
                            vessel_hypothesis_id=active,
                            proposal_source="MULTI_EXPERT_UNION",
                            candidate_sources=query.sources,
                            source_proposal_ids=[
                                p.proposal_id for p in query.proposals
                            ],
                            candidate_state="PARTIAL",
                            containment=row["containment"],
                            exterior=row["exterior"],
                            deck_ring=row["deck_ring"],
                            layout=row["layout"],
                            review_support_side_count=support,
                            independently_supported_sides=supported_sides,
                            coverage_status="FULL_OPENING_COVERAGE_UNVERIFIED",
                            individual_hatch_count=None,
                        )
                        if support < 3:
                            deferred.append(
                                _deferred(
                                    row,
                                    vessel.axes,
                                    "INSUFFICIENT_FINAL_PHYSICAL_SIDES",
                                )
                            )
                            if not row.get("alternate_hypothesis"):
                                scheduler.alternate(query, row)
                            continue
                        result["status"] = "PARTIAL_HATCH"
                        item["hatches"].append(result)
                        scheduler.record(query, row, result)
                        uv = np.asarray(result["boundary_uv"])
                        measured_family.append(
                            dict(
                                bounds=(*uv.min(0), *uv.max(0)),
                                independent_support_sides=support,
                            )
                        )
                        layout.fit(measured_family)
                        scheduler.update_layout(layout)
                for result in item["hatches"]:
                    uv = np.asarray(result["boundary_uv"])
                    result["layout"] = layout.review((*uv.min(0), *uv.max(0)))
            profiler = dict(
                measurement.counters,
                proposal_count=len(union),
                physical_query_count=len(queries),
                precision_query_count=len(refiner.refinement_cache),
            )
        except PerformanceBudgetExceeded as error:
            failures.append(
                dict(
                    stage=str(error).split(":")[-1] or "EXECUTION_BUDGET",
                    code=str(error),
                )
            )
            # Validation results survive a precision interruption. The final
            # output always contains input/scene/selected-candidate diagnostics.
            if validation is None:
                deferred.extend(
                    _deferred(
                        dict(
                            p.evidence(),
                            bounds_axial=list(p.bounds),
                            proposal_id=p.proposal_id,
                            candidate_sources=[p.expert],
                        ),
                        vessel.axes,
                        "PERFORMANCE_BUDGET_EXCEEDED",
                    )
                    for p in proposals_seen
                )
            if validation is not None:
                known = {r["proposal_id"] for r in deferred}
                deferred.extend(
                    _deferred(r, vessel.axes, "PERFORMANCE_BUDGET_EXCEEDED")
                    for r in validation.decisions
                    if r["state"] != "REJECTED"
                    and r["proposal_id"] not in known
                    and not any(
                        h.get("proposal_id") == r["query_id"] for h in item["hatches"]
                    )
                )
        if measurement is not None:
            profiler.update(measurement.counters)
        if measurement is not None:
            profiler.update(
                proposal_count=len(proposals_seen),
                physical_query_count=len(queries),
                precision_query_count=len(precision_attempts),
                precision_incomplete_count=int(bool(failures)),
                frozen_provider_normals="LEGACY_PROVIDER_REQUIRED; NOT_INCLUDED_IN_PRECISION_NORMAL_COUNTERS",
                compiled_group_count=len(getattr(cache, "_v21_compiled_registry", [])),
                exact_duplicate_count=getattr(cache, "_v21_exact_duplicates", 0),
            )
            completed = [
                a for a in precision_attempts if a.get("status") == "COMPLETED"
            ]
            slowest = max(
                completed,
                key=lambda a: a["result"]
                .get("precision_profiler", {})
                .get("per_candidate_elapsed_s", 0),
                default=None,
            )
            profiler["slowest_candidate"] = (
                dict(
                    query_id=slowest["query_id"],
                    elapsed_s=slowest["result"]["precision_profiler"][
                        "per_candidate_elapsed_s"
                    ],
                )
                if slowest
                else None
            )
            side_times = [
                (s.get("elapsed_s", 0), q["query_id"], name)
                for q in query_rows
                for name, s in q.get("raw_probe", {}).get("sides", {}).items()
            ]
            profiler["slowest_probe_side"] = max(side_times, default=None)
            full_times = [
                (elapsed, a["query_id"], name, "FULL_PRECISION")
                for a in completed
                for name, elapsed in a["result"]["precision_profiler"]
                .get("per_side_elapsed_s", {})
                .items()
            ]
            profiler["slowest_side"] = max(
                full_times + [(*value, "RAW_EDGE_PROBE") for value in side_times],
                default=None,
            )
            candidate_times = []
            for row in query_rows:
                elapsed = row.get("raw_probe", {}).get("elapsed_s", 0) + row.get(
                    "local_profile_elapsed_s", 0
                )
                elapsed += sum(
                    a["result"]
                    .get("precision_profiler", {})
                    .get("per_candidate_elapsed_s", 0)
                    for a in completed
                    if a["query_id"] == row["query_id"]
                )
                candidate_times.append((elapsed, row["query_id"]))
            if candidate_times:
                elapsed, query_id = max(candidate_times)
                profiler["slowest_candidate"] = dict(
                    query_id=query_id, elapsed_s=elapsed
                )
        diagnostics = precision_diagnostics.report()
        profiler.update({k: v for k, v in diagnostics.items() if k != "candidates"})
        slowest = max(
            diagnostics["candidates"], key=lambda c: c["elapsed_total"], default=None
        )
        profiler["slowest_candidate"] = (
            dict(
                query_id=slowest["query_id"],
                candidate_id=slowest["candidate_id"],
                elapsed_s=slowest["elapsed_total"],
                timeout_stage=slowest["timeout_stage"],
            )
            if slowest
            else None
        )
        item["proposal_registry"] = [
            dict(
                proposal_id=p.proposal_id,
                expert=p.expert,
                bounds_axial=list(p.bounds),
                evidence=p.evidence(),
            )
            for p in proposals_seen
        ]
        processed = {r["query_id"] for r in query_rows}
        for query in queries:
            if query.query_id not in processed:
                pending = dict(
                    query_id=query.query_id,
                    proposal_id=query.query_id,
                    vessel_hypothesis_id=active,
                    bounds_axial=list(query.bounds),
                    candidate_sources=query.sources,
                    source_proposal_ids=[p.proposal_id for p in query.proposals],
                    state="PROPOSED",
                    precision_allowed=False,
                    reason="PERFORMANCE_BUDGET_EXCEEDED",
                )
                query_rows.append(pending)
                deferred.append(_deferred(pending, vessel.axes))
        if cache is not None:
            compiled_registry = getattr(cache, "_v21_compiled_registry", [])
            item["legacy_post_b2_evidence"] = (
                list(adapter.evidence.values()) if adapter is not None else []
            )
            item["boundary_line_graph"] = getattr(cache, "_v2_boundary_graph", {})
            item["provider_cache"] = dict(
                raw3d_build_count=cache.provider_build_count,
                bev_build_count=cache.provider_build_count,
                grid_build_counts=cache.grid_build_counts,
            )
        item["hatches"], suppressed = precision_suppression(
            item["hatches"], vessel.axes, config
        )
        item["precision_suppressed_candidates"] = bounded(suppressed)
        item["unresolved_review_proposals"] = bounded(deferred)
        item["status"] = "REVIEW_CANDIDATES" if item["hatches"] else "UNRESOLVED"
        for result in item["hatches"]:
            with timing.stage("CORNER_HEIGHT_OUTPUT"):
                result["corner_heights"] = CornerHeightMeasurement(
                    measurement, config
                ).measure(result)
            if layout is not None:
                uv = np.asarray(result["boundary_uv"])
                result["layout"] = layout.review((*uv.min(0), *uv.max(0)))
            result.update(
                hatch_id="H%03d" % len(hatches),
                input_frame=frame.frame_id,
                transform_provenance="FROZEN_SELECTED_VESSEL_UV_TO_INPUT_XY",
                candidate_blockers=["STEEL_INNER_EDGE_UNVERIFIED"]
                + (["UNDER_COVERAGE_RISK"] if result["undercoverage_risk"] else [])
                + (
                    ["INCOMPLETE_FOUR_SIDE_PHYSICAL_SUPPORT"]
                    if result["review_support_side_count"] < 4
                    else []
                ),
            )
            hatches.append(result)
        if measurement is not None:
            profiler.update(measurement.counters)
        item["independent_groups"] = [
            dict(
                group_id="G%03d" % i,
                hatch_ids=[r["hatch_id"]],
                source_proposal_ids=r["source_proposal_ids"],
            )
            for i, r in enumerate(item["hatches"])
        ]
        item["aggregate_summary"] = dict(
            group_count=len(item["independent_groups"]),
            hatch_count=len(item["hatches"]),
        )
    intensity = admission_stats(
        frame.intensity, frame.provenance.get("intensity_dtype")
    )
    intensity.update(
        used=False, production_scoring_enabled=False, geometry_mode="XYZ_ONLY"
    )
    performance = budget.report()
    performance.update(
        core_execution_budget_s=55.0,
        core_budget_exceeded=performance["total_budget_exceeded"],
        render_reserve_s=5.0,
        total_budget_s=60.0,
        total_budget_exceeded=performance["total_elapsed_s"] > 60.0,
    )
    profiler.update(
        {k: v for k, v in precision_diagnostics.report().items() if k != "candidates"}
    )
    identity = source_identity()
    return dict(
        schema="ship_perception.hatch_once.2",
        pipeline_version="multi-expert-recall-perf.2.1",
        production_geometry_mode="XYZ_ONLY",
        _ownership_indexes={v.vessel_id: v.point_indexes for v in context.vessels},
        input=dict(
            semantic_sha256=frame.sha256,
            provenance=frame.provenance,
            point_count=len(frame.xyz),
            source_point_count=frame.source_point_count,
            bbox_xyz=[frame.xyz.min(0).tolist(), frame.xyz.max(0).tolist()],
            fields=list(frame.fields),
            frame_id=frame.frame_id,
            timestamp=frame.stamp,
        ),
        scene_context=context.audit(),
        scene=thaw(context.scene),
        berth_selection=selection,
        selected_raw_context=selected_audit,
        config_sha256=config_sha,
        software_git_sha=context.software_sha,
        software_source_identity=identity,
        git_head_sha=identity.get("git_head_sha"),
        git_worktree_dirty=identity.get("git_worktree_dirty"),
        git_diff_sha256=identity.get("git_diff_sha256"),
        runtime_source_sha256=identity["runtime_source_sha256"],
        input_sha256=frame.provenance.get("wire_sha256", frame.sha256),
        _runtime_patch=runtime_patch,
        research_config_sha256=hashlib.sha256(research_bytes).hexdigest(),
        runtime=dict(
            frame.provenance.get(
                "runtime", dict(transport=frame.provenance.get("transport", "OFFLINE"))
            ),
            numerical_environment=numerical_environment(),
        ),
        intensity_status=intensity,
        intensity_aux_weight=0.0,
        stage_elapsed=timing.elapsed,
        per_vessel_status=by_vessel,
        per_hatch_status={h["hatch_id"]: h["status"] for h in hatches},
        per_side_uncertainty={
            h["hatch_id"]: h["per_side_uncertainty"] for h in hatches
        },
        grid_phase_stability={
            h["hatch_id"]: {k: s["grid_phase"] for k, s in h["sides"].items()}
            for h in hatches
        },
        undercoverage_risk={h["hatch_id"]: h["undercoverage_risk"] for h in hatches},
        geometry_only=dict(
            hatch_count=len(hatches),
            review_candidate_count=len(hatches),
            hatch_count_status="REVIEW_ONLY",
            hatches=hatches,
            intensity_used=False,
        ),
        pipeline_health=dict(
            status="REVIEW_REQUIRED",
            first_failure_stage=failures[0]["stage"] if failures else None,
            isolated_failures=failures,
            b1_execution_count=1,
            steel_precision_status="UNVERIFIED_NO_INDEPENDENT_INNER_EDGE_GOLDEN",
            legacy_expert_enabled=legacy_evidence,
        ),
        legacy_proposal_batches=legacy_batches,
        physical_validation=bounded(query_rows),
        physical_candidate_registry=query_rows,
        precision_attempts=precision_attempts,
        precision_candidates=precision_diagnostics.report()["candidates"],
        precision_attempt_registry_includes_interrupted=True,
        compiled_candidate_registry=compiled_registry,
        precision_profiler=profiler,
        performance_budget=performance,
        rejected_exterior_candidates=bounded(
            [r for r in rejected if r["reason"] == "STRONG_EXTERIOR_WATER_CONNECTION"]
        ),
        rejected_outside_vessel=bounded(
            [
                r
                for r in rejected
                if r["reason"]
                in ("CONFIRMED_OUTSIDE_SELECTED_VESSEL", "CONFIRMED_OTHER_VESSEL")
            ]
        ),
        scene_exterior_low_regions=dict(
            policy="CONFIRMED_EXTERIOR_ONLY; UNKNOWN_IS_NOT_EXTERIOR"
        ),
    )
