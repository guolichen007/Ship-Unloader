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
    queries = physical_queries(
        proposals, cache.config, lambda p: resolve_physical_modes(p, cache)
    )
    families = {
        f["provider_ids"]: f for group in line_families(cache).values() for f in group
    }
    for group in role_families(cache).values():
        for family in group:
            for sign in family["supported_signs"]:
                families[family["provider_ids"] + (("SIGNED_RAW_ROLE", sign),)] = family
    output = []
    for q in queries:
        if len(q.mode_signature) == 4 and all(m is not None for m in q.mode_signature):
            u0, u1, v0, v1 = [families[m]["rho"] for m in q.mode_signature]
            q = replace(q, bounds=(u0, v0, u1, v1))
        output.append(q)
    return tuple(output)


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
                    elif row["precision_allowed"]:
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
                    combined = _canonical_queries(
                        tuple(p for q in queries for p in q.proposals) + recovery, cache
                    )
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
                        if row["precision_allowed"]:
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
            with timing.stage("FINAL_PRECISION"), budget.stage(
                "FINAL_PRECISION", hard=True
            ):
                measured_family = []
                for position, (query, row) in enumerate(ready):
                    try:
                        budget.check()
                        proposal = dict(
                            row,
                            measured_side_candidates=cache.side_candidates(
                                query.bounds
                            ),
                        )
                        # All nominal inputs remain available for directional
                        # movement guards even after physical-mode dedup.
                        boxes = np.array([p.bounds for p in query.proposals])
                        proposal["bounds_before_separator"] = [
                            *boxes[:, :2].min(0),
                            *boxes[:, 2:].max(0),
                        ]
                        result = refiner.refine(proposal, vessel.axes)
                    except PerformanceBudgetExceeded:
                        deferred.extend(
                            _deferred(r, vessel.axes, "PERFORMANCE_BUDGET_EXCEEDED")
                            for _, r in ready[position:]
                        )
                        raise
                    precision_attempts.append(
                        dict(
                            query_id=query.query_id,
                            bounds_axial=list(query.bounds),
                            result=result,
                        )
                    )
                    if not result.get("boundary_xy"):
                        deferred.append(
                            _deferred(row, vessel.axes, "RAW_PRECISION_UNRESOLVED")
                        )
                        continue
                    supported_sides = final_support_sides(result, config)
                    support = len(supported_sides)
                    result.update(
                        proposal_id=query.query_id,
                        vessel_hypothesis_id=active,
                        proposal_source="MULTI_EXPERT_UNION",
                        candidate_sources=query.sources,
                        source_proposal_ids=[p.proposal_id for p in query.proposals],
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
                                row, vessel.axes, "INSUFFICIENT_FINAL_PHYSICAL_SIDES"
                            )
                        )
                        continue
                    result["status"] = "PARTIAL_HATCH"
                    item["hatches"].append(result)
                    uv = np.asarray(result["boundary_uv"])
                    measured_family.append(
                        dict(
                            bounds=(*uv.min(0), *uv.max(0)),
                            independent_support_sides=support,
                        )
                    )
                    layout.fit(measured_family)
                    for _, pending in ready[position + 1 :]:
                        pending["layout"] = layout.review(pending["bounds_axial"])
                    ready[position + 1 :] = sorted(
                        ready[position + 1 :],
                        key=lambda qr: (
                            qr[1]["layout"]["ranking_penalty"],
                            -sum(
                                s["vertical_face_support"]
                                for s in qr[1]["deck_ring"]["sides"].values()
                            ),
                            -qr[1]["deck_ring"]["positive_height_side_count"],
                            qr[0].query_id,
                        ),
                    )
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
                    state="PROPOSED",
                    precision_allowed=False,
                    reason="PERFORMANCE_BUDGET_EXCEEDED",
                )
                query_rows.append(pending)
                deferred.append(_deferred(pending, vessel.axes))
        if cache is not None:
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
    return dict(
        schema="ship_perception.hatch_once.2",
        pipeline_version="multi-expert-berth-raw-profile.2",
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
        software_source_identity=source_identity(),
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
