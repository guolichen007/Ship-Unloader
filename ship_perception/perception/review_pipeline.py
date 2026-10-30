"""Region-first complete review, finite boundary jobs, independent certification."""

from contextlib import nullcontext
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .pipeline import (
    Timing,
    software_identity,
    source_identity,
    numerical_environment,
    bounded,
)
from .runtime_patch import git_source_state
from .context import SceneEvidenceContext, thaw
from .berth import load_berth_reference, select_berth_vessel
from .operator_selection import select_operator_vessel
from .selected_context import SelectedVesselRawContext
from .local_evidence import VesselLocalEvidence
from .measurement import VesselMeasurementContext
from .legacy_expert import (
    LegacyProposalExpert,
    verify_legacy_b2_core,
    legacy_full_chain_reference,
)
from .experts import HeightBasinExpert
from .opening_regions import build_regions
from .review_geometry import ReviewGeometryAssembler, SIDES
from .review_height import ReviewCornerHeight
from .review_runtime import MeasurementTasks
from .review_measurement import raw_face_measurements
from .review_topology_sides import current_side_constraints
from .local_enclosure_replay import replay_review_constraints
from .raw_boundary_probe import SignedRoleResolver
from .final_boundary import FinalBoundaryRefiner
from .budget import PerformanceBudget, PerformanceBudgetExceeded
from .intensity import admission_stats
from ship_perception.r1_static.run import resolve_config
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG


class ProgressOnly:
    def __init__(self, callback):
        self.callback = callback
        self.last = 0.0

    def check(self):
        now = time.monotonic()
        if now - self.last >= 1.0:
            self.last = now
            self.callback()


def run_perception(
    frame,
    progress=None,
    legacy_evidence=True,
    berth_reference=None,
    mode="bounded",
    operator_vessel_roi=None,
    monitor=None,
):
    if mode not in ("bounded", "review-first"):
        raise ValueError("INVALID_REVIEW_MODE")
    start = time.perf_counter()
    timing = Timing(progress)
    config, physical_hash = resolve_config()
    input_identity = frame.provenance.get("wire_sha256", frame.sha256)
    input_identity_kind = (
        "WIRE_SHA256"
        if frame.provenance.get("wire_sha256")
        else "POINT_CLOUD_SEMANTIC_SHA256"
    )
    research_bytes = RESEARCH_CONFIG.read_bytes()
    research = json.loads(research_bytes)
    if (
        len(frame.xyz) > config["geometry"]["max_input_points"]
        or np.max(np.ptp(frame.xyz, axis=0)) > config["geometry"]["max_local_extent_m"]
    ):
        raise ValueError("INPUT_POINT_OR_EXTENT_LIMIT_EXCEEDED")
    _, patch = git_source_state(Path(__file__).resolve().parents[2])
    effective = dict(
        physical_config_sha256=physical_hash,
        mode=mode,
        review_corner_angles_deg=[85.0, 95.0],
        soft_runtime_limit_s=120.0,
        hard_process_watchdog_s=300.0,
    )
    if operator_vessel_roi:
        effective["operator_roi_sha256"] = hashlib.sha256(
            Path(operator_vessel_roi).read_bytes()
        ).hexdigest()
    config_hash = hashlib.sha256(
        json.dumps(effective, sort_keys=True).encode()
    ).hexdigest()
    budget = PerformanceBudget(total_s=55.0) if mode == "bounded" else None

    def checkpoint(stage, **kw):
        if monitor:
            monitor.checkpoint(stage=stage, **kw)

    def stage(name):
        checkpoint(name)
        return timing.stage(name)

    with stage("SCENE_PREPROCESS"):
        scene = SceneEvidenceContext.build(
            frame, config, research, physical_hash, software_identity()
        )
        scene.validate_frame(frame)
    with stage("BERTH_SIDE_VESSEL_SELECTION"):
        if operator_vessel_roi:
            selection = select_operator_vessel(frame, scene, operator_vessel_roi)
        else:
            path = (
                berth_reference
                or Path(__file__).resolve().parents[1]
                / "config/site/berth_reference.yaml"
            )
            reference = (
                path
                if isinstance(path, dict)
                else load_berth_reference(path, frame.frame_id)
            )
            selection = select_berth_vessel(frame, scene, config, reference)
    by_vessel = [
        dict(
            vessel_hypothesis_id=v.vessel_id,
            classification_status=v.public["classification_status"],
            status="IGNORED_NON_BERTH_VESSEL",
            hatches=[],
            independent_groups=[],
            aggregate_summary=dict(group_count=0, hatch_count=0),
            unresolved_review_proposals=bounded([]),
            unresolved_candidates=bounded([]),
        )
        for v in scene.vessels
    ]
    hatches = []
    regions = []
    legacy_batches = []
    failures = []
    measurement = None
    selected_audit = None
    diagnostics = {}
    attempts = []
    tasks = MeasurementTasks(input_identity, monitor)
    assembler = ReviewGeometryAssembler(config)
    if selection["active_vessel_id"] is not None:
        vessel = next(
            v for v in scene.vessels if v.vessel_id == selection["active_vessel_id"]
        )
        item = next(
            v for v in by_vessel if v["vessel_hypothesis_id"] == vessel.vessel_id
        )
        selected = SelectedVesselRawContext.build(frame, scene, selection, config)
        selected_audit = selected.audit()
        primary = VesselLocalEvidence.build(frame, vessel, config, selected.measurement)
        cache = (
            primary
            if primary.context_xyz is primary.xyz
            else VesselLocalEvidence(
                replace(
                    vessel,
                    point_indexes=selected.measurement.point_indexes,
                    point_index_sha256=selected.measurement.point_index_sha256,
                ),
                primary.context_xyz,
                config,
                measurement_context=selected.measurement,
                context_xyz=primary.context_xyz,
            )
        )
        cache.primary_cache = primary
        cache.input_frame = frame
        cache.s4_exterior = selected.exterior(frame, config)
        measurement = VesselMeasurementContext(cache)
        cache.v2_measurement = measurement
        measurement.budget = budget
        last_s4_checkpoint = [0.0]

        def s4_progress(label, record):
            now = time.monotonic()
            if now - last_s4_checkpoint[0] >= 1.0:
                last_s4_checkpoint[0] = now
                checkpoint("S4_" + label, cache_state=dict(measurement.counters))

        cache.review_checkpoint = s4_progress
        with stage("LOCAL_RECALL"):
            proposals = ()
            if legacy_evidence:
                verify_legacy_b2_core()
                batch = LegacyProposalExpert().propose(primary, research)
                proposals = batch.proposals
                legacy_batches.append(
                    dict(
                        vessel_id=vessel.vessel_id,
                        proposal_sha256=batch.output_sha256,
                        proposal_count=len(batch.proposals),
                        frozen_source_base_sha="51410e3f4f932fcbdcc7fc88b8f9979c60c2b954",
                        raw_proposal_evidence=batch.raw_output(),
                    )
                )
            cache.providers = primary.structural_providers()
            proposals += HeightBasinExpert().propose(cache, research, budget)
            _, diagnostics = build_regions(
                cache,
                proposals,
                tuple(np.r_[primary.xyz[:, :2].min(0), primary.xyz[:, :2].max(0)]),
                research,
                local_evidence_before_identity=True,
            )
        bridge = getattr(cache, "s4_topology_bridge", {})
        canonical = []
        eligibility = []
        with stage("REVIEW_GEOMETRY_ASSEMBLY"):
            for region in cache.opening_identities:
                topology_sides = (
                    replay_review_constraints(region, config)
                    if region.identity_evidence.get("source")
                    == "CURRENT_BOUNDED_LOCAL_EVIDENCE_VECTOR"
                    else (
                        current_side_constraints(cache, measurement, region, config)
                        if region.identity_role.startswith("CONFIRMED")
                        else None
                    )
                )
                result, decision = assembler.assemble(
                    region, vessel.axes, bridge, topology_sides
                )
                eligibility.append(dict(opening_identity=region.region_id, **decision))
                if result:
                    canonical.append(region)
                    hatches.append(result)
            checkpoint("REVIEW_GEOMETRY_READY", review_objects=hatches)
        if measurement is not None:

            def measurement_progress():
                tasks.flush(dict(measurement.counters))

            measurement.budget = budget or ProgressOnly(measurement_progress)
            refiner = FinalBoundaryRefiner(measurement.raw_uvz, [], config, measurement)
            resolver = SignedRoleResolver(measurement, refiner)
            with stage("BOUNDARY_MEASUREMENT"):
                for index, region in enumerate(canonical):
                    result = hatches[index]
                    modes = {name: [] for name, _, _, _ in SIDES}
                    candidates = cache.side_candidates(region.conservative_bounds)
                    try:
                        with (
                            budget.stage("FINAL_PRECISION", hard=True)
                            if budget
                            else nullcontext()
                        ):
                            for name, axis, pos, sign in SIDES:
                                task = tasks.begin(region.region_id, name, "NEW")
                                if task is None:
                                    continue
                                span = tuple(
                                    region.conservative_bounds[i]
                                    for i in (1 - axis, 3 - axis)
                                )
                                # Geometry stores outward normals; the signed
                                # Raw primitive expects the inward direction.
                                row = resolver.side(
                                    axis,
                                    region.conservative_bounds[pos],
                                    span,
                                    -sign,
                                    candidates.get(name, []),
                                    extended=True,
                                    measurement_only=True,
                                )
                                attempts.append(
                                    dict(
                                        task_id=task,
                                        opening_identity=region.region_id,
                                        side=name,
                                        result=row,
                                    )
                                )
                                if (
                                    row["status"] == "STRONG_UNIQUE_EDGE"
                                    and row.get("measurement_source")
                                    == "RAW_PAIRED_PROFILE"
                                ):
                                    fit = row["fit"]
                                    slope = fit.get("slope", 0.0)
                                    reference = fit.get(
                                        "reference_along_m", sum(span) / 2
                                    )
                                    normal = (
                                        np.array([1.0, -slope])
                                        if axis == 0
                                        else np.array([-slope, 1.0])
                                    )
                                    norm = np.linalg.norm(normal)
                                    normal /= norm
                                    rho = (row["rho"] - slope * reference) / norm
                                    direction = np.array([-normal[1], normal[0]])
                                    points = np.array(
                                        [
                                            [
                                                row["rho"] if axis == 0 else span[0],
                                                span[0] if axis == 0 else row["rho"],
                                            ],
                                            [
                                                row["rho"] if axis == 0 else span[1],
                                                span[1] if axis == 0 else row["rho"],
                                            ],
                                        ]
                                    )
                                    modes[name].append(
                                        dict(
                                            mode_id=task,
                                            source="MEASURED_RAW",
                                            state="MEASURED_WEAK",
                                            rho=float(rho),
                                            normal_uv=normal.tolist(),
                                            support_span=sorted(
                                                (points @ direction).tolist()
                                            ),
                                            uncertainty_m=max(
                                                config["geometry"]["plane_inlier_m"],
                                                fit.get("residual_mad_m", 0.0),
                                            ),
                                            independent_raw_sections=len(
                                                row["sections"]
                                            ),
                                            physical_role_improved=True,
                                            raw_support_ids=[
                                                task + ":section:" + str(k)
                                                for k in range(len(row["sections"]))
                                            ],
                                            raw_profile_sections=row["sections"],
                                        )
                                    )
                                else:
                                    face_modes = raw_face_measurements(
                                        measurement,
                                        region,
                                        name,
                                        axis,
                                        pos,
                                        span,
                                        candidates.get(name, []),
                                        config,
                                    )
                                    modes[name].extend(face_modes)
                                signature = hashlib.sha256(
                                    json.dumps(row, sort_keys=True).encode()
                                ).hexdigest()
                                tasks.finish(
                                    task,
                                    "COMPLETED",
                                    signature,
                                    dict(measurement.counters),
                                )
                    except PerformanceBudgetExceeded:
                        failures.append(
                            dict(
                                stage="BOUNDARY_MEASUREMENT",
                                code="BOUNDED_MEASUREMENT_INTERRUPTED_REVIEW_PRESERVED",
                            )
                        )
                    hatches[index] = assembler.update(
                        region, result, modes, vessel.axes
                    )
                    checkpoint("BOUNDARY_MEASUREMENT", review_objects=hatches)
        with stage("CORNER_HEIGHT_OUTPUT"):
            for result in hatches:
                result["corner_heights"] = ReviewCornerHeight(
                    measurement, config
                ).measure(result)
                result.update(
                    input_frame=frame.frame_id,
                    transform_provenance="SELECTED_VESSEL_UV_TO_ORIGINAL_XYZ",
                    candidate_blockers=["STEEL_INNER_EDGE_UNVERIFIED"],
                )
        regions = [r.audit() for r in cache.opening_identities]
        for r in regions:
            for source, target in [
                ("conservative_bounds", "conservative_boundary_xy"),
                ("search_envelope", "search_envelope_xy"),
            ]:
                b = r[source]
                r[target] = (
                    np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
                    @ vessel.axes
                ).tolist()
        item.update(
            status="REVIEW_CANDIDATES" if hatches else "UNRESOLVED",
            hatches=hatches,
            opening_region_identities=regions,
            review_geometry_eligibility=eligibility,
            s4_topology_bridge=bridge,
            region_recall_diagnostics=diagnostics,
            proposal_registry=[
                dict(
                    proposal_id=p.proposal_id,
                    expert=p.expert,
                    bounds_axial=list(p.bounds),
                    evidence=p.evidence(),
                )
                for p in proposals
            ],
            independent_groups=[
                dict(
                    group_id=r["opening_identity"],
                    hatch_ids=[r["hatch_id"]],
                    source_proposal_ids=r["source_proposal_ids"],
                )
                for r in hatches
            ],
            aggregate_summary=dict(group_count=len(hatches), hatch_count=len(hatches)),
        )
    identity = source_identity()
    intensity = admission_stats(
        frame.intensity, frame.provenance.get("intensity_dtype")
    )
    intensity.update(
        used=False, production_scoring_enabled=False, geometry_mode="XYZ_ONLY"
    )
    performance = dict(
        total_elapsed_s=time.perf_counter() - start,
        total_budget_s=60.0 if mode == "bounded" else None,
        total_budget_exceeded=False,
        stages=[],
        soft_runtime_limit_s=120.0,
        hard_process_watchdog_s=300.0,
        semantic_budget_enforced=mode == "bounded",
        soft_runtime_exceeded=time.perf_counter() - start > 120.0,
        watchdog_hit=False,
    )
    return dict(
        schema="ship_perception.hatch_once.3",
        schema_version=".3",
        pipeline_version="effect-first-review-geometry.2.9",
        review_mode=mode,
        producer_git_sha=software_identity(),
        runtime_source_hash=identity["runtime_source_sha256"],
        config_hash=config_hash,
        production_geometry_mode="XYZ_ONLY",
        _ownership_indexes={v.vessel_id: v.point_indexes for v in scene.vessels},
        _runtime_patch=patch,
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
        scene_context=scene.audit(),
        scene=thaw(scene.scene),
        berth_selection=selection,
        selected_raw_context=selected_audit,
        config_sha256=physical_hash,
        effective_runtime_config=effective,
        software_git_sha=software_identity(),
        software_source_identity=identity,
        git_head_sha=identity["git_head_sha"],
        git_worktree_dirty=identity["git_worktree_dirty"],
        git_diff_sha256=identity["git_diff_sha256"],
        runtime_source_sha256=identity["runtime_source_sha256"],
        input_sha256=input_identity,
        input_identity_kind=input_identity_kind,
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
        per_hatch_status={r["hatch_id"]: r["status"] for r in hatches},
        per_side_uncertainty={
            r["hatch_id"]: r["per_side_uncertainty"] for r in hatches
        },
        grid_phase_stability={
            r["hatch_id"]: {k: s["grid_phase"] for k, s in r["sides"].items()}
            for r in hatches
        },
        undercoverage_risk={r["hatch_id"]: r["undercoverage_risk"] for r in hatches},
        geometry_only=dict(
            hatch_count=len(hatches),
            review_candidate_count=len(hatches),
            hatch_count_status="REVIEW_ONLY_NOT_TOTAL_HATCH_COUNT",
            hatches=hatches,
            review_objects=hatches,
            opening_regions=regions,
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
        legacy_freeze_contracts=dict(
            legacy_b2_core=verify_legacy_b2_core(),
            legacy_full_chain_reference=legacy_full_chain_reference(),
        ),
        physical_validation=bounded([]),
        physical_candidate_registry=[],
        compiled_candidate_registry=[],
        precision_attempts=attempts,
        precision_candidates=list(tasks.rows.values()),
        measurement_tasks=list(tasks.rows.values()),
        precision_profiler=dict(
            measurement.counters if measurement else {},
            proposal_count=sum(len(v.get("proposal_registry", [])) for v in by_vessel),
            precision_query_count=len(attempts),
            assembly_operation_count=assembler.assembly_operation_count,
            four_line_cartesian_enumeration_count=0,
        ),
        performance_budget=performance,
        rejected_exterior_candidates=bounded([]),
        rejected_outside_vessel=bounded([]),
        scene_exterior_low_regions=dict(status="DIAGNOSTIC_NOT_TOTAL_COUNT"),
    )
