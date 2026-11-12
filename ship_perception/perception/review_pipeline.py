"""Region-first complete review, finite boundary jobs, independent certification."""

from contextlib import nullcontext, contextmanager
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
from .discovery_frame import conditioned_discovery
from .berth import load_berth_reference, select_berth_vessel
from .operator_selection import select_operator_vessel
from .selected_context import SelectedVesselRawContext
from .local_evidence import VesselLocalEvidence
from .measurement import VesselMeasurementContext, replicated_corner_view
from .legacy_expert import (
    LegacyProposalExpert,
    verify_legacy_b2_core,
    legacy_full_chain_reference,
)
from .experts import HeightBasinExpert
from .opening_regions import build_regions
from .opening_cells import opening_cells, coarse_review
from .local_rim_evidence import (
    LocalRimEvidence,
    shadow_rectangle_ab,
    shadow_review_only_cell,
)
from .vessel_rim_recovery import recover_vessel_sides
from .review_geometry_recovery import recover_review_geometries
from .recall_snapshot import RecallSnapshot
from .runtime_metrics import peak_rss_bytes
from .review_geometry import ReviewGeometryAssembler, SIDES
from .constrained_rectangle import RectangleInfeasible, constrain_review_geometry
from .review_height import ReviewCornerHeight
from .review_runtime import MeasurementTasks
from .opening_boundary_assignment import OpeningBoundaryAssignment
from .review_measurement import (
    raw_face_measurements,
    qualified_raw_pair_gain,
    raw_profile_line,
)
from .review_topology_sides import current_side_constraints
from .local_enclosure_replay import replay_review_constraints
from .review_side_capability import compile_constraints
from .raw_boundary_probe import SignedRoleResolver
from .final_boundary import FinalBoundaryRefiner
from .budget import PerformanceBudget, PerformanceBudgetExceeded
from .intensity import admission_stats
from ship_perception.r1_static.run import resolve_config
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG


class ProgressOnly:
    def __init__(self, callback, deadline=None):
        self.callback = callback
        self.deadline = deadline
        self.last = 0.0

    def check(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise PerformanceBudgetExceeded("PRECISION_DEFERRED_FOR_REVIEW_OUTPUT")
        now = time.monotonic()
        if now - self.last >= 1.0:
            self.last = now
            self.callback()


def review_precision_deadline(
    start, watchdog_s, output_reserve_s, process_deadline=None
):
    deadline = start + watchdog_s
    if process_deadline is not None:
        if not np.isfinite(process_deadline):
            raise ValueError("NONFINITE_REVIEW_PROCESS_DEADLINE")
        deadline = min(deadline, process_deadline)
    return deadline - output_reserve_s


def run_perception(
    frame,
    progress=None,
    legacy_evidence=True,
    berth_reference=None,
    mode="bounded",
    operator_vessel_roi=None,
    monitor=None,
    opening_topology_v2=False,
    opening_preview=False,
    local_rim_review=False,
    review_snapshot=None,
    review_process_deadline=None,
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
        final_review_model="FIVE_DOF_ORTHOGONAL_RECTANGLE",
        rectangle_source_rank_weights=[1.0, 0.5, 0.25, 0.125, 0.0625],
        rectangle_orientation_update_policy="PERSISTENT_RAW_LONG_SIDE_OR_OPPOSED_RAW_ENDS_ELSE_RETAIN_RAW_COMPATIBLE_CONSERVATIVE_DIRECTION",
        rectangle_conservative_bounds_role="ONE_SIDED_COVERAGE_ONLY",
        rectangle_numerical_orthogonality_abs_dot=1e-10,
        rectangle_numerical_parallel_error_rad=1e-10,
        rectangle_numerical_scaled_length_tolerance=1e-8,
        soft_runtime_limit_s=120.0,
        hard_process_watchdog_s=300.0,
        discovery_lattice="CURRENT_RAW_RIGID_FRAME_V1",
        provider_side_positions="EXACT_OBSERVED_FLOAT_NO_ABSOLUTE_ROUNDING",
    )
    if operator_vessel_roi:
        effective["operator_roi_sha256"] = hashlib.sha256(
            Path(operator_vessel_roi).read_bytes()
        ).hexdigest()
    opening_topology_v2 = opening_topology_v2 or opening_preview or local_rim_review
    if local_rim_review:
        effective["local_rim_review"] = "CELL_LOCAL_RAW_MULTISURFACE_V2"
        effective["long_side_recovery"] = "SAME_VESSEL_OWN_INTERIOR_RAW_V1"
        effective["review_output_reserve_s"] = 60.0
    if opening_topology_v2:
        effective["opening_topology_v2"] = True
        effective["opening_review_level"] = (
            "FAST_OPENING_REVIEW" if opening_preview else "PRECISION_RECTANGLE_REVIEW"
        )
    config_hash = hashlib.sha256(
        json.dumps(effective, sort_keys=True).encode()
    ).hexdigest()
    budget = PerformanceBudget(total_s=55.0) if mode == "bounded" else None
    precision_deadline = (
        review_precision_deadline(
            start,
            effective["hard_process_watchdog_s"],
            effective["review_output_reserve_s"],
            review_process_deadline,
        )
        if local_rim_review
        else None
    )

    def checkpoint(stage, **kw):
        if monitor:
            monitor.checkpoint(stage=stage, **kw)

    stage_resources = {}

    @contextmanager
    def stage(name):
        checkpoint(name)
        started_cpu = time.process_time()
        started_wall = time.perf_counter()
        try:
            with timing.stage(name):
                yield
        finally:
            stage_resources[name] = dict(
                wall_s=time.perf_counter() - started_wall,
                cpu_s=time.process_time() - started_cpu,
                process_peak_rss_bytes=peak_rss_bytes(),
            )

    reference_key = berth_reference
    if not isinstance(reference_key, dict):
        reference_path = (
            Path(reference_key)
            if reference_key
            else (
                Path(__file__).resolve().parents[1] / "config/site/berth_reference.yaml"
            )
        )
        reference_key = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    with stage("RECALL_SNAPSHOT_VALIDATE_AND_LOAD"):
        snapshot = RecallSnapshot(
            review_snapshot,
            dict(
                version="RECALL_SNAPSHOT_V1_B1_REVIEW_UV",
                input_sha256=input_identity,
                semantic_sha256=frame.sha256,
                frame_id=frame.frame_id,
                provenance=frame.provenance_json,
                physical_config_sha256=physical_hash,
                research_sha256=hashlib.sha256(research_bytes).hexdigest(),
                runtime_source_sha256=source_identity()["runtime_source_sha256"],
                berth_reference=reference_key,
                operator_roi=effective.get("operator_roi_sha256"),
                legacy_evidence=legacy_evidence,
            ),
        )
    with stage("SCENE_PREPROCESS"):
        scene = (
            snapshot.value["scene"]
            if snapshot.value
            else SceneEvidenceContext.build(
                frame,
                config,
                research,
                physical_hash,
                software_identity(),
                discovery=conditioned_discovery,
            )
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
    unresolved_rectangles = []
    selected_cells, cell_relations, coarse_boxes = [], [], []
    regions = []
    legacy_batches = []
    failures = []
    measurement = None
    precision_deferred = False

    def honor_precision_deadline(stage_name):
        nonlocal precision_deferred
        if (
            local_rim_review
            and not opening_preview
            and not precision_deferred
            and time.perf_counter() >= precision_deadline
        ):
            precision_deferred = True
            failures.append(
                dict(
                    stage=stage_name,
                    code="PRECISION_DEFERRED_FOR_REVIEW_OUTPUT",
                    budget_scope=(
                        "SHARED_PROCESS"
                        if review_process_deadline is not None
                        else "SINGLE_EXECUTE"
                    ),
                )
            )

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
        primary.exact_side_positions = True
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
        cache.exact_side_positions = True
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
        if snapshot.value:
            with stage("RECALL_SNAPSHOT_RESTORE"):
                state = snapshot.value
                RecallSnapshot.restore_cache(primary, state["primary"])
                RecallSnapshot.restore_cache(cache, state["cache"])
                proposals = state["proposals"]
                diagnostics = state["diagnostics"]
                legacy_batches = state["legacy_batches"]
        else:
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
            with stage("RECALL_SNAPSHOT_SAVE"):
                snapshot.save(
                    dict(
                        scene=scene,
                        primary=RecallSnapshot.cache_state(primary),
                        cache=RecallSnapshot.cache_state(cache),
                        proposals=proposals,
                        diagnostics=diagnostics,
                        legacy_batches=legacy_batches,
                    )
                )
        bridge = getattr(cache, "s4_topology_bridge", {})
        canonical = []
        eligibility = []
        with stage("REVIEW_GEOMETRY_ASSEMBLY"):
            initial_constraints = {
                r.region_id: (
                    replay_review_constraints(r, config)
                    if r.identity_evidence.get("source")
                    == "CURRENT_BOUNDED_LOCAL_EVIDENCE_VECTOR"
                    else current_side_constraints(cache, measurement, r, config)
                )
                for r in cache.opening_identities
                if not opening_preview and r.identity_role.startswith("CONFIRMED")
            }
            compiled_constraints = (
                {}
                if opening_preview
                else compile_constraints(
                    cache, cache.opening_identities, config, initial_constraints
                )
            )
            for region in cache.opening_identities:
                topology_sides = compiled_constraints.get(region.region_id)
                result, decision = assembler.assemble(
                    region, vessel.axes, bridge, topology_sides
                )
                eligibility.append(dict(opening_identity=region.region_id, **decision))
                if result:
                    canonical.append(region)
                    hatches.append(result)
            checkpoint("REVIEW_GEOMETRY_READY", review_objects=hatches)
        if opening_topology_v2:
            with stage("OPENING_CELL_OWNERSHIP"):
                selected_cells, cell_relations = opening_cells(
                    canonical,
                    bridge,
                    getattr(cache, "local_enclosure_evidence", ()),
                    config["geometry"]["coarse_voxel_m"],
                )
                precision_ids = {
                    cell["primary_region_id"]
                    for cell in selected_cells
                    if not opening_preview and not cell["expanded_review_only"]
                }
                paired = [
                    (region, hatch)
                    for region, hatch in zip(canonical, hatches)
                    if region.region_id in precision_ids
                ]
                canonical = [region for region, _ in paired]
                hatches = [hatch for _, hatch in paired]
        if measurement is not None and (canonical or not opening_topology_v2):

            def measurement_progress():
                tasks.flush(dict(measurement.counters))

            measurement.budget = budget or ProgressOnly(
                measurement_progress, precision_deadline
            )
            refiner = FinalBoundaryRefiner(measurement.raw_uvz, [], config, measurement)
            resolver = SignedRoleResolver(measurement, refiner)
            assignment = OpeningBoundaryAssignment(measurement, config)
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
                                assigned_profile = False
                                if (
                                    row["status"] == "STRONG_UNIQUE_EDGE"
                                    and row.get("measurement_source")
                                    == "RAW_PAIRED_PROFILE"
                                ):
                                    fit = row["fit"]
                                    line = raw_profile_line(row, axis, span)
                                    ownership = assignment.profile(region, row)
                                    row["opening_assignment"] = ownership
                                    assigned_profile = (
                                        ownership["state"] == "ASSIGNED_CURRENT_OPENING"
                                    )
                                    modes[name].append(
                                        dict(
                                            mode_id=task,
                                            source="MEASURED_RAW",
                                            state="MEASURED_WEAK",
                                            **line,
                                            uncertainty_m=max(
                                                config["geometry"]["plane_inlier_m"],
                                                fit.get("residual_mad_m", 0.0),
                                            ),
                                            independent_raw_sections=ownership[
                                                "supported_section_count"
                                            ],
                                            geometric_raw_section_count=len(
                                                row["sections"]
                                            ),
                                            physical_role_improved=True,
                                            independent_opening_role_gain=qualified_raw_pair_gain(
                                                row, config
                                            )
                                            and assigned_profile,
                                            physical_role_gain_basis="SIGNED_RAW_PAIRS_WITH_CURRENT_OPENING_MATERIAL_ASSIGNMENT",
                                            opening_assignment_state=ownership["state"],
                                            opening_assignment=ownership,
                                            ownership_rejected=not assigned_profile,
                                            raw_support_ids=[
                                                task + ":section:" + str(k)
                                                for k in range(len(row["sections"]))
                                            ],
                                            raw_profile_sections=row["sections"],
                                        )
                                    )
                                if not assigned_profile:
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
                                    if not any(
                                        m.get("independent_opening_role_gain")
                                        for m in face_modes
                                    ):
                                        complete_modes, face_audit = (
                                            assignment.complete_faces(
                                                region, name, axis, pos, span
                                            )
                                        )
                                        face_modes = (
                                            assignment.corroborate_primary_faces(
                                                face_modes, complete_modes
                                            )
                                        )
                                        row["opening_face_investigation"] = face_audit
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
                    except PerformanceBudgetExceeded as error:
                        precision_deferred = (
                            local_rim_review
                            and str(error) == "PRECISION_DEFERRED_FOR_REVIEW_OUTPUT"
                        )
                        for pending in list(tasks.rows.values()):
                            if pending["status"] == "RUNNING":
                                tasks.finish(
                                    pending["task_id"],
                                    "DEFERRED_PERFORMANCE",
                                    pending["progress_signature"],
                                    dict(measurement.counters),
                                )
                        failures.append(
                            dict(
                                stage="BOUNDARY_MEASUREMENT",
                                code=(
                                    "PRECISION_DEFERRED_FOR_REVIEW_OUTPUT"
                                    if precision_deferred
                                    else "BOUNDED_MEASUREMENT_INTERRUPTED_REVIEW_PRESERVED"
                                ),
                            )
                        )
                    hatches[index] = assembler.update(
                        region, result, modes, vessel.axes
                    )
                    checkpoint("BOUNDARY_MEASUREMENT", review_objects=hatches)
        local_rim_by_region = {}
        honor_precision_deadline("LOCAL_RIM_NOMINATION")
        if local_rim_review and not opening_preview and not precision_deferred:
            with stage("LOCAL_RIM_NOMINATION"):
                local_rim = LocalRimEvidence(measurement, config)
                region_by_id = {r.region_id: r for r in cache.opening_identities}
                candidate_by_id = {h["opening_identity"]: h for h in hatches}
                for cell in selected_cells:
                    key = cell["primary_region_id"]
                    candidate = candidate_by_id.get(key, {})
                    xy = candidate.get("boundary_xy")
                    if xy is not None and np.asarray(xy).shape == (4, 2):
                        polygon_uv = np.asarray(xy) @ vessel.axes.T
                    else:
                        u0, v0, u1, v1 = cell["bounds_uv"]
                        polygon_uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
                    region = region_by_id.get(key)
                    members, other = (
                        assignment.masks(region) if region is not None else (None, None)
                    )
                    evidence = {
                        name: local_rim.side(polygon_uv, name, members, other)
                        for name in ("U0", "U1", "V0", "V1")
                    }
                    cell["local_rim_evidence"] = evidence
                    local_rim_by_region[key] = evidence
        with stage("CONSTRAINED_REVIEW_RECTANGLE"):
            strict_hatches = []
            cell_by_region = {c["primary_region_id"]: c for c in selected_cells}
            for region, candidate in zip(canonical, hatches):
                try:
                    strict_hatches.append(
                        shadow_rectangle_ab(
                            region,
                            candidate,
                            local_rim_by_region.get(region.region_id, {}),
                            measurement,
                            assembler,
                            config,
                            bridge,
                            opening_cell=cell_by_region.get(region.region_id),
                        )
                        if local_rim_review and not precision_deferred
                        else constrain_review_geometry(
                            region, candidate, config, bridge
                        )
                    )
                except RectangleInfeasible as error:
                    failures.append(
                        dict(
                            stage="CONSTRAINED_REVIEW_RECTANGLE",
                            code=str(error),
                            opening_identity=region.region_id,
                        )
                    )
                    unresolved_rectangles.append(
                        dict(
                            opening_identity=region.region_id,
                            status="UNRESOLVED_REVIEW_CANDIDATE",
                            reason=str(error),
                            audit=error.audit,
                            pre_solve_candidate=candidate,
                            steel_certified=False,
                            certification_status="NOT_CERTIFIED",
                            corner_z_m=[None] * 4,
                        )
                    )
            if local_rim_review and not opening_preview and not precision_deferred:
                precision_keys = {r.region_id for r in canonical}
                for cell in selected_cells:
                    key = cell["primary_region_id"]
                    if key in precision_keys or key not in region_by_id:
                        continue
                    promoted = shadow_review_only_cell(
                        region_by_id[key],
                        cell,
                        local_rim_by_region.get(key, {}),
                        region_by_id,
                        measurement,
                        assembler,
                        config,
                        bridge,
                    )
                    if promoted is not None:
                        strict_hatches.append(promoted)
            hatches = strict_hatches
            checkpoint("RECTANGLE_REVIEW_READY", review_objects=hatches)
        honor_precision_deadline("SAME_VESSEL_RAW_SIDE_RECOVERY")
        if local_rim_review and not opening_preview and not precision_deferred:
            with stage("SAME_VESSEL_RAW_SIDE_RECOVERY"):
                hatches, width_audits = recover_vessel_sides(
                    selected_cells,
                    hatches,
                    candidate_by_id,
                    region_by_id,
                    local_rim,
                    assembler,
                    config,
                    bridge,
                )
                item["same_vessel_width_consistency"] = width_audits
                checkpoint("RAW_SIDE_RECOVERY_READY", review_objects=hatches)
        with stage("CORNER_HEIGHT_OUTPUT"):
            replication = (
                replicated_corner_view(measurement)
                if hatches and not precision_deferred
                else None
            )
            if measurement is not None and precision_deferred:
                measurement.budget = ProgressOnly(measurement_progress)
            for result in hatches:
                plane_sources = getattr(cache, "review_rim_plane_sources", {}).get(
                    result["opening_identity"], ()
                )
                use_view = replication is not None and not plane_sources
                height_context = replication[0] if use_view else measurement
                height_estimator = ReviewCornerHeight(height_context, config)
                if precision_deferred:
                    result["corner_heights"] = [
                        dict(
                            height_estimator._empty(),
                            z_probe_attempted=False,
                            reason="DEFERRED_PERFORMANCE",
                        )
                        for _ in range(4)
                    ]
                    height_estimator.audit = dict(status="DEFERRED_PERFORMANCE")
                else:
                    result["corner_heights"], height_deferred = (
                        height_estimator.measure_for_review(
                            result, plane_sources, allow_defer=local_rim_review
                        )
                    )
                    if height_deferred:
                        precision_deferred = True
                        failures.append(
                            dict(
                                stage="CORNER_HEIGHT_OUTPUT",
                                code="PRECISION_DEFERRED_FOR_REVIEW_OUTPUT",
                                opening_identity=result["opening_identity"],
                            )
                        )
                if use_view:
                    _, groups, replication_audit = replication
                    for corner in result["corner_heights"]:
                        sources = corner["source_point_indexes"]
                        if sources:
                            rows = np.isin(groups[:, 0], sources, assume_unique=True)
                            corner["exact_duplicate_raw_source_groups"] = groups[
                                rows
                            ].tolist()
                    height_estimator.audit["exact_replication"] = replication_audit
                elif replication is not None:
                    height_estimator.audit["exact_replication"] = dict(
                        replication[2], status="UPSTREAM_RIM_SOURCE_CONTRACT_RETAINED"
                    )
                result["z_measurement_audit"] = height_estimator.audit
                result.update(
                    input_frame=frame.frame_id,
                    transform_provenance="SELECTED_VESSEL_UV_TO_ORIGINAL_XYZ",
                    candidate_blockers=["STEEL_INNER_EDGE_UNVERIFIED"],
                )
        if opening_topology_v2:
            strict_by_id = {h["opening_identity"]: h for h in hatches}
            infeasible = {r["opening_identity"]: r for r in unresolved_rectangles}
            for cell in selected_cells:
                key = cell["primary_region_id"]
                if key in strict_by_id:
                    strict = strict_by_id[key]
                    cell.update(
                        strict_rectangle_state="FEASIBLE",
                        geometry=dict(
                            opening_identity=key,
                            hatch_id=strict["hatch_id"],
                            boundary_xy=strict["boundary_xy"],
                            corners_xyz=[
                                [x, y, height.get("z_m")]
                                for (x, y), height in zip(
                                    strict["boundary_xy"], strict["corner_heights"]
                                )
                            ],
                            geometry_role=strict["geometry_role"],
                            steel_certified=False,
                        ),
                    )
                    continue
                state = "INFEASIBLE" if key in infeasible else "NOT_RUN"
                review = coarse_review(cell, vessel.axes, measurement, state)
                cell.update(strict_rectangle_state=state, geometry=review)
                if review is not None:
                    coarse_boxes.append(review)
        honor_precision_deadline("EVIDENCE_GRADED_REVIEW_GEOMETRY")
        if local_rim_review and not opening_preview and not precision_deferred:
            with stage("EVIDENCE_GRADED_REVIEW_GEOMETRY"):
                recover_review_geometries(
                    selected_cells, region_by_id, measurement, config, local_rim
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
    cell_output = {}
    if opening_topology_v2:
        by_opening = {row["opening_identity"]: row for row in hatches + coarse_boxes}
        visible = [
            by_opening[c["primary_region_id"]]
            for c in selected_cells
            if c["primary_region_id"] in by_opening
        ]
        cell_output = dict(
            opening_review_level=effective["opening_review_level"],
            selected_opening_cells=selected_cells,
            opening_cell_relations=cell_relations,
            review_opening_count=len(visible),
            confirmed_opening_count=sum(
                cell["identity_state"] == "SUPPORTED" for cell in selected_cells
            ),
            strict_rectangle_count=len(hatches),
            coarse_opening_reviews=coarse_boxes,
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
        stage_resources=stage_resources,
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
            review_objects=visible if opening_topology_v2 else hatches,
            **cell_output,
            unresolved_rectangle_candidates=unresolved_rectangles,
            opening_regions=regions,
            intensity_used=False,
        ),
        recall_snapshot=snapshot.audit,
        pipeline_health=dict(
            precision_budget_scope=(
                "SHARED_PROCESS"
                if review_process_deadline is not None
                else "SINGLE_EXECUTE"
            ),
            precision_state=(
                "DEFERRED_PERFORMANCE" if precision_deferred else "COMPLETED"
            ),
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
