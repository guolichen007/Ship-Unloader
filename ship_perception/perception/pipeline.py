"""Transport-neutral recall, exterior validation and four-side precision."""

from contextlib import contextmanager
import hashlib
import json
import os
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import subprocess
import time

import numpy as np

from ship_perception.r1_static.run import DEFAULT_CONFIG, resolve_config
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG
from ship_perception.r1_static.pcd_showcase import prepare_geometry
from ship_perception.r1_static.corner_topology_review import (
    review_points,
    _separator_split,
)
from ship_perception.r1_static.height_grid import R1HeightGrid
from .context import SceneEvidenceContext, thaw
from .local_evidence import VesselLocalEvidence
from .recall import propose_openings, validate_and_select, box_overlap
from .validation import ExteriorTopology, vessel_containment, partial_context_review
from .final_boundary import FinalBoundaryRefiner, polygon
from .intensity import admission_stats
from ship_perception.r1_static.intensity_evidence import AUX_WEIGHT


def software_identity():
    package = Path(__file__).resolve().parents[1]
    root = package.parent
    if (root / ".git").exists():
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    metadata = package / "software_identity.json"
    return (
        json.loads(metadata.read_text())["software_git_sha"]
        if metadata.exists()
        else "UNKNOWN"
    )


def source_identity():
    """Fingerprint executed Python/config sources, including uncommitted runs."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = [
        p
        for folder in ("perception", "io", "r1_static", "ros2", "tools")
        for p in (package / folder).rglob("*.py")
    ] + [
        p
        for p in (package / "config").rglob("*")
        if p.suffix in (".json", ".yaml", ".yml")
    ]
    for path in sorted(paths):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    dirty = None
    if (package.parent / ".git").exists():
        dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=normal",
                    "--",
                    "ship_perception",
                    "scripts",
                    "requirements-ros.txt",
                ],
                cwd=str(package.parent),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    return dict(runtime_source_sha256=digest.hexdigest(), git_worktree_dirty=dirty)


def numerical_environment():
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "jsonschema"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "UNKNOWN"
    return dict(
        python=sys.version.split()[0],
        platform=sys.platform,
        packages=packages,
        numerical_threads={
            key: os.getenv(key)
            for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
    )


def bounded(rows, limit=64):
    """Only rejected/search samples are truncated; selected evidence is retained."""
    data = json.dumps(rows, sort_keys=True, allow_nan=False, default=_json_default)
    return dict(
        total_count=len(rows),
        samples=rows[:limit],
        omitted_count=max(0, len(rows) - limit),
        all_records_sha256=hashlib.sha256(data.encode()).hexdigest(),
    )


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def precision_suppression(rows, vessel_axes, config):
    """Compare physical side support after precision, never coarse box size."""
    from .opening_partitions import explained_envelopes

    rows, explained = explained_envelopes(rows, vessel_axes, config)

    def quality(row):
        sides = list(row["sides"].values())
        required = config["boundary"]["profile_min_sections"]
        signed = sum(
            s["positive_height_section_count"] >= required
            and s["raw_subvoxel_section_count"] >= required
            for s in sides
        )
        return (
            signed,
            min(s["support_section_ratio"] for s in sides),
            min(
                s["positive_height_section_count"]
                / max(1, s["phase_runs"][0]["physical_section_count"])
                for s in sides
            ),
            not row["undercoverage_risk"],
        )

    selected, suppressed = [], list(explained)

    def bounds(row):
        uv = np.asarray(row["boundary_xy"]) @ vessel_axes.T
        return [*uv.min(0), *uv.max(0)]

    for row in sorted(rows, key=quality, reverse=True):
        box = bounds(row)
        prior = next(
            (
                r
                for r in selected
                if box_overlap(box, bounds(r)) >= config["roi"]["min_enclosure_ratio"]
            ),
            None,
        )
        if prior is None:
            selected.append(row)
        else:
            suppressed.append(
                dict(
                    proposal_id=row["proposal_id"],
                    kept=prior["proposal_id"],
                    reason="POST_PRECISION_SAME_VESSEL_OVERLAP",
                )
            )
    selected.sort(
        key=lambda row: tuple(
            np.mean(np.asarray(row["boundary_xy"]) @ vessel_axes.T, axis=0)
        )
    )
    return selected, suppressed


class Timing:
    def __init__(self, progress):
        self.progress, self.elapsed = progress, {}

    @contextmanager
    def stage(self, name):
        from ship_perception.tools.stage_progress import stage

        started = time.perf_counter()
        try:
            with stage(self.progress, name):
                yield
        finally:
            self.elapsed[name] = time.perf_counter() - started


def _legacy_proposals(frame, context, config, research, timing, local_caches=None):
    """Independent compatibility evidence; local recall does not consume its gate."""
    measured = {}
    structural_cache = (
        {ident: cache.export_structural() for ident, cache in local_caches.items()}
        if local_caches is not None
        else None
    )
    with timing.stage("LEGACY_GEOMETRY_EVIDENCE"):
        b2, b4, counts = prepare_geometry(
            frame.xyz,
            "input",
            frame.sha256,
            config,
            research,
            scene_context=context,
            structural_cache=structural_cache,
            evidence_sink=measured,
        )
    with timing.stage("LEGACY_TOPOLOGY_EVIDENCE"):
        report, _, _ = review_points(
            "input",
            frame.xyz,
            frame.sha256,
            b2,
            b4,
            config,
            research,
            scene_context=context,
        )
    counts["per_vessel_status"] = [
        dict(
            vessel_hypothesis_id=v["vessel_hypothesis_id"],
            status=v.get("status"),
            hatches=[
                dict(hatch_id=h["hatch_id"], status=h["status"])
                for h in v["rectangles"]
            ],
        )
        for v in report["vessels"]
    ]
    rows = []
    for vessel in report["vessels"]:
        ident = vessel["vessel_hypothesis_id"]
        originals = {
            r["hatch_id"]: r
            for r in b4["rectangles"]
            if r.get("vessel_hypothesis_id") == ident
        }
        for row in vessel["rectangles"]:
            if not row["status"].startswith(("PARTIAL", "UNRESOLVED")):
                continue
            original = originals[row["hatch_id"]]
            axes = np.asarray(original["axes"])
            uv = np.asarray(row["polygon_after"]) @ axes.T
            rows.append(
                dict(
                    proposal_id="legacy:" + row["hatch_id"],
                    vessel_hypothesis_id=ident,
                    axes=axes.tolist(),
                    bounds_axial=[*uv.min(0), *uv.max(0)],
                    proposal_source="LEGACY_MEASURED_TOPOLOGY_COMPARISON",
                    inherited_review_status=row["status"],
                    prior_evidence=row,
                )
            )
    # Resolve each adjacent pair from two observed faces, not a shared station.
    for ident in sorted({r["vessel_hypothesis_id"] for r in rows}):
        local = sorted(
            [r for r in rows if r["vessel_hypothesis_id"] == ident],
            key=lambda r: r["bounds_axial"][0],
        )
        vessel = next(
            v
            for v in b2["scene"]["vessel_hypotheses"]
            if v["vessel_hypothesis_id"] == ident
        )
        for first, second in zip(local[:-1], local[1:]):
            if not np.allclose(first["axes"], second["axes"]):
                continue
            a, b = first["bounds_axial"], second["bounds_axial"]
            if abs(a[2] - b[0]) > config["boundary"]["corner_join_m"]:
                continue
            axes = np.asarray(first["axes"])
            xyz = np.column_stack((frame.xyz[:, :2] @ axes.T, frame.xyz[:, 2]))
            grid = R1HeightGrid.from_points(
                xyz, config["geometry"]["candidate_voxel_m"]
            )
            union = [a[0], min(a[1], b[1]), b[2], max(a[3], b[3])]
            boxes, separator = _separator_split(
                dict(bounds_axial=union),
                b2,
                dict(vessel, local_axes=axes.tolist()),
                grid,
                config,
            )
            if (
                isinstance(separator, list)
                and len(boxes) == 2
                and separator[1] - separator[0]
                <= config["boundary"]["beam_max_width_m"]
                and abs(np.mean(separator) - (a[2] + b[0]) / 2)
                <= config["roi"]["support_search_m"]
            ):
                for row, box in zip((first, second), boxes):
                    row["bounds_before_separator"] = row["bounds_axial"][:]
                    row["bounds_axial"] = box
                    row["measured_separator_faces"] = separator
    from ship_perception.r1_static.r2h_b5_corner_forensics import (
        _observations,
        _side_modes,
    )

    for row in rows:
        box = row["bounds_axial"]
        margin = config["roi"]["support_search_m"]
        observations = _observations(
            b2,
            row["vessel_hypothesis_id"],
            np.asarray(row["axes"]),
            [box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin],
        )
        sides = {}
        row["measured_side_candidates"] = {}
        for name, axis, position in (
            ("U0", 0, 0),
            ("U1", 0, 2),
            ("V0", 1, 1),
            ("V1", 1, 3),
        ):
            tolerance = config["roi"]["support_search_m"]
            modes = _side_modes(
                observations,
                axis,
                (box[1 - axis], box[3 - axis]),
                (box[position] - tolerance, box[position] + tolerance),
                config,
            )
            row["measured_side_candidates"][name] = modes
            modes = [
                m
                for m in modes
                if abs(m["rho"] - box[position]) <= config["boundary"]["corner_join_m"]
            ]
            minimum = config["roi"]["min_enclosure_ratio"]
            sides[name] = dict(
                raw3d_support=any(m["primary_raw3d"] >= minimum for m in modes),
                bev_support=any(m["primary_bev"] >= minimum for m in modes),
                source_vessel_ids=[row["vessel_hypothesis_id"]],
                steel_role_status="UNVERIFIED",
            )
        row["deck_ring"] = dict(sides=sides)
    from ship_perception.r1_static.boundary_topology import (
        _observations as profile_observations,
    )

    profiles, _ = profile_observations(
        measured.get("structural", {}).get("segment_resolutions", [])
    )
    source_xy = np.asarray([p["xy"] for p in profiles], dtype=float).reshape(-1, 2)
    profile_ownership = {}
    for row in rows:
        vessel = next(
            v for v in context.vessels if v.vessel_id == row["vessel_hypothesis_id"]
        )
        if vessel.vessel_id not in profile_ownership:
            if local_caches is not None and vessel.vessel_id in local_caches:
                owned = local_caches[vessel.vessel_id].grid("coarse")
            else:
                points = frame.xyz[vessel.point_indexes]
                owned = R1HeightGrid.from_points(
                    np.column_stack((points[:, :2] @ vessel.axes.T, points[:, 2])),
                    config["geometry"]["coarse_voxel_m"],
                )
            rr, cc = owned.cell_indices(source_xy @ vessel.axes.T)
            keep = (
                (rr >= 0)
                & (rr < owned.count.shape[0])
                & (cc >= 0)
                & (cc < owned.count.shape[1])
            )
            keep[keep] &= owned.occupancy[rr[keep], cc[keep]]
            profile_ownership[vessel.vessel_id] = np.flatnonzero(keep)
        indexes = profile_ownership[vessel.vessel_id]
        positions = source_xy[indexes] @ np.asarray(row["axes"]).T
        row["profile_observations"] = [
            dict(
                profiles[index],
                xy=xy.tolist(),
                source_vessel_id=vessel.vessel_id,
                point_index_sha256=vessel.point_index_sha256,
            )
            for index, xy in zip(indexes, positions)
        ]
    return rows, counts


def run_perception(frame, progress=None, legacy_evidence=True):
    timing = Timing(progress)
    config, config_sha = resolve_config(DEFAULT_CONFIG)
    research_bytes = RESEARCH_CONFIG.read_bytes()
    research = json.loads(research_bytes)
    with timing.stage("INPUT_VALIDATION"):
        if len(frame.xyz) > config["geometry"]["max_input_points"]:
            raise ValueError("INPUT_POINT_LIMIT_EXCEEDED")
        if np.max(np.ptp(frame.xyz, axis=0)) > config["geometry"]["max_local_extent_m"]:
            raise ValueError("INPUT_EXTENT_LIMIT_EXCEEDED")
        intensity = admission_stats(
            frame.intensity, frame.provenance.get("intensity_dtype")
        )
    with timing.stage("SCENE_VESSEL_DISCOVERY_ONCE"):
        context = SceneEvidenceContext.build(
            frame, config, research, config_sha, software_identity()
        )
        context.validate_frame(frame)
    by_vessel, caches, exteriors, rejected, isolated = [], {}, {}, [], []
    for vessel in context.vessels:
        ident = vessel.vessel_id
        item = dict(
            vessel_hypothesis_id=ident,
            classification_status=vessel.public["classification_status"],
            hatches=[],
            unresolved_candidates=[],
            unresolved_review_proposals=bounded([]),
            rejected_final_candidates=[],
            status="UNRESOLVED_VESSEL_REVIEW",
        )
        by_vessel.append(item)
        minimum = (
            research["min_points_per_coarse_cell"]
            * config["roi"]["min_deck_support_cells"]
        )
        if len(vessel.point_indexes) < minimum:
            item.update(
                status="INSUFFICIENT_SUPPORT",
                reason="INSUFFICIENT_RAW_SUPPORT_FOR_PROVIDERS",
            )
            continue
        try:
            with timing.stage(ident + "_LOCAL_RECALL_VALIDATE"):
                cache = VesselLocalEvidence.build(
                    frame, vessel, config, context.boundary_context(ident)
                )
                exterior = ExteriorTopology.build(
                    frame, vessel, config, context.boundary_context(ident)
                )
                caches[ident], exteriors[ident] = cache, exterior
                proposals, audit = propose_openings(cache, research)
                selected, selection = validate_and_select(cache, proposals, exterior)
                deferred = selection.pop("unresolved_proposals", [])
                for proposal in deferred:
                    boundary = polygon(proposal["bounds_axial"])
                    proposal.update(
                        proposal_boundary_uv=boundary.tolist(),
                        proposal_boundary_xy=(boundary @ vessel.axes).tolist(),
                        vessel_local_axes=vessel.axes.tolist(),
                        input_frame=frame.frame_id,
                        transform_provenance="FROZEN_VESSEL_CONTEXT; UV_TO_INPUT_XY",
                    )
                # Retain the physically qualified examples before weaker
                # alternatives in the bounded audit; area/count never rank them.
                deferred.sort(
                    key=lambda p: (
                        sum(
                            s["vertical_face_support"]
                            for s in p["deck_ring"]["sides"].values()
                        ),
                        p["deck_ring"]["measured_strong_side_count"],
                        p["deck_ring"]["dual_provider_side_count"],
                    ),
                    reverse=True,
                )
                item["unresolved_review_proposals"] = bounded(deferred)
                rejected.extend(audit["rejected_candidates"])
                item.update(
                    recall_audit={
                        k: (bounded(v) if isinstance(v, list) else v)
                        for k, v in audit.items()
                    },
                    validation_audit={
                        k: (bounded(v) if isinstance(v, list) else v)
                        for k, v in selection.items()
                    },
                )
                rejected.extend(selection["rejected"])
            with timing.stage(ident + "_LOCAL_FOUR_SIDE_PRECISION"):
                item["precision_admission"] = dict(
                    validated_proposal_count=len(selected),
                    ambiguous_topology_count=sum(
                        p.get("topology_status") == "AMBIGUOUS_TOPOLOGY"
                        for p in selected
                    ),
                    raw_refinement_attempt_count=0,
                )
                refiner = FinalBoundaryRefiner(cache.xyz, [], config)
                for proposal in selected:
                    try:
                        item["precision_admission"]["raw_refinement_attempt_count"] += 1
                        proposal["measured_side_candidates"] = cache.side_candidates(
                            proposal["bounds_axial"]
                        )
                        result = refiner.refine(proposal, vessel.axes)
                        result.update(
                            proposal_id=proposal["proposal_id"],
                            vessel_hypothesis_id=ident,
                            containment=proposal["containment"],
                            exterior=proposal["exterior"],
                            deck_ring=proposal["deck_ring"],
                            proposal_source="VESSEL_LOCAL_RECALL",
                            topology_status=proposal.get("topology_status"),
                            topology_ambiguity=proposal.get("topology_ambiguity"),
                            separator_evidence=proposal.get("measured_separator_faces"),
                        )
                        if not result.get("boundary_xy"):
                            item["unresolved_candidates"].append(result)
                            continue
                        measured = sum(
                            s["raw_subvoxel_section_count"]
                            >= config["boundary"]["profile_min_sections"]
                            and s["positive_height_section_count"]
                            >= config["boundary"]["profile_min_sections"]
                            for s in result["sides"].values()
                        )
                        review_sides = sum(
                            s["phase_runs"][0]["physical_section_count"] > 0
                            and min(
                                s["raw_subvoxel_section_count"],
                                s["positive_height_section_count"],
                            )
                            >= min(
                                config["boundary"]["profile_min_sections"],
                                s["phase_runs"][0]["physical_section_count"],
                            )
                            for s in result["sides"].values()
                        )
                        result.update(
                            full_support_side_count=measured,
                            review_support_side_count=review_sides,
                        )
                        contained = proposal["containment"]["status"] == "CONTAINED"
                        enclosed = (
                            proposal["exterior"]["status"]
                            == "ENCLOSED_OBSERVED_LOW_COMPONENT"
                        )
                        ambiguous_topology = (
                            proposal.get("topology_status") == "AMBIGUOUS_TOPOLOGY"
                        )
                        contextual = (
                            partial_context_review(
                                cache,
                                result["boundary_xy"],
                                proposal["containment"],
                                proposal["exterior"],
                            )
                            if review_sides >= 3 and not (contained and enclosed)
                            else None
                        )
                        if review_sides >= 3 and (
                            (contained and enclosed) or contextual
                        ):
                            if measured < 4 or result["undercoverage_risk"]:
                                result["status"] = "PARTIAL_HATCH"
                            if contextual:
                                result.update(
                                    status="PARTIAL_HATCH",
                                    boundary_context_review=contextual,
                                )
                            if ambiguous_topology:
                                # Its measured outer edges remain useful, but
                                # cannot prove this group is one hatch. Dropping
                                # it exposes competing cargo splits as separate
                                # hatches during overlap suppression.
                                result.update(
                                    status="UNRESOLVED",
                                    reason="HATCH_COUNT_AMBIGUOUS",
                                    candidate_type="UNRESOLVED_OPENING_GROUP",
                                    individual_hatch_count=None,
                                    geometry_role="UNRESOLVED_ENCLOSURE_REVIEW",
                                )
                            if (
                                vessel.public["classification_status"]
                                != "VESSEL_HYPOTHESIS"
                            ):
                                result["status"] = "UNRESOLVED_VESSEL_REVIEW"
                                result["vessel_role_confirmed"] = False
                            item["hatches"].append(result)
                        else:
                            result["status"] = (
                                "PARTIAL_HATCH"
                                if review_sides >= 3 and not ambiguous_topology
                                else "UNRESOLVED"
                            )
                            if ambiguous_topology:
                                result["reason"] = "HATCH_COUNT_AMBIGUOUS"
                            item["unresolved_candidates"].append(result)
                    except (ValueError, ArithmeticError) as error:
                        isolated.append(
                            dict(
                                vessel=ident,
                                hatch=proposal["proposal_id"],
                                stage="FINAL_BOUNDARY",
                                code=str(error),
                            )
                        )
        except (ValueError, ArithmeticError) as error:
            item.update(
                status="UNRESOLVED",
                first_failure_stage="LOCAL_RECALL_VALIDATE",
                error_code=str(error),
            )
            isolated.append(
                dict(vessel=ident, stage="LOCAL_RECALL_VALIDATE", code=str(error))
            )
    legacy_counts = None
    if legacy_evidence:
        # Legacy evidence remains explicit and independently disableable. It
        # cannot suppress, seed or restrict the vessel-local recall above.
        try:
            legacy, legacy_counts = _legacy_proposals(
                frame, context, config, research, timing, caches
            )
        except (ValueError, KeyError, ArithmeticError) as error:
            if any(
                code in str(error)
                for code in (
                    "PROVENANCE_MISMATCH",
                    "INPUT_SHA_MISMATCH",
                    "BEHAVIOR_CHANGED",
                    "CONFIG",
                )
            ):
                raise
            legacy, legacy_counts = [], dict(
                status="UNRESOLVED_LEGACY_EVIDENCE", code=str(error)
            )
            isolated.append(dict(stage="LEGACY_EVIDENCE", code=str(error)))
        previous_key, shared_refiner = None, None
        for proposal in legacy:
            ident = proposal["vessel_hypothesis_id"]
            if ident not in caches:
                continue
            vessel = next(v for v in context.vessels if v.vessel_id == ident)
            item = next(v for v in by_vessel if v["vessel_hypothesis_id"] == ident)
            axes = np.asarray(proposal["axes"])
            key = (ident, tuple(axes.ravel()))
            if key != previous_key:
                # A review may use eligible unresolved sibling measurements,
                # never every unresolved point in the whole scene.
                points = frame.xyz[caches[ident].measurement_context.point_indexes]
                uvz = np.column_stack((points[:, :2] @ axes.T, points[:, 2]))
                shared_refiner = FinalBoundaryRefiner(
                    uvz, proposal.get("profile_observations", []), config
                )
                previous_key = key
            proposal["raw_support_scope"] = (
                "FROZEN_CORE_AND_ELIGIBLE_UNRESOLVED_SIBLINGS; OTHER_SUPPORTS_EXCLUDED"
            )
            try:
                with timing.stage(ident + "_COMPARISON_BOUNDARY_PRECISION"):
                    result = shared_refiner.refine(proposal, axes)
            except (ValueError, ArithmeticError) as error:
                if any(
                    code in str(error)
                    for code in ("PROVENANCE", "CONFIG", "INVALID_HATCH_AXES")
                ):
                    raise
                failure = dict(
                    vessel=ident,
                    hatch=proposal["proposal_id"],
                    stage="COMPARISON_BOUNDARY_PRECISION",
                    code=str(error),
                )
                isolated.append(failure)
                item["unresolved_candidates"].append(
                    dict(
                        status="UNRESOLVED",
                        failure=failure,
                        proposal_id=proposal["proposal_id"],
                    )
                )
                continue
            if not result.get("boundary_xy"):
                result.update(
                    proposal_id=proposal["proposal_id"],
                    vessel_hypothesis_id=ident,
                    proposal_source=proposal["proposal_source"],
                )
                item["unresolved_candidates"].append(result)
                continue
            boundary = np.asarray(result["boundary_xy"]) @ vessel.axes.T
            box = [*boundary.min(0), *boundary.max(0)]
            containment = vessel_containment(caches[ident].grid("coarse"), box, config)
            # The ring's own observed exterior level is used, never a fixed Z.
            from ship_perception.r1_static.hatch_proposal import _opening_evidence

            opening = _opening_evidence(caches[ident].context_grid("fine"), box, config)
            levels = [
                v for v in opening.get("outside_side_median_z", []) if v is not None
            ]
            exterior = exteriors[ident].review(
                box, float(np.median(levels)) if levels else None
            )
            result.update(
                vessel_hypothesis_id=ident,
                proposal_id=proposal["proposal_id"],
                proposal_source=proposal["proposal_source"],
                containment=containment,
                exterior=exterior,
                separator_evidence=proposal.get("measured_separator_faces"),
                boundary_before_separator_uv=proposal.get("bounds_before_separator"),
                raw_roi_ownership="CORE_PLUS_UNRESOLVED_BOUNDARY_CONTEXT; NOT_CERTIFIED_STEEL",
                inherited_review_status=proposal.get("inherited_review_status"),
            )
            review_sides = sum(
                min(s["raw_subvoxel_section_count"], s["positive_height_section_count"])
                >= config["boundary"]["profile_min_sections"]
                for s in result["sides"].values()
            )
            result["review_support_side_count"] = review_sides
            contextual = (
                partial_context_review(
                    caches[ident], result["boundary_xy"], containment, exterior
                )
                if review_sides >= 3
                else None
            )
            if (
                containment["status"] == "REJECT_OUTSIDE_VESSEL"
                or exterior["status"] == "REJECT_EXTERIOR_LOW_REGION"
            ):
                result["status"] = "REJECTED"
                item["rejected_final_candidates"].append(result)
                rejected.append(
                    dict(
                        proposal_id=proposal["proposal_id"],
                        reason=(
                            containment["status"]
                            if containment["status"] == "REJECT_OUTSIDE_VESSEL"
                            else exterior["status"]
                        ),
                        boundary_xy=result["boundary_xy"],
                    )
                )
                continue
            if proposal.get("inherited_review_status", "").startswith("UNRESOLVED"):
                # Preserve historical uncertainty for inspection. It may only
                # enter hatch output after the same raw support and topology
                # requirements as a newly recalled local proposal.
                if review_sides < 3 or (
                    not contextual
                    and (
                        containment["status"] != "CONTAINED"
                        or exterior["status"] != "ENCLOSED_OBSERVED_LOW_COMPONENT"
                    )
                ):
                    result.update(
                        status="UNRESOLVED", review_support_side_count=review_sides
                    )
                    item["unresolved_candidates"].append(result)
                    continue
            if contextual:
                result.update(
                    status="PARTIAL_HATCH", boundary_context_review=contextual
                )
            if (
                exterior["status"] != "ENCLOSED_OBSERVED_LOW_COMPONENT"
                or result["undercoverage_risk"]
            ):
                result["status"] = "PARTIAL_HATCH"
            item["hatches"].append(result)
    hatches = []
    for item in by_vessel:
        vessel = next(
            v for v in context.vessels if v.vessel_id == item["vessel_hypothesis_id"]
        )
        item["hatches"], suppressed = precision_suppression(
            item["hatches"], vessel.axes, config
        )
        item["precision_suppressed_candidates"] = bounded(suppressed)
        if vessel.vessel_id in caches:
            cache = caches[vessel.vessel_id]
            item["provider_cache"] = dict(
                raw3d_build_count=cache.provider_build_count,
                bev_build_count=cache.provider_build_count,
                grid_build_counts=cache.grid_build_counts,
            )
        item["status"] = (
            (
                "REVIEW_CANDIDATES"
                if item["classification_status"] == "VESSEL_HYPOTHESIS"
                else "UNRESOLVED_VESSEL_REVIEW"
            )
            if item["hatches"]
            else item["status"]
        )
        for result in item["hatches"]:
            result.update(
                hatch_id="H%03d" % len(hatches),
                input_frame=frame.frame_id,
                transform_provenance="FROZEN_VESSEL_CONTEXT; UV_TO_INPUT_XY",
            )
            result["candidate_blockers"] = [
                s["status"] + ":" + name
                for name, s in result["sides"].items()
                if s["undercoverage_risk"]
                or s["grid_phase"]["status"] == "GRID_PHASE_SENSITIVE"
            ]
            if result.get("reason") == "HATCH_COUNT_AMBIGUOUS":
                result["candidate_blockers"].append("HATCH_COUNT_AMBIGUOUS")
            if result.get("boundary_context_review"):
                result["candidate_blockers"].append("BOUNDARY_OWNERSHIP_UNRESOLVED")
            hatches.append(result)
        groups = {}
        for result in item["hatches"]:
            separator = result.get("separator_evidence")
            key = tuple(separator) if separator else (result["hatch_id"],)
            group = groups.setdefault(
                key,
                dict(
                    group_id="G%03d" % len(groups),
                    hatch_ids=[],
                    source_proposal_ids=[],
                    measured_separator_faces=separator,
                    status=(
                        "HATCH_COUNT_AMBIGUOUS"
                        if result.get("reason") == "HATCH_COUNT_AMBIGUOUS"
                        else (
                            "SEPARATE_MEASURED_SEPARATOR_FACES"
                            if separator
                            else "INDEPENDENT_OPENING"
                        )
                    ),
                ),
            )
            group["hatch_ids"].append(result["hatch_id"])
            group["source_proposal_ids"].append(result["proposal_id"])
        item["independent_groups"] = list(groups.values())
        item["aggregate_summary"] = dict(
            group_count=len(groups), hatch_count=len(item["hatches"])
        )
        item.update(
            u_boundary_chain=None,
            selected_pair=None,
            v0_candidates=None,
            v1_candidates=None,
            section_rho_consensus=None,
        )
        # Curated comparison uncertainty must not disappear behind the many
        # coarse local alternatives in the bounded diagnostic sample.
        item["unresolved_candidates"].sort(
            key=lambda r: not r.get("proposal_id", "").startswith("legacy:")
        )
        item["unresolved_candidates"] = bounded(item["unresolved_candidates"])
    return dict(
        schema="ship_perception.hatch_once.2",
        pipeline_version="recall-validate-precision.1",
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
        intensity_aux_weight=AUX_WEIGHT,
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
            hatch_count_status=(
                "AMBIGUOUS"
                if any(h.get("reason") == "HATCH_COUNT_AMBIGUOUS" for h in hatches)
                else "REVIEW_ONLY"
            ),
            hatches=hatches,
            intensity_used=False,
        ),
        pipeline_health=dict(
            status="REVIEW_REQUIRED",
            first_failure_stage=(
                isolated[0]["stage"]
                if isolated
                else (
                    "OPENING_TOPOLOGY_VALIDATION"
                    if any(h.get("reason") == "HATCH_COUNT_AMBIGUOUS" for h in hatches)
                    or (
                        not hatches
                        and any(
                            v["unresolved_review_proposals"]["total_count"]
                            for v in by_vessel
                        )
                    )
                    else "LOCAL_RECALL_VALIDATE_PRECISION" if not hatches else None
                )
            ),
            isolated_failures=isolated,
            legacy_comparison_enabled=legacy_evidence,
            legacy_provider_counts=legacy_counts,
            b1_execution_count=1,
            steel_precision_status="UNVERIFIED_NO_INDEPENDENT_INNER_EDGE_GOLDEN",
        ),
        rejected_exterior_candidates=bounded(
            [r for r in rejected if r["reason"] == "REJECT_EXTERIOR_LOW_REGION"]
        ),
        rejected_outside_vessel=bounded(
            [r for r in rejected if r["reason"] == "REJECT_OUTSIDE_VESSEL"]
        ),
        scene_exterior_low_regions={
            ident: dict(
                observed_only=True,
                level_evaluations=e.level_evaluations,
                cached_level_count=len(e.components),
            )
            for ident, e in exteriors.items()
        },
    )
