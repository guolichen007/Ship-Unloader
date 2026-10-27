"""Single four-side refiner using existing raw and multi-section primitives."""

import warnings
import copy
import json
import time
import hashlib
from contextlib import contextmanager

from .measurement import RawSpatialIndex, RawSectionProfile, BudgetedLocalSurfaceNormals

import numpy as np
from scipy.stats import theilslopes

from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.height_rim_review import _section_transition
from ship_perception.r1_static.rectangle_refinement import (
    _continuous_rho,
    _raw_edge_rho,
)
from .sections import physical_sections, robust_height_support
from .line_geometry import SIDE_SPECS, guard_line_move, guarded_intersections
from .raw_face_refinement import (
    measured_faces,
    core_adjacency,
    same_face,
    public_face,
    face_position,
    raw_face_section,
)


def polygon(box):
    return np.array(
        ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])),
        dtype=float,
    )


def transpose_grid(grid):
    return R1HeightGrid(
        grid.y0,
        grid.x0,
        grid.cell_m,
        *(
            getattr(grid, k).T
            for k in ("count", "q10", "median", "q90", "minimum", "maximum")
        ),
    )


def guard_move(before, estimate, sign, sections, independent, config):
    inward = (estimate - before) * sign
    supported = sections >= config["boundary"]["profile_min_sections"] and independent
    if abs(inward) > config["boundary"]["corner_join_m"] and not supported:
        return before, "KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY", True
    return (
        estimate,
        "RAW_GEOMETRY_REVIEW" if supported else "PARTIAL_RAW_SUPPORT",
        not supported,
    )


class FinalBoundaryRefiner:
    def __init__(self, xyz, observations, config, measurement_context=None):
        self.xyz, self.observations, self.config = xyz, observations, config
        self.raw_section_cache = {}
        self.refinement_cache = {}
        self.refinement_cache_bytes = 0
        self.face_normals = None
        self.measurement_context = measurement_context
        self.profiler = (
            measurement_context.counters if measurement_context is not None else {}
        )
        self.spatial = (
            measurement_context.spatial
            if measurement_context is not None
            else RawSpatialIndex(xyz, self.profiler)
        )
        self.continuous_cache = {}
        self.height_cache = {}
        self.budget = getattr(measurement_context, "budget", None)
        self.side_elapsed = {}
        self.observation_key = ""

    @contextmanager
    def _timed_side(self, name, scope):
        started = time.perf_counter()
        try:
            yield
        finally:
            scopes = self.side_elapsed.setdefault(name, {})
            scopes[scope] = scopes.get(scope, 0.0) + time.perf_counter() - started

    def refine(self, proposal, axes):
        if self.budget is not None:
            self.budget.check()
        # Different proposal paths can converge to the exact same physical
        # query after separator discovery. Reuse that query, preserving each
        # caller's proposal identity outside this immutable geometry result.
        key = json.dumps(
            [
                np.asarray(axes).tolist(),
                {
                    k: proposal.get(k)
                    for k in (
                        "bounds_axial",
                        "bounds_before_separator",
                        "deck_ring",
                        "measured_side_candidates",
                        "vessel_hypothesis_id",
                        "raw_support_scope",
                        "opening_evidence",
                        "profile_observations",
                    )
                },
                {
                    name: (side.get("status"), side.get("rho"))
                    for name, side in proposal.get("raw_probe", {})
                    .get("sides", {})
                    .items()
                },
            ],
            sort_keys=True,
        )
        if key in self.refinement_cache:
            return copy.deepcopy(self.refinement_cache[key][0])
        started = time.perf_counter()
        self.observations = proposal.get("profile_observations", self.observations)
        self.observation_key = (
            hashlib.sha256(
                json.dumps(self.observations, sort_keys=True).encode()
            ).hexdigest()
            if self.observations
            else ""
        )
        self.side_elapsed = {}
        previous_counters = dict(self.profiler)
        result = self._refine_uncached(proposal, axes)
        result["precision_profiler"] = dict(
            per_candidate_elapsed_s=time.perf_counter() - started,
            per_side_elapsed_s={
                key: sum(value.values()) for key, value in self.side_elapsed.items()
            },
            per_side_elapsed_breakdown_s=copy.deepcopy(self.side_elapsed),
            per_side_raw_measurement_elapsed_s=self.profiler.get(
                "last_raw_side_elapsed_s", {}
            ),
            **{
                key: value - previous_counters.get(key, 0)
                for key, value in self.profiler.items()
                if isinstance(value, (int, float))
            },
        )
        size = len(json.dumps(result).encode()) + len(key.encode())
        budget = 64 * 1024 * 1024
        while self.refinement_cache and self.refinement_cache_bytes + size > budget:
            _, removed = self.refinement_cache.popitem()
            self.refinement_cache_bytes -= removed[1]
        if size <= budget:
            self.refinement_cache[key] = (copy.deepcopy(result), size)
            self.refinement_cache_bytes += size
        return result

    def _refine_uncached(self, proposal, axes):
        box = list(map(float, proposal["bounds_axial"]))
        guard_bounds = list(map(float, proposal.get("bounds_before_separator", box)))
        config = self.config
        axes = np.asarray(axes, dtype=float)
        if (
            axes.shape != (2, 2)
            or not np.isfinite(axes).all()
            or not np.allclose(axes @ axes.T, np.eye(2), atol=1e-10)
        ):
            raise ValueError("INVALID_HATCH_AXES")
        cell = config["geometry"]["refine_voxel_m"]
        margin = config["roi"]["boundary_search_m"] + config["boundary"]["side_probe_m"]
        # The inherited rectangle can be anchored on a shell rim. A structural
        # proposal may nominate a different physical line within the existing
        # support ROI; each nominated line still gets the unchanged raw search.
        reanchors = self._inner_role_anchors(proposal, box, cell)
        measured_faces_by_side = {
            name: row.pop("_raw_normal_face", None) for name, row in reanchors.items()
        }
        for name, _, position, _ in SIDE_SPECS:
            if reanchors[name]["status"] == "CORROBORATED_INNER_STEP":
                box[position] = reanchors[name]["rho"]
        points = self.xyz
        roi = self.spatial.query(
            [box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin]
        )
        if not len(roi):
            return dict(status="UNRESOLVED", error_code="NO_RAW_ROI_SUPPORT", sides={})
        if len(roi) > config["geometry"]["max_candidate_points"]:
            raise ValueError("FINAL_ROI_POINT_LIMIT_EXCEEDED")
        raw_sections = {}
        section_points = {}
        side_elapsed = {}
        for name, axis, position, sign in SIDE_SPECS:
            if self.budget is not None:
                self.budget.check()
            if self.budget is not None:
                self.budget.check()
            side_started = time.perf_counter()
            edges = physical_sections((box[1 - axis], box[3 - axis]), cell, config)
            for index, span in enumerate(zip(edges[:-1], edges[1:])):
                # Each primitive independently restricts the same along-line
                # interval. Retain those exact points once for all grid phases.
                section_points[name, index] = roi[
                    (roi[:, 1 - axis] >= span[0]) & (roi[:, 1 - axis] <= span[1])
                ]
                raw_sections[name, index] = self._raw_section(
                    section_points[name, index],
                    axis,
                    box[position],
                    span,
                    sign,
                    measured_face=measured_faces_by_side[name],
                )
            side_elapsed[name] = time.perf_counter() - side_started
        for name, elapsed in side_elapsed.items():
            self.side_elapsed.setdefault(name, {})["RAW_MEASUREMENT"] = elapsed
        from .raw_mode_consensus import resolve_section_modes

        for name, axis, _, _ in SIDE_SPECS:
            edges = physical_sections((box[1 - axis], box[3 - axis]), cell, config)
            spans = list(zip(edges[:-1], edges[1:]))
            records = [raw_sections[name, i] for i in range(len(spans))]
            for i, record in enumerate(
                resolve_section_modes(records, spans, config, self.budget)
            ):
                raw_sections[name, i] = record
        self.profiler["last_raw_side_elapsed_s"] = side_elapsed
        # Raster phases only query boundary bands. All original raw points
        # remain available to the spatial index and raw section measurements.
        raster_roi = roi[
            (abs(roi[:, 0] - box[0]) <= margin)
            | (abs(roi[:, 0] - box[2]) <= margin)
            | (abs(roi[:, 1] - box[1]) <= margin)
            | (abs(roi[:, 1] - box[3]) <= margin)
        ]
        origin = roi[:, :2].min(0) - cell
        phases = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
        phase_results = {name: [] for name, *_ in SIDE_SPECS}
        for phase in phases:
            if self.budget is not None:
                self.budget.check()
            grid = (
                self.measurement_context.rasters.grid(box, margin, phase)
                if self.measurement_context is not None
                else R1HeightGrid.from_points(
                    raster_roi, cell, origin_xy=origin + np.array(phase) * cell
                )
            )
            for name, axis, position, sign in SIDE_SPECS:
                with self._timed_side(name, "RASTER_EVIDENCE"):
                    span = (box[1 - axis], box[3 - axis])
                    edges = physical_sections(span, cell, config)
                    evidence = _section_transition(
                        transpose_grid(grid) if axis == 0 else grid,
                        span,
                        box[position],
                        "V0" if sign == 1 else "V1",
                        config,
                        section_edges=edges,
                    )
                    samples = []
                    for index, (begin, end) in enumerate(zip(edges[:-1], edges[1:])):
                        measured = [
                            s for s in evidence["samples"] if begin <= s[0] <= end
                        ]
                        # A sparse 0.05 m raster may have no paired occupied rows.
                        # It must not veto an observed raw step at the proposal.
                        anchor = (
                            float(np.median(np.array(measured)[:, 1]))
                            if measured
                            else box[position]
                        )
                        direct = raw_sections[name, index]
                        raw_points = section_points[name, index]
                        if direct["status"] == "UNIQUE_SIGNED_RAW_STEP":
                            rho, uncertainty, raw_count, paired = direct["continuous"]
                        else:
                            rho, uncertainty, raw_count, paired = (
                                self._continuous_measurement(
                                    raw_points,
                                    self.observations,
                                    axis,
                                    box[position],
                                    (begin, end),
                                    sign,
                                    config,
                                )
                            )
                        if not paired and anchor != box[position]:
                            prior_rho, prior_uncertainty, prior_count, prior_paired = (
                                self._continuous_measurement(
                                    raw_points,
                                    self.observations,
                                    axis,
                                    box[position],
                                    (begin, end),
                                    sign,
                                    config,
                                )
                            )
                            if prior_paired:
                                rho, uncertainty, raw_count, paired = (
                                    prior_rho,
                                    prior_uncertainty,
                                    prior_count,
                                    prior_paired,
                                )
                        height = (
                            direct["height"]
                            if direct["status"] == "UNIQUE_SIGNED_RAW_STEP"
                            else self._height_measurement(
                                raw_points, axis, rho, (begin, end), sign, config
                            )
                        )
                        profiles = [
                            r
                            for r in self.observations
                            if r.get("reference_role")
                            not in ("INTERIOR_SURFACE", "ROLE_AMBIGUOUS_OVERFLOW")
                            and begin <= r["xy"][1 - axis] <= end
                            and abs(r["xy"][axis] - rho)
                            <= config["boundary"]["corner_join_m"]
                        ]
                        samples.append(
                            dict(
                                section=index,
                                along=(begin + end) / 2,
                                rho=rho,
                                uncertainty_m=uncertainty,
                                raw_point_count=raw_count,
                                raw_subvoxel=bool(
                                    direct.get("raw_measurement_present", paired)
                                ),
                                surface_pair_present=bool(paired),
                                raw_normal_face_count=direct.get(
                                    "raw_normal_face_count", 0
                                ),
                                measurement_source=direct.get(
                                    "measurement_source",
                                    (
                                        "PAIRED_RAW_SURFACES"
                                        if paired
                                        else "PROFILE_OR_GRID"
                                    ),
                                ),
                                profile_observation_count=len(profiles),
                                profile_observation_ids=[
                                    r["observation_id"] for r in profiles
                                ],
                                height=height,
                                raw_search=direct,
                                raster_transition_support=bool(measured),
                            )
                        )
                    ring_side = (
                        proposal.get("deck_ring", {}).get("sides", {}).get(name, {})
                    )
                    for sample in samples:
                        sample["height_bev_corroborated"] = bool(
                            ring_side.get("bev_support")
                            and sample["height"].get("transition")
                            and min(sample["height"].get("counts", [0, 0]))
                            >= config["boundary"]["profile_min_points"]
                            and sample["raster_transition_support"]
                        )
                        if (
                            sample["height_bev_corroborated"]
                            and not sample["raw_subvoxel"]
                        ):
                            sample["measurement_source"] = "RAW_HEIGHT_PROFILE_AND_BEV"
                    good = [
                        s
                        for s in samples
                        if (
                            s["raw_subvoxel"]
                            or s["profile_observation_count"]
                            or s["height_bev_corroborated"]
                        )
                        and s["height"].get("transition", False)
                        and s["raw_search"]["status"] != "MULTIMODE_BOUNDARY_CONFLICT"
                    ]
                    estimate, fit = self._fit(good, box[position])
                    phase_results[name].append(
                        dict(
                            phase=list(phase),
                            rho=estimate,
                            samples=samples,
                            physical_section_count=len(edges) - 1,
                            support_section_count=len(good),
                            fit=fit,
                            height_transition_reason=evidence["reason"],
                        )
                    )
        sides, after = {}, box[:]
        for name, axis, position, sign in SIDE_SPECS:
            with self._timed_side(name, "LINE_FIT_AND_GUARDS"):
                runs = phase_results[name]
                reference = (box[1 - axis] + box[3 - axis]) / 2
                span = (
                    min(box[1 - axis], guard_bounds[1 - axis]),
                    max(box[3 - axis], guard_bounds[3 - axis]),
                )
                phase_slopes = np.array(
                    [
                        (
                            r["fit"].get("slope", 0.0)
                            if r["fit"]["status"] == "CONSENSUS"
                            else 0.0
                        )
                        for r in runs
                    ]
                )
                rhos = np.array(
                    [
                        (
                            r["fit"]["rho_at_reference_m"]
                            + m * (reference - r["fit"]["reference_along_m"])
                            if r["fit"]["status"] == "CONSENSUS"
                            else r["rho"]
                        )
                        for r, m in zip(runs, phase_slopes)
                    ]
                )
                estimate = float(np.median(rhos))
                proposed_slope = (
                    float(np.median(phase_slopes))
                    if all(r["fit"]["status"] == "CONSENSUS" for r in runs)
                    else 0.0
                )
                mad = float(np.median(abs(rhos - estimate)))
                maximum = float(max(abs(rhos - estimate)))
                proposed_endpoints = estimate + proposed_slope * (
                    np.asarray(span) - reference
                )
                phase_endpoints = rhos[:, None] + phase_slopes[:, None] * (
                    np.asarray(span)[None, :] - reference
                )
                endpoint_deviation = float(
                    np.max(abs(phase_endpoints - proposed_endpoints))
                )
                minimum_sections = min(r["support_section_count"] for r in runs)
                raw_sections = min(
                    sum(s["raw_subvoxel"] for s in r["samples"]) for r in runs
                )
                normal_face_sections = min(
                    sum(
                        s["raw_normal_face_count"]
                        >= config["boundary"]["line_min_points"]
                        for s in r["samples"]
                    )
                    for r in runs
                )
                paired_surface_sections = min(
                    sum(s["surface_pair_present"] for s in r["samples"]) for r in runs
                )
                profile_sections = min(
                    sum(s["profile_observation_count"] > 0 for s in r["samples"])
                    for r in runs
                )
                positive_sections = min(
                    sum(s["height"].get("transition", False) for s in r["samples"])
                    for r in runs
                )
                conflict = reanchors[name][
                    "status"
                ] == "MULTIMODE_BOUNDARY_CONFLICT" or any(
                    r["fit"]["status"] == "MULTIMODE_BOUNDARY_CONFLICT"
                    or sum(
                        s["raw_search"]["status"] == "MULTIMODE_BOUNDARY_CONFLICT"
                        for s in r["samples"]
                    )
                    >= config["boundary"]["profile_min_sections"]
                    for r in runs
                )
                ring_side = proposal.get("deck_ring", {}).get("sides", {}).get(name, {})
                dual_provider = ring_side.get("raw3d_support", False) and ring_side.get(
                    "bev_support", False
                )
                dual_provider |= reanchors[name]["status"] == "CORROBORATED_INNER_STEP"
                independent = raw_sections >= config["boundary"][
                    "profile_min_sections"
                ] and (
                    (
                        positive_sections >= config["boundary"]["profile_min_sections"]
                        and dual_provider
                    )
                    or profile_sections >= config["boundary"]["profile_min_sections"]
                )
                rho, slope, status, risk = guard_line_move(
                    guard_bounds[position],
                    estimate,
                    proposed_slope,
                    reference,
                    span,
                    sign,
                    minimum_sections,
                    independent and not conflict,
                    config,
                )
                if minimum_sections < config["boundary"]["profile_min_sections"]:
                    rho = (
                        guard_bounds[position]
                        if abs(box[position] - guard_bounds[position])
                        > config["boundary"]["corner_join_m"]
                        else box[position]
                    )
                    slope = 0.0
                    status, risk = "KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY", True
                if conflict:
                    rho, status, risk = (
                        guard_bounds[position],
                        "MULTIMODE_BOUNDARY_CONFLICT",
                        True,
                    )
                    slope = 0.0
                applied_endpoints = rho + slope * (np.asarray(span) - reference)
                sample_displacement = max(
                    (
                        abs(s["rho"] - (rho + slope * (s["along"] - reference)))
                        for r in runs
                        for s in r["samples"]
                        if s["height"].get("transition", False)
                        and (s["raw_subvoxel"] or s["profile_observation_count"])
                    ),
                    default=0.0,
                )
                uncertainty = max(
                    config["geometry"]["plane_inlier_m"],
                    maximum,
                    endpoint_deviation,
                    float(np.max(abs(proposed_endpoints - applied_endpoints))),
                    sample_displacement,
                    max(
                        (s["uncertainty_m"] for r in runs for s in r["samples"]),
                        default=cell,
                    ),
                    abs(estimate - rho),
                )
                if minimum_sections < config["boundary"]["profile_min_sections"]:
                    uncertainty = max(uncertainty, config["roi"]["boundary_search_m"])
                after[position] = rho
                sides[name] = dict(
                    rho_before_m=guard_bounds[position],
                    rho_search_anchor_m=box[position],
                    rho_after_m=rho,
                    side_inward_move_m=(rho - guard_bounds[position]) * sign,
                    side_max_endpoint_inward_move_m=float(
                        np.max(sign * (applied_endpoints - guard_bounds[position]))
                    ),
                    proposed_max_endpoint_inward_move_m=float(
                        np.max(sign * (proposed_endpoints - guard_bounds[position]))
                    ),
                    independent_geometry_support=bool(
                        independent
                        and not conflict
                        and minimum_sections
                        >= config["boundary"]["profile_min_sections"]
                    ),
                    line=dict(
                        normal_axis=axis,
                        slope=slope,
                        reference_along_m=reference,
                        rho_at_reference_m=rho,
                        source=(
                            "RAW_SECTION_ROBUST_LINE"
                            if minimum_sections
                            >= config["boundary"]["profile_min_sections"]
                            and not conflict
                            and rho == estimate
                            and slope == proposed_slope
                            else "CONSERVATIVE_AXIS_LINE"
                        ),
                    ),
                    proposed_line=dict(
                        normal_axis=axis,
                        slope=proposed_slope,
                        reference_along_m=reference,
                        rho_at_reference_m=estimate,
                    ),
                    support_section_count=minimum_sections,
                    support_section_ratio=minimum_sections
                    / max(1, runs[0]["physical_section_count"]),
                    raw_subvoxel_section_count=raw_sections,
                    raw_normal_face_section_count=normal_face_sections,
                    paired_surface_section_count=paired_surface_sections,
                    independent_profile_section_count=profile_sections,
                    positive_height_section_count=positive_sections,
                    boundary_uncertainty_m=uncertainty,
                    undercoverage_risk=risk,
                    status=status,
                    grid_phase=dict(
                        status=(
                            "GRID_PHASE_STABLE"
                            if max(maximum, endpoint_deviation)
                            <= config["boundary"]["line_inlier_m"]
                            else "GRID_PHASE_SENSITIVE"
                        ),
                        median_m=estimate,
                        mad_m=mad,
                        max_deviation_m=maximum,
                        endpoint_max_deviation_m=endpoint_deviation,
                    ),
                    phase_runs=runs,
                    steel_edge_certified=False,
                    opening_role_reanchor=reanchors[name],
                    source_vessel_id=proposal["vessel_hypothesis_id"],
                    raw_source_scope=proposal.get(
                        "raw_support_scope", "FROZEN_VESSEL_OWNED_POINTS"
                    ),
                )
        if after[0] >= after[2] or after[1] >= after[3]:
            return dict(
                status="UNRESOLVED", error_code="DEGENERATE_FINAL_BOUNDARY", sides=sides
            )
        try:
            best, radii, corner_geometry = guarded_intersections(
                sides, guard_bounds, config
            )
        except ValueError as error:
            return dict(status="UNRESOLVED", error_code=str(error), sides=sides)
        uncertainty = np.linalg.norm(radii, axis=1).tolist()
        # Coverage is an uncertainty envelope, never a replacement steel line.
        original = polygon(guard_bounds)
        cover = polygon(
            [
                *np.minimum(original.min(0), (best - radii).min(0)),
                *np.maximum(original.max(0), (best + radii).max(0)),
            ]
        )
        return dict(
            status="HATCH_REVIEW_CANDIDATE",
            boundary_uv=best.tolist(),
            boundary_xy=(best @ axes).tolist(),
            best_estimate_boundary_xy=(best @ axes).tolist(),
            review_coverage_boundary_xy=(cover @ axes).tolist(),
            display_boundary_xy=(best @ axes).tolist(),
            sides=sides,
            per_side_uncertainty={
                k: s["boundary_uncertainty_m"] for k, s in sides.items()
            },
            undercoverage_risk=any(s["undercoverage_risk"] for s in sides.values()),
            corners=[
                dict(
                    corner_xy=xy,
                    uncertainty_m=u,
                    source="FINAL_FOUR_LINE_INTERSECTION",
                    steel_edge_certified=False,
                )
                for xy, u in zip((best @ axes).tolist(), uncertainty)
            ],
            corner_geometry=corner_geometry,
            vessel_local_axes=np.asarray(axes).tolist(),
            geometry_role="REVIEW_NOT_CERTIFIED_STEEL",
        )

    def _inner_role_anchors(self, proposal, box, cell):
        """Reject wrong-polarity shell lines before counting geometric support.

        A long profile line alone cannot turn an outward drop into a hatch rim.
        A replacement requires same-vessel Raw3D+BEV and signed raw support in
        the existing number of independent physical sections. Coordinates and
        scene identity are never supplied as hints.
        """
        config = self.config
        minimum = config["boundary"]["profile_min_sections"]
        support = config["roi"]["min_enclosure_ratio"]
        output = {}
        for name, axis, position, sign in SIDE_SPECS:
            with self._timed_side(name, "ANCHOR_ROLE"):
                span = (box[1 - axis], box[3 - axis])
                edges = physical_sections(span, cell, config)
                spans = list(zip(edges[:-1], edges[1:]))
                band = (
                    config["roi"]["support_search_m"]
                    + max(
                        config["roi"]["boundary_search_m"],
                        config["roi"]["support_search_m"] / 2,
                    )
                    + config["boundary"]["side_probe_m"]
                )
                query_box = [0.0, 0.0, 0.0, 0.0]
                query_box[axis], query_box[axis + 2] = (
                    box[position] - band,
                    box[position] + band,
                )
                query_box[1 - axis], query_box[3 - axis] = span
                points = self.spatial.query(query_box)
                prior = [
                    self._height_measurement(
                        points, axis, box[position], s, sign, config
                    )
                    for s in spans
                ]
                prior_positive = sum(s["transition"] for s in prior)
                state = dict(
                    status="UNCHANGED_ANCHOR",
                    rho=box[position],
                    prior_positive_sections=prior_positive,
                )
                output[name] = state
                opening = proposal.get("opening_evidence", {})
                has_low_context = opening.get("status") == "OPENING_LOWER"
                prior_face = dict(
                    normal_axis=axis,
                    a_xy=(
                        [box[position], span[0]]
                        if axis == 0
                        else [span[0], box[position]]
                    ),
                    b_xy=(
                        [box[position], span[1]]
                        if axis == 0
                        else [span[1], box[position]]
                    ),
                )
                prior_adjacent = (
                    not has_low_context
                    or sum(
                        r["supported"]
                        for r in core_adjacency(
                            points, prior_face, box, spans, sign, config, opening
                        )
                    )
                    >= minimum
                )
                if prior_positive >= minimum and prior_adjacent:
                    prior_raw = [
                        self._raw_section(points, axis, box[position], s, sign)
                        for s in spans
                    ]
                    from .raw_mode_consensus import resolve_section_modes

                    prior_raw = resolve_section_modes(
                        prior_raw, spans, config, self.budget
                    )
                    unique_raw = sum(
                        r["status"] == "UNIQUE_SIGNED_RAW_STEP"
                        and r["height"]["transition"]
                        and r["continuous"][3]
                        for r in prior_raw
                    )
                    state["prior_unique_raw_sections"] = unique_raw
                    # Height polarity can be clear while two step positions
                    # remain ambiguous. Only unique Raw geometry may skip
                    # independent structural/normal-face resolution.
                    if unique_raw >= minimum:
                        continue
                candidates = proposal.get("measured_side_candidates", {}).get(name, [])
                measured = []
                candidate_heights = {}
                investigated = (
                    proposal.get("raw_probe", {}).get("sides", {}).get(name, {})
                )
                ring = proposal.get("deck_ring", {}).get("sides", {}).get(name, {})
                if (
                    investigated.get("status") == "STRONG_UNIQUE_EDGE"
                    and ring.get("raw3d_support")
                    and ring.get("bev_support")
                ):
                    anchor = investigated.get("rho", box[position])
                    middle = (box[axis] + box[axis + 2]) / 2
                    if (
                        abs(anchor - box[position]) <= config["roi"]["support_search_m"]
                        and (anchor - middle) * sign < 0
                    ):
                        from .raw_mode_consensus import resolve_section_modes, supported

                        raw = resolve_section_modes(
                            [
                                self._raw_section(points, axis, anchor, s, sign)
                                for s in spans
                            ],
                            spans,
                            config,
                            self.budget,
                        )
                        rows = [
                            dict(along=sum(s) / 2, rho=r["continuous"][0])
                            for s, r in zip(spans, raw)
                            if supported(r)
                        ]
                        rho, fit = self._fit(rows, anchor)
                        if fit["status"] == "CONSENSUS":
                            rho += fit["slope"] * (
                                (span[0] + span[1]) / 2 - fit["reference_along_m"]
                            )
                            line = dict(
                                normal_axis=axis,
                                a_xy=(
                                    [
                                        rho + fit["slope"] * (span[0] - sum(span) / 2),
                                        span[0],
                                    ]
                                    if axis == 0
                                    else [
                                        span[0],
                                        rho + fit["slope"] * (span[0] - sum(span) / 2),
                                    ]
                                ),
                                b_xy=(
                                    [
                                        rho + fit["slope"] * (span[1] - sum(span) / 2),
                                        span[1],
                                    ]
                                    if axis == 0
                                    else [
                                        span[1],
                                        rho + fit["slope"] * (span[1] - sum(span) / 2),
                                    ]
                                ),
                            )
                            if (
                                not has_low_context
                                or sum(
                                    r["supported"]
                                    for r in core_adjacency(
                                        points, line, box, spans, sign, config, opening
                                    )
                                )
                                >= minimum
                            ):
                                measured.append(
                                    dict(
                                        rho=rho,
                                        support_section_count=len(rows),
                                        fit=fit,
                                        provider_ids=ring.get(
                                            "supporting_provider_ids", []
                                        ),
                                        measurement_source="RAW_PROBE_ANCHOR_REMEASURED_IN_FINAL_PRECISION",
                                    )
                                )
                for candidate in candidates:
                    if self.budget is not None:
                        self.budget.check()
                    anchor = candidate["rho"]
                    # A replacement belongs to this side's half of the opening.
                    middle = (box[axis] + box[axis + 2]) / 2
                    if (anchor - middle) * sign >= 0 or abs(
                        anchor - box[position]
                    ) > config["roi"]["support_search_m"]:
                        continue
                    if (
                        min(candidate["primary_raw3d"], candidate["primary_bev"])
                        < support
                    ):
                        continue
                    heights = [
                        self._height_measurement(points, axis, anchor, s, sign, config)
                        for s in spans
                    ]
                    candidate_heights[anchor] = heights
                    raw = [
                        self._raw_section(points, axis, anchor, s, sign) for s in spans
                    ]
                    from .raw_mode_consensus import resolve_section_modes

                    raw = resolve_section_modes(raw, spans, config, self.budget)
                    unique = [
                        s
                        for s in raw
                        if s["status"] == "UNIQUE_SIGNED_RAW_STEP"
                        and s["height"]["transition"]
                        and s["continuous"][3]
                    ]
                    if len(unique) < minimum:
                        continue
                    fit_rows = [
                        dict(along=(span[0] + span[1]) / 2, rho=record["continuous"][0])
                        for span, record in zip(spans, raw)
                        if record in unique
                    ]
                    rho, fit = self._fit(fit_rows, anchor)
                    if fit["status"] != "CONSENSUS":
                        continue
                    rho += fit["slope"] * (
                        (span[0] + span[1]) / 2 - fit["reference_along_m"]
                    )
                    if has_low_context:
                        ends = [
                            rho + fit["slope"] * (along - (span[0] + span[1]) / 2)
                            for along in span
                        ]
                        fitted_face = dict(
                            normal_axis=axis,
                            a_xy=(
                                [ends[0], span[0]] if axis == 0 else [span[0], ends[0]]
                            ),
                            b_xy=(
                                [ends[1], span[1]] if axis == 0 else [span[1], ends[1]]
                            ),
                        )
                        if (
                            sum(
                                r["supported"]
                                for r in core_adjacency(
                                    points,
                                    fitted_face,
                                    box,
                                    spans,
                                    sign,
                                    config,
                                    opening,
                                )
                            )
                            < minimum
                        ):
                            continue
                    measured.append(
                        dict(
                            rho=rho,
                            support_section_count=len(unique),
                            provider_ids=candidate["provider_ids"],
                            primary_raw3d=candidate["primary_raw3d"],
                            primary_bev=candidate["primary_bev"],
                            fit=fit,
                        )
                    )
                if not measured:
                    face = self._normal_face_reanchor(
                        proposal,
                        name,
                        axis,
                        box,
                        spans,
                        sign,
                        candidates,
                        candidate_heights,
                        points,
                    )
                    if face is not None:
                        if "unresolved_face_candidates" in face:
                            state.update(
                                status="MULTIMODE_BOUNDARY_CONFLICT",
                                raw_normal_face_modes=face[
                                    "unresolved_face_candidates"
                                ],
                            )
                            continue
                        state.update(
                            status="CORROBORATED_INNER_STEP",
                            rho=face_position(face, (span[0] + span[1]) / 2),
                            raw_normal_face=public_face(face),
                            _raw_normal_face=face,
                        )
                        continue
                    state["status"] = "NO_CORROBORATED_INNER_REPLACEMENT"
                    continue
                positions = np.array([s["rho"] for s in measured])
                if np.ptp(positions) > config["boundary"]["corner_join_m"]:
                    state.update(
                        status="MULTIMODE_BOUNDARY_CONFLICT", measured_modes=measured
                    )
                    continue
                state.update(
                    status="CORROBORATED_INNER_STEP",
                    rho=float(np.median(positions)),
                    measured_modes=measured,
                )
        return output

    def _normal_face_reanchor(
        self,
        proposal,
        name,
        axis,
        box,
        spans,
        sign,
        candidates,
        candidate_heights,
        points,
    ):
        """Use a unique observed 3D face when mixed returns veto paired profiles."""
        opening = proposal.get("opening_evidence", {})
        lower_context = opening.get("status") == "OPENING_LOWER"
        local_profiles = (
            self.measurement_context.local_deck_profiles
            if self.measurement_context is not None
            else None
        )
        if not lower_context and local_profiles is None:
            return None
        minimum = self.config["boundary"]["profile_min_sections"]
        support = self.config["roi"]["min_enclosure_ratio"]
        span = (box[1 - axis], box[3 - axis])
        qualified = []
        face_lines = None
        for candidate in candidates:
            anchor = candidate["rho"]
            middle = (box[axis] + box[axis + 2]) / 2
            position = 0 if name in ("U0", "V0") else 2
            position += axis
            if (anchor - middle) * sign >= 0 or abs(
                anchor - box[position]
            ) > self.config["roi"]["support_search_m"]:
                continue
            if min(candidate["primary_raw3d"], candidate["primary_bev"]) < support:
                continue
            heights = candidate_heights.get(anchor, [])
            usable = sum(
                h["transition"]
                or (
                    h.get("z_mad_m") is not None
                    and h["signed_contrast_m"] - sum(h["z_mad_m"])
                    >= self.config["roi"]["opening_drop_m"]
                )
                for h in heights
            )
            standard_support = lower_context and usable >= minimum
            if not standard_support and local_profiles is None:
                continue
            if not standard_support:
                providers = {
                    p["provider_id"]: p
                    for p in self.measurement_context.local.structural_providers()
                }
                if not any(
                    "RAW3D_VERIFIED_VERTICAL_FACE"
                    in providers.get(pid, {}).get("evidence_types", [])
                    for pid in candidate["provider_ids"]
                ):
                    continue
                # Cheap local Raw profiles precede any new normal queries.
                # Unexplained nominal frames must not trigger costly geometry.
                evidence = local_profiles.evidence(axis, anchor, span, sign)
                if sum(e["supported"] for e in evidence) < minimum:
                    continue
            # This query is lazy and cached per refiner, never per grid phase.
            if self.face_normals is None:
                from ship_perception.r1_static.raw_surface_normals import (
                    LocalSurfaceNormals,
                )

                self.face_normals = (
                    self.measurement_context.local_surface_normals
                    if self.measurement_context is not None
                    else None
                )
                if self.face_normals is None:
                    self.face_normals = (
                        BudgetedLocalSurfaceNormals(
                            self.xyz, self.config, self.budget, self.profiler
                        )
                        if self.measurement_context is not None
                        else LocalSurfaceNormals(self.xyz, self.config)
                    )
                    if self.measurement_context is not None:
                        self.measurement_context.local_surface_normals = (
                            self.face_normals
                        )
            if face_lines is None:
                key = (axis, box[position], tuple(span), tuple(spans))
                faces = (
                    self.measurement_context.face_evidence
                    if self.measurement_context is not None
                    else {}
                )
                if key not in faces:
                    if self.budget is not None:
                        self.budget.check()
                    self.profiler["normal_query_count"] = (
                        self.profiler.get("normal_query_count", 0) + 1
                    )
                    before_normal = int(np.count_nonzero(self.face_normals.computed))
                    faces[key] = measured_faces(
                        self.face_normals, axis, box[position], span, spans, self.config
                    )
                    if not isinstance(self.face_normals, BudgetedLocalSurfaceNormals):
                        self.profiler["normal_points_evaluated"] = self.profiler.get(
                            "normal_points_evaluated", 0
                        ) + max(
                            0,
                            int(np.count_nonzero(self.face_normals.computed))
                            - before_normal,
                        )
                else:
                    self.profiler["normal_face_cache_hit"] = (
                        self.profiler.get("normal_face_cache_hit", 0) + 1
                    )
                face_lines = copy.deepcopy(faces[key])
            for face in face_lines:
                if (
                    abs(face_position(face, (span[0] + span[1]) / 2) - anchor)
                    > self.config["boundary"]["corner_join_m"]
                ):
                    continue
                adjacency = []
                if lower_context:
                    adjacency = core_adjacency(
                        points, face, box, spans, sign, self.config, opening
                    )
                    if sum(r["supported"] for r in adjacency) < minimum:
                        continue
                standard_sections = (
                    sum(
                        raw_face_section(points, face, axis, s, sign, self.config)
                        is not None
                        for s in spans
                    )
                    if standard_support
                    else 0
                )
                if standard_sections < minimum:
                    if local_profiles is None:
                        continue
                    context = self.measurement_context.local.measurement_context
                    ids = face["point_indexes"]
                    owned = (
                        context.sources[0].point_indexes
                        if context is not None
                        else self.measurement_context.raw_point_indexes
                    )
                    primary_ids = ids[
                        np.isin(self.measurement_context.raw_point_indexes[ids], owned)
                    ]
                    # point_indexes are sorted; face.points retain fit order.
                    # Index the original Raw array to preserve their association.
                    face["_primary_normal_points"] = self.xyz[primary_ids]
                    face["local_deck_profile_support"] = True
                    face["local_deck_profile_reference"] = dict(
                        rho=face_position(face, sum(span) / 2), span=list(span)
                    )
                    if (
                        sum(
                            raw_face_section(
                                points, face, axis, s, sign, self.config, local_profiles
                            )
                            is not None
                            for s in spans
                        )
                        < minimum
                    ):
                        continue
                face.update(
                    core_adjacency=adjacency,
                    provider_ids=candidate["provider_ids"],
                    primary_raw3d=candidate["primary_raw3d"],
                    primary_bev=candidate["primary_bev"],
                )
                if not any(same_face(face, prior) for prior in qualified):
                    qualified.append(face)
        # A different observed face remains an unresolved physical alternative.
        if len(qualified) > 1:
            return dict(unresolved_face_candidates=[public_face(f) for f in qualified])
        return qualified[0] if qualified else None

    def _raw_section(self, roi, axis, anchor, span, sign, measured_face=None):
        """Search the physical ROI with the existing raw midpoint primitive.

        The fine raster only suggests a band. It cannot veto sparse raw pairs.
        Distinct measured modes are retained and never averaged together.
        """
        config = self.config
        key = (
            axis,
            anchor,
            span[0],
            span[1],
            sign,
            (
                None
                if measured_face is None
                else (
                    measured_face["point_index_sha256"],
                    json.dumps(
                        measured_face.get("local_deck_profile_reference"),
                        sort_keys=True,
                    ),
                )
            ),
            self.observation_key,
        )
        if key not in self.raw_section_cache:
            result = (
                None
                if measured_face is None
                else raw_face_section(
                    roi,
                    measured_face,
                    axis,
                    span,
                    sign,
                    config,
                    (
                        self.measurement_context.local_deck_profiles
                        if self.measurement_context is not None
                        else None
                    ),
                )
            )
            if result is None:
                result = self._search_raw_section(roi, axis, anchor, span, sign)
            # Eviction only repeats measurements. It never changes a gate or
            # removes a candidate. Cache scalars/diagnostics, not point arrays.
            if len(self.raw_section_cache) >= 8192:
                self.raw_section_cache.pop(next(iter(self.raw_section_cache)))
            self.raw_section_cache[key] = result
        else:
            self.profiler["raw_profile_cache_hit"] = (
                self.profiler.get("raw_profile_cache_hit", 0) + 1
            )
        return self.raw_section_cache[key]

    def _search_raw_section(self, roi, axis, anchor, span, sign):
        config = self.config
        search = config["roi"]["boundary_search_m"]
        band = search + config["boundary"]["side_probe_m"]
        # One original-point extraction, followed by one normal-coordinate sort.
        profile = (
            self.measurement_context.side_measurements.profile(axis, anchor, span)
            if self.measurement_context is not None
            else None
        )
        selected = (
            profile.points
            if profile is not None
            else roi[
                (abs(roi[:, axis] - anchor) <= band)
                & (roi[:, 1 - axis] >= span[0])
                & (roi[:, 1 - axis] <= span[1])
            ]
        )
        if (
            not len(selected)
            or np.ptp(selected[:, 2]) < config["boundary"]["profile_min_drop_m"]
        ):
            return dict(status="INSUFFICIENT_RAW_SUPPORT", modes=[])
        if profile is None:
            profile = RawSectionProfile(selected, axis, tuple(span), config)
            self.profiler["raw_profile_build_count"] = (
                self.profiler.get("raw_profile_build_count", 0) + 1
            )
        nominated = profile.edge_modes(anchor, sign)
        self.profiler["raw_edge_mode_count"] = self.profiler.get(
            "raw_edge_mode_count", 0
        ) + len(nominated)
        modes = []
        for mode in nominated:
            if self.budget is not None:
                self.budget.check()
            # Several distinct modes are retained. Only a small number of exact
            # paired-return refinements are needed for each measured change.
            for rho in dict.fromkeys((mode["center_rho"], mode["peak_rho"])):
                width = config["geometry"]["candidate_voxel_m"]
                points = profile.window(rho - width, rho + width)
                self.profiler["continuous_refinement_count"] = (
                    self.profiler.get("continuous_refinement_count", 0) + 1
                )
                raw, count = _raw_edge_rho(points, axis, rho, span, sign, config)
                if raw is not None and profile.height(raw, sign)["transition"]:
                    if not any(
                        abs(raw - old) <= config["geometry"]["plane_inlier_m"]
                        for old in modes
                    ):
                        modes.append(float(raw))
                    break
        groups = []
        for rho in sorted(modes):
            if groups and rho - groups[-1][0] <= config["boundary"]["corner_join_m"]:
                groups[-1].append(rho)
            else:
                groups.append([rho])
        measured = [float(np.median(group)) for group in groups]
        if len(measured) != 1:
            candidates = []
            for mode_rho in measured:
                width = config["geometry"]["candidate_voxel_m"]
                continuous = _continuous_rho(
                    profile.window(mode_rho - width, mode_rho + width),
                    self.observations,
                    axis,
                    mode_rho,
                    span,
                    sign,
                    config,
                )
                height = profile.height(continuous[0], sign)
                if (
                    continuous[1] <= config["boundary"]["corner_join_m"]
                    and continuous[3]
                    and height["transition"]
                ):
                    candidates.append(
                        dict(
                            status="UNIQUE_SIGNED_RAW_STEP",
                            rho=mode_rho,
                            modes=[mode_rho],
                            continuous=list(continuous),
                            height=height,
                            profile_method="EXACT_RAW_MODE_ALTERNATIVE",
                        )
                    )
            return dict(
                status=(
                    "MULTIMODE_BOUNDARY_CONFLICT"
                    if measured
                    else "INSUFFICIENT_RAW_SUPPORT"
                ),
                modes=measured,
                mode_candidates=candidates,
            )
        width = config["geometry"]["candidate_voxel_m"]
        continuous = _continuous_rho(
            profile.window(measured[0] - width, measured[0] + width),
            self.observations,
            axis,
            measured[0],
            span,
            sign,
            config,
        )
        if continuous[1] > config["boundary"]["corner_join_m"]:
            return dict(
                status="MULTIMODE_BOUNDARY_CONFLICT", modes=[measured[0], continuous[0]]
            )
        return dict(
            status="UNIQUE_SIGNED_RAW_STEP",
            rho=measured[0],
            modes=measured,
            continuous=list(continuous),
            height=profile.height(continuous[0], sign),
            profile_method="SORTED_RAW_PROFILE_WITH_PAIRED_MODE_REFINEMENT",
        )

    def _search_raw_section_v1(self, roi, axis, anchor, span, sign):
        config = self.config
        search = config["roi"]["boundary_search_m"]
        selected = roi[
            (abs(roi[:, axis] - anchor) <= search + config["boundary"]["side_probe_m"])
            & (roi[:, 1 - axis] >= span[0])
            & (roi[:, 1 - axis] <= span[1])
        ]
        if (
            not len(selected)
            or np.ptp(selected[:, 2]) < config["boundary"]["profile_min_drop_m"]
        ):
            return dict(status="INSUFFICIENT_RAW_SUPPORT", modes=[])
        modes = []
        for rho in np.arange(
            anchor - search,
            anchor + search + config["boundary"]["profile_step_m"] / 2,
            config["boundary"]["profile_step_m"],
        ):
            raw, count = _raw_edge_rho(selected, axis, float(rho), span, sign, config)
            if raw is None:
                continue
            support = robust_height_support(selected, axis, raw, span, sign, config)
            if not support["transition"]:
                continue
            if not any(
                abs(raw - old) <= config["geometry"]["plane_inlier_m"] for old in modes
            ):
                modes.append(float(raw))
        groups = []
        for rho in sorted(modes):
            if groups and rho - groups[-1][0] <= config["boundary"]["corner_join_m"]:
                groups[-1].append(rho)
            else:
                groups.append([rho])
        measured = [float(np.median(group)) for group in groups]
        if len(measured) != 1:
            return dict(
                status=(
                    "MULTIMODE_BOUNDARY_CONFLICT"
                    if measured
                    else "INSUFFICIENT_RAW_SUPPORT"
                ),
                modes=measured,
            )
        continuous = _continuous_rho(
            selected, self.observations, axis, measured[0], span, sign, config
        )
        if continuous[1] > config["boundary"]["corner_join_m"]:
            return dict(
                status="MULTIMODE_BOUNDARY_CONFLICT", modes=[measured[0], continuous[0]]
            )
        return dict(
            status="UNIQUE_SIGNED_RAW_STEP",
            rho=measured[0],
            modes=measured,
            continuous=list(continuous),
            height=robust_height_support(
                selected, axis, continuous[0], span, sign, config
            ),
        )

    def _height_measurement(self, points, axis, rho, span, sign, config):
        key = (axis, float(rho), tuple(span), sign)
        if key not in self.height_cache:
            self.height_cache[key] = robust_height_support(
                points, axis, rho, span, sign, config
            )
        return self.height_cache[key]

    def _continuous_measurement(
        self, points, observations, axis, rho, span, sign, config
    ):
        key = (axis, float(rho), tuple(span), sign, self.observation_key)
        if key not in self.continuous_cache:
            self.continuous_cache[key] = _continuous_rho(
                points, observations, axis, rho, span, sign, config
            )
        return self.continuous_cache[key]

    def _fit(self, rows, fallback):
        from .consensus import SpatialEvidenceConsensus

        return SpatialEvidenceConsensus(self.config).fit(rows, fallback)
