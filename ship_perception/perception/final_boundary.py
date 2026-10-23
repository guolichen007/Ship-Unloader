"""Single four-side refiner using existing raw and multi-section primitives."""

import warnings
import copy
import json

import numpy as np
from scipy.stats import theilslopes

from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.height_rim_review import _section_transition
from ship_perception.r1_static.rectangle_refinement import (
    _continuous_rho,
    _raw_edge_rho,
)
from .sections import physical_sections, robust_height_support


SIDE_SPECS = (("U0", 0, 0, 1), ("U1", 0, 2, -1), ("V0", 1, 1, 1), ("V1", 1, 3, -1))


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
    if inward > config["boundary"]["corner_join_m"] and not supported:
        return before, "KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY", True
    return (
        estimate,
        "RAW_GEOMETRY_REVIEW" if supported else "PARTIAL_RAW_SUPPORT",
        not supported,
    )


class FinalBoundaryRefiner:
    def __init__(self, xyz, observations, config):
        self.xyz, self.observations, self.config = xyz, observations, config
        self.raw_section_cache = {}
        self.refinement_cache = {}
        self.refinement_cache_bytes = 0

    def refine(self, proposal, axes):
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
                    )
                },
            ],
            sort_keys=True,
        )
        if key in self.refinement_cache:
            return copy.deepcopy(self.refinement_cache[key][0])
        result = self._refine_uncached(proposal, axes)
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
        for name, _, position, _ in SIDE_SPECS:
            if reanchors[name]["status"] == "CORROBORATED_INNER_STEP":
                box[position] = reanchors[name]["rho"]
        points = self.xyz
        roi = points[
            (points[:, 0] >= box[0] - margin)
            & (points[:, 0] <= box[2] + margin)
            & (points[:, 1] >= box[1] - margin)
            & (points[:, 1] <= box[3] + margin)
        ]
        if not len(roi):
            return dict(status="UNRESOLVED", error_code="NO_RAW_ROI_SUPPORT", sides={})
        if len(roi) > config["geometry"]["max_candidate_points"]:
            raise ValueError("FINAL_ROI_POINT_LIMIT_EXCEEDED")
        raw_sections = {}
        for name, axis, position, sign in SIDE_SPECS:
            edges = physical_sections((box[1 - axis], box[3 - axis]), cell, config)
            for index, span in enumerate(zip(edges[:-1], edges[1:])):
                raw_sections[name, index] = self._raw_section(
                    roi, axis, box[position], span, sign
                )
        origin = roi[:, :2].min(0) - cell
        phases = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
        phase_results = {name: [] for name, *_ in SIDE_SPECS}
        for phase in phases:
            grid = R1HeightGrid.from_points(
                roi, cell, origin_xy=origin + np.array(phase) * cell
            )
            for name, axis, position, sign in SIDE_SPECS:
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
                    measured = [s for s in evidence["samples"] if begin <= s[0] <= end]
                    # A sparse 0.05 m raster may have no paired occupied rows.
                    # It must not veto an observed raw step at the proposal.
                    anchor = (
                        float(np.median(np.array(measured)[:, 1]))
                        if measured
                        else box[position]
                    )
                    rho, uncertainty, raw_count, paired = _continuous_rho(
                        roi, self.observations, axis, anchor, (begin, end), sign, config
                    )
                    direct = raw_sections[name, index]
                    if direct["status"] == "UNIQUE_SIGNED_RAW_STEP":
                        rho, uncertainty, raw_count, paired = direct["continuous"]
                    if not paired and anchor != box[position]:
                        prior_rho, prior_uncertainty, prior_count, prior_paired = (
                            _continuous_rho(
                                roi,
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
                        else robust_height_support(
                            roi, axis, rho, (begin, end), sign, config
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
                            raw_subvoxel=paired,
                            profile_observation_count=len(profiles),
                            profile_observation_ids=[
                                r["observation_id"] for r in profiles
                            ],
                            height=height,
                            raw_search=direct,
                        )
                    )
                good = [
                    s
                    for s in samples
                    if (s["raw_subvoxel"] or s["profile_observation_count"])
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
        sides, after, coverage = {}, box[:], box[:]
        for name, axis, position, sign in SIDE_SPECS:
            runs = phase_results[name]
            rhos = np.array([r["rho"] for r in runs])
            estimate = float(np.median(rhos))
            mad = float(np.median(abs(rhos - estimate)))
            maximum = float(max(abs(rhos - estimate)))
            minimum_sections = min(r["support_section_count"] for r in runs)
            raw_sections = min(
                sum(s["raw_subvoxel"] for s in r["samples"]) for r in runs
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
            rho, status, risk = guard_move(
                guard_bounds[position],
                estimate,
                sign,
                minimum_sections,
                independent and not conflict,
                config,
            )
            if minimum_sections < config["boundary"]["profile_min_sections"]:
                rho = (
                    guard_bounds[position]
                    if (box[position] - guard_bounds[position]) * sign
                    > config["boundary"]["corner_join_m"]
                    else box[position]
                )
                status, risk = "KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY", True
            if conflict:
                rho, status, risk = (
                    guard_bounds[position],
                    "MULTIMODE_BOUNDARY_CONFLICT",
                    True,
                )
            uncertainty = max(
                config["geometry"]["plane_inlier_m"],
                maximum,
                max(
                    (s["uncertainty_m"] for r in runs for s in r["samples"]),
                    default=cell,
                ),
                abs(estimate - rho),
            )
            if minimum_sections < config["boundary"]["profile_min_sections"]:
                uncertainty = max(uncertainty, config["roi"]["boundary_search_m"])
            after[position] = rho
            coverage[position] = (
                min(guard_bounds[position], rho - uncertainty)
                if sign == 1
                else max(guard_bounds[position], rho + uncertainty)
            )
            sides[name] = dict(
                rho_before_m=guard_bounds[position],
                rho_search_anchor_m=box[position],
                rho_after_m=rho,
                side_inward_move_m=(rho - guard_bounds[position]) * sign,
                support_section_count=minimum_sections,
                support_section_ratio=minimum_sections
                / max(1, runs[0]["physical_section_count"]),
                raw_subvoxel_section_count=raw_sections,
                independent_profile_section_count=profile_sections,
                positive_height_section_count=positive_sections,
                boundary_uncertainty_m=uncertainty,
                undercoverage_risk=risk,
                status=status,
                grid_phase=dict(
                    status=(
                        "GRID_PHASE_STABLE"
                        if maximum <= config["boundary"]["line_inlier_m"]
                        else "GRID_PHASE_SENSITIVE"
                    ),
                    median_m=estimate,
                    mad_m=mad,
                    max_deviation_m=maximum,
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
        # Four final constrained line intersections. The model constraint is
        # explicit; it does not create observation for an unsupported side.
        best, cover = polygon(after), polygon(coverage)
        uncertainty = [
            float(
                np.hypot(
                    sides[a]["boundary_uncertainty_m"],
                    sides[b]["boundary_uncertainty_m"],
                )
            )
            for a, b in (("U0", "V0"), ("U1", "V0"), ("U1", "V1"), ("U0", "V1"))
        ]
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
            corner_geometry=dict(
                orthogonality_deg=90.0,
                parallelism_deg=0.0,
                rectangle_constraint_applied=True,
                intersection_stable=True,
            ),
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
            span = (box[1 - axis], box[3 - axis])
            edges = physical_sections(span, cell, config)
            spans = list(zip(edges[:-1], edges[1:]))
            points = self.xyz[
                (self.xyz[:, 1 - axis] >= span[0])
                & (self.xyz[:, 1 - axis] <= span[1])
                & (
                    abs(self.xyz[:, axis] - box[position])
                    <= config["roi"]["support_search_m"]
                    + config["roi"]["boundary_search_m"]
                    + config["boundary"]["side_probe_m"]
                )
            ]
            prior = [
                robust_height_support(points, axis, box[position], s, sign, config)
                for s in spans
            ]
            prior_positive = sum(s["transition"] for s in prior)
            state = dict(
                status="UNCHANGED_ANCHOR",
                rho=box[position],
                prior_positive_sections=prior_positive,
            )
            output[name] = state
            if prior_positive >= minimum:
                continue
            candidates = proposal.get("measured_side_candidates", {}).get(name, [])
            measured = []
            for candidate in candidates:
                anchor = candidate["rho"]
                # A replacement belongs to this side's half of the opening.
                middle = (box[axis] + box[axis + 2]) / 2
                if (anchor - middle) * sign >= 0 or abs(
                    anchor - box[position]
                ) > config["roi"]["support_search_m"]:
                    continue
                if min(candidate["primary_raw3d"], candidate["primary_bev"]) < support:
                    continue
                positive = sum(
                    robust_height_support(points, axis, anchor, s, sign, config)[
                        "transition"
                    ]
                    for s in spans
                )
                if positive < minimum:
                    continue
                raw = [self._raw_section(points, axis, anchor, s, sign) for s in spans]
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
                measured.append(
                    dict(
                        rho=rho,
                        support_section_count=len(unique),
                        provider_ids=candidate["provider_ids"],
                        primary_raw3d=candidate["primary_raw3d"],
                        primary_bev=candidate["primary_bev"],
                    )
                )
            if not measured:
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

    def _raw_section(self, roi, axis, anchor, span, sign):
        """Search the physical ROI with the existing raw midpoint primitive.

        The fine raster only suggests a band. It cannot veto sparse raw pairs.
        Distinct measured modes are retained and never averaged together.
        """
        config = self.config
        key = (axis, anchor, span[0], span[1], sign)
        if key not in self.raw_section_cache:
            result = self._search_raw_section(roi, axis, anchor, span, sign)
            # Eviction only repeats measurements. It never changes a gate or
            # removes a candidate. Cache scalars/diagnostics, not point arrays.
            if len(self.raw_section_cache) >= 8192:
                self.raw_section_cache.pop(next(iter(self.raw_section_cache)))
            self.raw_section_cache[key] = result
        return self.raw_section_cache[key]

    def _search_raw_section(self, roi, axis, anchor, span, sign):
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

    def _fit(self, rows, fallback):
        if not rows:
            return fallback, dict(status="INSUFFICIENT_SUPPORT")
        x = np.array([r["along"] for r in rows])
        y = np.array([r["rho"] for r in rows])
        if len(rows) < self.config["boundary"]["profile_min_sections"]:
            return float(np.median(y)), dict(status="INSUFFICIENT_SUPPORT")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fit = theilslopes(y, x)
        residual = y - (fit[0] * x + fit[1])
        mad = float(np.median(abs(residual)))
        consistent = abs(residual) <= self.config["boundary"]["corner_join_m"]
        status = (
            "CONSENSUS"
            if (
                consistent.all()
                and abs(np.degrees(np.arctan(fit[0])))
                <= self.config["boundary"]["merge_angle_deg"]
            )
            else "MULTIMODE_BOUNDARY_CONFLICT"
        )
        return float(np.median(y)), dict(
            status=status,
            residual_mad_m=mad,
            unconstrained_slope_deg=float(np.degrees(np.arctan(fit[0]))),
            inlier_section_count=int(consistent.sum()),
        )
