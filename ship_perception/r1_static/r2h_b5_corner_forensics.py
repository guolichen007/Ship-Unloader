"""Read-only B5 corner and lineage forensics; never changes B4 geometry.

This deliberately reports measured *intersection candidates*, not certified
Hatch steel corners. Height-level persistence and the opening quadrant remain
diagnostics until the B5 joint solver and independent inner-edge Golden exist.
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

from ship_perception.tools.v15_pcd import decode

from .batch import NORMAL6
from .opening_semantics import _box, _iou
from .raw_topview_evidence import build_topview
from .run import DEFAULT_CONFIG, REPO, _atomic_json, resolve_config


def _world_box(bounds, axes):
    u0, v0, u1, v1 = bounds
    return np.asarray([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]) @ axes


def _overlap_fraction(first, second):
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    area = max(1e-9, (second[2] - second[0]) * (second[3] - second[1]))
    return float(overlap / area)


def audit_lineage(b4, b2, config):
    """Compare old independent IoU links with a one-to-one forensic assignment."""
    vessels = {
        row["vessel_hypothesis_id"]: row for row in b2["scene"]["vessel_hypotheses"]
    }
    contexts = {row["vessel_hypothesis_id"]: row for row in b2["per_vessel"]}
    old_links = [row.get("hatch_hypothesis_id") for row in b4["rectangles"]]
    duplicates = sorted(
        {value for value in old_links if value and old_links.count(value) > 1}
    )
    candidates = []
    for hatch in b4["rectangles"]:
        vessel_id = hatch.get("vessel_hypothesis_id")
        if vessel_id not in contexts:
            candidates.append([])
            continue
        axes = np.asarray(hatch["axes"])
        rect_box = _box(hatch["polygon_before"], axes)
        rect_center = np.mean(np.asarray(hatch["polygon_before"]), axis=0)
        local_axes = np.asarray(vessels[vessel_id]["local_axes"])
        rows = []
        for proposal in contexts[vessel_id]["hatch_hypotheses"]:
            bounds = proposal.get("bounds_axial")
            if bounds is None:
                continue
            corners = _world_box(bounds, local_axes)
            candidate_box = _box(corners, axes)
            center = corners.mean(axis=0)
            rows.append(
                dict(
                    proposal_id=proposal["hatch_hypothesis_id"],
                    state=proposal["state"],
                    iou=_iou(rect_box, candidate_box),
                    center_distance_m=float(np.linalg.norm(center - rect_center)),
                    opening_overlap_fraction=_overlap_fraction(rect_box, candidate_box),
                    axis_compatibility=float(abs(np.dot(axes[0], local_axes[0]))),
                )
            )
        rows.sort(
            key=lambda row: (-row["iou"], row["center_distance_m"], row["proposal_id"])
        )
        candidates.append(rows)
    minimum = config["roi"]["min_enclosure_ratio"]
    shared = {}
    for rows in candidates:
        for row in rows:
            if row["iou"] >= minimum:
                shared[row["proposal_id"]] = shared.get(row["proposal_id"], 0) + 1
    composite = sorted(key for key, value in shared.items() if value > 1)
    # Forensic exact assignment is small (one ship's visible Hatch set). An
    # unmatched slot is always allowed and composite proposals are ineligible.
    options = [
        [None]
        + [
            row
            for row in rows
            if row["iou"] >= minimum
            and row["proposal_id"] not in composite
            and row["state"] in ("HATCH_PROPOSAL", "PARTIAL_HATCH")
        ]
        for rows in candidates
    ]
    best_key = None
    best = None
    for choice in itertools.product(*options):
        ids = [row["proposal_id"] for row in choice if row is not None]
        if len(ids) != len(set(ids)):
            continue
        key = (
            len(ids),
            sum(row["iou"] for row in choice if row is not None),
            -sum(row["center_distance_m"] for row in choice if row is not None),
        )
        if best_key is None or key > best_key:
            best_key, best = key, choice
    return dict(
        duplicated_b4_links=duplicates,
        composite_b2_proposals=composite,
        independent_current_links=[
            dict(
                hatch_id=hatch["hatch_id"],
                proposal_id=hatch.get("hatch_hypothesis_id"),
                iou=hatch.get("hatch_hypothesis_iou"),
            )
            for hatch in b4["rectangles"]
        ],
        one_to_one_forensic_assignment=[
            dict(
                hatch_id=hatch["hatch_id"],
                proposal_id=row["proposal_id"] if row else None,
                status="MATCHED" if row else "UNMATCHED_OPENING",
            )
            for hatch, row in zip(b4["rectangles"], best or [])
        ],
        pair_candidates={
            hatch["hatch_id"]: rows[:8]
            for hatch, rows in zip(b4["rectangles"], candidates)
        },
        algorithm_changed=False,
    )


def audit_final_side_sources(b4, b2):
    """Expose B4 sides whose selected lines include unresolved SceneSupport."""
    status = {
        row["vessel_hypothesis_id"]: row["classification_status"]
        for row in b2["scene"]["vessel_hypotheses"]
    }
    result = []
    for hatch in b4["rectangles"]:
        for side_id, side in hatch["sides"].items():
            sources = side.get("selected_source_vessel_ids", [])
            if not sources:
                continue
            unresolved = [
                source for source in sources if status.get(source) == "UNRESOLVED"
            ]
            result.append(
                dict(
                    hatch_id=hatch["hatch_id"],
                    side_id=side_id,
                    selected_source_vessel_ids=sources,
                    selected_source_statuses={
                        source: status.get(source) for source in sources
                    },
                    unresolved_source_ids=unresolved,
                    unresolved_only=bool(unresolved)
                    and len(unresolved) == len(sources),
                    used_as_formal_b4_side=bool(unresolved),
                )
            )
    return result


def _observations(b2, vessel_id, axes, roi):
    """Project all B2 measured lines in a bounded opening ROI, no old-side rho gate."""
    observed = []
    statuses = {
        row["vessel_hypothesis_id"]: row["classification_status"]
        for row in b2["scene"]["vessel_hypotheses"]
    }
    for context in b2["per_vessel"]:
        source_id = context["vessel_hypothesis_id"]
        status = statuses.get(source_id, "UNRESOLVED")
        if status == "VESSEL_HYPOTHESIS" and source_id != vessel_id:
            continue
        for row in context["provider_evidence"]:
            if row["geometry_type"] != "AXIAL_LINE_INTERVAL" or row[
                "provider_type"
            ] not in ("RAW3D_STRUCTURAL_PROVIDER", "BEV_RECTILINEAR_PROVIDER"):
                continue
            endpoints = np.asarray(row["source_geometry"]["endpoints_xy"], dtype=float)
            if endpoints.shape != (2, 2):
                continue
            local = endpoints @ axes.T
            normal_axis = int(row["rho_or_region"]["normal_axis"])
            rho = float(np.mean(local[:, normal_axis]))
            along = sorted(local[:, 1 - normal_axis].tolist())
            if not (
                roi[normal_axis] <= rho <= roi[normal_axis + 2]
                and along[1] >= roi[1 - normal_axis]
                and along[0] <= roi[3 - normal_axis]
            ):
                continue
            observed.append(
                dict(
                    normal_axis=normal_axis,
                    rho=rho,
                    along=along,
                    provider_type=row["provider_type"],
                    provider_id=row["provider_id"],
                    source_vessel_id=source_id,
                    source_support_ids=row["source_support_ids"],
                    source_scope=(
                        "SAME_VESSEL" if source_id == vessel_id else "CONTEXT_ONLY"
                    ),
                    coverage=float(row["coverage"]),
                )
            )
    return observed


def _side_modes(observations, normal_axis, span, band, config):
    """Survey measured rho modes across an opening band without B4's ±4 m anchor."""
    join = config["boundary"]["corner_join_m"]
    min_line = config["boundary"]["line_min_length_m"]
    candidates = []
    source = [
        row
        for row in observations
        if row["normal_axis"] == normal_axis
        and band[0] <= row["rho"] <= band[1]
        and min(span[1], row["along"][1]) - max(span[0], row["along"][0]) >= min_line
    ]
    for rho in sorted({round(row["rho"], 2) for row in source}):
        nearby = [row for row in source if abs(row["rho"] - rho) <= join]
        levels = {}
        for scope in ("SAME_VESSEL", "CONTEXT_ONLY"):
            for provider in ("RAW3D_STRUCTURAL_PROVIDER", "BEV_RECTILINEAR_PROVIDER"):
                levels[(scope, provider)] = max(
                    (
                        row["coverage"]
                        * min(
                            1.0,
                            max(
                                0.0,
                                min(span[1], row["along"][1])
                                - max(span[0], row["along"][0]),
                            )
                            / max(span[1] - span[0], 1e-9),
                        )
                        for row in nearby
                        if row["source_scope"] == scope
                        and row["provider_type"] == provider
                    ),
                    default=0.0,
                )
        candidates.append(
            dict(
                rho=float(rho),
                normal_axis=normal_axis,
                primary_raw3d=levels[("SAME_VESSEL", "RAW3D_STRUCTURAL_PROVIDER")],
                primary_bev=levels[("SAME_VESSEL", "BEV_RECTILINEAR_PROVIDER")],
                context_raw3d=levels[("CONTEXT_ONLY", "RAW3D_STRUCTURAL_PROVIDER")],
                context_bev=levels[("CONTEXT_ONLY", "BEV_RECTILINEAR_PROVIDER")],
                provider_ids=[row["provider_id"] for row in nearby],
                source_support_ids=sorted(
                    {sid for row in nearby for sid in row["source_support_ids"]}
                ),
                source_vessel_ids=sorted({row["source_vessel_id"] for row in nearby}),
                provider_scope=(
                    "MIXED_PRIMARY_AND_CONTEXT"
                    if min(
                        levels[("SAME_VESSEL", "RAW3D_STRUCTURAL_PROVIDER")],
                        levels[("SAME_VESSEL", "BEV_RECTILINEAR_PROVIDER")],
                    )
                    > 0
                    and max(
                        levels[("CONTEXT_ONLY", "RAW3D_STRUCTURAL_PROVIDER")],
                        levels[("CONTEXT_ONLY", "BEV_RECTILINEAR_PROVIDER")],
                    )
                    > 0
                    else (
                        "PRIMARY_CURRENT_VESSEL"
                        if min(
                            levels[("SAME_VESSEL", "RAW3D_STRUCTURAL_PROVIDER")],
                            levels[("SAME_VESSEL", "BEV_RECTILINEAR_PROVIDER")],
                        )
                        > 0
                        else "CONTEXT_ONLY"
                    )
                ),
            )
        )
    # Retain physically distinct rho hypotheses; this is a display shortlist,
    # not B4's proposal dedup or a decision about an inner edge.
    candidates.sort(
        key=lambda row: (
            -min(row["primary_raw3d"], row["primary_bev"]),
            -max(row["primary_raw3d"], row["primary_bev"]),
            row["rho"],
        )
    )
    shortlist = []
    for row in candidates:
        if all(
            abs(row["rho"] - prior["rho"]) > config["boundary"]["line_inlier_m"]
            for prior in shortlist
        ):
            shortlist.append(row)
        if len(shortlist) >= 16:
            break
    return shortlist


def _quadrants(grid, u0, v0, width):
    u, v = grid.centers()
    values = []
    for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        mask = (
            (su * (u - u0) > 0)
            & (su * (u - u0) <= width)
            & (sv * (v - v0) > 0)
            & (sv * (v - v0) <= width)
            & grid.occupancy
        )
        values.append(
            dict(
                valid_cells=int(mask.sum()),
                median_z=float(np.median(grid.median[mask])) if mask.any() else None,
                density=float(np.median(grid.count[mask])) if mask.any() else None,
                vertical_span=(
                    float(np.median((grid.maximum - grid.minimum)[mask]))
                    if mask.any()
                    else None
                ),
            )
        )
    return values


def _corner(u_mode, v_mode, observations, grid, height_levels, config, inside_index):
    u0, v0 = u_mode["rho"], v_mode["rho"]
    join = config["boundary"]["corner_join_m"]

    def support(axis, rho, coordinate, provider):
        rows = [
            row
            for row in observations
            if row["normal_axis"] == axis
            and row["source_scope"] == "SAME_VESSEL"
            and row["provider_type"] == provider
            and abs(row["rho"] - rho) <= join
            and row["along"][0] - join <= coordinate <= row["along"][1] + join
        ]
        return max((row["coverage"] for row in rows), default=0.0), sorted(
            {row["provider_id"] for row in rows}
        )

    raw_u, raw_u_ids = support(0, u0, v0, "RAW3D_STRUCTURAL_PROVIDER")
    bev_u, bev_u_ids = support(0, u0, v0, "BEV_RECTILINEAR_PROVIDER")
    raw_v, raw_v_ids = support(1, v0, u0, "RAW3D_STRUCTURAL_PROVIDER")
    bev_v, bev_v_ids = support(1, v0, u0, "BEV_RECTILINEAR_PROVIDER")
    quadrants = _quadrants(grid, u0, v0, config["roi"]["support_search_m"] / 2)
    inside = quadrants[inside_index]
    neighbors = [
        quadrants[i]
        for i in range(4)
        if i != inside_index and quadrants[i]["median_z"] is not None
    ]
    other_z = (
        float(np.median([q["median_z"] for q in neighbors])) if neighbors else None
    )
    contrast = (
        abs(inside["median_z"] - other_z)
        if inside["median_z"] is not None and other_z is not None
        else None
    )
    z_values = [q["median_z"] for q in quadrants if q["median_z"] is not None]
    persistent_levels = [
        level
        for level in height_levels
        if len(z_values) >= 3
        and min(z_values) < level < max(z_values)
        and contrast is not None
        and contrast >= config["roi"]["opening_drop_m"]
    ]
    structural = min(max(raw_u, bev_u), max(raw_v, bev_v))
    outer_pattern = sum(q["valid_cells"] == 0 for q in quadrants) >= 2
    return dict(
        corner_xy=[float(u0), float(v0)],
        u_line_support_raw3d=raw_u,
        u_line_support_bev=bev_u,
        v_line_support_raw3d=raw_v,
        v_line_support_bev=bev_v,
        source_provider_ids=raw_u_ids + bev_u_ids + raw_v_ids + bev_v_ids,
        median_z_quadrants=[q["median_z"] for q in quadrants],
        density_quadrants=[q["density"] for q in quadrants],
        vertical_span_quadrants=[q["vertical_span"] for q in quadrants],
        vessel_support_quadrants=[q["valid_cells"] for q in quadrants],
        opening_quadrant=inside_index,
        local_height_contrast_m=contrast,
        height_threshold_support_levels=persistent_levels,
        height_threshold_level_count=len(persistent_levels),
        position_dispersion_m=None,
        spatial_persistence_status="NOT_MEASURED_BY_B4_OR_S3_BBOX",
        outer_hull_conflict=outer_pattern,
        role=(
            "OUTER_HULL_PATTERN"
            if outer_pattern
            else (
                "MEASURED_INTERSECTION_CANDIDATE"
                if structural > 0
                and contrast is not None
                and contrast >= config["roi"]["opening_drop_m"]
                else "HEIGHT_OR_STRUCTURE_ONLY_CANDIDATE"
            )
        ),
        structural_support=structural,
    )


def inspect_scene(scene, points, b2, b4, s3, config):
    lineage = audit_lineage(b4, b2, config)
    results = []
    for vessel_id in sorted(
        {
            row.get("vessel_hypothesis_id")
            for row in b4["rectangles"]
            if row.get("vessel_hypothesis_id")
        }
    ):
        hatches = [
            row
            for row in b4["rectangles"]
            if row.get("vessel_hypothesis_id") == vessel_id
        ]
        axes = np.asarray(hatches[0]["axes"])
        boxes = [_box(row["polygon_before"], axes) for row in hatches]
        b = [
            min(x[0] for x in boxes),
            min(x[1] for x in boxes),
            max(x[2] for x in boxes),
            max(x[3] for x in boxes),
        ]
        s3_boxes = []
        for row in s3.get("regions", []):
            sb = _box(row["polygon_xy"], axes)
            if _iou(b, sb) > 0:
                s3_boxes.append(sb)
        margin = config["roi"]["support_search_m"]
        roi = [
            min([b[0]] + [x[0] for x in s3_boxes]) - margin,
            min([b[1]] + [x[1] for x in s3_boxes]) - margin,
            max([b[2]] + [x[2] for x in s3_boxes]) + margin,
            max([b[3]] + [x[3] for x in s3_boxes]) + margin,
        ]
        observations = _observations(b2, vessel_id, axes, roi)
        grid = build_topview(points, axes, config)["coarse"]
        span_u = (b[0], b[2])
        span_v = (b[1], b[3])
        all_u = _side_modes(observations, 0, span_v, (roi[0], roi[2]), config)
        u_chain = sorted({round(x, 2) for box in boxes for x in (box[0], box[2])})
        v_mid = (b[1] + b[3]) / 2
        v0 = _side_modes(observations, 1, span_u, (roi[1], v_mid), config)
        v1 = _side_modes(observations, 1, span_u, (v_mid, roi[3]), config)
        # Preserve the old modes alongside active B2 surveys so the baseline
        # is visible even if it is not among the strongest measured lines.
        for side, rows in (("V0", v0), ("V1", v1)):
            for hatch in hatches:
                old = hatch["sides"][side]["rho_after"]
                local_modes = _side_modes(
                    observations, 1, span_u, (old - margin, old + margin), config
                )
                for local in sorted(local_modes, key=lambda row: abs(row["rho"] - old))[
                    :4
                ]:
                    if not any(
                        abs(row["rho"] - local["rho"])
                        <= config["boundary"]["line_inlier_m"]
                        for row in rows
                    ):
                        rows.append(local)
                if not any(
                    abs(row["rho"] - old) <= config["boundary"]["line_inlier_m"]
                    for row in rows
                ):
                    rows.append(
                        dict(
                            rho=float(old),
                            normal_axis=1,
                            primary_raw3d=0.0,
                            primary_bev=0.0,
                            context_raw3d=0.0,
                            context_bev=0.0,
                            provider_ids=[],
                            source_support_ids=[],
                            source_vessel_ids=[],
                            provider_scope="OLD_R2G_BASELINE_ONLY",
                        )
                    )
        # U candidates are measured independently; the old chain merely picks
        # which transverse stations form the current two-cell graph.
        u_modes = []
        for old in u_chain:
            local_modes = _side_modes(
                observations, 0, span_v, (old - margin, old + margin), config
            )
            qualifying = [
                row
                for row in local_modes
                if min(row["primary_raw3d"], row["primary_bev"])
                >= config["roi"]["min_enclosure_ratio"]
            ]
            found = (
                min(qualifying, key=lambda row: abs(row["rho"] - old))
                if qualifying
                else None
            )
            u_modes.append(
                found
                if found and abs(found["rho"] - old) <= margin
                else dict(rho=float(old), provider_scope="OLD_R2G_BASELINE_ONLY")
            )
        all_u = sorted(
            {row["rho"]: row for row in all_u + u_modes}.values(),
            key=lambda row: row["rho"],
        )
        levels = s3.get("levels_m", [])
        pair_rows = []
        for lower, upper in itertools.product(v0, v1):
            if lower["rho"] >= upper["rho"]:
                continue
            cells = []
            for index in range(len(u_modes) - 1):
                left, right = u_modes[index : index + 2]
                cells.extend(
                    [
                        _corner(left, lower, observations, grid, levels, config, 2),
                        _corner(right, lower, observations, grid, levels, config, 3),
                        _corner(right, upper, observations, grid, levels, config, 0),
                        _corner(left, upper, observations, grid, levels, config, 1),
                    ]
                )
            unique = {
                (round(c["corner_xy"][0], 2), round(c["corner_xy"][1], 2)): c
                for c in cells
            }
            corners = list(unique.values())
            measured = sum(
                c["role"] == "MEASURED_INTERSECTION_CANDIDATE" for c in corners
            )
            height = sum(c["height_threshold_level_count"] > 0 for c in corners)
            outer = sum(c["outer_hull_conflict"] for c in corners)
            score = (
                measured,
                -outer,
                height,
                round(min((c["structural_support"] for c in corners), default=0), 3),
            )
            pair_rows.append(
                dict(
                    v0=lower["rho"],
                    v1=upper["rho"],
                    diagnostic_four_corner_score=list(score),
                    measured_intersection_candidate_count=measured,
                    height_threshold_corner_count=height,
                    outer_hull_conflict_count=outer,
                    corners=corners,
                )
            )
        pair_rows.sort(
            key=lambda row: tuple(row["diagnostic_four_corner_score"]), reverse=True
        )
        best = pair_rows[0] if pair_rows else None
        tied_top_pairs = (
            sum(
                row["diagnostic_four_corner_score"]
                == best["diagnostic_four_corner_score"]
                for row in pair_rows
            )
            if best
            else 0
        )
        shared = []
        if len(u_modes) >= 3 and best:
            separator = u_modes[1]["rho"]
            shared = [
                row
                for row in best["corners"]
                if abs(row["corner_xy"][0] - separator)
                < config["boundary"]["corner_join_m"]
            ]
        results.append(
            dict(
                vessel_hypothesis_id=vessel_id,
                search_roi_axial=roi,
                search_roi_source="S3_OPENING_REGION_UNION_R2G_HATCH_ENVELOPE",
                unresolved_support_policy="CONTEXT_ONLY_NOT_FINAL_EDGE",
                observation_count=len(observations),
                u_boundary_candidates=all_u,
                u_boundary_chain=u_modes,
                v0_candidates=v0,
                v1_candidates=v1,
                pair_ranking=pair_rows,
                diagnostic_best_pair=best,
                tied_top_pair_count=tied_top_pairs,
                pair_resolution_status=(
                    "NO_PAIR"
                    if best is None
                    else (
                        "NO_FOUR_MEASURED_INTERSECTIONS"
                        if len(u_modes) == 2
                        and best["measured_intersection_candidate_count"] < 4
                        else (
                            "AMBIGUOUS_TIED_SCORE"
                            if tied_top_pairs > 1
                            else "DIAGNOSTIC_ONLY_NOT_CERTIFIED"
                        )
                    )
                ),
                shared_separator_corners=shared,
                measured_corner_count=(
                    best["measured_intersection_candidate_count"] if best else 0
                ),
                persistent_corner_count="UNVERIFIED_SPATIAL_PERSISTENCE",
                length_prior_used=False,
            )
        )
    return dict(
        schema="ship_perception.v15r.r2h_b5_corner_forensics.1",
        scene_id=scene,
        algorithm_changed=False,
        corner_diagnostic_only_current_b4=True,
        old_height_hierarchy_reused="PARTIAL_LEVELS_AND_REGION_ROI_ONLY",
        current_b4_status=[row["status"] for row in b4["rectangles"]],
        lineage=lineage,
        final_side_source_audit=audit_final_side_sources(b4, b2),
        vessels=results,
    )


def render(scene, points, b4, report, output, config):
    """Produce stage evidence pictures, not a new calibrated rectangle."""
    output.mkdir(parents=True, exist_ok=True)
    mpl_dir = output / ".mplconfig"
    mpl_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not report["vessels"]:
        return []
    vessel = report["vessels"][0]
    axes = np.asarray(
        next(
            row["axes"]
            for row in b4["rectangles"]
            if row.get("vessel_hypothesis_id") == vessel["vessel_hypothesis_id"]
        )
    )
    grid = build_topview(points, axes, config)["coarse"]
    extent = (
        grid.x0,
        grid.x0 + grid.cell_m * grid.count.shape[1],
        grid.y0,
        grid.y0 + grid.cell_m * grid.count.shape[0],
    )
    paths = []
    for label in (
        "A_median_z",
        "B_all_line_modes",
        "C_corner_candidates",
        "D_corner_graph",
        "E_before_after",
        "F_role_overlay",
    ):
        fig, ax = plt.subplots(figsize=(16, 6), dpi=130)
        ax.imshow(
            np.ma.masked_invalid(grid.median),
            extent=extent,
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            aspect="equal",
        )
        for row in b4["rectangles"]:
            if row.get("vessel_hypothesis_id") != vessel["vessel_hypothesis_id"]:
                continue
            for kind, color in (
                ("polygon_before", "white"),
                ("polygon_after", "orange"),
            ):
                xy = np.asarray(row[kind]) @ axes.T
                ax.plot(
                    *np.vstack((xy, xy[0])).T,
                    color=color,
                    linewidth=0.8 if kind == "polygon_before" else 1.5,
                )
        if label in (
            "B_all_line_modes",
            "C_corner_candidates",
            "D_corner_graph",
            "F_role_overlay",
        ):
            for item in vessel["u_boundary_candidates"][:16]:
                ax.axvline(item["rho"], color="#80d8ff", alpha=0.25, lw=0.7)
            for item in vessel["v0_candidates"][:16] + vessel["v1_candidates"][:16]:
                ax.axhline(item["rho"], color="#eebf64", alpha=0.25, lw=0.7)
        best = vessel.get("diagnostic_best_pair")
        if best and label in (
            "C_corner_candidates",
            "D_corner_graph",
            "F_role_overlay",
        ):
            for i, corner in enumerate(best["corners"]):
                x, y = corner["corner_xy"]
                color = (
                    "#34e0af"
                    if corner["role"] == "MEASURED_INTERSECTION_CANDIDATE"
                    else "#e2676b"
                )
                ax.scatter([x], [y], s=22, color=color, zorder=5)
                ax.text(x, y, f"C{i}:{corner['role'][:4]}", color="white", fontsize=6)
            if label == "D_corner_graph":
                for item in vessel["u_boundary_chain"]:
                    ax.axvline(item["rho"], color="#34e0af", lw=1.3)
                for rho in (best["v0"], best["v1"]):
                    ax.axhline(rho, color="#34e0af", lw=1.3)
        ax.set(
            xlabel="Vessel U (m)",
            ylabel="Vessel V (m)",
            title=f"{scene} | {label} | FORENSIC ONLY - no geometry change",
        )
        ax.set_xlim(vessel["search_roi_axial"][0], vessel["search_roi_axial"][2])
        ax.set_ylim(vessel["search_roi_axial"][1], vessel["search_roi_axial"][3])
        fig.tight_layout()
        path = output / (label + ".png")
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", action="append", choices=("08-01", "08-08"), required=True
    )
    parser.add_argument("--b2-root", type=Path, required=True)
    parser.add_argument("--b4-root", type=Path, required=True)
    parser.add_argument("--s3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config, _ = resolve_config(DEFAULT_CONFIG)
    manifest = json.loads(
        (REPO / "ship_perception/datasets/v15_dataset_manifest.json").read_text(
            encoding="utf8"
        )
    )
    scans = {row["scan_id"]: row for row in manifest["scans"]}
    for scene in args.scene:
        record = scans[NORMAL6[scene]]
        source = REPO / "Ship-Unloader-Data/legacy/hold_detector" / record["pcd_path"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        b2 = json.loads(
            (args.b2_root / scene / "hatch_proposals.json").read_text(encoding="utf8")
        )
        b4 = json.loads(
            (args.b4_root / scene / "opening_semantics.json").read_text(encoding="utf8")
        )
        s3 = json.loads(
            (args.s3_root / scene / "hatch_regions.json").read_text(encoding="utf8")
        )
        if (
            digest != record["pcd_sha256"]
            or digest != b2["input_sha256"]
            or digest != s3["input_sha256"]
        ):
            raise ValueError("INPUT_SHA_MISMATCH:" + scene)
        points, _ = decode(source)
        result = inspect_scene(scene, points, b2, b4, s3, config)
        destination = args.output_root / scene
        destination.mkdir(parents=True, exist_ok=True)
        result["png_outputs"] = render(scene, points, b4, result, destination, config)
        _atomic_json(destination / "corner_forensics.json", result)
        print(
            scene,
            result["lineage"]["duplicated_b4_links"],
            [
                (row["vessel_hypothesis_id"], len(row["pair_ranking"]))
                for row in result["vessels"]
            ],
            flush=True,
        )


if __name__ == "__main__":
    main()
