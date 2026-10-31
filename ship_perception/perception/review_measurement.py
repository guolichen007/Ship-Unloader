"""Independent primary-Raw normal-face measurements, not steel certification."""

import numpy as np
from .measurement import BudgetedLocalSurfaceNormals
from .sections import physical_sections
from .raw_face_refinement import measured_faces
from .raw_mode_consensus import supported


def qualified_raw_pair_gain(row, config):
    """Raw owns its physical qualification; a provider flag alone is no gain."""
    sections = row.get("sections", [])
    return bool(
        row.get("status") == "STRONG_UNIQUE_EDGE"
        and row.get("measurement_source") == "RAW_PAIRED_PROFILE"
        and len(sections) >= config["boundary"]["profile_min_sections"]
        and all(supported(s.get("section_record", {})) for s in sections)
    )


def raw_profile_line(row, axis, span):
    """Normal form and measured intervals use the fitted line's own tangent.

    Section centres are fit observations; only the actual inlier section
    intervals can support extrapolation toward a review corner. The nominal
    full-side query span is not itself a measured interval.
    """
    fit = row["fit"]
    slope = fit.get("slope", 0.0)
    reference = fit.get("reference_along_m", sum(span) / 2)
    normal = np.array([1.0, -slope]) if axis == 0 else np.array([-slope, 1.0])
    length = np.linalg.norm(normal)
    normal /= length
    rho = (row["rho"] - slope * reference) / length
    tangent = np.array([-normal[1], normal[0]])
    excluded = set(fit.get("outlier_section_indexes", []))
    intervals = []
    for k, section in enumerate(row["sections"]):
        if k in excluded:
            continue
        along = np.asarray(
            section.get("section_span_m", [section["along"], section["along"]]), float
        )
        coordinate = row["rho"] + slope * (along - reference)
        points = np.c_[coordinate, along] if axis == 0 else np.c_[along, coordinate]
        intervals.append(sorted((points @ tangent).tolist()))
    return dict(
        normal_uv=normal.tolist(),
        rho=float(rho),
        support_span=[min(i[0] for i in intervals), max(i[1] for i in intervals)],
        measured_intervals=intervals,
        support_parameter_contract="DOT_ACTUAL_INLIER_SECTION_ENDPOINTS_WITH_LINE_TANGENT",
    )


def raised_opposed_end_gain(region, name, linked_provider_ids):
    """A raised interior needs independently observed, distinct end enclosures.

    This corroborates an already measured end face. It does not create an
    opening, make lateral cargo ridges into rims, or certify steel. A local
    face inside a low multi-lobe basin cannot authorize truncating that basin.
    """
    evidence = region.identity_evidence
    if (
        not name.startswith("U")
        or evidence.get("nomination", {}).get("polarity") != "RAISED"
    ):
        return False
    sides = evidence.get("sides", {})
    ends = [
        set(sides.get(n, {}).get("verified_face_provider_ids", []))
        for n in ("U0", "U1")
    ]
    return bool(
        all(ends)
        and not ends[0].intersection(ends[1])
        and set(linked_provider_ids).intersection(ends[int(name[-1])])
    )


def raw_face_measurements(
    measurement, region, name, axis, position, span, candidates, config
):
    primary = getattr(measurement.local, "primary_cache", measurement.local)
    providers = {p["provider_id"]: p for p in primary.structural_providers()}
    # A nomination need not already have passed the normal estimator. The
    # estimator below independently qualifies the actual complete Raw face;
    # requiring a previously verified face would prevent that investigation.
    nominated = [
        c
        for c in candidates
        if min(c.get("primary_raw3d", 0), c.get("primary_bev", 0))
        >= config["roi"]["min_enclosure_ratio"]
    ]
    if not nominated:
        return []
    normals = getattr(measurement, "review_primary_normals", None)
    if normals is None:
        normals = BudgetedLocalSurfaceNormals(
            primary.xyz, config, measurement.budget, measurement.counters
        )
        measurement.review_primary_normals = normals
    edges = physical_sections(span, config["geometry"]["refine_voxel_m"], config)
    faces = measured_faces(
        normals,
        axis,
        region.conservative_bounds[position],
        span,
        list(zip(edges[:-1], edges[1:])),
        config,
    )
    output = []
    center = (
        region.conservative_bounds[axis] + region.conservative_bounds[axis + 2]
    ) / 2
    for face in faces:
        a, b = np.asarray(face["a_xy"]), np.asarray(face["b_xy"])
        direction = (b - a) / np.linalg.norm(b - a)
        normal = np.array([direction[1], -direction[0]])
        if normal[axis] < 0:
            normal = -normal
        rho = float(normal @ a)
        along = (span[0] + span[1]) / 2
        axis_rho = float(
            a[axis]
            + (along - a[1 - axis]) * (b[axis] - a[axis]) / (b[1 - axis] - a[1 - axis])
        )
        if (axis_rho - center) * (1 if name.endswith("0") else -1) >= 0:
            continue
        linked = [
            c
            for c in nominated
            if abs(c["rho"] - axis_rho)
            <= config["geometry"]["coarse_voxel_m"] / 2
            + config["boundary"]["line_inlier_m"]
        ]
        if not linked:
            continue
        d = np.array([-normal[1], normal[0]])
        support = face["points"][:, :2] @ d
        raw_ids = primary.vessel.point_indexes[np.asarray(face["point_indexes"])]
        linked_ids = sorted({pid for c in linked for pid in c.get("provider_ids", [])})
        end_gain = raised_opposed_end_gain(region, name, linked_ids)
        intervals = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            observed = face["points"][
                (face["points"][:, 1 - axis] >= lo)
                & (face["points"][:, 1 - axis] <= hi)
            ]
            if len(observed) >= config["boundary"]["line_min_points"]:
                coordinate = observed[:, :2] @ d
                intervals.append([float(coordinate.min()), float(coordinate.max())])
        output.append(
            dict(
                mode_id="RAWFACE:" + face["point_index_sha256"],
                source="MEASURED_RAW",
                state="MEASURED_WEAK",
                rho=rho,
                normal_uv=normal.tolist(),
                support_span=[float(support.min()), float(support.max())],
                measured_intervals=intervals,
                uncertainty_m=max(
                    config["geometry"]["plane_inlier_m"], face["residual_p95_m"]
                ),
                independent_raw_sections=sum(
                    n >= config["boundary"]["line_min_points"]
                    for n in face["section_point_counts"]
                ),
                physical_role_improved=True,
                independent_opening_role_gain=end_gain,
                physical_role_gain_basis=(
                    "RAISED_INTERIOR_WITH_DISTINCT_OPPOSED_VERIFIED_END_ENCLOSURES"
                    if end_gain
                    else "LOCAL_RAW_FACE_OPENING_ROLE_UNRESOLVED"
                ),
                physical_role="ASSIGNED_OPENING_SIDE_RAW_FACE_NOT_CERTIFIED_STEEL",
                raw_support_ids=["RAWFACE:" + face["point_index_sha256"]],
                source_point_indexes=raw_ids.tolist(),
                nominated_provider_ids=sorted(
                    {pid for c in linked for pid in c.get("provider_ids", [])}
                ),
                section_point_counts=face["section_point_counts"],
            )
        )
    return output
