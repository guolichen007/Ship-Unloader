"""Independent primary-Raw normal-face measurements, not steel certification."""

import numpy as np
from .measurement import BudgetedLocalSurfaceNormals
from .sections import physical_sections
from .raw_face_refinement import measured_faces


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
        output.append(
            dict(
                mode_id="RAWFACE:" + face["point_index_sha256"],
                source="MEASURED_RAW",
                state="MEASURED_WEAK",
                rho=rho,
                normal_uv=normal.tolist(),
                support_span=[float(support.min()), float(support.max())],
                uncertainty_m=max(
                    config["geometry"]["plane_inlier_m"], face["residual_p95_m"]
                ),
                independent_raw_sections=sum(
                    n >= config["boundary"]["line_min_points"]
                    for n in face["section_point_counts"]
                ),
                physical_role_improved=True,
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
