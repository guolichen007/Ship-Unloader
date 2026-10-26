"""Measured local-normal faces resolve a mixed-return height profile, not steel role."""

import hashlib

import numpy as np

from ship_perception.r1_static.boundary_refinement import _fit_line
from ship_perception.r1_static.rectangle_refinement import _continuous_rho
from .sections import robust_height_support


def face_position(face, along):
    axis = face["normal_axis"]
    a, b = np.asarray(face["a_xy"]), np.asarray(face["b_xy"])
    return float(
        a[axis]
        + (along - a[1 - axis]) * (b[axis] - a[axis]) / (b[1 - axis] - a[1 - axis])
    )


def measured_faces(normals, axis, anchor, span, spans, config):
    """Reuse the existing robust line fit on independently qualified 3D points."""
    # Fit complete faces within the existing side search ROI. Clipping each
    # fit to one coarse line's half-cell can splice parts of two real faces.
    points, indexes, _ = normals.vertical_points(
        axis, anchor, span, band_m=config["roi"]["support_search_m"]
    )
    geometry, boundary = config["geometry"], config["boundary"]
    remaining = np.arange(len(points))
    faces = []
    while len(remaining) >= geometry["min_plane_points"]:
        line = _fit_line(points[remaining, :2], config)
        if line is None:
            break
        a, b = line["a_xy"], line["b_xy"]
        tangent = (b - a) / np.linalg.norm(b - a)
        distance = abs(np.cross(tangent, points[remaining, :2] - a))
        member = remaining[distance <= boundary["line_inlier_m"]]
        remaining = remaining[distance > boundary["line_inlier_m"]]
        observed = points[member]
        if len(member) < geometry["min_plane_points"]:
            continue
        if np.ptp(observed[:, 2]) < boundary["face_vertical_span_min_m"]:
            continue
        angle = np.degrees(np.arctan2(abs(tangent[axis]), abs(tangent[1 - axis])))
        if angle > boundary["merge_angle_deg"]:
            continue
        counts = [
            int(
                np.count_nonzero(
                    (observed[:, 1 - axis] >= lo) & (observed[:, 1 - axis] <= hi)
                )
            )
            for lo, hi in spans
        ]
        if (
            sum(n >= boundary["line_min_points"] for n in counts)
            < boundary["profile_min_sections"]
        ):
            continue
        ids = np.sort(indexes[member])
        identity = hashlib.sha256(ids.astype("<i8", copy=False).tobytes()).hexdigest()
        faces.append(
            dict(
                normal_axis=axis,
                a_xy=a.tolist(),
                b_xy=b.tolist(),
                points=observed,
                point_indexes=ids,
                point_index_sha256=identity,
                point_count=len(ids),
                section_point_counts=counts,
                residual_p95_m=line["residual_p95"],
                physical_role="UNKNOWN",
            )
        )
    return faces


def core_adjacency(xyz, face, box, spans, sign, config, opening):
    """An elevated terrace must not stand in for the deep interior's first wall.

    The reference is the proposal's measured outside-ring height, also used
    by exterior-low validation. An isolated cargo ridge returns to that low
    family on its far side and cannot authorize an inward crop.
    """
    axis = face["normal_axis"]
    probe = config["boundary"]["side_probe_m"]
    far = config["roi"]["support_search_m"] / 2
    levels = [z for z in opening.get("outside_side_median_z", []) if z is not None]
    level = None if not levels else float(np.median(levels))
    minimum = config["boundary"]["profile_min_points"]
    drop = config["roi"]["opening_drop_m"]
    rows = []
    for lo, hi in spans:
        rho = face_position(face, (lo + hi) / 2)
        local = xyz[(xyz[:, 1 - axis] >= lo) & (xyz[:, 1 - axis] <= hi)]
        normal = sign * (local[:, axis] - rho)
        values = [
            local[(normal >= 0) & (normal <= probe), 2],
            local[(normal <= -far) & (normal >= -far - probe), 2],
        ]
        if level is None or min(map(len, values)) < minimum:
            rows.append(
                dict(supported=False, reason="INSUFFICIENT_CORE_OR_FAR_SUPPORT")
            )
            continue
        inside, outside = (float(np.median(z)) for z in values)
        rows.append(
            dict(
                supported=inside < level - drop and outside >= level - drop,
                observed_ring_reference_z_m=level,
                near_inside_median_z_m=inside,
                far_outside_median_z_m=outside,
            )
        )
    return rows


def same_face(first, second):
    return first["point_index_sha256"] == second["point_index_sha256"]


def public_face(face):
    return {k: v for k, v in face.items() if k not in ("points", "point_indexes")}


def raw_face_section(xyz, face, axis, span, sign, config):
    """A raw-normal measurement is distinct from a paired-height midpoint."""
    points = face["points"]
    selected = points[
        (points[:, 1 - axis] >= span[0]) & (points[:, 1 - axis] <= span[1])
    ]
    if len(selected) < config["boundary"]["line_min_points"]:
        return None
    rho = face_position(face, (span[0] + span[1]) / 2)
    observations = [
        dict(xy=p[:2].tolist(), reference_role="RAW_NORMAL_FACE_ROLE_UNVERIFIED")
        for p in selected
    ]
    continuous = _continuous_rho(xyz, observations, axis, rho, span, sign, config)
    if abs(continuous[0] - rho) > config["boundary"]["corner_join_m"]:
        return None
    # The surface step may lie within the measured face's residual band. Keep
    # its probe position separate from the raw 3D line's geometric position.
    radius = max(config["geometry"]["plane_inlier_m"], face["residual_p95_m"])
    probes = [continuous[0], rho - radius, rho + radius] + np.arange(
        rho - radius, rho + radius, config["boundary"]["profile_step_m"]
    ).tolist()
    probes.sort(key=lambda value: abs(value - continuous[0]))
    height = None
    for position in probes:
        measured = robust_height_support(xyz, axis, position, span, sign, config)
        spread = measured.get("z_mad_m")
        dominant = (
            None if spread is None else measured["signed_contrast_m"] - sum(spread)
        )
        if measured["transition"] or (
            dominant is not None and dominant >= config["roi"]["opening_drop_m"]
        ):
            height = dict(measured, support_rho_m=float(position))
            break
    if height is None:
        return None
    height = dict(
        height,
        quartile_transition=height["transition"],
        transition=True,
        dominant_family_lower_contrast_m=dominant,
        support_basis="INDEPENDENT_RAW_NORMAL_FACE_AND_HEIGHT_FAMILY",
    )
    continuous = list(continuous)
    continuous[1] = max(continuous[1], face["residual_p95_m"])
    return dict(
        status="UNIQUE_SIGNED_RAW_STEP",
        rho=rho,
        modes=[rho],
        continuous=continuous,
        height=height,
        raw_measurement_present=True,
        raw_normal_face_count=len(selected),
        measurement_source="RAW_NORMAL_FACE",
        surface_pair_present=bool(continuous[3]),
        raw_normal_face_identity=face["point_index_sha256"],
    )
