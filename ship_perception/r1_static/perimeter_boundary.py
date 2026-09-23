"""Observed profile/face refinement with each perimeter side's own deck plane."""
import numpy as np
from scipy.spatial import cKDTree

from .boundary_refinement import (_face_normal_supported, _fit_line, _intersect,
                                  _profile, _raw3)


def refine_perimeter_boundaries(points, seed, segments, segment_decks, config, tree=None):
    """Never complete an unobserved side or a scan-edge opening."""
    if tree is None:
        tree = cKDTree(points[:, :2])
    boundary = config["boundary"]
    spacing = max(boundary["profile_step_m"],
                  min(boundary["max_support_gap_m"] * .8,
                      config["geometry"]["coarse_voxel_m"] / 2))
    edges, profiles = [], []
    for segment in segments:
        deck, plane, _ = segment_decks[segment.segment_id]
        if deck["status"] != "RESOLVED" or plane is None:
            continue
        start = np.asarray(segment.rough_start, dtype=float)
        tangent = np.asarray(segment.tangent, dtype=float)
        inward = -np.asarray(segment.outward_normal, dtype=float)
        positions = np.arange(spacing / 2, segment.length_m, spacing)
        rows = [_profile(points, tree, start + tangent * position, tangent, inward,
                         plane, config, "%s-p%d" % (segment.segment_id, index))
                for index, position in enumerate(positions)]
        profiles.extend(rows)
        breaks = np.asarray([row["break_position"] for row in rows
                             if row["valid"]], dtype=float).reshape(-1, 2)
        faces = np.asarray([row["face_position"] for row in rows
                            if row["face_position"] is not None], dtype=float).reshape(-1, 2)
        break_line = _fit_line(breaks, config)
        face_line = _fit_line(faces, config)
        if face_line is not None and not _face_normal_supported(points, face_line, plane, config):
            face_line = None
        if break_line is not None and face_line is not None:
            separation = np.linalg.norm((break_line["a_xy"] + break_line["b_xy"] -
                                         face_line["a_xy"] - face_line["b_xy"]) / 2)
            if separation <= max(boundary["line_inlier_m"], config["roi"]["support_band_m"]):
                chosen = _fit_line(np.vstack((breaks, faces)), config) or break_line
                evidence = "OBSERVED_PROFILE_BREAK+OBSERVED_3D_FACE"
            else:
                chosen, evidence = break_line, "OBSERVED_PROFILE_BREAK"
        elif break_line is not None:
            chosen, evidence = break_line, "OBSERVED_PROFILE_BREAK"
        elif face_line is not None:
            chosen, evidence = face_line, "OBSERVED_3D_FACE"
        else:
            continue
        edges.append(dict(edge_id=segment.segment_id, segment_id=segment.segment_id,
                          a_raw=_raw3(chosen["a_xy"], plane),
                          b_raw=_raw3(chosen["b_xy"], plane),
                          evidence_type=evidence, support_count=chosen["support_count"],
                          observed_support_length=chosen["support_length"],
                          fit_residual_p50_m=chosen["residual_p50"],
                          fit_residual_p95_m=chosen["residual_p95"],
                          visibility="VISIBLE", uncertainty_m=chosen["normal_uncertainty"],
                          coverage=min(1.0, chosen["support_length"] / segment.length_m),
                          reason="RAW_3D_OBSERVED_SUPPORT"))
    polygon = None
    by_id = {edge["segment_id"]: edge for edge in edges}
    if len(segments) >= 3 and len(edges) == len(segments):
        corners = []
        for previous, current in zip(segments, segments[1:] + segments[:1]):
            first, second = by_id[previous.segment_id], by_id[current.segment_id]
            intersection = _intersect(first, second)
            if intersection is None:
                break
            endpoints = [np.asarray(first[side][:2]) for side in ("a_raw", "b_raw")]
            endpoints += [np.asarray(second[side][:2]) for side in ("a_raw", "b_raw")]
            if max(min(np.linalg.norm(intersection - point) for point in endpoints[:2]),
                   min(np.linalg.norm(intersection - point) for point in endpoints[2:])) > boundary["corner_join_m"]:
                break
            first_plane = segment_decks[previous.segment_id][1]
            second_plane = segment_decks[current.segment_id][1]
            z_first = _raw3(intersection, first_plane)[2]
            z_second = _raw3(intersection, second_plane)[2]
            if abs(z_first - z_second) > config["roi"]["support_band_m"]:
                break
            corners.append([float(intersection[0]), float(intersection[1]),
                            float((z_first + z_second) / 2)])
        if len(corners) == len(segments):
            polygon = corners
    status = ("COMPLETE_OBSERVED" if polygon is not None and
              seed.fov_status == "FULLY_OBSERVED" else
              "PARTIAL" if edges else "UNRESOLVED")
    if status != "COMPLETE_OBSERVED":
        polygon = None
    x0, y0, x1, y1 = seed.bbox_xy
    center_xy = np.array(((x0 + x1) / 2, (y0 + y1) / 2))
    planes = [segment_decks[segment.segment_id][1] for segment in segments
              if segment_decks[segment.segment_id][1] is not None]
    if planes:
        center_z = float(np.median([_raw3(center_xy, plane)[2] for plane in planes]))
    else:
        inside = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
                  (points[:, 1] >= y0) & (points[:, 1] <= y1))
        center_z = float(np.median(points[inside, 2])) if inside.any() else float(np.median(points[:, 2]))
    return dict(status=status, boundaries=edges, polygon_raw=polygon,
                center_raw=[float(center_xy[0]), float(center_xy[1]), center_z],
                center_source="COARSE_OPENING_SEED", profiles=profiles,
                fov_status=seed.fov_status)
