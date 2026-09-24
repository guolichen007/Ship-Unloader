"""P1.5 synthetic adversarial scenes for the plane-consensus forensics.

Three required counter-examples, all diagnostic-only:

* Case A — a smooth full-cargo slope crossing many segments must be a stable
  plane family but must NOT be classified as steel on coplanarity alone.
* Case B — cargo overflowing a coaming (interior -> boundary -> exterior, no
  narrow strip) must stay ROLE_AMBIGUOUS_OVERFLOW / PARTIAL, never a steel edge.
* Case C — rolling / pitching / yawing / translating the whole scene must not
  change family membership or role relationship (rotation-covariant n·p+d).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .plane_consensus import centered_offset, normalize_normal, run_consensus
from .run import REPO, _atomic_json

STEEL_ROLES = ("BROAD_PERIMETER_SUPPORT", "NARROW_BOUNDARY_STRIP")


def _points_on_plane(normal, offset, x0, x1, y0, y1, spacing, noise, rng):
    normal = normalize_normal(normal)
    xs = np.arange(x0, x1, spacing)
    ys = np.arange(y0, y1, spacing)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    z = -(normal[0] * xy[:, 0] + normal[1] * xy[:, 1] + offset) / normal[2]
    points = np.column_stack((xy, z))
    if noise:
        points = points + rng.normal(0.0, noise, points.shape)
    return points


def _assemble(patches, node_bboxes):
    rng = np.random.default_rng(2026)
    point_parts, candidates = [], []
    cursor = 0
    for patch in patches:
        points = _points_on_plane(patch["normal"], patch["offset"], patch["x0"], patch["x1"],
                                  patch["y0"], patch["y1"], patch["spacing"],
                                  patch.get("noise", 0.0), rng)
        point_parts.append(points)
        raw_ids = np.arange(cursor, cursor + len(points), dtype=np.int64)
        cursor += len(points)
        candidates.append(dict(
            scene_id="synthetic", node_id=patch["node_id"], segment_id=patch["segment_id"],
            candidate_index=patch["candidate_index"], normal=normalize_normal(patch["normal"]),
            offset=float(patch["offset"]),
            support_count=int(len(raw_ids)), outward_width=patch["width"],
            along_span=patch["span"], along_coverage=patch.get("coverage", 1.0),
            raw_ids=raw_ids,
        ))
    points = np.vstack(point_parts) if point_parts else np.empty((0, 3))
    centroid = points.mean(axis=0) if len(points) else np.zeros(3)
    for candidate in candidates:
        candidate["d_centered"] = centered_offset(candidate["offset"], candidate["normal"], centroid)
    return points.astype(np.float32), candidates, node_bboxes


def build_case_a():
    """Smooth cargo slope across several segments plus a real deck ring."""
    node_bboxes = {"n0": (0.0, 0.0, 10.0, 8.0)}
    patches = []
    # Deck candidates on four sides of the node (boundary ring support).
    deck_sides = [(-3.0, 13.0, -3.0, 0.0), (-3.0, 13.0, 8.0, 11.0),
                  (-3.0, 0.0, 0.0, 8.0), (10.0, 13.0, 0.0, 8.0)]
    for index, (x0, x1, y0, y1) in enumerate(deck_sides):
        patches.append(dict(normal=(0, 0, 1), offset=0.0, x0=x0, x1=x1, y0=y0, y1=y1,
                            spacing=0.5, node_id="n0", segment_id="s_deck_%d" % index,
                            candidate_index=0, width=3.0, span=10.0))
    # One smooth tilted cargo slope crossing interior -> boundary -> exterior.
    patches.append(dict(normal=(0.18, 0.08, 0.98), offset=0.2, x0=-2.0, x1=12.0,
                        y0=-2.0, y1=10.0, spacing=0.5, node_id="n0", segment_id="s_slope",
                        candidate_index=0, width=4.0, span=14.0))
    # The same slope normal appears as a candidate on two other segments, so it
    # clusters as a stable cross-segment family.
    for extra in ("s_slope_2", "s_slope_3"):
        patches.append(dict(normal=(0.18, 0.08, 0.98), offset=0.2, x0=-1.0, x1=11.0,
                            y0=-1.0, y1=9.0, spacing=0.5, node_id="n0", segment_id=extra,
                            candidate_index=0, width=4.0, span=12.0))
    return _assemble(patches, node_bboxes)


def build_case_b():
    """Cargo overflowing a coaming: continuous interior->boundary->exterior."""
    node_bboxes = {"n0": (0.0, 0.0, 10.0, 8.0)}
    patches = []
    # Overflow cargo: a single smooth plane spanning all three zones (interior,
    # the 4 m boundary band, and well beyond); no narrow coaming strip is
    # separately observed, so it must stay fail-closed.
    patches.append(dict(normal=(0.1, 0.0, 0.995), offset=0.1, x0=-8.0, x1=18.0,
                        y0=-6.0, y1=14.0, spacing=0.5, node_id="n0", segment_id="s_ovf",
                        candidate_index=0, width=4.5, span=26.0))
    return _assemble(patches, node_bboxes)


def build_case_c():
    """A clean deck + narrow elevated coaming strip for the invariance test."""
    node_bboxes = {"n0": (0.0, 0.0, 10.0, 8.0)}
    patches = []
    deck_sides = [(-3.0, 13.0, -3.0, 0.0), (-3.0, 13.0, 8.0, 11.0),
                  (-3.0, 0.0, 0.0, 8.0), (10.0, 13.0, 0.0, 8.0)]
    for index, (x0, x1, y0, y1) in enumerate(deck_sides):
        patches.append(dict(normal=(0, 0, 1), offset=0.0, x0=x0, x1=x1, y0=y0, y1=y1,
                            spacing=0.5, node_id="n0", segment_id="s_deck_%d" % index,
                            candidate_index=0, width=3.0, span=10.0))
    # Narrow coaming strip hugging the node boundary (elevated, thin).
    strips = [(-0.6, 10.6, -0.6, 0.0), (-0.6, 10.6, 8.0, 8.6),
              (-0.6, 0.0, 0.0, 8.0), (10.0, 10.6, 0.0, 8.0)]
    for index, (x0, x1, y0, y1) in enumerate(strips):
        patches.append(dict(normal=(0, 0, 1), offset=-0.6, x0=x0, x1=x1, y0=y0, y1=y1,
                            spacing=0.25, node_id="n0", segment_id="s_coam_%d" % index,
                            candidate_index=0, width=0.6, span=10.0))
    return _assemble(patches, node_bboxes)


def _rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return rz @ ry @ rx


def apply_se3(points, candidates, node_bboxes, rotation, translation):
    points = np.asarray(points, dtype=float) @ rotation.T + translation
    transformed = []
    for candidate in candidates:
        normal = rotation @ np.asarray(candidate["normal"], dtype=float)
        offset = float(candidate["offset"] - normal @ translation)
        transformed.append(dict(candidate, normal=normal, offset=offset,
                                raw_ids=candidate["raw_ids"]))
    centroid = points.mean(axis=0)
    for candidate in transformed:
        candidate["d_centered"] = centered_offset(candidate["offset"], candidate["normal"], centroid)
    transformed_boxes = {}
    for node_id, bbox in node_bboxes.items():
        from .plane_consensus import _corners_of
        corners = _corners_of(bbox)
        corners3 = np.column_stack((corners, np.zeros(len(corners))))
        rotated = corners3 @ rotation.T + translation
        transformed_boxes[node_id] = [list(point) for point in rotated[:, :2]]
    return points.astype(np.float32), transformed, transformed_boxes


def _family_signature(families):
    """Membership partition + role, used for rotation invariance comparison.

    The representative normal is frame-dependent and deliberately excluded:
    invariance is checked on membership and role relationship, not raw normals.
    """
    signature = []
    for family in families:
        signature.append((family["role_hypothesis"],
                          tuple(sorted(family["segment_ids"])),
                          tuple(sorted(family["node_ids"]))))
    return sorted(signature, key=lambda row: (row[0], row[1]))


def run_synthetic_adversarial(config, angle_deg=3.0, separation_m=0.05):
    results = {}
    # Case A — smooth cargo slope is a stable family but not steel.
    points_a, candidates_a, boxes_a = build_case_a()
    families_a, _ = run_consensus(candidates_a, points_a, boxes_a, config, angle_deg, separation_m)
    slope_families = [family for family in families_a if "s_slope" in family["segment_ids"]]
    stable_slope = any(family["unique_segment_votes"] >= 3 for family in slope_families)
    slope_steel = any(family["role_hypothesis"] in STEEL_ROLES for family in slope_families)
    results["smooth_cargo_slope"] = dict(
        stable_family_detected=bool(stable_slope),
        slope_family_roles=[family["role_hypothesis"] for family in slope_families],
        not_classified_as_steel=bool(stable_slope and not slope_steel),
        pass_=bool(stable_slope and not slope_steel),
    )
    # Case B — overflow cargo stays fail-closed.
    points_b, candidates_b, boxes_b = build_case_b()
    families_b, _ = run_consensus(candidates_b, points_b, boxes_b, config, angle_deg, separation_m)
    overflow = [family for family in families_b
                if family["role_hypothesis"] == "ROLE_AMBIGUOUS_OVERFLOW"]
    steel = [family for family in families_b if family["role_hypothesis"] in STEEL_ROLES]
    results["overflow_cargo"] = dict(
        overflow_role_count=len(overflow),
        steel_role_count=len(steel),
        pass_=bool(overflow and not steel),
    )
    # Case C — roll/pitch/yaw invariance.
    points_c, candidates_c, boxes_c = build_case_c()
    families_c, _ = run_consensus(candidates_c, points_c, boxes_c, config, angle_deg, separation_m)
    rotation = _rotation_matrix(4.0, 2.0, 7.0)
    translation = np.array((3.0, -2.0, 5.0))
    points_r, candidates_r, boxes_r = apply_se3(points_c, candidates_c, boxes_c, rotation, translation)
    families_r, _ = run_consensus(candidates_r, points_r, boxes_r, config, angle_deg, separation_m)
    baseline_signature = _family_signature(families_c)
    rotated_signature = _family_signature(families_r)
    results["roll_pitch_yaw_invariance"] = dict(
        baseline_family_count=len(families_c),
        rotated_family_count=len(families_r),
        membership_and_role_invariant=bool(baseline_signature == rotated_signature),
        pass_=bool(baseline_signature == rotated_signature),
    )
    results["schema"] = "ship_perception.v15r.p15_synthetic_adversarial.1"
    results["overall"] = "ALL_PASS" if all(results[case]["pass_"] for case in
                                           ("smooth_cargo_slope", "overflow_cargo",
                                            "roll_pitch_yaw_invariance")) else "SOME_FAIL"
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO / "Ship-Unloader-Work/r1_plane_consensus/synthetic_adversarial_results.json")
    args = parser.parse_args()
    config = json.loads((REPO / "ship_perception/config/v15.json").read_text(encoding="utf-8"))
    results = run_synthetic_adversarial(config)
    _atomic_json(args.output, results)
    print(json.dumps({key: value for key, value in results.items() if key != "schema"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
