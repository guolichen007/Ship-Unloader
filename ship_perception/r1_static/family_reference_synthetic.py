"""S4-R1 synthetic adversarial scenes for deterministic family consensus.

Cases A/B/C reuse the P1.5 builders; Case D adds an irregular cargo heap of
multiple low-residual local patches with drifting plane parameters, freezing
the P1 conclusion that low residual does not imply steel.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .family_consensus import (deterministic_families, normalize_normal,
                               role_from_zones, zone_counts)
from .plane_consensus import _corners_of, _distance_to_polygon, _point_in_polygon
from .plane_consensus_synthetic import (_rotation_matrix, apply_se3, build_case_a,
                                        build_case_b, build_case_c)
from .run import REPO, _atomic_json

REFERENCE_ROLE = "BROAD_PERIMETER_SUPPORT"


def _polygon(bbox):
    return _corners_of(bbox)


def _signed_distance(points_xy, polygon):
    points_xy = np.asarray(points_xy, dtype=float)
    inside = _point_in_polygon(points_xy, polygon)
    distance = _distance_to_polygon(points_xy, polygon)
    return np.where(inside, -distance, distance)


def run_synthetic_consensus(points, candidates, node_polygons, config):
    strict_angle = config["frame"]["normal_refine_deg"]
    strict_sep = config["geometry"]["plane_inlier_m"]
    families, family_of = deterministic_families(candidates, strict_angle, strict_sep,
                                                 2 * strict_angle, 2 * strict_sep)
    inner_band = config["roi"]["boundary_search_m"]
    outer_band = config["roi"]["support_search_m"]
    widths = []
    for family in families:
        if family["members"]:
            widths.append(float(np.median([candidates[i]["outward_width"]
                                          for i in family["members"]])))
    narrow_threshold = 0.5 * float(np.median(widths)) if widths else 0.0
    records = []
    for family_index, family in enumerate(families):
        members = family["members"]
        interior_deep = boundary_inner = boundary_outer = exterior_remote = 0
        for index in members:
            polygon = node_polygons.get(candidates[index]["node_id"])
            if polygon is None:
                continue
            signed = _signed_distance(points[candidates[index]["raw_ids"], :2], polygon)
            counts = zone_counts(signed, inner_band, outer_band)
            interior_deep += counts[0]
            boundary_inner += counts[1]
            boundary_outer += counts[2]
            exterior_remote += counts[3]
        median_width = float(np.median([candidates[i]["outward_width"] for i in members])) \
            if members else 0.0
        role = role_from_zones(interior_deep, boundary_inner, boundary_outer, exterior_remote,
                               median_width, narrow_threshold)
        records.append(dict(
            family_id="f%03d" % family_index,
            unique_segment_votes=len({candidates[i]["segment_id"] for i in members}),
            role_hypothesis=role,
            median_transverse_width=median_width,
            segment_ids=sorted({candidates[i]["segment_id"] for i in members}),
            member_candidate_indexes=[int(i) for i in members],
        ))
    return records


def build_case_d():
    """Irregular cargo heap: several low-residual patches with drifting planes."""
    rng = np.random.default_rng(2026)
    node_bboxes = {"n0": (0.0, 0.0, 10.0, 8.0)}
    patches = []
    drifts = [(0.0, 0.0, 0.0), (0.05, 0.02, 0.06), (0.08, 0.05, 0.10),
              (0.03, 0.07, 0.04), (0.06, 0.04, 0.12)]
    for index, (dx, dy, dz) in enumerate(drifts):
        normal = normalize_normal((dx, dy, 1.0 - dz))
        # Low-residual plane: points exactly on the plane (tiny noise).
        patches.append(dict(normal=normal, offset=0.0, x0=-1.0, x1=11.0, y0=-1.0, y1=9.0,
                            spacing=0.5, node_id="n0", segment_id="s_heap_%d" % index,
                            candidate_index=0, width=2.0, span=10.0, noise=0.005))
    from .plane_consensus_synthetic import _assemble
    return _assemble(patches, node_bboxes)


def run_synthetic_regression(config):
    results = {}
    # Case A: smooth cargo slope stable but not steel.
    points_a, candidates_a, boxes_a = build_case_a()
    polygons_a = {node_id: _polygon(box) for node_id, box in boxes_a.items()}
    families_a = run_synthetic_consensus(points_a, candidates_a, polygons_a, config)
    slope = [family for family in families_a if any("s_slope" in s for s in family["segment_ids"])]
    stable = any(family["unique_segment_votes"] >= 3 for family in slope)
    not_steel = all(family["role_hypothesis"] != REFERENCE_ROLE for family in slope)
    results["smooth_cargo_slope"] = dict(
        stable_family=bool(stable), roles=[family["role_hypothesis"] for family in slope],
        not_reference_capable=bool(stable and not_steel), pass_=bool(stable and not_steel))

    # Case B: overflow fail-closed.
    points_b, candidates_b, boxes_b = build_case_b()
    polygons_b = {node_id: _polygon(box) for node_id, box in boxes_b.items()}
    families_b = run_synthetic_consensus(points_b, candidates_b, polygons_b, config)
    overflow = [family for family in families_b
                if family["role_hypothesis"] == "ROLE_AMBIGUOUS_OVERFLOW"]
    steel = [family for family in families_b if family["role_hypothesis"] == REFERENCE_ROLE]
    results["overflow_cargo"] = dict(
        overflow_count=len(overflow), steel_count=len(steel),
        pass_=bool(overflow and not steel))

    # Case C: roll/pitch/yaw invariance of membership + role.
    points_c, candidates_c, boxes_c = build_case_c()
    polygons_c = {node_id: _polygon(box) for node_id, box in boxes_c.items()}
    families_c = run_synthetic_consensus(points_c, candidates_c, polygons_c, config)
    rotation = _rotation_matrix(4.0, 2.0, 7.0)
    translation = np.array((3.0, -2.0, 5.0))
    points_r, candidates_r, boxes_r = apply_se3(points_c, candidates_c, boxes_c,
                                                rotation, translation)
    polygons_r = {node_id: _polygon(box) for node_id, box in boxes_r.items()}
    families_r = run_synthetic_consensus(points_r, candidates_r, polygons_r, config)
    before = sorted((f["role_hypothesis"], tuple(sorted(f["segment_ids"])))
                    for f in families_c)
    after = sorted((f["role_hypothesis"], tuple(sorted(f["segment_ids"])))
                   for f in families_r)
    results["roll_pitch_yaw_invariance"] = dict(
        baseline_family_count=len(families_c), rotated_family_count=len(families_r),
        membership_and_role_invariant=bool(before == after), pass_=bool(before == after))

    # Case D: irregular cargo — no reference-capable strict family.
    points_d, candidates_d, boxes_d = build_case_d()
    polygons_d = {node_id: _polygon(box) for node_id, box in boxes_d.items()}
    families_d = run_synthetic_consensus(points_d, candidates_d, polygons_d, config)
    reference = [family for family in families_d
                 if family["role_hypothesis"] == REFERENCE_ROLE
                 and family["unique_segment_votes"] >= 2]
    results["irregular_cargo"] = dict(
        family_count=len(families_d), reference_capable_family_count=len(reference),
        pass_=bool(not reference))

    results["schema"] = "ship_perception.v15r.s4r1_synthetic_regression.1"
    results["overall"] = "ALL_PASS" if all(results[case]["pass_"] for case in
                                           ("smooth_cargo_slope", "overflow_cargo",
                                            "roll_pitch_yaw_invariance", "irregular_cargo")) \
        else "SOME_FAIL"
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO / "Ship-Unloader-Work/r1_family_reference/synthetic_regression.json")
    args = parser.parse_args()
    config = json.loads((REPO / "ship_perception/config/v15.json").read_text(encoding="utf-8"))
    results = run_synthetic_regression(config)
    _atomic_json(args.output, results)
    print(json.dumps({key: value for key, value in results.items() if key != "schema"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
