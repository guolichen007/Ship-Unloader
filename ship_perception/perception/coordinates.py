"""Explicit four-corner export in the original input frame, without new geometry."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


CORNER_INTERSECTIONS = ("U0/V0", "U1/V0", "U1/V1", "U0/V1")


def corner_coordinates(payload, source_result_sha256):
    """Project final review results into a small coordinate-only document.

    Coordinates come from the same final line intersections used by the full
    scene image. Ambiguous opening groups retain their status and unknown
    individual count; this export never splits them or synthesizes a Z value.
    """
    if payload["schema"] != "ship_perception.hatch_once.2":
        raise ValueError("UNSUPPORTED_CORNER_EXPORT_SOURCE_SCHEMA")
    result = payload["geometry_only"]
    rows = []
    for hatch in result["hatches"]:
        polygon = np.asarray(hatch["boundary_xy"], dtype=float)
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            raise ValueError("INVALID_FOUR_CORNER_COORDINATES")
        if "best_estimate_boundary_xy" in hatch and not np.array_equal(
            polygon, np.asarray(hatch["best_estimate_boundary_xy"])
        ):
            raise ValueError("FINAL_BOUNDARY_COORDINATE_MISMATCH")
        evidence = hatch.get("corners", [])
        corners = []
        for index, (point, intersection) in enumerate(
            zip(polygon, CORNER_INTERSECTIONS)
        ):
            corners.append(
                dict(
                    corner_id="C" + str(index),
                    side_intersection=intersection,
                    x_m=float(point[0]),
                    y_m=float(point[1]),
                    uncertainty_m=(
                        evidence[index].get("uncertainty_m")
                        if len(evidence) == 4
                        else None
                    ),
                )
            )
        rows.append(
            dict(
                hatch_id=hatch["hatch_id"],
                vessel_hypothesis_id=hatch["vessel_hypothesis_id"],
                status=hatch["status"],
                candidate_type=hatch.get("candidate_type", "HATCH_REVIEW_CANDIDATE"),
                individual_hatch_count=hatch.get("individual_hatch_count"),
                geometry_role=hatch["geometry_role"],
                candidate_blockers=list(hatch.get("candidate_blockers", [])),
                undercoverage_risk=hatch["undercoverage_risk"],
                boundary_xy_m=polygon.tolist(),
                corners=corners,
            )
        )
    return dict(
        schema="ship_perception.hatch_corners.1",
        source_result_schema=payload["schema"],
        source_result_sha256=source_result_sha256,
        software_git_sha=payload["software_git_sha"],
        software_source_identity=copy.deepcopy(payload["software_source_identity"]),
        config_sha256=payload["config_sha256"],
        input=copy.deepcopy(payload["input"]),
        coordinate_frame=payload["input"]["frame_id"],
        coordinate_system="ORIGINAL_INPUT_XY",
        coordinate_dimensions=["x", "y"],
        units="m",
        output_scope="FULL_SCENE_ALL_FINAL_REVIEW_CANDIDATES",
        corner_order=list(CORNER_INTERSECTIONS),
        polygon_closure="IMPLICIT_LAST_TO_FIRST",
        steel_edge_certified=False,
        hatch_count_status=result["hatch_count_status"],
        review_candidate_count=len(rows),
        hatches=rows,
    )


def main():
    import argparse

    from ship_perception.r1_static.run import _atomic_json

    parser = argparse.ArgumentParser(
        description="Export four XY corners from a completed hatch review JSON"
    )
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.result.resolve() == args.output.resolve():
        parser.error("OUTPUT_MUST_DIFFER_FROM_SOURCE_RESULT")
    source = args.result.read_bytes()
    document = corner_coordinates(
        json.loads(source), hashlib.sha256(source).hexdigest()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, document)
    print("HATCH_CORNERS_JSON=" + str(args.output.resolve()))
    print("REVIEW_CANDIDATE_COUNT=" + str(document["review_candidate_count"]))


if __name__ == "__main__":
    main()
