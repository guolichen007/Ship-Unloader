"""Explicit four-corner export in the original input frame, without new geometry."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


CORNER_INTERSECTIONS = ("U0/V0", "U1/V0", "U1/V1", "U0/V1")


def boundary_dimensions(boundary_xy):
    """Measure the final polygon in XY, including slanted/nonparallel edges.

    Length follows vessel U and width follows vessel V. Neither is an XY
    bounding-box extent or a claim about an independently certified steel edge.
    """
    polygon = np.asarray(boundary_xy, dtype=float)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError("INVALID_FOUR_CORNER_COORDINATES")
    edges = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
    return dict(
        length_m=float((edges[0] + edges[2]) / 2),
        width_m=float((edges[1] + edges[3]) / 2),
        side_lengths_m={
            "V0": float(edges[0]),
            "U1": float(edges[1]),
            "V1": float(edges[2]),
            "U0": float(edges[3]),
        },
        method="MEAN_OPPOSITE_EDGE_LENGTHS",
        coordinate_system="ORIGINAL_INPUT_XY",
        geometry_role="FINAL_REVIEW_BOUNDARY_NOT_CERTIFIED_STEEL",
    )


def corner_coordinates(payload, source_result_sha256):
    """Project final review results into a small coordinate-only document.

    Coordinates come from the same final line intersections used by the full
    scene image. Ambiguous opening groups retain their status and unknown
    individual count. Z is copied only from local Raw surface measurements;
    unavailable heights remain null and never become an invented coordinate.
    """
    if payload["schema"] not in (
        "ship_perception.hatch_once.2",
        "ship_perception.hatch_once.3",
    ):
        raise ValueError("UNSUPPORTED_CORNER_EXPORT_SOURCE_SCHEMA")
    v3 = payload["schema"] == "ship_perception.hatch_once.3"
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
        heights = hatch.get("corner_heights", [])
        corners = []
        for index, (point, intersection) in enumerate(
            zip(polygon, CORNER_INTERSECTIONS)
        ):
            height = heights[index] if len(heights) == 4 else {}
            z = height.get("z_m")
            if z is not None and not np.isfinite(z):
                raise ValueError("INVALID_CORNER_HEIGHT")
            corners.append(
                dict(
                    corner_id=("P" + str(index + 1)) if v3 else "C" + str(index),
                    display_label="P" + str(index + 1),
                    side_intersection=intersection,
                    x_m=float(point[0]),
                    y_m=float(point[1]),
                    z_m=float(z) if z is not None else None,
                    z_status=height.get("z_status", "UNOBSERVED"),
                    z_uncertainty_m=height.get("z_uncertainty_m"),
                    z_definition=height.get(
                        "z_definition", "LOCAL_COAMING_OR_DECK_UPPER_SURFACE"
                    ),
                    z_measurement=copy.deepcopy(height),
                    uncertainty_m=(
                        evidence[index].get("uncertainty_m")
                        if len(evidence) == 4
                        else None
                    ),
                )
            )
            if v3:
                corner = corners[-1]
                xy = evidence[index] if len(evidence) == 4 else {}
                corner.update(
                    x=corner["x_m"],
                    y=corner["y_m"],
                    z=corner["z_m"],
                    legacy_corner_id="C" + str(index),
                    xy_state=xy.get("xy_state", "ESTIMATED"),
                    xy_source=xy.get("xy_source", "LINE_INTERSECTION"),
                    xy_uncertainty_m=xy.get("uncertainty_m"),
                    corner_state=xy.get("corner_state", "LINE_INTERSECTION"),
                    z_state=height.get("z_state", "UNOBSERVED"),
                    z_observed=height.get("z_observed"),
                    z_estimate=height.get("z_estimate"),
                    z_source=height.get("z_source"),
                    steel_certified=False,
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
                boundary_xyz_m=[[p["x_m"], p["y_m"], p["z_m"]] for p in corners],
                dimensions=boundary_dimensions(polygon),
                corners=corners,
            )
        )
        if v3:
            rows[-1].update(
                opening_identity=hatch["opening_identity"],
                topology_status=hatch["topology_status"],
                geometry_status=hatch["geometry_status"],
                certification_status=hatch["certification_status"],
                sides=copy.deepcopy(hatch["sides"]),
            )
            for key in (
                "raw_evidence_geometry",
                "constrained_review_rectangle",
                "boundary_measurement_status",
                "rectangle_constraint_conflict_count",
                "position_review_status",
                "calibration_scope",
            ):
                if key in hatch:
                    rows[-1][key] = copy.deepcopy(hatch[key])
    return dict(
        schema=(
            "ship_perception.hatch_corners.3"
            if v3
            else "ship_perception.hatch_corners.2"
        ),
        schema_version=".3" if v3 else ".2",
        producer_git_sha=payload["software_git_sha"],
        runtime_source_hash=payload["software_source_identity"][
            "runtime_source_sha256"
        ],
        config_hash=payload.get("config_hash", payload["config_sha256"]),
        input_sha256=payload.get(
            "input_sha256", payload["input"].get("semantic_sha256")
        ),
        review_mode=payload.get("review_mode", "bounded"),
        source_result_schema=payload["schema"],
        source_result_sha256=source_result_sha256,
        software_git_sha=payload["software_git_sha"],
        software_source_identity=copy.deepcopy(payload["software_source_identity"]),
        config_sha256=payload["config_sha256"],
        input=copy.deepcopy(payload["input"]),
        coordinate_frame=payload["input"]["frame_id"],
        coordinate_system="ORIGINAL_INPUT_XYZ",
        coordinate_dimensions=["x", "y", "z"],
        z_definition=(
            "HATCH_INNER_RIM_TOP_ELEVATION"
            if v3
            else "LOCAL_COAMING_OR_DECK_UPPER_SURFACE"
        ),
        unobserved_z_representation="null",
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
        description="Export four XYZ corners; unobserved Z remains null"
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
