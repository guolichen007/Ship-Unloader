"""Evaluate independent visible inner-edge annotations; never called by runtime."""

import argparse
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator


def evaluate(manifest, result, config):
    source = Path(__file__).resolve().parents[1]
    Draft202012Validator(
        json.loads((source / "config/hatch_golden.schema.json").read_text())
    ).validate(manifest)
    if result["input"]["provenance"].get("wire_sha256") != manifest["pcd_sha256"]:
        raise ValueError("GOLDEN_PCD_IDENTITY_MISMATCH")
    if result["input"]["frame_id"] != manifest["coordinate_frame"]:
        raise ValueError("GOLDEN_COORDINATE_FRAME_MISMATCH")
    predictions = {h["hatch_id"]: h for h in result["geometry_only"]["hatches"]}
    rows = []
    for annotation in manifest["hatches"]:
        ident = annotation["prediction_hatch_id"]
        if ident not in predictions:
            rows.append(dict(hatch_id=ident, status="MISSING_HATCH"))
            continue
        prediction = predictions[ident]
        poly = np.asarray(prediction["best_estimate_boundary_xy"], dtype=float)
        indexes = {"U0": (0, 3), "U1": (1, 2), "V0": (0, 1), "V1": (3, 2)}
        edges, corner_errors = {}, []
        for name, edge in annotation["edges"].items():
            if edge["visibility"] != "VISIBLE":
                edges[name] = dict(status="NOT_EVALUATED_" + edge["visibility"])
                continue
            normal = np.array(edge["inward_normal_xy"], dtype=float)
            length = np.linalg.norm(normal)
            if length < np.finfo(float).eps:
                raise ValueError("INVALID_GOLDEN_NORMAL")
            normal /= length
            expected = np.array([edge["start_xy"], edge["end_xy"]], dtype=float)
            start, end = poly[list(indexes[name])]
            fractions = np.linspace(0.0, 1.0, 101)
            keep = np.ones(len(fractions), bool)
            for low, high in edge["ignore_segments"]:
                if high < low:
                    raise ValueError("INVALID_IGNORE_SEGMENT")
                keep &= ~((fractions >= low) & (fractions <= high))
            if not keep.any():
                edges[name] = dict(status="NOT_EVALUATED_ALL_IGNORED")
                continue
            # Compare signed normal distance along corresponding annotated side.
            # Positive inward displacement is UNDER_COVERAGE.
            estimate = start + (end - start) * fractions[keep, None]
            target = expected[0] + (expected[1] - expected[0]) * fractions[keep, None]
            delta = (estimate - target) @ normal
            inward, outward = float(max(0.0, max(delta))), float(max(0.0, max(-delta)))
            error = float(np.percentile(abs(delta), 95))
            gate = config["evaluation"]["boundary_p95_m"]
            edges[name] = dict(
                status=(
                    "UNDER_COVERAGE"
                    if inward > gate
                    else "BOUNDARY_ERROR" if error > gate else "PASS"
                ),
                error_p95_m=error,
                undercoverage_m=inward,
                overcoverage_m=outward,
                uncertainty_m=prediction["per_side_uncertainty"][name],
            )
            for index in (0, -1):
                if keep[index]:
                    corner_errors.append(
                        float(
                            np.linalg.norm(
                                (start if index == 0 else end) - expected[index]
                            )
                        )
                    )
        corner_max = max(corner_errors, default=None)
        rows.append(
            dict(
                hatch_id=ident,
                edges=edges,
                corner_error_max_m=corner_max,
                status=(
                    "UNDER_COVERAGE"
                    if any(e["status"] == "UNDER_COVERAGE" for e in edges.values())
                    else (
                        "CORNER_ERROR"
                        if corner_max is not None
                        and corner_max > config["evaluation"]["corner_p95_m"]
                        else (
                            "REVIEW"
                            if any(e["status"] != "PASS" for e in edges.values())
                            else "PASS"
                        )
                    )
                ),
            )
        )
    return dict(
        schema="ship_perception.hatch_precision.1",
        annotation_version=manifest["annotation_version"],
        pcd_sha256=manifest["pcd_sha256"],
        reviewer=manifest["reviewer"],
        release_evidence=manifest.get("review_status") == "ADJUDICATED",
        review_status=manifest.get("review_status", "DRAFT"),
        hatches=rows,
    )


def main():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--manifest", required=True, type=Path)
    command.add_argument("--result", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    args = command.parse_args()
    from ship_perception.r1_static.run import resolve_config, _atomic_json

    config, _ = resolve_config()
    _atomic_json(
        args.output,
        evaluate(
            json.loads(args.manifest.read_text()),
            json.loads(args.result.read_text()),
            config,
        ),
    )


if __name__ == "__main__":
    main()
