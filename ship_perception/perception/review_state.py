"""Separate measured geometry, independent steel benchmark and operator review."""

import numpy as np


def apply_review_state(hatch):
    sides = hatch.get("sides", {})
    polygon = np.asarray(hatch.get("boundary_xy", []))
    complete = (
        hatch.get("review_support_side_count") == 4
        and not hatch.get("undercoverage_risk", True)
        and polygon.shape == (4, 2)
        and np.isfinite(polygon).all()
        and hatch.get("corner_geometry", {}).get("intersection_stable", False)
        and all(
            sides.get(name, {}).get("independent_geometry_support", False)
            and not sides[name].get("undercoverage_risk", True)
            and sides[name].get("line", {}).get("source") == "RAW_SECTION_ROBUST_LINE"
            for name in ("U0", "U1", "V0", "V1")
        )
    )
    hatch.update(
        geometry_status=(
            "VALIDATED"
            if complete
            else (
                "PARTIAL"
                if hatch.get("review_support_side_count", 0) >= 3
                else "UNRESOLVED"
            )
        ),
        steel_edge_benchmark_status="NOT_CERTIFIED",
        operator_confirmation="PENDING",
    )
    if complete:
        hatch["status"] = "HATCH_REVIEW_CANDIDATE"
    return hatch
