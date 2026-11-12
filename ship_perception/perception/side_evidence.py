"""Pure current-provider projection and side modes, without diagnostic imports.

Equivalent local capabilities extracted from the existing forensic helper.
No scene inventory, historical polygon, lineage solver or rectangle search
is imported by this module.
"""

import numpy as np


def project_provider_observations(b2, vessel_id, axes, roi):
    """Project all B2 measured lines in a bounded opening ROI, no old-side rho gate."""
    observed = []
    statuses = {
        row["vessel_hypothesis_id"]: row["classification_status"]
        for row in b2["scene"]["vessel_hypotheses"]
    }
    for context in b2["per_vessel"]:
        source_id = context["vessel_hypothesis_id"]
        status = statuses.get(source_id, "UNRESOLVED")
        if status == "VESSEL_HYPOTHESIS" and source_id != vessel_id:
            continue
        for row in context["provider_evidence"]:
            if row["geometry_type"] != "AXIAL_LINE_INTERVAL" or row[
                "provider_type"
            ] not in ("RAW3D_STRUCTURAL_PROVIDER", "BEV_RECTILINEAR_PROVIDER"):
                continue
            endpoints = np.asarray(row["source_geometry"]["endpoints_xy"], dtype=float)
            if endpoints.shape != (2, 2):
                continue
            local = endpoints @ axes.T
            normal_axis = int(row["rho_or_region"]["normal_axis"])
            rho = float(np.mean(local[:, normal_axis]))
            along = sorted(local[:, 1 - normal_axis].tolist())
            if not (
                roi[normal_axis] <= rho <= roi[normal_axis + 2]
                and along[1] >= roi[1 - normal_axis]
                and (along[0] <= roi[3 - normal_axis])
            ):
                continue
            observed.append(
                dict(
                    normal_axis=normal_axis,
                    rho=rho,
                    along=along,
                    provider_type=row["provider_type"],
                    provider_id=row["provider_id"],
                    source_vessel_id=source_id,
                    source_support_ids=row["source_support_ids"],
                    source_scope=(
                        "SAME_VESSEL" if source_id == vessel_id else "CONTEXT_ONLY"
                    ),
                    coverage=float(row["coverage"]),
                )
            )
    return observed


def bounded_side_modes(observations, normal_axis, span, band, config):
    """Survey measured rho modes across an opening band without B4's ±4 m anchor."""
    join = config["boundary"]["corner_join_m"]
    min_line = config["boundary"]["line_min_length_m"]
    candidates = []
    source = [
        row
        for row in observations
        if row["normal_axis"] == normal_axis
        and band[0] <= row["rho"] <= band[1]
        and (min(span[1], row["along"][1]) - max(span[0], row["along"][0]) >= min_line)
    ]
    for rho in sorted({round(row["rho"], 2) for row in source}):
        nearby = [row for row in source if abs(row["rho"] - rho) <= join]
        levels = {}
        for scope in ("SAME_VESSEL", "CONTEXT_ONLY"):
            for provider in ("RAW3D_STRUCTURAL_PROVIDER", "BEV_RECTILINEAR_PROVIDER"):
                levels[scope, provider] = max(
                    (
                        row["coverage"]
                        * min(
                            1.0,
                            max(
                                0.0,
                                min(span[1], row["along"][1])
                                - max(span[0], row["along"][0]),
                            )
                            / max(span[1] - span[0], 1e-09),
                        )
                        for row in nearby
                        if row["source_scope"] == scope
                        and row["provider_type"] == provider
                    ),
                    default=0.0,
                )
        candidates.append(
            dict(
                rho=float(rho),
                normal_axis=normal_axis,
                primary_raw3d=levels["SAME_VESSEL", "RAW3D_STRUCTURAL_PROVIDER"],
                primary_bev=levels["SAME_VESSEL", "BEV_RECTILINEAR_PROVIDER"],
                context_raw3d=levels["CONTEXT_ONLY", "RAW3D_STRUCTURAL_PROVIDER"],
                context_bev=levels["CONTEXT_ONLY", "BEV_RECTILINEAR_PROVIDER"],
                provider_ids=[row["provider_id"] for row in nearby],
                source_support_ids=sorted(
                    {sid for row in nearby for sid in row["source_support_ids"]}
                ),
                source_vessel_ids=sorted({row["source_vessel_id"] for row in nearby}),
                provider_scope=(
                    "MIXED_PRIMARY_AND_CONTEXT"
                    if min(
                        levels["SAME_VESSEL", "RAW3D_STRUCTURAL_PROVIDER"],
                        levels["SAME_VESSEL", "BEV_RECTILINEAR_PROVIDER"],
                    )
                    > 0
                    and max(
                        levels["CONTEXT_ONLY", "RAW3D_STRUCTURAL_PROVIDER"],
                        levels["CONTEXT_ONLY", "BEV_RECTILINEAR_PROVIDER"],
                    )
                    > 0
                    else (
                        "PRIMARY_CURRENT_VESSEL"
                        if min(
                            levels["SAME_VESSEL", "RAW3D_STRUCTURAL_PROVIDER"],
                            levels["SAME_VESSEL", "BEV_RECTILINEAR_PROVIDER"],
                        )
                        > 0
                        else "CONTEXT_ONLY"
                    )
                ),
            )
        )
    candidates.sort(
        key=lambda row: (
            -min(row["primary_raw3d"], row["primary_bev"]),
            -max(row["primary_raw3d"], row["primary_bev"]),
            row["rho"],
        )
    )
    shortlist = []
    for row in candidates:
        if all(
            (
                abs(row["rho"] - prior["rho"]) > config["boundary"]["line_inlier_m"]
                for prior in shortlist
            )
        ):
            shortlist.append(row)
        if len(shortlist) >= 16:
            break
    return shortlist
