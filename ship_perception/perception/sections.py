"""Physical, occupied section support shared by all four final sides."""

import math
import numpy as np


def physical_sections(span, cell_m, config):
    length = float(span[1] - span[0])
    minimum = max(
        config["boundary"]["line_min_length_m"],
        3 * cell_m,
        config["boundary"]["max_support_gap_m"]
        * min(
            config["boundary"]["line_min_points"],
            config["boundary"]["profile_min_sections"],
        ),
    )
    if length < minimum:
        return []
    target = max(2 * config["boundary"]["profile_half_length_m"], minimum)
    count = max(1, int(math.floor(length / target)))
    count = max(
        count,
        min(
            config["boundary"]["profile_min_sections"],
            int(math.floor(length / minimum)),
        ),
    )
    edges = np.linspace(span[0], span[1], count + 1)
    return edges.tolist()


def robust_height_support(axial_points, axis, rho, span, sign, config):
    normal = sign * (axial_points[:, axis] - rho)
    along = axial_points[:, 1 - axis]
    step = config["boundary"]["side_probe_m"]
    section = (along >= span[0]) & (along <= span[1])
    bands = [
        section & (normal >= 0) & (normal <= step),
        section & (normal < 0) & (normal >= -step),
    ]
    values = [axial_points[mask, 2] for mask in bands]
    counts = [len(v) for v in values]
    if min(counts) < config["boundary"]["profile_min_points"]:
        return dict(
            status="INSUFFICIENT_SUPPORT",
            counts=counts,
            signed_contrast_m=None,
            robust_lower_contrast_m=None,
            z_mad_m=None,
            transition=False,
        )
    medians = [float(np.median(v)) for v in values]
    mads = [float(np.median(abs(v - m))) for v, m in zip(values, medians)]
    lower = float(np.quantile(values[1], 0.25) - np.quantile(values[0], 0.75))
    return dict(
        status="MEASURED",
        counts=counts,
        signed_contrast_m=medians[1] - medians[0],
        robust_lower_contrast_m=lower,
        z_mad_m=mads,
        transition=bool(lower >= config["roi"]["opening_drop_m"]),
    )
