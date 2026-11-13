"""Spatially distributed mixed evidence, with robust line-mode consensus.

Raw pairs, raw faces and height/BEV observations remain individually typed;
consensus does not promote their physical role to certified steel.
"""

import warnings
import numpy as np
from scipy.stats import theilslopes


class SpatialEvidenceConsensus:
    def __init__(self, config):
        self.config = config

    def fit(self, rows, fallback):
        families = {
            row.get("physical_line_family")
            for row in rows
            if row.get("physical_line_family") is not None
        }
        if len(families) > 1:
            return fallback, dict(
                status="MULTIMODE_BOUNDARY_CONFLICT",
                reason="DISTINCT_PHYSICAL_LINE_FAMILIES",
            )
        value, detail = self._fit(rows, fallback)
        detail["evidence_sources"] = sorted(
            {row.get("measurement_source", "UNSPECIFIED") for row in rows}
        )
        detail["steel_edge_certified"] = False
        return value, detail

    def _fit(self, rows, fallback):
        if not rows:
            return fallback, dict(status="INSUFFICIENT_SUPPORT")
        x = np.array([r["along"] for r in rows])
        y = np.array([r["rho"] for r in rows])
        if len(rows) < self.config["boundary"]["profile_min_sections"]:
            return float(np.median(y)), dict(status="INSUFFICIENT_SUPPORT")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reference, center_rho = float(np.median(x)), float(np.median(y))
            fit = theilslopes(y - center_rho, x - reference)
        if not np.isfinite(np.asarray(fit)).all():
            return center_rho, dict(
                status="MULTIMODE_BOUNDARY_CONFLICT", reason="DEGENERATE_SECTION_LINE"
            )
        rho_at_reference = center_rho + float(fit[1])
        residual = y - (rho_at_reference + fit[0] * (x - reference))
        mad = float(np.median(abs(residual)))
        tolerance = self.config["boundary"]["corner_join_m"]
        minimum = self.config["boundary"]["profile_min_sections"]
        consistent = abs(residual) <= tolerance
        # Deterministic robust consensus. A contaminated section is not itself
        # a second physical edge; two spatially supported lines remain modes.
        hypotheses = [consistent]
        for a in range(len(x)):
            for b in range(a + 1, len(x)):
                if abs(x[b] - x[a]) < self.config["roi"]["min_deck_span_m"]:
                    continue
                slope = (y[b] - y[a]) / (x[b] - x[a])
                if (
                    abs(np.degrees(np.arctan(slope)))
                    > self.config["boundary"]["merge_angle_deg"]
                ):
                    continue
                hypotheses.append(abs(y - (y[a] + slope * (x - x[a]))) <= tolerance)
        consistent = max(
            hypotheses,
            key=lambda mask: (
                int(mask.sum()),
                float(np.ptp(x[mask])) if mask.any() else 0,
            ),
        )
        enough = (
            consistent.sum() >= minimum
            and consistent.mean() >= 1 - self.config["roi"]["min_enclosure_ratio"]
            and np.ptp(x[consistent]) >= self.config["roi"]["min_deck_span_m"]
        )
        second = False
        remaining = ~consistent
        if (
            remaining.sum() >= minimum
            and np.ptp(x[remaining]) >= self.config["roi"]["min_deck_span_m"]
        ):
            other = theilslopes(y[remaining], x[remaining])
            other_residual = abs(y[remaining] - (other[1] + other[0] * x[remaining]))
            second = (
                int((other_residual <= tolerance).sum()) >= minimum
                and abs(np.degrees(np.arctan(other[0])))
                <= self.config["boundary"]["merge_angle_deg"]
            )
        if enough and not second:
            reference = float(np.median(x[consistent]))
            center_rho = float(np.median(y[consistent]))
            fit = theilslopes(y[consistent] - center_rho, x[consistent] - reference)
            rho_at_reference = center_rho + float(fit[1])
            residual = y - (rho_at_reference + fit[0] * (x - reference))
            mad = float(np.median(abs(residual[consistent])))
        status = (
            "CONSENSUS"
            if enough
            and not second
            and abs(np.degrees(np.arctan(fit[0])))
            <= self.config["boundary"]["merge_angle_deg"]
            else "MULTIMODE_BOUNDARY_CONFLICT"
        )
        return rho_at_reference, dict(
            status=status,
            slope=float(fit[0]),
            low_slope=float(fit[2]),
            high_slope=float(fit[3]),
            reference_along_m=reference,
            rho_at_reference_m=rho_at_reference,
            residual_mad_m=mad,
            unconstrained_slope_deg=float(np.degrees(np.arctan(fit[0]))),
            inlier_section_count=int(consistent.sum()),
            inlier_fraction=float(consistent.mean()),
            inlier_spatial_span_m=(
                float(np.ptp(x[consistent])) if consistent.any() else 0.0
            ),
            outlier_section_indexes=np.flatnonzero(~consistent).tolist(),
            outlier_status=(
                "OUTLIER_SECTION" if (~consistent).any() and not second else None
            ),
            second_stable_line_mode=bool(second),
        )
