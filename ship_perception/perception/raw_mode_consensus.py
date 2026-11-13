"""Resolve local step alternatives only from independently unique Raw sections."""

import numpy as np
from scipy.stats import theilslopes

from .consensus import SpatialEvidenceConsensus


def supported(record):
    return (
        record.get("status") == "UNIQUE_SIGNED_RAW_STEP"
        and record.get("height", {}).get("transition", False)
        and record.get("continuous", [0, 0, 0, False])[3]
    )


def _second_family(alternatives, spans, config, budget):
    minimum = config["boundary"]["profile_min_sections"]
    indexes = [i for i, modes in enumerate(alternatives) if modes]
    if len(indexes) < minimum:
        return False
    along = [sum(span) / 2 for span in spans]
    join = config["boundary"]["corner_join_m"]
    for offset, i in enumerate(indexes):
        for j in indexes[offset + 1 :]:
            if abs(along[j] - along[i]) < config["roi"]["min_deck_span_m"]:
                continue
            for a in alternatives[i]:
                for b in alternatives[j]:
                    if budget is not None:
                        budget.check()
                    slope = (b - a) / (along[j] - along[i])
                    if (
                        abs(np.degrees(np.arctan(slope)))
                        > config["boundary"]["merge_angle_deg"]
                    ):
                        continue
                    rows = []
                    for k in indexes:
                        predicted = a + slope * (along[k] - along[i])
                        rho = min(alternatives[k], key=lambda r: abs(r - predicted))
                        if abs(rho - predicted) <= join:
                            rows.append(dict(along=along[k], rho=rho))
                    if len(rows) >= minimum:
                        _, fit = SpatialEvidenceConsensus(config).fit(rows, a)
                        if fit["status"] == "CONSENSUS":
                            return True
    return False


def resolve_section_modes(records, spans, config, budget=None):
    """Never fill missing sections or average distinct physical alternatives.

    Two spatially separated, already unique Raw sections nominate a line.
    An ambiguous section joins it only when exactly one of its measured modes
    agrees. The complete selection still passes the unchanged final consensus.
    """
    if not any(r.get("mode_candidates") for r in records):
        return records
    seeds = [
        (sum(s) / 2, r["continuous"][0]) for s, r in zip(spans, records) if supported(r)
    ]
    if len(seeds) < 2:
        return records
    x, y = np.asarray(seeds).T
    if np.ptp(x) < config["roi"]["min_deck_span_m"]:
        return records
    slope, intercept, _, _ = theilslopes(y, x)
    if abs(np.degrees(np.arctan(slope))) > config["boundary"]["merge_angle_deg"]:
        return records
    join = config["boundary"]["corner_join_m"]
    if np.any(abs(y - (intercept + slope * x)) > join):
        return records
    selected = list(records)
    for i, (span, record) in enumerate(zip(spans, records)):
        candidates = [
            m
            for m in record.get("mode_candidates", [])
            if supported(m)
            and abs(m["continuous"][0] - (intercept + slope * sum(span) / 2)) <= join
        ]
        if len(candidates) == 1:
            selected[i] = dict(
                candidates[0],
                local_mode_resolution="UNIQUE_RAW_SECTION_LINE_ANCHORS",
                original_section_modes=record.get("modes", []),
            )
    rows = [
        dict(
            along=sum(span) / 2,
            rho=r["continuous"][0],
            measurement_source="PAIRED_RAW_SURFACES",
        )
        for span, r in zip(spans, selected)
        if supported(r)
    ]
    _, fit = SpatialEvidenceConsensus(config).fit(rows, float(np.median(y)))
    if fit["status"] != "CONSENSUS":
        return records
    alternatives = []
    for span, record in zip(spans, records):
        predicted = fit["rho_at_reference_m"] + fit["slope"] * (
            sum(span) / 2 - fit["reference_along_m"]
        )
        modes = record.get("mode_candidates", [])
        alternatives.append(
            [
                r["continuous"][0]
                for r in modes
                if supported(r) and abs(r["continuous"][0] - predicted) > join
            ]
        )
    return records if _second_family(alternatives, spans, config, budget) else selected
