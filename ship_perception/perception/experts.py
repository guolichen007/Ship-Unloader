"""Independent measured proposal experts; no expert can veto another source."""

from dataclasses import replace
import numpy as np
from .candidate_union import CandidateProposal, canonical
from .local_evidence import local_height_recall


def _proposal(expert, cache, bounds, index, **evidence):
    row = dict(evidence)
    # Derived closure evidence can carry its own diagnostic geometry role.
    # The new expert record still owns its identity and remains a proposal.
    row.update(
        proposal_id=expert + str(index),
        bounds_axial=list(map(float, bounds)),
        vessel_hypothesis_id=cache.vessel.vessel_id,
        geometry_role="PROPOSAL_NOT_STEEL",
    )
    return CandidateProposal.from_evidence(expert, cache.vessel.vessel_id, row, index)


def line_families(cache):
    if hasattr(cache, "_v2_line_families"):
        return cache._v2_line_families
    output = {0: [], 1: []}
    tolerance = cache.config["boundary"]["corner_join_m"]
    for axis in (0, 1):
        providers = sorted(
            (
                p
                for p in cache.structural_providers()
                if p["rho_or_region"].get("normal_axis") == axis
            ),
            key=lambda p: p["rho_or_region"]["rho"],
        )
        groups = []
        for p in providers:
            if (
                groups
                and p["rho_or_region"]["rho"] - groups[-1][0]["rho_or_region"]["rho"]
                <= tolerance
            ):
                groups[-1].append(p)
            else:
                groups.append([p])
        for group in groups:
            start = min(p["support_interval"][0] for p in group)
            end = max(p["support_interval"][1] for p in group)
            row = dict(
                rho=float(np.median([p["rho_or_region"]["rho"] for p in group])),
                interval=(start, end),
                provider_ids=tuple(sorted(p["provider_id"] for p in group)),
                raw3d=any(
                    p["provider_type"] == "RAW3D_STRUCTURAL_PROVIDER" for p in group
                ),
                bev=any(
                    p["provider_type"] == "BEV_RECTILINEAR_PROVIDER" for p in group
                ),
                normal_face=any(
                    "RAW3D_VERIFIED_VERTICAL_FACE" in p["evidence_types"] for p in group
                ),
            )
            output[axis].append(row)
    cache._v2_line_families = output
    return output


def line_height(cache, axis, rho, span, sign):
    grid = cache.context_grid("fine")
    u, v = grid.axis_centers()
    normal, along = (u, v) if axis == 0 else (v, u)
    step = cache.config["boundary"]["side_probe_m"]
    mask = (along >= span[0]) & (along <= span[1])
    med = grid.median.T if axis == 0 else grid.median
    count = grid.count.T if axis == 0 else grid.count
    key = (axis, float(rho), tuple(span), sign)
    if not hasattr(cache, "_v2_line_height_cache"):
        cache._v2_line_height_cache = {}
    if key in cache._v2_line_height_cache:
        return cache._v2_line_height_cache[key]
    from .sections import physical_sections

    values = []
    edges = physical_sections(span, grid.cell_m, cache.config)
    for begin, end in zip(edges[:-1], edges[1:]):
        section = mask & (along >= begin) & (along <= end)
        pair = []
        point_counts = []
        for a, b in ((0, step), (-step, 0)):
            rows = (sign * (normal - rho) >= a) & (sign * (normal - rho) < b)
            region = med[np.ix_(rows, section)]
            occupied = count[np.ix_(rows, section)] > 0
            pair.append(region[occupied])
            point_counts.append(int(count[np.ix_(rows, section)].sum()))
        if (
            all(len(x) > 0 for x in pair)
            and min(point_counts) >= cache.config["boundary"]["profile_min_points"]
        ):
            values.append(
                float(np.quantile(pair[1], 0.25) - np.quantile(pair[0], 0.75))
            )
    result = (
        float(np.median(values))
        if len(values) >= cache.config["boundary"]["profile_min_sections"]
        else None
    )
    cache._v2_line_height_cache[key] = result
    return result


def structural_enclosures(cache, budget=None):
    if hasattr(cache, "_v2_structural_enclosures"):
        return cache._v2_structural_enclosures
    from .boundary_line_graph import BoundaryLineGraph

    graph = BoundaryLineGraph(
        line_families(cache),
        cache.config,
        lambda axis, rho, span, sign: line_height(cache, axis, rho, span, sign),
        budget,
    )
    candidates = graph.enclosures()
    cache._v2_boundary_graph = dict(
        node_count=graph.nodes,
        intersection_count=graph.edges,
        adjacent_closure_count=len(candidates),
        raw_role_queries=0,
    )
    cache._v2_structural_enclosures = candidates
    return candidates


def role_families(cache, budget=None):
    """Nominal structural lines nominate exact signed raw modes, never boxes.

    A coarse line may lie between return rows. Query its existing search band
    once per physical section before associating four sides. This prevents a
    coarse raster station from becoming an unmeasured final hatch boundary.
    """
    if hasattr(cache, "_v2_role_families"):
        return cache._v2_role_families
    from .final_boundary import FinalBoundaryRefiner
    from .sections import physical_sections
    from .measurement import VesselMeasurementContext
    from .consensus import SpatialEvidenceConsensus

    measurement = getattr(cache, "v2_measurement", None)
    if measurement is None:
        measurement = VesselMeasurementContext(cache)
    refiner = getattr(measurement, "refiner", None)
    if refiner is None:
        refiner = FinalBoundaryRefiner(
            measurement.raw_uvz, [], cache.config, measurement
        )
        measurement.refiner = refiner
    result = {0: [], 1: []}
    config = cache.config
    for axis, families in line_families(cache).items():
        for family in families:
            if budget is not None:
                budget.check()
            if not (family["raw3d"] and family["bev"]):
                continue
            if not family["normal_face"]:
                # Cheap whole-interval nomination precedes section-level raw
                # work. Basins and loaded enclosures remain independent
                # experts; failure here cannot veto their proposals.
                grid = cache.context_grid("fine")
                u, v = grid.axis_centers()
                normal, along = (u, v) if axis == 0 else (v, u)
                med = grid.median.T if axis == 0 else grid.median
                count = grid.count.T if axis == 0 else grid.count
                span_mask = (along >= family["interval"][0]) & (
                    along <= family["interval"][1]
                )
                width = config["boundary"]["side_probe_m"]
                rho = family["rho"]
                pair = []
                for low, high in ((rho - width, rho), (rho, rho + width)):
                    rows = (normal >= low) & (normal < high)
                    values = med[np.ix_(rows, span_mask)]
                    live = count[np.ix_(rows, span_mask)] > 0
                    pair.append(values[live])
                if not all(len(x) > 0 for x in pair):
                    continue
                if (
                    max(
                        float(np.quantile(pair[0], 0.25) - np.quantile(pair[1], 0.75)),
                        float(np.quantile(pair[1], 0.25) - np.quantile(pair[0], 0.75)),
                    )
                    < config["roi"]["opening_drop_m"]
                ):
                    continue
            span = family["interval"]
            band = (
                config["roi"]["boundary_search_m"] + config["boundary"]["side_probe_m"]
            )
            box = [0.0, 0.0, 0.0, 0.0]
            box[axis] = family["rho"] - band
            box[axis + 2] = family["rho"] + band
            box[1 - axis], box[3 - axis] = span
            points = measurement.spatial.query(box)
            edges = physical_sections(
                span, config["geometry"]["refine_voxel_m"], config
            )
            for sign in (1, -1):
                rows = []
                for a, b in zip(edges[:-1], edges[1:]):
                    direct = refiner._raw_section(
                        points, axis, family["rho"], (a, b), sign
                    )
                    if direct["status"] == "UNIQUE_SIGNED_RAW_STEP":
                        rows.append(
                            dict(
                                along=(a + b) / 2,
                                rho=direct["continuous"][0],
                                measurement_source="RAW_PAIRED_PROFILE",
                            )
                        )
                rho, fit = SpatialEvidenceConsensus(config).fit(rows, family["rho"])
                if fit["status"] != "CONSENSUS":
                    continue
                observed = dict(
                    family,
                    rho=float(rho),
                    supported_signs=[sign],
                    role_support_sections=fit["inlier_section_count"],
                    raw_role_fit=fit,
                )
                existing = next(
                    (
                        r
                        for r in result[axis]
                        if abs(r["rho"] - rho) <= config["geometry"]["plane_inlier_m"]
                        and r["provider_ids"] == family["provider_ids"]
                    ),
                    None,
                )
                if existing:
                    existing["supported_signs"].append(sign)
                else:
                    result[axis].append(observed)
            if family["normal_face"] and not any(
                r["provider_ids"] == family["provider_ids"] for r in result[axis]
            ):
                # A loaded interior may hide the signed height step. Preserve
                # its measured face with unknown role; validation/precision
                # must resolve it, and no height or steel role is invented.
                result[axis].append(
                    dict(
                        family,
                        supported_signs=[1, -1],
                        role_support_sections=0,
                        raw_role_fit=dict(status="NORMAL_FACE_ROLE_UNRESOLVED"),
                    )
                )
    for rows in result.values():
        rows.sort(key=lambda r: r["rho"])
    cache._v2_role_families = result
    return result


class HeightBasinExpert:
    name = "HEIGHT_BASIN"

    def propose(self, cache, research, budget=None):
        if not hasattr(cache, "_v2_height_recall"):
            cache._v2_height_recall = local_height_recall(cache, research)
        return tuple(
            CandidateProposal.from_evidence(self.name, cache.vessel.vessel_id, row, i)
            for i, row in enumerate(cache._v2_height_recall[0])
            if row.get("polarity") != "RAISED"
        )


class LoadedHoldExpert(HeightBasinExpert):
    name = "LOADED_HOLD"

    def propose(self, cache, research, budget=None):
        if not hasattr(cache, "_v2_height_recall"):
            cache._v2_height_recall = local_height_recall(cache, research)
        # Interior sand elevation proposes a search region; it cannot assert a
        # beam or discard a measured enclosure from a different expert.
        rows = [
            row for row in cache._v2_height_recall[0] if row.get("polarity") == "RAISED"
        ]
        height = tuple(
            CandidateProposal.from_evidence(self.name, cache.vessel.vessel_id, row, i)
            for i, row in enumerate(rows)
        )
        structural = tuple(
            replace(
                p,
                expert=self.name,
                proposal_id=self.name + ":STRUCTURAL:" + p.proposal_id,
            )
            for p in StructuralLineExpert().propose(cache, research, budget)
        )
        return height + structural


class StructuralLineExpert:
    name = "STRUCTURAL_LINE"

    def propose(self, cache, research, budget=None):
        if not hasattr(cache, "_v21_structural_proposals"):
            cache._v21_structural_proposals = tuple(
                CandidateProposal.from_evidence(
                    self.name, cache.vessel.vessel_id, row, i
                )
                for i, row in enumerate(structural_enclosures(cache, budget))
            )
        return cache._v21_structural_proposals


class LocalEnclosureExpert(StructuralLineExpert):
    name = "LOCAL_ENCLOSURE"

    def propose(self, cache, research, budget=None):
        return tuple(
            replace(p, expert=self.name, proposal_id=self.name + ":" + p.proposal_id)
            for p in StructuralLineExpert().propose(cache, research, budget)
            if p.evidence()["normal_face_sides"] >= 3
        )


class LargeSandHoldExpert(StructuralLineExpert):
    name = "LARGE_SAND_HOLD"

    def propose(self, cache, research, budget=None):
        # Interpret the same immutable enclosure; no second geometry search,
        # outer-box Cartesian product or separate Raw measurement is created.
        return tuple(
            replace(
                p,
                expert=self.name,
                proposal_id=self.name + ":" + p.proposal_id,
                evidence_json=canonical(
                    dict(
                        p.evidence(),
                        interior_height_hard_gate=False,
                        interpretation="CONTINUOUS_STRUCTURAL_ENCLOSURE_WITH_UNKNOWN_LOAD",
                    )
                ),
            )
            for p in StructuralLineExpert().propose(cache, research, budget)
        )


class LayoutRecoveryExpert:
    name = "LAYOUT_RECOVERY"

    def search(self, cache, validated, research, budget=None):
        strong = sorted(
            (v for v in validated if v.get("independent_support_sides", 0) >= 3),
            key=lambda v: v["bounds"][0],
        )
        diagnostics = []
        proposals = []
        for left, right in zip(strong, strong[1:]):
            if right["bounds"][0] <= left["bounds"][2]:
                continue
            roi = [
                left["bounds"][2],
                min(left["bounds"][1], right["bounds"][1]),
                right["bounds"][0],
                max(left["bounds"][3], right["bounds"][3]),
            ]
            found = []
            for row in structural_enclosures(cache, budget):
                box = row["bounds_axial"]
                if roi[0] <= box[0] < box[2] <= roi[2]:
                    found.append(
                        _proposal(
                            self.name,
                            cache,
                            box,
                            len(proposals),
                            **{k: v for k, v in row.items() if k != "bounds_axial"},
                        )
                    )
            proposals.extend(found)
            diagnostics.append(
                dict(
                    status=(
                        "MEASURED_LAYOUT_GAP_PROPOSALS"
                        if found
                        else "UNRESOLVED_LAYOUT_GAP"
                    ),
                    search_roi=roi,
                    physical_edges_created_by_layout=False,
                )
            )
        return tuple(proposals), diagnostics


class MultiHoldLayoutExpert(LayoutRecoveryExpert):
    name = "MULTI_HOLD_LAYOUT"

    def search(self, cache, validated, research, budget=None):
        proposals, diagnostics = super().search(cache, validated, research, budget)
        from .layout import cross_deck_evidence

        output = list(proposals)
        seen = set()
        rows = structural_enclosures(cache, budget)
        for parent in validated:
            outer = parent["bounds"]
            children = sorted(
                (
                    r
                    for r in rows
                    if outer[0]
                    <= r["bounds_axial"][0]
                    < r["bounds_axial"][2]
                    <= outer[2]
                    and outer[1]
                    <= r["bounds_axial"][1]
                    < r["bounds_axial"][3]
                    <= outer[3]
                ),
                key=lambda r: (r["bounds_axial"][0], r["bounds_axial"][2]),
            )
            for left in children:
                if budget is not None:
                    budget.check()
                rights = [
                    r
                    for r in children
                    if r["bounds_axial"][0] > left["bounds_axial"][2]
                ]
                if not rights:
                    continue
                right = min(rights, key=lambda r: r["bounds_axial"][0])
                key = (tuple(left["bounds_axial"]), tuple(right["bounds_axial"]))
                if key in seen:
                    continue
                seen.add(key)
                evidence = cross_deck_evidence(cache, *key)
                diagnostics.append(
                    dict(
                        status=(
                            "SPLIT_HYPOTHESIS"
                            if evidence["can_split"]
                            else "UNRESOLVED_CROSS_DECK_OR_CARGO_RIDGE"
                        ),
                        parent_bounds=list(outer),
                        cross_deck=evidence,
                        physical_edges_created_by_layout=False,
                    )
                )
                if evidence["can_split"]:
                    for child in (left, right):
                        detail = {k: v for k, v in child.items() if k != "bounds_axial"}
                        output.append(
                            _proposal(
                                self.name,
                                cache,
                                child["bounds_axial"],
                                len(output),
                                **detail,
                                split_hypothesis=dict(
                                    parent_bounds=list(outer), cross_deck=evidence
                                ),
                            )
                        )
        return tuple(output), diagnostics
