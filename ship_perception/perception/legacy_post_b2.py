"""Selected-vessel post-B2 semantics, reusing measured providers and grids.

This adapter is not an expert. Frozen B2 proposals remain immutable and are
retained in every derived physical candidate's provenance. No global legacy
pipeline or legacy final precision is executed here.
"""

from .precision_diagnostics import stage as precision_stage, side_completed

import hashlib
import numpy as np

from .current_separator import (
    _initial_seed_groups,
    _separator_split,
)
from .candidate_union import PhysicalBoundaryQuery, canonical
from .context import thaw
from .line_geometry import SIDE_SPECS
from .layout import cross_deck_evidence


class LegacyPostB2EvidenceAdapter:
    def __init__(self, cache):
        self.cache, self.config = cache, cache.config
        self.evidence = {}
        self.profile_cache = {}

    @precision_stage("raw_profile")
    def measure_profiles(self, query, probe, measurement):
        """Reuse the historical per-segment Raw profile primitive locally."""
        from scipy.spatial import cKDTree
        from ship_perception.r1_static.model import PerimeterSegment
        from ship_perception.r1_static.perimeter_deck import estimate_segment_deck
        from ship_perception.r1_static.boundary_refinement import _profile
        from .sections import physical_sections

        output = []
        for name, axis, position, sign in SIDE_SPECS:
            if measurement.budget is not None:
                measurement.budget.check()
            side = probe["sides"][name]
            rho = side.get("rho", query.bounds[position])
            span = (query.bounds[1 - axis], query.bounds[3 - axis])
            key = (axis, rho, span, sign)
            if key not in self.profile_cache:
                a, b = [0.0, 0.0], [0.0, 0.0]
                a[axis] = b[axis] = rho
                a[1 - axis], b[1 - axis] = span
                tangent, outward = [0.0, 0.0], [0.0, 0.0]
                tangent[1 - axis], outward[axis] = 1.0, -sign
                segment = PerimeterSegment(
                    "POST_B2:" + name,
                    query.query_id,
                    tuple(a),
                    tuple(b),
                    tuple(tangent),
                    tuple(outward),
                    span[1] - span[0],
                    "UNKNOWN",
                )
                band = (
                    self.config["roi"]["support_search_m"]
                    + self.config["boundary"]["profile_half_width_m"]
                )
                box = [0.0] * 4
                box[axis], box[axis + 2] = rho - band, rho + band
                box[1 - axis], box[3 - axis] = span
                points = measurement.spatial.query(box)
                rows = []
                if len(points):
                    tree = cKDTree(points[:, :2])
                    deck, plane, _ = estimate_segment_deck(
                        points, segment, self.config, tree
                    )
                    if plane is not None:
                        edges = physical_sections(
                            span, self.config["geometry"]["refine_voxel_m"], self.config
                        )
                        for index, (start, end) in enumerate(
                            zip(edges[:-1], edges[1:])
                        ):
                            center = np.asarray(a)
                            center[1 - axis] = (start + end) / 2
                            measured = _profile(
                                points,
                                tree,
                                center,
                                np.asarray(tangent),
                                -np.asarray(outward),
                                plane,
                                self.config,
                                str(index),
                            )
                            if measured["valid"]:
                                rows.append(
                                    dict(
                                        observation_id="POST_B2:"
                                        + hashlib.sha256(
                                            canonical([key, index]).encode()
                                        ).hexdigest()[:16],
                                        xy=measured["break_position"],
                                        evidence_type="OBSERVED_PROFILE_BREAK",
                                        reference_role="LOCAL_MEASURED_DECK",
                                        source_vessel_id=query.vessel_id,
                                        point_index_sha256=self.cache.vessel.point_index_sha256,
                                        local_plane_status=deck["status"],
                                    )
                                )
                self.profile_cache[key] = rows
            output.extend(self.profile_cache[key])
        if query.query_id in self.evidence:
            self.evidence[query.query_id].update(
                profile_observations=output,
                profile_observations_status=(
                    "OBSERVED_LOCAL_PROFILE" if output else "INSUFFICIENT_LOCAL_PROFILE"
                ),
            )
        return output

    def adapt(self, batch, budget=None):
        cache, config = self.cache, self.config
        vessel = thaw(cache.vessel.public)
        vessel["local_axes"] = [[1, 0], [0, 1]]
        raw = batch.raw_output()
        proposals = {p.proposal_id: p for p in batch.proposals}
        b2 = dict(
            scene=dict(vessel_hypotheses=[vessel]),
            per_vessel=[
                dict(
                    vessel_hypothesis_id=cache.vessel.vessel_id,
                    provider_evidence=cache.structural_providers(),
                    hatch_hypotheses=raw["hatch_hypotheses"],
                )
            ],
        )
        output = []
        groups = _initial_seed_groups(raw["hatch_hypotheses"], config)
        grouped_ids = {m["hatch_hypothesis_id"] for _, group in groups for m in group}
        groups.extend(
            (row, [row])
            for row in raw["hatch_hypotheses"]
            if row["hatch_hypothesis_id"] not in grouped_ids
            and row.get("state") in ("HATCH_PROPOSAL", "PARTIAL_HATCH", "UNRESOLVED")
            and row.get("bounds_axial") is not None
            and row.get("supporting_provider_ids")
        )
        for seed, members in groups:
            if budget is not None:
                budget.check()
            ids = sorted("LEGACY:" + m["hatch_hypothesis_id"] for m in members)
            sources = tuple(proposals[pid] for pid in ids)
            envelope = list(seed["bounds_axial"])
            # Preserve the B2 search extent. Height persistence can nominate
            # additional side modes later, but does not rewrite this geometry.
            nominal = envelope
            semantic = dict(status="LOCAL_SIDE_MODES_WITH_UNCHANGED_B2_ANCHORS")
            split, separator = _separator_split(
                dict(bounds_axial=nominal),
                b2,
                vessel,
                cache.context_grid("fine"),
                config,
            )
            cross = None
            if isinstance(separator, list) and len(split) == 2:
                cross = cross_deck_evidence(cache, split[0], split[1])
                if not cross["can_split"]:
                    split = [nominal]
            else:
                split = [nominal]
            for index, box in enumerate(split):
                ident = canonical([cache.vessel.vessel_id, ids, box, separator, index])
                qid = "L" + hashlib.sha256(ident.encode()).hexdigest()[:16]
                query = PhysicalBoundaryQuery(
                    qid, cache.vessel.vessel_id, tuple(box), sources, ()
                )
                sides = cache.side_candidates(box)
                availability = {
                    name: any(
                        min(m["primary_raw3d"], m["primary_bev"])
                        >= config["roi"]["min_enclosure_ratio"]
                        and abs(m["rho"] - box[position])
                        <= config["boundary"]["corner_join_m"]
                        for m in sides[name]
                    )
                    for name, _, position, _ in SIDE_SPECS
                }
                evidence = dict(
                    query_id=qid,
                    source_proposal_ids=ids,
                    bounds_axial=list(box),
                    original_group_envelope=envelope,
                    separator_topology=(
                        "TWO_MEASURED_FACES" if len(split) == 2 else "UNSPLIT"
                    ),
                    measured_separator_faces=separator if len(split) == 2 else None,
                    separator_diagnostic=separator,
                    cross_deck=cross,
                    side_evidence_availability=availability,
                    support_side_count=sum(availability.values()),
                    derived_search_evidence=True,
                    measured_side_candidates=sides,
                    local_physical_candidate_available=True,
                    profile_observations=[],
                    profile_observations_status="LOCAL_RAW_MEASUREMENT_PENDING",
                    point_index_sha256=cache.vessel.point_index_sha256,
                    semantic_review=semantic,
                    geometry_role="POST_B2_SEARCH_HYPOTHESIS_NOT_STEEL",
                )
                self.evidence[qid] = evidence
                output.append(query)
        return tuple(output)
