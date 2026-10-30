"""Compile owned S4 observations into the existing opening identity contract.

No final rectangles, four-line products, historical coordinates or precision
measurements are inputs. Missing separators remain unknown.
"""

import hashlib
import itertools
import math
import time

import numpy as np
from scipy import ndimage

from ship_perception.r1_static.boundary_topology import (
    FORBIDDEN_REFERENCE_ROLES,
    _observations,
    _candidate_modes,
    _observed_intervals,
    structural_axis_estimate,
    orientation_search_budget,
)
from ship_perception.r1_static.family_reference import analyze_scene_s4r1
from ship_perception.r1_static.height_structure_forensics import _hierarchy
from .candidate_union import canonical
from .region_model import OpeningRegion, TopologySupport, EvidenceSource
from .structural_family import merge_intervals


def _ident(prefix, value):
    return prefix + hashlib.sha256(canonical(value).encode()).hexdigest()[:16]


class S4ObservationDeferred(RuntimeError):
    """The existing total budget must still leave time for shared precision."""


def topology_relation(first, second, shared_families, continuity, split_evidence):
    """Positive shared structure AND observed continuity are required to connect."""
    split = [r for r in split_evidence if r.get("qualified")]
    same = first["vessel_id"] == second["vessel_id"]
    if not same:
        decision = "SPLIT"
    elif split:
        decision = "SPLIT"
    elif shared_families and continuity.get("observed_continuity"):
        decision = "CONNECT"
    else:
        decision = "AMBIGUOUS"
    return dict(
        node_ids=[first["node_id"], second["node_id"]],
        ownership=dict(
            same_active_vessel=same,
            vessel_ids=[first["vessel_id"], second["vessel_id"]],
        ),
        shared_family_evidence=list(shared_families),
        separator_evidence=split_evidence,
        continuity_evidence=continuity,
        relation_decision=decision,
        unobserved_is_no_separator=False,
    )


def topology_components(vertices, relations):
    """Do not let transitive CONNECT edges bypass a measured SPLIT."""
    groups = [{v["node_id"]} for v in vertices]
    split = {
        frozenset(r["node_ids"]) for r in relations if r["relation_decision"] == "SPLIT"
    }
    for relation in sorted(relations, key=lambda r: tuple(r["node_ids"])):
        if relation["relation_decision"] != "CONNECT":
            continue
        a, b = relation["node_ids"]
        left = next(g for g in groups if a in g)
        right = next(g for g in groups if b in g)
        if left is right:
            continue
        if any(frozenset((x, y)) in split for x in left for y in right):
            continue
        left.update(right)
        groups.remove(right)
    return sorted(groups, key=lambda g: sorted(g))


def member_cell_assignment(xy, member_masks, region_bounds, nomination_bounds):
    """Map a nomination to an already compiled component using observed cells.

    The conservative display uses common interior. Ownership must include all
    observed members that the graph already connected, including higher layers.
    This function cannot create a CONNECT decision or expand region geometry.
    """
    in_region = np.all((xy >= region_bounds[:2]) & (xy <= region_bounds[2:]), axis=-1)
    in_nomination = np.all(
        (xy >= nomination_bounds[:2]) & (xy <= nomination_bounds[2:]), axis=-1
    )
    counts = {
        n: int(np.count_nonzero(mask & in_region & in_nomination))
        for n, mask in sorted(member_masks.items())
    }
    return dict(
        member_cell_counts={n: count for n, count in counts.items() if count},
        mapping_basis="OBSERVED_CELLS_OF_EXISTING_TOPOLOGY_COMPONENT",
    )


class S4TopologyBridge:
    def __init__(self, cache, research, budget=None, nominations=()):
        self.cache = cache
        self.primary = getattr(cache, "primary_cache", cache)
        self.config = cache.config
        self.vessel_id = cache.vessel.vessel_id
        self.budget = budget
        self.legacy_nominations = [
            p.evidence() for p in nominations if p.expert == "LEGACY"
        ]
        self.node_partition_audit = []
        self.observation_progress = dict(
            completed_deck_segments=[], completed_profile_segments=[]
        )
        # Only immutable active-vessel primary points own topology. Nearby
        # UNKNOWN context is available to precision, never silently promoted.
        self.xyz = self.primary.xyz
        self.grid = None
        self.nodes = {}
        self.families, self.vertices, self.relations = [], [], []
        self.raw_record_count = 0
        self.observations, self.regions = [], []
        try:
            self._checkpoint("BEFORE_HEIGHT_HIERARCHY", None)
            height, auxiliary, nodes = _hierarchy(self.xyz, self.config, research)
            self.grid = auxiliary["coarse"]
            self.nodes = {n.node_id: n for n in nodes}
            self.s4 = analyze_scene_s4r1(
                self.xyz,
                self.config,
                research,
                height,
                checkpoint=self._checkpoint if budget is not None else None,
            )
        except S4ObservationDeferred:
            if self.grid is None:
                self.grid = self.primary.grid("coarse")
            self.axis = dict(
                status="DEFERRED_PERFORMANCE", reason="SHARED_PRECISION_TIME_RESERVED"
            )
            self.s4 = dict(
                edges=[],
                segment_resolutions=self.observation_progress[
                    "completed_profile_segments"
                ],
            )
            self.observations, _ = _observations(self.s4["segment_resolutions"])
            self.raw_record_count = len(self.observations)
            return
        forbidden = {
            s["segment_id"]
            for s in self.s4["segment_resolutions"]
            if s.get("reference_role") in FORBIDDEN_REFERENCE_ROLES
        }
        self.axis = structural_axis_estimate(
            [e for e in self.s4["edges"] if e["segment_id"] not in forbidden],
            2 * self.config["boundary"]["merge_angle_deg"],
        )
        self.families = []
        self.vertices = []
        self.relations = []
        self.raw_record_count = 0
        self.observations = []
        self.regions = []
        if self.axis["status"] != "AXIS_ESTIMATED":
            return
        axes = [np.asarray(self.axis[k]) for k in ("axis_a", "axis_b")]
        # B1 labels the already estimated orthogonal pair; it never changes it.
        longitudinal = int(np.argmax([abs(a[0]) for a in axes]))
        self.axes = [axes[longitudinal], axes[1 - longitudinal]]
        if self.axes[0][0] < 0:
            self.axes[0] = -self.axes[0]
        if self.axes[1][1] < 0:
            self.axes[1] = -self.axes[1]
        u, v = self.grid.centers()
        self.xy = np.stack((u, v), axis=-1)
        self.projected = self.xy @ np.asarray(self.axes).T
        self._families()
        self._partition_nodes()
        self._vertices()
        self._relations()

    def _partition_nodes(self):
        nominations = getattr(self, "legacy_nominations", ())
        if not nominations:
            return
        from .region_partition import partition_current_nodes

        self.nodes, self.node_partition_audit = partition_current_nodes(
            self.nodes, self.xy, self.families, self.primary, nominations
        )

    def _checkpoint(self, stage, record):
        """Keep finished evidence; incomplete S4 never authorizes a new merge."""
        self.observation_progress["last_stage"] = stage
        if stage == "SEGMENT_DECK":
            self.observation_progress["completed_deck_segments"].append(record)
        elif stage == "PROFILE":
            self.observation_progress["completed_profile_segments"].append(record)
        callback = getattr(getattr(self, "cache", None), "review_checkpoint", None)
        if callback is not None:
            callback(stage, record)
        if self.budget is None:
            return
        reserve = (
            self.budget.LIMITS["FINAL_PRECISION"]
            + self.budget.LIMITS["UNIFIED_PHYSICAL_VALIDATION"]
        )
        if time.perf_counter() >= self.budget.started + self.budget.total_s - reserve:
            raise S4ObservationDeferred("S4_OBSERVATION_DEFERRED_FOR_SHARED_PRECISION")
        self.budget.check()

    def _families(self):
        observations, _ = _observations(self.s4["segment_resolutions"])
        for observation in observations:
            observation["observation_id"] = canonical(
                [
                    self.vessel_id,
                    observation["segment_id"],
                    observation["evidence_type"],
                    observation["source_index"],
                ]
            )
        self.observations = observations
        self.raw_record_count = len(observations)
        modes = _candidate_modes(observations, self.axes, self.config)
        primitives = []
        gap = self.config["boundary"]["max_support_gap_m"]
        segments = {s["segment_id"]: s for s in self.s4["segment_resolutions"]}
        for (node, axis), rows in sorted(modes.items()):
            for row in rows:
                primitives.append(
                    dict(
                        row,
                        axis=axis,
                        node_id=node,
                        source_ids=row["observation_ids"],
                        evidence_types=sorted(
                            {
                                observations[i]["evidence_type"]
                                for i in row["observation_indexes"]
                            }
                        ),
                    )
                )
        # Fitted S4 edges retain their actual supporting profile intervals,
        # rather than treating the whole endpoint span as observed support.
        for edge in self.s4["edges"]:
            segment = segments[edge["segment_id"]]
            if segment.get("reference_role") in FORBIDDEN_REFERENCE_ROLES:
                continue
            a, b = np.asarray(edge["a_raw"][:2]), np.asarray(edge["b_raw"][:2])
            direction = (b - a) / np.linalg.norm(b - a)
            axis = int(np.argmax([abs(direction @ ax) for ax in self.axes]))
            if abs(direction @ self.axes[1 - axis]) > math.sin(
                math.radians(2 * self.config["boundary"]["merge_angle_deg"])
            ):
                continue
            ids = [
                i
                for i, o in enumerate(observations)
                if o["segment_id"] == edge["segment_id"]
                and abs(np.cross(direction, np.asarray(o["xy"]) - a))
                <= self.config["boundary"]["line_inlier_m"]
            ]
            if not ids:
                continue
            xy = np.asarray([observations[i]["xy"] for i in ids])
            rho = xy @ self.axes[1 - axis]
            along = xy @ self.axes[axis]
            primitives.append(
                dict(
                    axis=axis,
                    node_id=edge["node_id"],
                    rho=float(np.median(rho)),
                    residual_p95_m=max(
                        edge["fit_residual_p95_m"], float(np.ptp(rho) / 2)
                    ),
                    along_values=along.tolist(),
                    source_segment_ids=[edge["segment_id"]],
                    source_ids=[observations[i]["observation_id"] for i in ids],
                    evidence_types=[edge["evidence_type"]],
                )
            )
        for axis in (0, 1):
            groups = []
            for row in sorted(
                (r for r in primitives if r["axis"] == axis),
                key=lambda r: (r["rho"], r["source_segment_ids"]),
            ):
                if groups and abs(
                    row["rho"] - np.median([r["rho"] for r in groups[-1]])
                ) <= (
                    self.config["boundary"]["corner_join_m"]
                    + row["residual_p95_m"]
                    + max(r["residual_p95_m"] for r in groups[-1])
                ):
                    groups[-1].append(row)
                else:
                    groups.append([row])
            for group in groups:
                ids = sorted({i for r in group for i in r["source_ids"]})
                segment_ids = sorted(
                    {i for r in group for i in r["source_segment_ids"]}
                )
                values = [v for r in group for v in r["along_values"]]
                intervals = merge_intervals(_observed_intervals(values, gap))
                if not intervals:
                    continue
                rho = float(np.median([r["rho"] for r in group]))
                uncertainty = max(
                    abs(r["rho"] - rho) + r["residual_p95_m"] for r in group
                )
                self.families.append(
                    dict(
                        family_id=_ident(
                            "S4F", [self.vessel_id, axis, segment_ids, ids]
                        ),
                        vessel_id=self.vessel_id,
                        axis_role="LONGITUDINAL" if axis == 0 else "TRANSVERSE",
                        coordinate_frame="SELECTED_VESSEL_S4_AXES",
                        direction_uv=self.axes[axis].tolist(),
                        normal_uv=self.axes[1 - axis].tolist(),
                        direction_axis=axis,
                        rho=rho,
                        rho_uncertainty=uncertainty,
                        observed_intervals=[list(i) for i in intervals],
                        support_span=[min(values), max(values)],
                        source_height_nodes=sorted({r["node_id"] for r in group}),
                        source_segment_ids=segment_ids,
                        source_provider_ids=sorted(
                            {
                                self.vessel_id
                                + ":S4:"
                                + observations[i]["segment_id"]
                                + ":"
                                + observations[i]["evidence_type"]
                                for i, o in enumerate(observations)
                                if o["observation_id"] in ids
                            }
                        ),
                        source_observation_ids=ids,
                        evidence_types=sorted(
                            {t for r in group for t in r["evidence_types"]}
                        ),
                        topology_support="SUPPORTING",
                        steel_edge_certified=False,
                    )
                )

    def _family_touch(self, mask, family):
        axis = family["direction_axis"]
        band = max(self.grid.cell_m, family["rho_uncertainty"])
        near = abs(self.projected[:, :, 1 - axis] - family["rho"]) <= band
        supported = np.zeros(mask.shape, bool)
        for lo, hi in family["observed_intervals"]:
            supported |= (self.projected[:, :, axis] >= lo - self.grid.cell_m) & (
                self.projected[:, :, axis] <= hi + self.grid.cell_m
            )
        # A coaming can sit immediately outside a height mask. This is the
        # existing coarse-cell neighbourhood, not unobserved gap completion.
        return bool(
            np.any(ndimage.binary_dilation(mask, iterations=1) & near & supported)
        )

    def _vertices(self):
        for ident, node in sorted(self.nodes.items()):
            families = [
                f["family_id"]
                for f in self.families
                if self._family_touch(node.mask, f)
                or self._terminal_search_support(node.mask, f)
            ]
            xy = self.xy[node.mask]
            self.vertices.append(
                dict(
                    node_id=ident,
                    vessel_id=self.vessel_id,
                    source_height_nodes=[ident],
                    side_family_ids=families,
                    centroid=np.mean(xy, axis=0).tolist(),
                    opening_interior_cells_rc=np.argwhere(node.mask).tolist(),
                    height_level_m=float(node.level_m),
                )
            )
        original = {v["node_id"]: list(v["side_family_ids"]) for v in self.vertices}
        for vertex in self.vertices:
            node = self.nodes[vertex["node_id"]]
            if self._opposite_terminals(original[vertex["node_id"]], node.mask):
                continue
            owners = []
            for parent in self.vertices:
                ancestor = self.nodes[parent["node_id"]]
                if ancestor.level_m <= node.level_m:
                    continue
                if getattr(ancestor, "opening_domain", None) != getattr(
                    node, "opening_domain", None
                ):
                    continue
                overlap = int(np.count_nonzero(ancestor.mask & node.mask))
                if (
                    overlap
                    < node.mask.sum() * self.config["roi"]["min_enclosure_ratio"]
                ):
                    continue
                terminals = self._opposite_terminals(
                    original[parent["node_id"]], ancestor.mask
                )
                if terminals:
                    owners.append(
                        dict(
                            node_id=parent["node_id"],
                            observed_member_overlap=overlap,
                            terminal_family_ids=terminals,
                        )
                    )
            if owners:
                # Ownership follows measured member cells, not bbox overlap.
                # This attaches existing end-wall search evidence; it does not
                # fabricate Raw support along unobserved parts of those walls.
                vertex["side_family_ids"] = sorted(
                    set(vertex["side_family_ids"])
                    | {f for owner in owners for f in owner["terminal_family_ids"]}
                )
                vertex["terminal_topology_assignment"] = owners

    def _terminal_search_support(self, mask, family):
        """Associate an observed end wall with a receding interior height lobe.

        This is a topology search association, not measured support at a gap.
        The existing structural search radius bounds it. A lone longitudinal
        cargo contour cannot supply the second end or connect two openings.
        """
        if family["axis_role"] != "TRANSVERSE" or not set(
            family["evidence_types"]
        ).intersection(("OBSERVED_3D_FACE", "OBSERVED_PROFILE_BREAK")):
            return False
        points = self.projected[mask]
        low, high = points.min(0), points.max(0)
        distance = min(abs(family["rho"] - low[0]), abs(family["rho"] - high[0]))
        overlap = sum(
            max(0.0, min(hi, high[1]) - max(lo, low[1]))
            for lo, hi in family["observed_intervals"]
        )
        return bool(
            distance <= self.config["roi"]["support_search_m"]
            and overlap
            >= (high[1] - low[1]) * self.config["roi"]["min_enclosure_ratio"]
        )

    def _opposite_terminals(self, family_ids, mask):
        """Two current observed terminal walls, never a four-line product."""
        if not mask.any():
            return []
        points = self.projected[mask]
        center = float(np.mean(points[:, 0]))
        eligible = [
            f
            for f in self.families
            if f["family_id"] in family_ids
            and f["axis_role"] == "TRANSVERSE"
            and set(f["evidence_types"]).intersection(
                ("OBSERVED_3D_FACE", "OBSERVED_PROFILE_BREAK")
            )
        ]
        left = [f for f in eligible if f["rho"] < center]
        right = [f for f in eligible if f["rho"] > center]
        if not left or not right:
            return []
        # Extent is constrained by observed terminals. Candidate modes remain
        # separate; selecting a review rho is not this compiler's job.
        if max(f["rho"] for f in right) - min(f["rho"] for f in left) < (
            self.config["roi"]["min_deck_span_m"]
        ):
            return []
        return sorted(f["family_id"] for f in left + right)

    def _split_evidence(self, first, second):
        a, b = np.asarray(first["centroid"]), np.asarray(second["centroid"])
        result = []
        # A measured transverse face between disjoint interiors is positive
        # split evidence. Nested height levels are not split by their end rim.
        overlap = self.nodes[first["node_id"]].mask & self.nodes[second["node_id"]].mask
        for family in self.families:
            if family["axis_role"] != "TRANSVERSE":
                continue
            signs = [(p @ self.axes[0]) - family["rho"] for p in (a, b)]
            normal_gate = (
                self.config["boundary"]["corner_join_m"] + family["rho_uncertainty"]
            )
            separated = (
                signs[0] * signs[1] < 0 and min(abs(s) for s in signs) > normal_gate
            )
            masks = [
                self.projected[self.nodes[v["node_id"]].mask, 1]
                for v in (first, second)
            ]
            low, high = max(m.min() for m in masks), min(m.max() for m in masks)
            covered = sum(
                max(0, min(hi, high) - max(lo, low))
                for lo, hi in family["observed_intervals"]
            )
            raw_face = "OBSERVED_3D_FACE" in family["evidence_types"]
            qualified = (
                separated
                and not overlap.any()
                and raw_face
                and high > low
                and covered >= (high - low) * self.config["roi"]["min_enclosure_ratio"]
            )
            result.append(
                dict(
                    kind="TRANSVERSE_COAMING_OR_END_WALL",
                    family_id=family["family_id"],
                    qualified=bool(qualified),
                    status="SPLIT_SUPPORT" if qualified else "UNOBSERVED_OR_INCOMPLETE",
                )
            )
            profile_supported = "OBSERVED_PROFILE_BREAK" in family["evidence_types"]
            result.append(
                dict(
                    kind="STRONG_SEPARATOR_PROFILE",
                    family_id=family["family_id"],
                    qualified=bool(qualified and profile_supported),
                    status=(
                        "SPLIT_SUPPORT"
                        if qualified and profile_supported
                        else "UNOBSERVED_OR_INCOMPLETE"
                    ),
                    corroboration="VALID_S4_PROFILE_CONSENSUS_AND_RAW_FACE",
                )
            )
        boxes = [self.nodes[v["node_id"]].bbox_xy for v in (first, second)]
        left, right = sorted(boxes, key=lambda box: box[0])
        from .layout import cross_deck_evidence

        cross = cross_deck_evidence(self.cache, left, right)
        result.append(
            dict(kind="CROSS_DECK", qualified=bool(cross["can_split"]), evidence=cross)
        )
        exterior = getattr(self.cache, "s4_exterior", None)
        confirmed = np.zeros(overlap.shape, bool)
        if exterior is not None:
            level = max(first["height_level_m"], second["height_level_m"])
            labels, outside_ids, _ = exterior.at_level(level)
            rows, cols = exterior.grid.cell_indices(self.xy.reshape(-1, 2))
            valid = (
                (rows >= 0)
                & (cols >= 0)
                & (rows < labels.shape[0])
                & (cols < labels.shape[1])
            )
            flat = confirmed.ravel()
            flat[valid] = np.isin(labels[rows[valid], cols[valid]], list(outside_ids))
        masks = [self.nodes[v["node_id"]].mask & ~confirmed for v in (first, second)]
        remaining, _ = ndimage.label(
            masks[0] | masks[1], structure=np.ones((3, 3), bool)
        )
        a_ids = set(np.unique(remaining[masks[0]])) - {0}
        b_ids = set(np.unique(remaining[masks[1]])) - {0}
        exterior_split = bool(
            confirmed.any() and a_ids and b_ids and not a_ids.intersection(b_ids)
        )
        result.append(
            dict(
                kind="CONFIRMED_EXTERIOR_STRIP",
                qualified=exterior_split,
                confirmed_cells=int(confirmed.sum()),
                status=(
                    "SPLIT_SUPPORT"
                    if exterior_split
                    else "UNOBSERVED_OR_NO_PROVEN_BARRIER"
                ),
            )
        )
        return result

    def _relations(self):
        for first, second in itertools.combinations(self.vertices, 2):
            shared = sorted(
                set(first["side_family_ids"]) & set(second["side_family_ids"])
            )
            shared_longitudinal = [
                i
                for i in shared
                if next(f for f in self.families if f["family_id"] == i)["axis_role"]
                == "LONGITUDINAL"
            ]
            a, b = self.nodes[first["node_id"]].mask, self.nodes[second["node_id"]].mask
            overlap = a & b
            labels, _ = ndimage.label(a | b, structure=np.ones((3, 3), bool))
            common_components = set(np.unique(labels[a])) & set(
                np.unique(labels[b])
            ) - {0}
            shared_terminals = self._opposite_terminals(shared, a | b)
            positive_shared = sorted(set(shared_longitudinal + shared_terminals))
            first_domain = getattr(self.nodes[first["node_id"]], "opening_domain", None)
            second_domain = getattr(
                self.nodes[second["node_id"]], "opening_domain", None
            )
            domains_compatible = first_domain == second_domain
            if (first_domain or second_domain) and not domains_compatible:
                positive_shared = []
            continuity = dict(
                observed_continuity=bool(common_components and positive_shared),
                overlapping_interior_cells=int(overlap.sum()),
                observed_connected_components=sorted(int(i) for i in common_components),
                geometry_source="ACTUAL_HEIGHT_MASK_CELLS_NOT_BBOX_OVERLAP",
                shared_longitudinal_family_ids=shared_longitudinal,
                shared_opposite_terminal_family_ids=shared_terminals,
                partition_ownership_compatible=domains_compatible,
            )
            split = self._split_evidence(first, second)
            if (
                first_domain
                and second_domain
                and first_domain["domain_id"] == second_domain["domain_id"]
                and first_domain["side"] != second_domain["side"]
            ):
                split.append(
                    dict(
                        kind="CURRENT_RAW3D_BEV_TWO_FACE_SEPARATOR",
                        qualified=True,
                        separator_faces=first_domain["separator_faces"],
                    )
                )
            self.relations.append(
                topology_relation(
                    first,
                    second,
                    positive_shared,
                    continuity,
                    split,
                )
            )

    def compile_regions(self, existing):
        """Called by the V2 compiler; preserve old nominations and all provenance."""
        if self.axis["status"] != "AXIS_ESTIMATED":
            self.regions = list(existing)
            return self.regions
        by_id = {v["node_id"]: v for v in self.vertices}
        consumed = set()
        result = []
        groups = topology_components(self.vertices, self.relations)
        components = [set(g) for g in groups]
        ambiguous_members = set()
        split_pairs = {
            frozenset(r["node_ids"])
            for r in self.relations
            if r["relation_decision"] == "SPLIT"
        }
        for relation in self.relations:
            if (
                relation["relation_decision"] != "AMBIGUOUS"
                or not relation["shared_family_evidence"]
            ):
                continue
            a, b = relation["node_ids"]
            left = next(g for g in groups if a in g)
            right = next(g for g in groups if b in g)
            if left is right or any(
                frozenset((x, y)) in split_pairs for x in left for y in right
            ):
                continue
            # Unassigned height nominations cannot consume an independently
            # attributed opening through an AMBIGUOUS relation.
            if any(
                getattr(self.nodes[n], "opening_domain", None) for n in left | right
            ):
                continue
            left.update(right)
            groups.remove(right)
            ambiguous_members.update(left)
        for members in groups:
            compound = bool(members.intersection(ambiguous_members))
            attached = sorted({i for n in members for i in by_id[n]["side_family_ids"]})
            families = [f for f in self.families if f["family_id"] in attached]
            longitudinal = [f for f in families if f["axis_role"] == "LONGITUDINAL"]
            observed = np.logical_or.reduce(
                [self.nodes[n].mask for n in sorted(members)]
            )
            terminals = self._opposite_terminals(attached, observed)
            if not longitudinal and not terminals:
                continue
            # Geometry is measured interior cells after a positive graph
            # decision. Bounds never decide whether two vertices connect.
            masks = [self.nodes[n].mask for n in sorted(members)]
            observed = np.logical_or.reduce(masks)
            common = np.logical_and.reduce(masks)
            if terminals:
                # Opposite current end-wall evidence owns the widthwise lobes.
                # Their observed union is not the empty intersection of top
                # and bottom cargo-height slices. No unobserved cells are added.
                interior = observed
            else:
                low = min(f["rho"] - f["rho_uncertainty"] for f in longitudinal)
                high = max(f["rho"] + f["rho_uncertainty"] for f in longitudinal)
                corridor = (self.projected[:, :, 1] >= low - self.grid.cell_m) & (
                    self.projected[:, :, 1] <= high + self.grid.cell_m
                )
                interior = (observed if compound else common) & corridor
            if not interior.any():
                continue
            xy = self.xy[interior]
            bounds = tuple(
                np.r_[
                    xy.min(0) - self.grid.cell_m / 2, xy.max(0) + self.grid.cell_m / 2
                ]
            )
            family_xy = [
                along * self.axes[f["direction_axis"]]
                + (f["rho"] + offset) * self.axes[1 - f["direction_axis"]]
                for f in families
                for along in f["support_span"]
                for offset in (-f["rho_uncertainty"], f["rho_uncertainty"])
            ]
            all_xy = np.vstack((self.xy[observed], family_xy))
            angular = orientation_search_budget(
                self.axis,
                bounds,
                self.config["roi"]["support_search_m"],
                self.config["boundary"]["profile_half_width_m"],
            )
            margin = angular["required_search_margin_m"]
            envelope = tuple(
                np.r_[
                    np.minimum(all_xy.min(0), bounds[:2]) - margin,
                    np.maximum(all_xy.max(0), bounds[2:]) + margin,
                ]
            )
            source = self._source()
            relation_rows = [
                r for r in self.relations if set(r["node_ids"]).issubset(members)
            ]
            detail = dict(
                member_height_nodes=sorted(members),
                side_family_ids=attached,
                relations=relation_rows,
                orientation_budget=angular,
                interior_cells_rc=np.argwhere(interior).tolist(),
                bounds_source="OBSERVED_INTERIOR_AFTER_TOPOLOGY_CONNECT",
                opposite_terminal_family_ids=terminals,
                search_envelope_source="MEMBER_MASKS_S4_FAMILIES_AND_UNCERTAINTY",
            )
            if compound:
                detail["member_components"] = [
                    sorted(g) for g in components if g.issubset(members)
                ]
                detail["relationship_decision"] = (
                    "AMBIGUOUS_SEARCH_GROUP_NOT_A_MERGED_HATCH"
                )
            support = TopologySupport(
                "S4_OPENING_TOPOLOGY", "SUPPORTING", tuple(attached), source, detail
            )
            ident = _ident("S4R", [self.vessel_id, sorted(members), attached])
            region = OpeningRegion(
                ident,
                self.vessel_id,
                bounds,
                envelope,
                (support,),
                tuple(sorted(members)),
                tuple(self.nodes[n].bbox_xy for n in sorted(members)),
                "REGION_AMBIGUOUS" if compound else "REGION_PARTIAL",
                precision_status="NOT_CERTIFIED",
            )
            region.relationship = (
                "COMPOUND_OPENING_GROUP"
                if compound
                else "INDEPENDENT_OPENING_CANDIDATE"
            )
            region.identity_role = (
                "COMPOUND_SEARCH_GROUP"
                if compound
                else (
                    "CONFIRMED_CONNECTED_OPENING"
                    if len(members) > 1
                    else "CONFIRMED_INDEPENDENT_OPENING"
                )
            )
            independent_scope = any(
                int(
                    np.count_nonzero(
                        observed
                        & np.all(
                            (self.xy >= row["bounds_axial"][:2])
                            & (self.xy <= row["bounds_axial"][2:]),
                            axis=-1,
                        )
                    )
                )
                >= max(
                    self.config["roi"]["min_opening_cells"],
                    observed.sum() * self.config["roi"]["min_enclosure_ratio"],
                )
                for row in getattr(self, "legacy_nominations", ())
                if row.get("bounds_axial")
                and row.get("measured_strong_side_count", 0) >= 3
                and row.get("opening_evidence", {}).get("status") == "OPENING_LOWER"
                and not row.get("role_conflicts")
            )
            if not compound and not terminals and not independent_scope:
                region.identity_role = "WEAK_UNASSIGNED_REGION"
            region.identity_evidence = dict(
                source="CURRENT_S4_TOPOLOGY",
                member_height_nodes=sorted(members),
                opposite_terminal_family_ids=terminals,
                member_support_cells_rc=np.argwhere(interior).tolist(),
                member_support_grid=dict(
                    x0=self.grid.x0, y0=self.grid.y0, cell_m=self.grid.cell_m
                ),
                support_snapshot_sha256=hashlib.sha256(
                    np.argwhere(interior).astype("<i8").tobytes()
                ).hexdigest(),
                current_independent_enclosure_scope=independent_scope,
            )
            # Associate a recall nomination only through its observed interior
            # cells in this proven component. No nomination bbox expands it.
            for prior in existing:
                assignment = member_cell_assignment(
                    self.xy,
                    {n: self.nodes[n].mask for n in members},
                    bounds,
                    prior.bounds,
                )
                if assignment["member_cell_counts"]:
                    consumed.add(prior.region_id)
                    detail.setdefault("associated_region_nominations", []).append(
                        dict(
                            region_id=prior.region_id,
                            source_proposal_ids=list(prior.source_proposal_ids),
                            **assignment,
                        )
                    )
                    region.source_proposal_ids = tuple(
                        sorted(
                            set(region.source_proposal_ids)
                            | set(prior.source_proposal_ids)
                        )
                    )
            # Every side retains its prior/unknown measurement. Topology
            # support does not convert a raster or a height into measured steel.
            for side, axis, coordinate in (
                ("U0", 1, bounds[0]),
                ("U1", 1, bounds[2]),
                ("V0", 0, bounds[1]),
                ("V1", 0, bounds[3]),
            ):
                normal = 0 if side.startswith("U") else 1
                near = []
                for family in families:
                    if family["direction_axis"] != axis:
                        continue
                    anchor = (
                        np.mean(family["support_span"]) * self.axes[axis]
                        + family["rho"] * self.axes[1 - axis]
                    )
                    if (
                        abs(anchor[normal] - coordinate)
                        <= self.config["roi"]["support_search_m"]
                        + family["rho_uncertainty"]
                    ):
                        near.append(family["family_id"])
                region.side_topology_support[side] = TopologySupport(
                    "STRUCTURAL_LINE_FAMILY",
                    "SUPPORTING" if near else "UNOBSERVED",
                    tuple(near),
                    source,
                )
                region.boundary_measurements[side].detail["topology_family_ids"] = near
            result.append(region)
        result.extend(r for r in existing if r.region_id not in consumed)
        self.regions = result
        return result

    def _source(self):
        frame = getattr(self.cache, "input_frame", None)
        transform = np.eye(4)
        transform[:2, :2] = self.cache.vessel.axes
        return EvidenceSource(
            frame.frame_id if frame else "UNKNOWN",
            str(frame.provenance.get("sensor_id", "UNKNOWN")) if frame else "UNKNOWN",
            frame.stamp if frame else "UNKNOWN",
            None,
            tuple(tuple(row) for row in transform),
            (self.vessel_id,),
        )

    def audit(self):
        return dict(
            status=(
                "S4_EVIDENCE_COMPILED"
                if self.axis["status"] == "AXIS_ESTIMATED"
                else (
                    "S4_EVIDENCE_DEFERRED_PERFORMANCE"
                    if self.axis["status"] == "DEFERRED_PERFORMANCE"
                    else "AXIS_AMBIGUOUS"
                )
            ),
            vessel_id=self.vessel_id,
            raw_record_count=self.raw_record_count,
            side_family_count=len(self.families),
            topology_vertex_count=len(self.vertices),
            topology_relation_count=len(self.relations),
            compiled_region_count=sum(
                any(s.source_type == "S4_OPENING_TOPOLOGY" for s in r.topology_support)
                for r in self.regions
            ),
            registry_region_count=len(self.regions),
            four_line_combination_enumeration=False,
            legacy_rectangle_input_count=0,
            side_families=self.families,
            topology_vertices=self.vertices,
            raw_observations=self.observations,
            topology_relations=self.relations,
            structural_axis=self.axis,
            observation_progress=getattr(self, "observation_progress", {}),
            node_partition_audit=getattr(self, "node_partition_audit", []),
            observation_grid=dict(
                x0=self.grid.x0, y0=self.grid.y0, cell_m=self.grid.cell_m
            ),
            s4_axes_in_vessel_uv=(
                [a.tolist() for a in self.axes] if hasattr(self, "axes") else None
            ),
            selected_primary_point_index_sha256=self.primary.vessel.point_index_sha256,
            geometry_role="TOPOLOGY_EVIDENCE_NOT_FINAL_PRECISION",
        )
