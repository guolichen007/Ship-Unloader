"""Candidate-local Raw deck families; ambiguous planes never become one deck."""

from .precision_diagnostics import stage as precision_stage, side_completed

from collections import OrderedDict
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree

from ship_perception.r1_static.model import PerimeterSegment
from ship_perception.r1_static.perimeter_deck import estimate_segment_deck
from .plane_profile import observed_profile
from .sections import physical_sections
from .raw_face_refinement import face_position


class LocalDeckProfileCache:
    def __init__(self, measurement, config):
        self.measurement, self.config = measurement, config
        self.entries = OrderedDict()
        self.bytes = 0
        self.enabled = True

    def _check(self):
        budget = getattr(self.measurement, "budget", None)
        if budget is not None:
            budget.check()

    def _count(self, name):
        counters = self.measurement.counters
        counters[name] = counters.get(name, 0) + 1

    @precision_stage("height")
    def section(self, axis, anchor, span, sign, profile_at=None):
        key = (axis, float(anchor), tuple(span), sign, profile_at)
        if self.enabled and key in self.entries:
            self.entries.move_to_end(key)
            self._count("local_deck_profile_cache_hit")
            return self.entries[key]
        self._check()
        self._count("local_deck_profile_build_count")
        config = self.config
        band = (
            config["roi"]["support_search_m"]
            + config["boundary"]["profile_half_length_m"]
        )
        half = config["boundary"]["profile_half_width_m"]
        box = [0.0] * 4
        box[axis], box[axis + 2] = anchor - band, anchor + band
        box[1 - axis], box[3 - axis] = span[0] - half, span[1] + half
        indexes = self.measurement.spatial.query_indexes(box)
        points = self.measurement.raw_uvz[indexes]
        entry = dict(points=points, indexes=indexes, profiles=[], planes=[])
        if len(points):
            tree = cKDTree(points[:, :2])
            start, end, tangent, outward = (np.zeros(2) for _ in range(4))
            start[axis] = end[axis] = anchor
            start[1 - axis], end[1 - axis] = span
            tangent[1 - axis], outward[axis] = 1.0, -sign
            segment = PerimeterSegment(
                "LOCAL_RAW",
                "LOCAL_RAW",
                tuple(start),
                tuple(end),
                tuple(tangent),
                tuple(outward),
                span[1] - span[0],
                "UNKNOWN",
            )
            deck, _, forensic = estimate_segment_deck(points, segment, config, tree)
            entry["deck_status"] = deck["status"]
            spacing = max(
                config["boundary"]["profile_step_m"],
                min(
                    config["boundary"]["max_support_gap_m"] * 0.8,
                    config["geometry"]["coarse_voxel_m"] / 2,
                ),
            )
            for i, value in enumerate(deck["candidate_plane_normals"]):
                self._check()
                ids = forensic["candidate_raw_ids"][i]
                if (
                    deck["candidate_outward_widths"][i]
                    < config["roi"]["deck_patch_width_m"]
                    or deck["candidate_along_coverages"][i]
                    < config["roi"]["deck_patch_fill_min"]
                    or np.ptp(points[ids, 1 - axis]) < config["roi"]["min_deck_span_m"]
                ):
                    continue
                normal = np.asarray(value)
                offset = -float(
                    normal[:2] @ ((start + end) / 2)
                    + normal[2] * deck["candidate_plane_heights"][i]
                )
                global_ids = self.measurement.raw_point_indexes[indexes[ids]]
                plane = dict(
                    normal=normal.tolist(),
                    offset=offset,
                    raw_support_count=len(ids),
                    raw_support_sha256=hashlib.sha256(
                        np.sort(global_ids).astype("<i8", copy=False).tobytes()
                    ).hexdigest(),
                    outward_width_m=deck["candidate_outward_widths"][i],
                    along_coverage=deck["candidate_along_coverages"][i],
                    residual_p95_m=float(
                        np.quantile(abs(points[ids] @ normal + offset), 0.95)
                    ),
                )
                plane_index = len(entry["planes"])
                entry["planes"].append(plane)
                positions = (
                    [profile_at]
                    if profile_at is not None
                    else np.arange(span[0] + spacing / 2, span[1], spacing)
                )
                for along in positions:
                    self._check()
                    self._count("local_deck_raw_profile_query_count")
                    center = start.copy()
                    center[1 - axis] = along
                    observed = observed_profile(
                        points,
                        tree,
                        center,
                        tangent,
                        -outward,
                        (normal, offset),
                        config,
                    )
                    if observed is None:
                        continue
                    drop = (
                        observed["break_strength"]
                        * 2
                        * observed["residual"]
                        / normal[2]
                    )
                    if drop < config["roi"]["opening_drop_m"]:
                        continue
                    entry["profiles"].append(
                        dict(
                            along=float(along),
                            rho=observed["break_position"][axis],
                            drop_m=float(drop),
                            uncertainty_m=observed["residual"],
                            plane_index=plane_index,
                            query_anchor_m=float(anchor),
                        )
                    )
        size = (
            points.nbytes
            + indexes.nbytes
            + len(
                json.dumps(
                    {k: v for k, v in entry.items() if k not in ("points", "indexes")}
                ).encode()
            )
        )
        while self.entries and self.bytes + size > 64 * 1024 * 1024:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= removed["cache_bytes"]
        if self.enabled and size <= 64 * 1024 * 1024:
            entry["cache_bytes"] = size
            self.entries[key] = entry
            self.bytes += size
        return entry

    def evidence(self, axis, anchor, span, sign, face=None):
        config = self.config
        edges = physical_sections(span, config["geometry"]["refine_voxel_m"], config)
        whole = self.section(axis, anchor, span, sign)
        return [
            self.section_evidence(axis, anchor, (lo, hi), sign, face, whole)
            for lo, hi in zip(edges[:-1], edges[1:])
        ]

    @precision_stage("height")
    def section_evidence(
        self, axis, anchor, span, sign, face=None, whole=None, patches=None
    ):
        config = self.config
        rows = []
        for lo, hi in [span]:
            self._check()
            rho = anchor if face is None else face_position(face, (lo + hi) / 2)
            entry = self.section(axis, rho, (lo, hi), sign)
            points, indexes = entry["points"], entry["indexes"]
            outside, inside = np.zeros(len(points), bool), np.zeros(len(points), bool)
            near_outside = np.zeros(len(points), bool)
            plane_residuals = {}
            used = []
            used_planes = set()
            along_values = points[:, 1 - axis]
            if face is None:
                physical_rho = rho
            else:
                a, b = np.asarray(face["a_xy"]), np.asarray(face["b_xy"])
                slope = (b[axis] - a[axis]) / (b[1 - axis] - a[1 - axis])
                physical_rho = rho + slope * (along_values - (lo + hi) / 2)
            signed = sign * (points[:, axis] - physical_rho)
            observations = [
                (profile, entry["planes"][profile["plane_index"]])
                for profile in entry["profiles"]
            ]
            if whole is not None:
                observations.extend(
                    (profile, whole["planes"][profile["plane_index"]])
                    for profile in whole["profiles"]
                    if lo <= profile["along"] <= hi
                )
            for patch in patches or []:
                observations.extend(
                    (profile, patch["planes"][profile["plane_index"]])
                    for profile in patch["profiles"]
                )
            for profile, plane in observations:
                expected = (
                    rho if face is None else face_position(face, profile["along"])
                )
                if abs(profile["rho"] - expected) > config["boundary"]["corner_join_m"]:
                    continue
                normal = np.asarray(plane["normal"])
                plane_key = (tuple(plane["normal"]), plane["offset"])
                if plane_key not in plane_residuals:
                    plane_residuals[plane_key] = (
                        points @ normal + plane["offset"]
                    ) / normal[2]
                residual = plane_residuals[plane_key]
                tube = (
                    abs(along_values - profile["along"])
                    <= config["boundary"]["profile_half_width_m"]
                ) & (
                    abs(points[:, axis] - profile.get("query_anchor_m", rho))
                    <= config["boundary"]["profile_half_length_m"]
                )
                # Profile construction sees padded neighbours, but each
                # physical section must own its counted Raw support itself.
                tube &= (along_values >= lo) & (along_values <= hi)
                observed_signed = (
                    sign * (points[:, axis] - profile["rho"])
                    if face is None
                    else signed
                )
                near_outside |= (
                    tube
                    & (observed_signed <= 0)
                    & (abs(residual) <= config["roi"]["support_band_m"])
                )
                # The wall supplies geometric position; reference height is
                # independently supported by the measured outside plane in the
                # existing support ROI. Keep these actual Raw counts distinct
                # from near-step pairs rather than inventing missing returns.
                outside |= (
                    (along_values >= lo)
                    & (along_values <= hi)
                    & (observed_signed <= 0)
                    & (observed_signed >= -config["roi"]["support_search_m"])
                    & (
                        abs(residual) * normal[2]
                        <= config["geometry"]["plane_inlier_m"]
                    )
                )
                inside |= (
                    tube
                    & (observed_signed > 0)
                    & (residual <= -config["roi"]["opening_drop_m"])
                )
                used.append(profile)
                used_planes.add(plane["raw_support_sha256"])
            counts = [int(outside.sum()), int(inside.sum())]
            enough = (
                bool(used) and min(counts) >= config["boundary"]["profile_min_points"]
            )
            unique = (
                face is not None
                or not used
                or np.ptp([p["rho"] for p in used])
                <= config["boundary"]["corner_join_m"]
            )
            if face is None and used:
                rho = float(np.median([p["rho"] for p in used]))
            if not (enough and unique) and patches is None:
                spacing = max(
                    config["boundary"]["profile_step_m"],
                    min(
                        config["boundary"]["max_support_gap_m"] * 0.8,
                        config["geometry"]["coarse_voxel_m"] / 2,
                    ),
                )
                length = (
                    config["roi"]["min_deck_span_m"]
                    + 2 * config["boundary"]["profile_half_width_m"]
                )
                patches = []
                for along in np.arange(lo + spacing / 2, hi, spacing):
                    position = anchor if face is None else face_position(face, along)
                    patches.append(
                        self.section(
                            axis,
                            position,
                            (float(along - length / 2), float(along + length / 2)),
                            sign,
                            float(along),
                        )
                    )
                return self.section_evidence(
                    axis, anchor, span, sign, face, whole, patches
                )
            ids = self.measurement.raw_point_indexes[indexes[outside | inside]]
            rows.append(
                dict(
                    span=[lo, hi],
                    rho=float(rho),
                    supported=bool(enough and unique),
                    geometry_status=(
                        "OBSERVED_PROFILE_OR_FACE"
                        if unique and enough
                        else (
                            "MULTIMODE_PROFILE_GEOMETRY" if not unique else "UNOBSERVED"
                        )
                    ),
                    profile_count=len(used),
                    counts=counts,
                    near_profile_support_counts=[
                        int(near_outside.sum()),
                        int(inside.sum()),
                    ],
                    deck_reference_support_count=counts[0],
                    height_support_basis="MEASURED_OUTSIDE_LOCAL_PLANE_AND_OBSERVED_NEAR_FACE_LOWER_RETURNS",
                    min_observed_drop_m=min((p["drop_m"] for p in used), default=None),
                    uncertainty_m=max((p["uncertainty_m"] for p in used), default=0.0),
                    profile_raw_support_sha256=hashlib.sha256(
                        np.sort(ids).astype("<i8", copy=False).tobytes()
                    ).hexdigest(),
                    measured_plane_count=len(used_planes),
                    plane_support_sha256=sorted(used_planes),
                    source="OBSERVED_SECTION_LOCAL_DECK_TO_LOWER_PROFILE_FAMILIES",
                    geometry_inferred=False,
                )
            )
        return rows[0]
