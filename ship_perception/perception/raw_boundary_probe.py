"""Cheap exact-XYZ edge investigation, without rasters, normals or corners."""

from .precision_diagnostics import stage as precision_stage, side_completed

import time
import copy
import numpy as np

from .line_geometry import SIDE_SPECS
from .sections import physical_sections
from .consensus import SpatialEvidenceConsensus


class SignedRoleResolver:
    """Resolve only candidate-local nearby lines after nominal compilation."""

    def __init__(self, measurement, refiner):
        self.measurement, self.refiner = measurement, refiner
        self.config = refiner.config
        self._nomination_progress = {}
        self._anchor_results = {}
        self._section_results = {}

    def raw_nominations(self, axis, anchor, span, sign):
        """Nominate within the existing structural ROI, then refine normally."""
        from .measurement import RawSectionProfile

        config = self.config
        key = ("RAW_NOMINATION", axis, anchor, tuple(span), sign)
        cache = self.measurement.raw_edge_modes
        if key in cache:
            return cache[key]
        search = config["roi"]["support_search_m"]
        fine = config["roi"]["boundary_search_m"]
        probe = config["boundary"]["side_probe_m"]
        state = self._nomination_progress.get(key)
        if state is None:
            box = [0.0] * 4
            box[axis], box[axis + 2] = anchor - search - probe, anchor + search + probe
            box[1 - axis], box[3 - axis] = span
            ids = self.measurement.spatial.query_indexes(box)
            points = self.measurement.raw_uvz[ids]
            profile = RawSectionProfile(points, axis, tuple(span), config)
            self.measurement.counters["raw_profile_build_count"] += 1
            context = self.measurement.local.measurement_context
            owned = (
                context.sources[0].point_indexes
                if context is not None
                else self.measurement.raw_point_indexes
            )
            state = dict(
                profile=profile,
                points=points,
                primary=np.isin(self.measurement.raw_point_indexes[ids], owned),
                centers=np.arange(anchor - search, anchor + search + fine, 2 * fine),
                next_center=0,
                modes=[],
            )
            self._nomination_progress[key] = state
        points, profile, primary = state["points"], state["profile"], state["primary"]
        edges = physical_sections(span, config["geometry"]["refine_voxel_m"], config)
        modes = state["modes"]
        while state["next_center"] < len(state["centers"]):
            center = state["centers"][state["next_center"]]
            if self.refiner.budget is not None:
                self.refiner.budget.check()
            for mode in profile.edge_modes(center, sign):
                rho = mode["peak_rho"]
                if abs(rho - anchor) > search or any(
                    abs(rho - old) <= config["boundary"]["corner_join_m"]
                    for old in modes
                ):
                    continue
                near = primary & (
                    abs(points[:, axis] - rho)
                    <= config["geometry"]["candidate_voxel_m"]
                )
                count = sum(
                    np.count_nonzero(
                        near & (points[:, 1 - axis] >= lo) & (points[:, 1 - axis] <= hi)
                    )
                    >= config["boundary"]["line_min_points"]
                    for lo, hi in zip(edges[:-1], edges[1:])
                )
                if count >= config["boundary"]["profile_min_sections"]:
                    modes.append(float(rho))
            state["next_center"] += 1
        self._nomination_progress.pop(key, None)
        cache[key] = tuple(modes)
        return cache[key]

    @precision_stage("raw_edge_search")
    def side(
        self, axis, anchor, span, sign, candidates, extended=False, normal_range=None,
        measurement_only=False,
    ):
        config = self.config
        anchors = [anchor]
        if extended:
            for rho in self.raw_nominations(axis, anchor, span, sign):
                if (
                    normal_range is not None
                    and (rho - sum(normal_range) / 2) * sign >= 0
                ):
                    continue
                if not any(
                    abs(rho - old) <= config["boundary"]["corner_join_m"]
                    for old in anchors
                ):
                    anchors.append(rho)
        for candidate in candidates:
            rho = candidate["rho"]
            if (
                min(candidate.get("primary_raw3d", 0), candidate.get("primary_bev", 0))
                < config["roi"]["min_enclosure_ratio"]
                or abs(rho - anchor) > config["roi"]["support_search_m"]
            ):
                continue
            if not any(
                abs(rho - old) <= config["boundary"]["corner_join_m"] for old in anchors
            ):
                anchors.append(rho)
        edges = physical_sections(span, config["geometry"]["refine_voxel_m"], config)
        spans = list(zip(edges[:-1], edges[1:]))
        from ship_perception.r1_static.rectangle_refinement import _raw_edge_rho

        providers = {
            p["provider_id"]: p for p in self.measurement.local.structural_providers()
        }
        face_anchors = {
            c["rho"]
            for c in candidates
            if any(
                "RAW3D_VERIFIED_VERTICAL_FACE"
                in providers.get(pid, {}).get("evidence_types", [])
                for pid in c.get("provider_ids", [])
            )
        }
        paired, faces, evidence = [], [], []
        observed, partial = False, False
        for rho in anchors:
            anchor_key = (axis, rho, tuple(span), sign, rho in face_anchors, extended, measurement_only)
            if anchor_key in self._anchor_results:
                old = self._anchor_results[anchor_key]
                evidence.append(copy.deepcopy(old["evidence"]))
                paired.extend(copy.deepcopy(old["paired"]))
                faces.extend(copy.deepcopy(old["faces"]))
                observed |= old["observed"]
                partial |= old["partial"]
                continue
            rows, face_rows = [], []
            anchor_observed = False
            for section in spans:
                section_key = (
                    axis,
                    rho,
                    tuple(section),
                    sign,
                    rho in face_anchors,
                    extended,
                    measurement_only,
                )
                if section_key in self._section_results:
                    cached = self._section_results[section_key]
                    rows.extend(copy.deepcopy(cached["rows"]))
                    face_rows.extend(copy.deepcopy(cached["face_rows"]))
                    anchor_observed |= cached["observed"]
                    continue
                if self.refiner.budget is not None:
                    self.refiner.budget.check()
                row_start, face_start = len(rows), len(face_rows)
                profile = self.measurement.side_measurements.profile(axis, rho, section)
                section_observed = (
                    len(profile.points) >= config["boundary"]["profile_min_points"]
                )
                anchor_observed |= section_observed
                width = config["geometry"]["candidate_voxel_m"]
                raw, count = _raw_edge_rho(
                    profile.window(rho - width, rho + width),
                    axis,
                    rho,
                    section,
                    sign,
                    config,
                )
                if raw is None and extended:
                    searched = self.refiner._raw_section(
                        self.measurement.raw_uvz, axis, rho, section, sign
                    )
                    if searched["status"] == "UNIQUE_SIGNED_RAW_STEP":
                        raw, count = (
                            searched["continuous"][0],
                            searched["continuous"][2],
                        )
                if raw is not None and (measurement_only or profile.height(raw, sign)["transition"]):
                    rows.append(
                        dict(
                            along=sum(section) / 2,
                            rho=float(raw),
                            measurement_source="RAW_PAIRED_PROFILE",
                            raw_point_count=int(count),
                        )
                    )
                elif rho in face_anchors and profile.height(rho, sign)["transition"]:
                    face_rows.append(
                        dict(
                            along=sum(section) / 2,
                            rho=float(rho),
                            measurement_source="RAW_HEIGHT_WITH_EXISTING_STRUCTURAL_FACE",
                            search_anchor_only=True,
                        )
                    )
                self._section_results[section_key] = dict(
                    rows=copy.deepcopy(rows[row_start:]),
                    face_rows=copy.deepcopy(face_rows[face_start:]),
                    observed=section_observed,
                )
            observed |= anchor_observed
            paired_start, faces_start = len(paired), len(faces)
            partial |= bool(rows or face_rows)
            estimate, fit = SpatialEvidenceConsensus(config).fit(rows, rho)
            face_estimate, face_fit = SpatialEvidenceConsensus(config).fit(
                face_rows, rho
            )
            evidence.append(
                dict(
                    anchor_m=rho,
                    supported_sections=len(rows),
                    structural_height_sections=len(face_rows),
                    fit=fit,
                )
            )
            if fit["status"] == "CONSENSUS":
                paired.append(
                    dict(
                        rho=float(estimate),
                        fit=fit,
                        sections=rows,
                        source="RAW_PAIRED_PROFILE",
                    )
                )
            elif face_fit["status"] == "CONSENSUS":
                faces.append(
                    dict(
                        rho=float(face_estimate),
                        fit=face_fit,
                        sections=face_rows,
                        source="RAW_HEIGHT_WITH_EXISTING_STRUCTURAL_FACE",
                    )
                )
            self._anchor_results[anchor_key] = dict(
                evidence=copy.deepcopy(evidence[-1]),
                paired=copy.deepcopy(paired[paired_start:]),
                faces=copy.deepcopy(faces[faces_start:]),
                observed=anchor_observed,
                partial=bool(rows or face_rows),
            )
        # A unique paired Raw edge outranks nominal face/height nominations.
        qualified = paired or faces
        join = config["boundary"]["corner_join_m"]
        if qualified and np.ptp([r["rho"] for r in qualified]) <= join:
            best = max(qualified, key=lambda r: len(r["sections"]))
            return dict(
                status="STRONG_UNIQUE_EDGE",
                rho=best["rho"],
                fit=best["fit"],
                sections=best["sections"],
                anchor_evidence=evidence,
                measurement_source=best["source"],
                final_coordinate=False,
            )
        return dict(
            status=(
                "MULTIMODE_CONFLICT"
                if qualified
                else (
                    "AMBIGUOUS_EDGE"
                    if partial
                    else "NO_EDGE" if observed else "UNOBSERVED"
                )
            ),
            anchor_evidence=evidence,
            sections=[],
            final_coordinate=False,
        )


class RawBoundaryProbe:
    def __init__(self, measurement, refiner):
        self.measurement, self.refiner = measurement, refiner
        self.resolver = SignedRoleResolver(measurement, refiner)

    @precision_stage("raw_profile")
    def evaluate(self, query, row, side_candidates):
        box = query.bounds
        started = time.perf_counter()
        sides = {}
        report = dict(
            sides=sides,
            probe_status="INTERRUPTED",
            full_precision_allowed=False,
            reason="RAW_PROBE_BUDGET_INTERRUPTED",
            raw_probe_support_side_count=0,
            raw_probe_conflict_side_count=0,
            raw_probe_missing_side_count=0,
        )
        row["raw_probe"] = report
        cached = getattr(self.measurement, "investigated_sides", {}).get(
            (query.vessel_id, query.query_id, tuple(query.bounds)), {}
        )
        for name, axis, position, sign in SIDE_SPECS:
            side_start = time.perf_counter()
            span = (box[1 - axis], box[3 - axis])
            # A side's physical candidates must remain on its own half.
            candidates = [
                c
                for c in side_candidates.get(name, [])
                if (c["rho"] - (box[axis] + box[axis + 2]) / 2) * sign < 0
            ]
            sides[name] = (
                copy.deepcopy(cached[name])
                if name in cached
                else self.resolver.side(axis, box[position], span, sign, candidates)
            )
            side_completed(name, "RAW_PROBE", sides[name]["status"])
            sides[name]["elapsed_s"] = time.perf_counter() - side_start
            report["raw_probe_support_side_count"] = sum(
                s["status"] == "STRONG_UNIQUE_EDGE" for s in sides.values()
            )
            report["elapsed_s"] = time.perf_counter() - started
        support = sum(s["status"] == "STRONG_UNIQUE_EDGE" for s in sides.values())
        structural = row.get("legacy_post_b2", {}).get("support_side_count", 0)
        coaming = sum(
            s.get("vertical_face_support", False)
            for s in row.get("deck_ring", {}).get("sides", {}).values()
        )
        structural_enclosure = (
            coaming >= 3 and row.get("independent_support_sides", 0) >= 3
        )
        investigation_support = max(structural, coaming if structural_enclosure else 0)
        target = 2 if investigation_support >= 3 else 3
        possible = list(box)
        for name, axis, position, sign in SIDE_SPECS:
            anchors = [
                c["rho"]
                for c in side_candidates.get(name, [])
                if min(c.get("primary_raw3d", 0), c.get("primary_bev", 0))
                >= self.refiner.config["roi"]["min_enclosure_ratio"]
                and abs(c["rho"] - box[position])
                <= self.refiner.config["roi"]["support_search_m"]
                and (c["rho"] - (box[axis] + box[axis + 2]) / 2) * sign < 0
            ]
            # Include the full existing Raw search around every structural
            # nomination. This is a search envelope, never a final rectangle.
            search = self.refiner.config["roi"]["boundary_search_m"]
            possible[position] = (
                min([box[position]] + anchors) - search
                if sign == 1
                else max([box[position]] + anchors) + search
            )
        capacities = [
            max(
                0,
                len(
                    physical_sections(
                        (possible[1 - axis], possible[3 - axis]),
                        self.refiner.config["geometry"]["refine_voxel_m"],
                        self.refiner.config,
                    )
                )
                - 1,
            )
            for axis in (0, 1)
        ]
        full_measurement_possible = (
            sum(
                2 * (count >= self.refiner.config["boundary"]["profile_min_sections"])
                for count in capacities
            )
            >= 3
        )
        # Escalate only unexplained nearby sides, and stop extra probing once
        # the documented investigation admission is met. This is still solely
        # Raw profiles: no raster, corner reconstruction or normal estimator.
        for name, axis, position, sign in SIDE_SPECS:
            if not full_measurement_possible:
                break
            if support >= target:
                break
            if sides[name]["status"] == "STRONG_UNIQUE_EDGE":
                continue
            candidates = side_candidates.get(name, [])
            side_start = time.perf_counter()
            replacement = self.resolver.side(
                axis,
                box[position],
                (box[1 - axis], box[3 - axis]),
                sign,
                candidates,
                extended=True,
                normal_range=(box[axis], box[axis + 2]),
            )
            replacement["elapsed_s"] = (
                sides[name]["elapsed_s"] + time.perf_counter() - side_start
            )
            replacement["extended_raw_search"] = True
            sides[name] = replacement
            support = sum(s["status"] == "STRONG_UNIQUE_EDGE" for s in sides.values())
        conflicts = sum(s["status"] == "MULTIMODE_CONFLICT" for s in sides.values())
        independent_remaining = all(
            s["status"] == "STRONG_UNIQUE_EDGE"
            or (
                row.get("deck_ring", {})
                .get("sides", {})
                .get(name, {})
                .get("raw3d_support", False)
                and row.get("deck_ring", {})
                .get("sides", {})
                .get(name, {})
                .get("bev_support", False)
            )
            for name, s in sides.items()
        )
        allowed = support >= 3 or (
            support >= 2 and investigation_support >= 3 and independent_remaining
        )
        cross = row.get("legacy_post_b2", {}).get("cross_deck") or {}
        cross_partial = (
            cross.get("can_split", False)
            and support >= 2
            and row.get("independent_support_sides", 0) >= 3
            and any(s["status"] == "UNOBSERVED" for s in sides.values())
        )
        allowed = (allowed or cross_partial) and full_measurement_possible
        reason = (
            "RAW_THREE_SIDE_INVESTIGATION"
            if support >= 3
            else (
                "CROSS_DECK_WITH_UNOBSERVED_SIDE_INVESTIGATION"
                if cross_partial
                else (
                    "POST_B2_TWO_RAW_SIDES_WITH_STRUCTURAL_REMAINDER"
                    if allowed
                    else (
                        "RAW_PROBE_CONFLICT"
                        if conflicts
                        else "INSUFFICIENT_RAW_PROBE_SUPPORT"
                    )
                )
            )
        )
        if not full_measurement_possible:
            reason = "INSUFFICIENT_PHYSICAL_SECTION_CAPACITY_FOR_THREE_FINAL_SIDES"
        elif allowed and support < 3 and structural_enclosure and not cross_partial:
            reason = "STRUCTURAL_TWO_RAW_SIDES_WITH_MEASURED_FACE_REMAINDER"
        report.update(
            sides=sides,
            raw_probe_support_side_count=support,
            raw_probe_conflict_side_count=conflicts,
            raw_probe_missing_side_count=sum(
                s["status"] in ("NO_EDGE", "UNOBSERVED") for s in sides.values()
            ),
            full_precision_allowed=bool(allowed),
            reason=reason,
            elapsed_s=time.perf_counter() - started,
            raster_build_count=0,
            normal_query_count=0,
            final_coordinates_created=False,
            full_measurement_possible=full_measurement_possible,
            structural_raw_search_envelope=possible,
            probe_status="COMPLETED",
        )
        return report
