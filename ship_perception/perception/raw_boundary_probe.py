"""Cheap exact-XYZ edge investigation, without rasters, normals or corners."""

import time
import numpy as np

from .line_geometry import SIDE_SPECS
from .sections import physical_sections
from .consensus import SpatialEvidenceConsensus


class SignedRoleResolver:
    """Resolve only candidate-local nearby lines after nominal compilation."""

    def __init__(self, measurement, refiner):
        self.measurement, self.refiner = measurement, refiner
        self.config = refiner.config

    def side(self, axis, anchor, span, sign, candidates, extended=False):
        config = self.config
        anchors = [anchor]
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
            if self.refiner.budget is not None:
                self.refiner.budget.check()
            rows, face_rows = [], []
            for section in spans:
                profile = self.measurement.side_measurements.profile(axis, rho, section)
                observed |= (
                    len(profile.points) >= config["boundary"]["profile_min_points"]
                )
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
                if raw is not None and profile.height(raw, sign)["transition"]:
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
        for name, axis, position, sign in SIDE_SPECS:
            side_start = time.perf_counter()
            span = (box[1 - axis], box[3 - axis])
            # A side's physical candidates must remain on its own half.
            candidates = [
                c
                for c in side_candidates.get(name, [])
                if (c["rho"] - (box[axis] + box[axis + 2]) / 2) * sign < 0
            ]
            sides[name] = self.resolver.side(
                axis, box[position], span, sign, candidates
            )
            sides[name]["elapsed_s"] = time.perf_counter() - side_start
            report["raw_probe_support_side_count"] = sum(
                s["status"] == "STRONG_UNIQUE_EDGE" for s in sides.values()
            )
            report["elapsed_s"] = time.perf_counter() - started
        support = sum(s["status"] == "STRONG_UNIQUE_EDGE" for s in sides.values())
        structural = row.get("legacy_post_b2", {}).get("support_side_count", 0)
        target = 2 if structural >= 3 else 3
        # Escalate only unexplained nearby sides, and stop extra probing once
        # the documented investigation admission is met. This is still solely
        # Raw profiles: no raster, corner reconstruction or normal estimator.
        for name, axis, position, sign in SIDE_SPECS:
            if support >= target:
                break
            if sides[name]["status"] == "STRONG_UNIQUE_EDGE":
                continue
            candidates = side_candidates.get(name, [])
            if not candidates:
                continue
            side_start = time.perf_counter()
            replacement = self.resolver.side(
                axis,
                box[position],
                (box[1 - axis], box[3 - axis]),
                sign,
                candidates,
                extended=True,
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
            support >= 2 and structural >= 3 and independent_remaining
        )
        cross = row.get("legacy_post_b2", {}).get("cross_deck") or {}
        cross_partial = (
            cross.get("can_split", False)
            and support >= 2
            and row.get("independent_support_sides", 0) >= 3
            and any(s["status"] == "UNOBSERVED" for s in sides.values())
        )
        allowed |= cross_partial
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
            probe_status="COMPLETED",
        )
        return report
