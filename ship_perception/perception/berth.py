"""Site-referenced berth selection; vessel size/count never defines the quay."""

from pathlib import Path
import json
import hashlib
import numpy as np


def load_berth_reference(path, frame_id):
    path = Path(path)
    raw = path.read_bytes()
    try:
        reference = json.loads(raw)
    except ValueError:
        import yaml

        reference = yaml.safe_load(raw)
    allowed = {
        "schema",
        "enabled",
        "frame_id",
        "quay_line",
        "quay_normal_direction",
        "working_longitudinal_range",
        "working_polygon",
        "expected_ship_side",
        "observer_xy",
        "provenance",
    }
    if not isinstance(reference, dict) or set(reference) - allowed:
        raise ValueError("INVALID_BERTH_REFERENCE_FIELDS")
    reference["reference_sha256"] = hashlib.sha256(raw).hexdigest()
    if not reference.get("enabled", False):
        return reference
    if reference.get("frame_id") != frame_id:
        raise ValueError("BERTH_REFERENCE_FRAME_MISMATCH")
    line = np.asarray(reference.get("quay_line"), dtype=float)
    normal = np.asarray(reference.get("quay_normal_direction"), dtype=float)
    if (
        line.shape != (2, 2)
        or normal.shape != (2,)
        or not np.isfinite(line).all()
        or not np.isfinite(normal).all()
    ):
        raise ValueError("INVALID_BERTH_LINE")
    tangent = line[1] - line[0]
    if np.linalg.norm(tangent) == 0 or np.linalg.norm(normal) == 0:
        raise ValueError("DEGENERATE_BERTH_LINE")
    tangent /= np.linalg.norm(tangent)
    normal /= np.linalg.norm(normal)
    if abs(np.dot(tangent, normal)) > 1e-6:
        raise ValueError("BERTH_NORMAL_NOT_PERPENDICULAR")
    bounds = reference.get("working_longitudinal_range")
    if bounds is not None and (
        len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]
    ):
        raise ValueError("INVALID_BERTH_WORK_RANGE")
    polygon = reference.get("working_polygon")
    if polygon is not None and (
        np.asarray(polygon).ndim != 2
        or np.asarray(polygon).shape[1] != 2
        or len(polygon) < 3
        or not np.isfinite(polygon).all()
    ):
        raise ValueError("INVALID_BERTH_WORK_POLYGON")
    if reference.get("expected_ship_side", "WATER_SIDE") != "WATER_SIDE":
        raise ValueError("BERTH_NORMAL_MUST_POINT_TO_WATER")
    reference.update(
        tangent=tangent.tolist(),
        normal=normal.tolist(),
        origin=line[0].tolist(),
        source="SURVEYED_SITE_REFERENCE",
    )
    return reference


class AutomaticQuayDetector:
    """Frontal BEV contour with robust line/support tests, or explicit unknown.

    Observer position must come from acquisition/site metadata. A map origin
    is not silently assumed to be the current sensor position. Long narrow
    rails/fenders alone fail the landward surface-support requirement.
    """

    def __init__(self, config):
        self.config = config

    def detect(self, frame, observer_xy=None):
        if observer_xy is None:
            return dict(
                status="SITE_REFERENCE_UNRESOLVED",
                reason="NO_TRUSTED_QUAY_OBSERVER_POSITION",
                candidates=[],
            )
        from ship_perception.r1_static.height_grid import R1HeightGrid

        grid = R1HeightGrid.from_points(
            frame.xyz, self.config["geometry"]["coarse_voxel_m"]
        )
        u, v = grid.centers()
        mask = grid.occupancy
        xy = np.column_stack((u[mask], v[mask]))
        observer = np.asarray(observer_xy, dtype=float)
        if observer.shape != (2,) or not np.isfinite(observer).all():
            raise ValueError("INVALID_QUAY_OBSERVER")
        vectors = xy - observer
        radius = np.linalg.norm(vectors, axis=1)
        angular_step = grid.cell_m / max(grid.cell_m, float(np.median(radius)))
        bins = np.floor(np.arctan2(vectors[:, 1], vectors[:, 0]) / angular_step).astype(
            np.int64
        )
        order = np.lexsort((radius, bins))
        _, first = np.unique(bins[order], return_index=True)
        contour = xy[order[first]]
        threshold = self.config["geometry"]["plane_inlier_m"] + grid.cell_m
        minimum_span = (
            self.config["roi"]["min_deck_span_m"]
            * self.config["boundary"]["profile_min_sections"]
        )
        candidates = []
        rng = np.random.default_rng(0)
        if len(contour) < self.config["boundary"]["line_min_points"]:
            return dict(
                status="SITE_REFERENCE_UNRESOLVED",
                reason="INSUFFICIENT_FRONTAL_CONTOUR",
                candidates=[],
            )
        for _ in range(min(128, len(contour) * 2)):
            a, b = contour[rng.choice(len(contour), 2, replace=False)]
            if np.linalg.norm(b - a) < minimum_span:
                continue
            tangent = (b - a) / np.linalg.norm(b - a)
            normal = np.array([-tangent[1], tangent[0]])
            if np.dot(observer - a, normal) > 0:
                normal = -normal
            distance = (contour - a) @ normal
            inliers = contour[abs(distance) <= threshold]
            along = (inliers - a) @ tangent
            if (
                len(inliers) < self.config["boundary"]["line_min_points"]
                or np.ptp(along) < minimum_span
            ):
                continue
            # Broad landward occupied surface, rather than a thin line object.
            d = (xy - a) @ normal
            t = (xy - a) @ tangent
            land = (
                (d < -grid.cell_m)
                & (d >= -self.config["roi"]["support_search_m"])
                & (t >= along.min())
                & (t <= along.max())
            )
            if (
                not land.any()
                or np.ptp(d[land]) < self.config["roi"]["min_deck_span_m"]
            ):
                continue
            stations = np.floor(
                (along - along.min()) / self.config["roi"]["min_deck_span_m"]
            )
            distribution = len(np.unique(stations)) / max(
                1, np.ceil(np.ptp(along) / self.config["roi"]["min_deck_span_m"])
            )
            if distribution < 1 - self.config["roi"]["min_enclosure_ratio"]:
                continue
            if any(
                abs(np.dot(normal, np.array(c["normal"])))
                > np.cos(np.radians(self.config["boundary"]["merge_angle_deg"]))
                and abs(np.dot(a - np.array(c["origin"]), normal)) <= threshold
                for c in candidates
            ):
                continue
            candidates.append(
                dict(
                    origin=a.tolist(),
                    tangent=tangent.tolist(),
                    normal=normal.tolist(),
                    source="SITE_REFERENCE_FALLBACK",
                    distribution_score=distribution,
                    span_m=float(np.ptp(along)),
                    landward_support_cells=int(land.sum()),
                    working_longitudinal_range=[float(along.min()), float(along.max())],
                )
            )
        if len(candidates) != 1:
            return dict(
                status="SITE_REFERENCE_UNRESOLVED",
                reason=(
                    "AMBIGUOUS_QUAY_LINES" if candidates else "NO_SUPPORTED_QUAY_LINE"
                ),
                candidates=candidates,
            )
        return dict(
            status="QUAY_REFERENCE_REVIEW",
            reference=candidates[0],
            candidates=candidates,
        )


def select_berth_vessel(frame, context, config, reference=None):
    resolved = [
        v
        for v in context.vessels
        if v.public["classification_status"] == "VESSEL_HYPOTHESIS"
    ]
    if reference is None or not reference.get("enabled", True):
        auto = AutomaticQuayDetector(config).detect(
            frame, (reference or {}).get("observer_xy")
        )
        if auto.get("reference"):
            reference = auto["reference"]
        elif len(resolved) == 1:
            return dict(
                status="SINGLE_VESSEL_REVIEW_NO_BERTH_CERTIFICATION",
                active_vessel_id=resolved[0].vessel_id,
                ignored_vessel_ids=[],
                reference_status=auto,
                measurements=[],
                berth_certified=False,
            )
        else:
            return dict(
                status="MULTI_VESSEL_BERTH_AMBIGUOUS",
                active_vessel_id=None,
                ignored_vessel_ids=[],
                reference_status=auto,
                measurements=[],
                berth_certified=False,
            )
    tangent = np.asarray(reference["tangent"])
    normal = np.asarray(reference["normal"])
    origin = np.asarray(reference["origin"])
    measurements = []
    for vessel in resolved:
        points = frame.xyz[vessel.point_indexes, :2]
        along = (points - origin) @ tangent
        cross = (points - origin) @ normal
        limits = reference.get("working_longitudinal_range") or [
            float(along.min()),
            float(along.max()),
        ]
        overlap = max(
            0.0, min(float(along.max()), limits[1]) - max(float(along.min()), limits[0])
        )
        angle = float(
            np.degrees(np.arccos(np.clip(abs(np.dot(vessel.axes[0], tangent)), 0, 1)))
        )
        span = config["roi"]["min_deck_span_m"]
        groups = np.floor((along - along.min()) / span).astype(int)
        order = np.argsort(groups, kind="stable")
        _, starts = np.unique(groups[order], return_index=True)
        distances = np.minimum.reduceat(cross[order], starts).astype(float).tolist()
        distance = float(np.median(distances))
        mad = float(np.median(abs(np.asarray(distances) - distance)))
        inside_work = True
        if reference.get("working_polygon") is not None:
            polygon = np.asarray(reference["working_polygon"], dtype=float)
            contained = np.zeros(len(points), dtype=bool)
            for first, last in zip(polygon, np.roll(polygon, -1, axis=0)):
                crosses = (first[1] > points[:, 1]) != (last[1] > points[:, 1])
                if first[1] != last[1]:
                    boundary_x = (last[0] - first[0]) * (points[:, 1] - first[1]) / (
                        last[1] - first[1]
                    ) + first[0]
                    contained ^= crosses & (points[:, 0] < boundary_x)
            inside_work = bool(contained.any())
        eligible = (
            overlap >= span
            and angle <= config["boundary"]["merge_angle_deg"]
            and distance >= -config["boundary"]["corner_join_m"]
            and inside_work
        )
        measurements.append(
            dict(
                vessel_id=vessel.vessel_id,
                distance_to_quay_m=distance,
                distance_mad_m=mad,
                parallelism_angle_deg=angle,
                longitudinal_overlap_m=overlap,
                physical_length_m=float(np.ptp(along)),
                point_support=len(points),
                deck_support=vessel.public.get("classification_evidence", {}),
                eligible=bool(eligible),
            )
        )
    eligible = sorted(
        (m for m in measurements if m["eligible"]),
        key=lambda m: m["distance_to_quay_m"],
    )
    ambiguous = not eligible
    if len(eligible) > 1:
        first, second = eligible[:2]
        ambiguous = (
            second["distance_to_quay_m"] - second["distance_mad_m"]
            <= first["distance_to_quay_m"]
            + first["distance_mad_m"]
            + config["boundary"]["corner_join_m"]
        )
    active = None if ambiguous else eligible[0]["vessel_id"]
    return dict(
        status="MULTI_VESSEL_BERTH_AMBIGUOUS" if ambiguous else "ACTIVE_BERTH_VESSEL",
        active_vessel_id=active,
        ignored_vessel_ids=[
            v.vessel_id
            for v in resolved
            if active is not None and v.vessel_id != active
        ],
        reference=reference,
        measurements=measurements,
        berth_certified=False,
    )
