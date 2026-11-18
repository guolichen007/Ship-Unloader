"""Conservative Raw profiles, deterministic section consensus and held-out evidence."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .contract import RAW_ROWS, digest


@dataclass(frozen=True)
class Config:
    move_m: float = 2.0
    angle_deg: float = 5.0
    slab_m: float = 0.75
    corner_trim_m: float = 0.6
    normal_bin_m: float = 0.10
    relief_m: float = 0.48
    cap_band_m: float = 0.07
    residual_m: float = 0.065
    coverage: float = 0.60
    sections_per_split: int = 3
    peak_limit: int = 3
    roi_points: int = 200_000
    slab_points: int = 24_000
    max_observations: int = 1200
    seconds: float = 5.0

    def audit(self):
        value = asdict(self)
        return {**value, "config_sha256": digest(value)}


CONFIG = Config()


class ExecutionStop(Exception):
    pass


def checkpoint(deadline, cancelled):
    if cancelled():
        raise ExecutionStop("CANCELLED")
    if perf_counter() >= deadline:
        raise ExecutionStop("TIMEOUT")


def point_rows(ids):
    return np.sort(ids).tolist()


def profile(points, ids, edge, tangent, normal, length, search, cfg, check):
    """Offer modes per fixed slab; preserve evidence from the original frame."""
    delta = points[:, :2] - edge
    t, d = delta @ tangent, delta @ normal
    trim = min(cfg.corner_trim_m, length * 0.08)
    mask = (
        (t >= trim) & (t < length - trim) & (d >= -search - 0.68) & (d < search + 0.68)
    )
    rows = np.flatnonzero(mask)
    if not len(rows):
        return [], []
    slab_id = np.floor((t[rows] - trim) / cfg.slab_m).astype(int)
    order = np.argsort(slab_id, kind="stable")
    slab_id, rows = slab_id[order], rows[order]
    sections, modes = [], []
    for group in np.split(np.arange(len(rows)), np.flatnonzero(np.diff(slab_id)) + 1):
        check()
        s = int(slab_id[group[0]])
        rows_s = rows[group]
        if len(rows_s) > cfg.slab_points:
            raise ExecutionStop("SLAB_POINT_BUDGET")
        # Multiplicity must not buy qualification. Keep the complete source rows
        # for evidence, but give each coincident XYZ a single vote in estimation.
        local = np.column_stack((t[rows_s], d[rows_s], points[rows_s, 2]))
        unique, inverse = np.unique(local, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        sections.append(
            {
                "slab": s,
                "split": "train" if s % 2 == 0 else "heldout",
                "t_m": [
                    trim + s * cfg.slab_m,
                    min(length - trim, trim + (s + 1) * cfg.slab_m),
                ],
                "raw_count": len(rows_s),
                "independent_xyz": len(unique),
            }
        )
        modes.extend(slab_modes(unique, ids[rows_s], inverse, s, search, cfg))
        if len(modes) > cfg.max_observations:
            raise ExecutionStop("SECTION_OBSERVATION_BUDGET")
    return sections, modes


def slab_modes(unique, raw_ids, inverse, slab, search, cfg):
    t, d, z = unique.T
    out = []
    accepted = []
    # Two bin phases cover vertical returns split across a bin boundary.
    for phase in (0.0, cfg.normal_bin_m / 2):
        bins = np.floor((d + phase) / cfg.normal_bin_m).astype(np.int64)
        order = np.argsort(bins, kind="stable")
        for group in np.split(order, np.flatnonzero(np.diff(bins[order])) + 1):
            if len(group) < 5:
                continue
            height = z[group]
            if np.ptp(height) < cfg.relief_m:
                continue
            low = float(np.quantile(height, 0.10))
            if np.count_nonzero(height > low + cfg.relief_m) < 2:
                continue
            high = float(np.quantile(height, 0.9))
            mid = group[(height >= low + 0.10) & (height <= high - 0.10)]
            if (
                len(mid) < 3
                or len(np.unique(np.floor((z[mid] - low) / 0.13))) < 2
                or np.ptp(t[mid]) < 0.16
            ):
                continue
            offset = float(np.median(d[mid]))
            if abs(offset) > search + cfg.residual_m:
                continue
            if any(abs(offset - other) < 0.085 for other in accepted):
                continue
            top = z[group].max()
            outer = (d > offset - 0.64) & (d < offset - 0.12)
            inner = (d > offset + 0.12) & (d < offset + 0.62)
            cap = outer & (np.abs(z - top) <= cfg.cap_band_m)
            lower = inner & (z <= top - cfg.relief_m * 0.8)
            contender = inner & (z >= top - cfg.cap_band_m)
            cap_spans = (
                np.count_nonzero(cap) >= 4
                and np.ptp(d[cap]) >= 0.13
                and np.ptp(t[cap]) >= 0.16
            )
            inner_spans = np.count_nonzero(lower) >= 3 and np.ptp(t[lower]) >= 0.16
            role = (
                "INNER_RIM_SUPPORTED"
                if cap_spans and inner_spans and np.count_nonzero(contender) < 4
                else (
                    "CONFLICTING"
                    if np.count_nonzero(contender) >= 4
                    else "OTHER_STRUCTURE"
                )
            )
            selected = np.zeros(len(unique), dtype=bool)
            selected[mid] = True
            observation = dict(
                slab=slab,
                t_m=float(np.median(t[mid])),
                d_m=offset,
                top_m=float(top),
                relief_m=high - low,
                role=role,
                face=point_rows(raw_ids[selected[inverse]]),
                cap=point_rows(raw_ids[cap[inverse]]),
                interior=point_rows(raw_ids[lower[inverse]]),
                competing_interior=point_rows(raw_ids[contender[inverse]]),
                unique_face_count=len(mid),
            )
            if role == "INNER_RIM_SUPPORTED":
                accepted.append(offset)
            out.append(observation)
    return out


def select_support(rows, slope, intercept, cfg):
    support = {}
    for row in rows:
        residual = abs(row["d_m"] - intercept - slope * row["t_m"])
        slab = row["slab"]
        if residual <= cfg.residual_m and (
            slab not in support or residual < support[slab][0]
        ):
            support[slab] = residual, row
    return [item[1] for _, item in sorted(support.items())]


def line_fit(rows):
    t = np.array([r["t_m"] for r in rows])
    d = np.array([r["d_m"] for r in rows])
    tc = t - t.mean()
    slope = float(tc @ (d - d.mean()) / (tc @ tc))
    return slope, float(d.mean() - slope * t.mean())


def fit_modes(qualified, length, cfg, check):
    train = [r for r in qualified if r["slab"] % 2 == 0]
    if len(train) < cfg.sections_per_split:
        return []
    found = {}
    t = np.array([r["t_m"] for r in train])
    d = np.array([r["d_m"] for r in train])
    gate = np.tan(np.deg2rad(cfg.angle_deg))
    # A coarse deterministic Hough vote bounds slope searches and never uses heldout.
    for nominal in np.linspace(-gate, gate, 21):
        check()
        offsets = d - nominal * (t - length / 2)
        for phase in (0, cfg.residual_m):
            bins = np.floor((offsets + phase) / (cfg.residual_m * 2)).astype(int)
            for key in np.unique(bins):
                subset = np.flatnonzero(bins == key)
                if len(subset) < cfg.sections_per_split:
                    continue
                support = select_support(
                    train,
                    nominal,
                    float(np.median(offsets[subset]) - nominal * length / 2),
                    cfg,
                )
                if (
                    len(support) < cfg.sections_per_split
                    or support[-1]["t_m"] - support[0]["t_m"]
                    < length * cfg.coverage * 0.72
                ):
                    continue
                slope, intercept = line_fit(support)
                if (
                    abs(slope) > gate
                    or max(abs(intercept), abs(intercept + slope * length)) > cfg.move_m
                ):
                    continue
                support = select_support(train, slope, intercept, cfg)
                if len(support) < cfg.sections_per_split:
                    continue
                signature = tuple((r["slab"], r["d_m"]) for r in support)
                residual = float(
                    np.sqrt(
                        np.mean(
                            [
                                (r["d_m"] - slope * r["t_m"] - intercept) ** 2
                                for r in support
                            ]
                        )
                    )
                )
                found[signature] = slope, intercept, support, residual
    candidates = []
    for fit in sorted(found.values(), key=lambda f: (-len(f[2]), f[3], f[1], f[0])):
        m, b, _, _ = fit
        if any(
            max(abs(b - old[1]), abs((m - old[0]) * length + b - old[1]))
            <= 2 * cfg.residual_m
            for old in candidates
        ):
            continue
        candidates.append(fit)
    return candidates


def separation(train, heldout, length, cfg):
    if len(heldout) < cfg.sections_per_split:
        return "HELDOUT_SUPPORT_INSUFFICIENT"
    stations = [r["t_m"] for r in train + heldout]
    if max(stations) - min(stations) < cfg.coverage * length:
        return "SUPPORT_SPAN_INSUFFICIENT"
    if raw_set(train) & raw_set(heldout):
        return "HELDOUT_NOT_INDEPENDENT"
    return "QUALIFIED"


def raw_set(rows):
    return {
        value
        for row in rows
        for k in ("face", "cap", "interior", "competing_interior")
        for value in row[k]
    }


def analyze_side(points, ids, edge, tangent, normal, length, cfg, check):
    sections, observed = profile(
        points, ids, edge, tangent, normal, length, cfg.move_m, cfg, check
    )
    qualified = [r for r in observed if r["role"] == "INNER_RIM_SUPPORTED"]
    fits = fit_modes(qualified, length, cfg, check)
    heldout = [r for r in qualified if r["slab"] % 2 == 1]
    modes = []
    for slope, intercept, train, rms in fits[: cfg.peak_limit]:
        verify = select_support(heldout, slope, intercept, cfg)
        reason = separation(train, verify, length, cfg)
        modes.append(
            dict(
                slope=slope,
                intercept_m=intercept,
                offset_m=intercept + slope * length / 2,
                rms_m=rms,
                train=train,
                heldout=verify,
                reason=reason,
                train_sha256=digest(train),
                heldout_sha256=digest(verify),
            )
        )
    if len(fits) > cfg.peak_limit:
        role, reason = "CONFLICTING", "CANDIDATE_BUDGET_EXCEEDED"
    elif len(modes) > 1:
        # A plausible competing training line cannot be erased by missing data.
        role, reason = "CONFLICTING", "MULTIPLE_PHYSICAL_LINES"
    elif len(modes) == 1 and modes[0]["reason"] == "QUALIFIED":
        role, reason = "INNER_RIM_SUPPORTED", "QUALIFIED"
    else:
        role, reason = "INSUFFICIENT", (
            modes[0]["reason"] if modes else "NO_DISTRIBUTED_INNER_RIM"
        )
    return dict(
        role=role,
        reason=reason,
        qualified=role == "INNER_RIM_SUPPORTED",
        observed=bool(sections),
        sections=sections,
        modes=modes,
        observations=observed,
        omitted_modes=max(0, len(fits) - cfg.peak_limit),
        index_contract=RAW_ROWS,
    )
