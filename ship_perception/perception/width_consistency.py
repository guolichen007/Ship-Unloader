"""Same-vessel side corridors nominate investigations, never measured edges."""

import numpy as np


LONG_SIDES = {"V0": (0, 1), "V1": (3, 2)}


def side_line(polygon, side):
    points = np.asarray(polygon, dtype=float)[list(LONG_SIDES[side])]
    tangent = points[1] - points[0]
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    if normal[1] < 0:
        normal = -normal
    return dict(
        normal_uv=normal.tolist(),
        rho=float(points[0] @ normal),
        points_uv=points.tolist(),
        span_m=float(np.linalg.norm(points[1] - points[0])),
    )


def line_separation(a, b):
    """Compare at the same physical station, along each line's own normal."""
    pa, pb = np.asarray(a["points_uv"]), np.asarray(b["points_uv"])
    values = []
    for first, second in ((a, b), (b, a)):
        n, m = np.asarray(first["normal_uv"]), np.asarray(second["normal_uv"])
        tangent = np.array([n[1], -n[0]])
        system = np.vstack((m, tangent))
        if abs(np.linalg.det(system)) <= np.sqrt(np.finfo(float).eps):
            return None
        station = float((pa.mean(0) + pb.mean(0)) @ tangent / 2)
        point = np.linalg.solve(system, [second["rho"], station])
        values.append(abs(float(point @ n - first["rho"])))
    return max(values)


def _trusted_mode(evidence, side):
    entry = evidence.get(side, {})
    modes = [
        m
        for m in entry.get("side_candidates", ())
        if m.get("qualified_for_shadow") and m.get("source_point_indexes")
    ]
    return (
        modes[0]
        if len(modes) == 1 and len(entry.get("side_candidates", ())) == 1
        else None
    )


def _uncertainty(line, mode, config, station):
    sigma = (
        float(mode["uncertainty_m"])
        if mode
        else config["geometry"]["coarse_voxel_m"] / 2
    )
    sigma = max(sigma, config["geometry"]["plane_inlier_m"])
    points = np.asarray(line["points_uv"])
    # Extrapolation grows from observed span; no fixed metre width tolerance.
    leverage = max(0.0, abs(station - points[:, 0].mean()) - np.ptp(points[:, 0]) / 2)
    return sigma * np.sqrt(1 + (leverage / max(line["span_m"], sigma)) ** 2)


def robust_corridor(observations, config):
    """Physical sections/Cells, rather than Raw point density, own the weights."""
    if not observations:
        return dict(status="NO_INDEPENDENT_RAW_CORRIDOR", observed=False)
    points, sigmas, owners = [], [], []
    for row in observations:
        n = np.asarray(row["mode"]["normal_uv"])
        along = np.asarray(row["mode"]["support_span"])
        p = n * row["mode"]["rho"] + along[:, None] * np.array([-n[1], n[0]])
        points.extend(p.tolist())
        sigmas.extend([row["mode"]["uncertainty_m"]] * 2)
        owners.extend([row["cell_id"]] * 2)
    xy = np.asarray(points)
    origin = xy.mean(0)
    design = np.column_stack((xy[:, 0] - origin[0], np.ones(len(xy))))
    uncertainty = np.maximum(sigmas, config["geometry"]["plane_inlier_m"])
    weight = 1 / uncertainty**2
    fit = np.array([0.0, np.median(xy[:, 1]) - origin[1]])
    for _ in range(config["geometry"]["max_plane_hypotheses"]):
        fitted = np.linalg.lstsq(
            design * np.sqrt(weight[:, None]),
            (xy[:, 1] - origin[1]) * np.sqrt(weight),
            rcond=None,
        )[0]
        residual = xy[:, 1] - origin[1] - design @ fitted
        updated = (
            np.minimum(
                1.0, uncertainty / np.maximum(abs(residual), np.finfo(float).eps)
            )
            / uncertainty**2
        )
        fit = fitted
        if np.allclose(updated, weight, rtol=0, atol=np.finfo(float).eps):
            break
        weight = updated
    n = np.array([-fit[0], 1.0])
    n /= np.linalg.norm(n)
    rho = float((origin + np.array([0.0, fit[1]])) @ n)
    residual = xy @ n - rho
    sigma = max(float(np.median(uncertainty)), float(np.quantile(abs(residual), 0.95)))
    inliers = sorted(
        {
            owner
            for owner, error, u in zip(owners, residual, uncertainty)
            if abs(error) <= u
        }
    )
    return dict(
        status="RAW_SUPPORTED_SEARCH_CORRIDOR",
        observed=True,
        normal_uv=n.tolist(),
        rho=rho,
        uncertainty_m=sigma,
        support_u_span_m=[float(xy[:, 0].min()), float(xy[:, 0].max())],
        cell_ids=sorted(set(owners)),
        inlier_cell_ids=inliers,
        outlier_cell_ids=sorted(set(owners) - set(inliers)),
        measurement_transfer_allowed=False,
        steel_certified=False,
    )


def width_consistency(cells, polygons_uv, config):
    """Return per-Cell side investigations and vessel-specific prior receipts.

    Two widths never vote for the narrower one. Even a supported corridor is
    only a search prior; a new line must be measured in the receiving Cell.
    """
    groups = {}
    for cell in cells:
        groups.setdefault(cell["vessel_id"], []).append(cell)
    audits, requests = {}, {c["cell_id"]: set() for c in cells}
    for vessel_id, group in groups.items():
        if len(group) < 2:
            audits[vessel_id] = dict(
                status="SINGLE_OPENING_CROSS_CELL_SKIPPED",
                cell_count=len(group),
                common_width_m=None,
                prior_used=False,
            )
            continue
        lines, modes, widths = {}, {}, {}
        for cell in group:
            key = cell["cell_id"]
            p = polygons_uv[key]
            lines[key] = {s: side_line(p, s) for s in LONG_SIDES}
            modes[key] = {
                s: _trusted_mode(cell.get("local_rim_evidence", {}), s)
                for s in LONG_SIDES
            }
            widths[key] = float(
                np.mean(
                    np.linalg.norm(
                        np.asarray(p)[[3, 2]] - np.asarray(p)[[0, 1]], axis=1
                    )
                )
            )
        relations = []
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                ka, kb = a["cell_id"], b["cell_id"]
                station = float(
                    np.mean([np.mean(polygons_uv[k], axis=0)[0] for k in (ka, kb)])
                )
                distances = {
                    s: line_separation(lines[ka][s], lines[kb][s]) for s in LONG_SIDES
                }
                if any(value is None for value in distances.values()):
                    relations.append(
                        dict(
                            cell_ids=[ka, kb],
                            status="LONG_SIDE_DIRECTION_AMBIGUOUS",
                            geometry_changed=False,
                            narrower_width_vote=False,
                        )
                    )
                    continue
                intervals = {
                    s: float(
                        np.hypot(
                            _uncertainty(lines[ka][s], modes[ka][s], config, station),
                            _uncertainty(lines[kb][s], modes[kb][s], config, station),
                        )
                    )
                    for s in LONG_SIDES
                }
                weak, other = sorted(
                    LONG_SIDES, key=lambda s: distances[s], reverse=True
                )
                asymmetric = (
                    distances[weak] - distances[other]
                    > intervals[weak] + intervals[other]
                )
                if asymmetric:
                    for key in (ka, kb):
                        if modes[key][weak] is None:
                            requests[key].add(weak)
                relations.append(
                    dict(
                        cell_ids=[ka, kb],
                        side_distance_m=distances,
                        propagated_uncertainty_m=intervals,
                        status=(
                            "ASYMMETRIC_SIDE_DISAGREEMENT"
                            if asymmetric
                            else "NO_SIDE_OUTLIER_PROVEN"
                        ),
                        investigation_side=weak if asymmetric else None,
                        smaller_side_within_uncertainty=distances[other]
                        <= intervals[other],
                        geometry_changed=False,
                        narrower_width_vote=False,
                    )
                )
        for cell in group:
            for side in LONG_SIDES:
                entry = cell.get("local_rim_evidence", {}).get(side, {})
                if (
                    entry.get("state")
                    == "STRUCTURAL_CONTINUATION_WITHOUT_DECK_CONFIRMATION"
                    and modes[cell["cell_id"]][side] is None
                ):
                    requests[cell["cell_id"]].add(side)
        corridors = {
            s: robust_corridor(
                [
                    dict(cell_id=c["cell_id"], mode=modes[c["cell_id"]][s])
                    for c in group
                    if modes[c["cell_id"]][s]
                ],
                config,
            )
            for s in LONG_SIDES
        }
        full = [key for key in modes if all(modes[key].values())]
        common = None
        if len(full) >= 3:
            measured = np.array([widths[key] for key in full])
            center = float(np.median(measured))
            tolerance = np.array(
                [
                    sum(modes[key][s]["uncertainty_m"] for s in LONG_SIDES)
                    for key in full
                ]
            )
            if np.all(abs(measured - center) <= tolerance):
                common = center
        audits[vessel_id] = dict(
            status="SAME_VESSEL_SIDE_CORRIDORS",
            cell_count=len(group),
            widths_m=widths,
            relations=relations,
            corridors=corridors,
            independently_supported_openings=full,
            common_width_m=common,
            common_width_status=(
                "OBSERVED_COMPATIBLE_FAMILY"
                if common is not None
                else "NO_UNIQUE_COMMON_WIDTH"
            ),
            unequal_widths_allowed=True,
            two_width_vote_allowed=False,
            geometry_changed=False,
            prior_used=any(requests[c["cell_id"]] for c in group),
        )
    return audits, {key: sorted(value) for key, value in requests.items()}
