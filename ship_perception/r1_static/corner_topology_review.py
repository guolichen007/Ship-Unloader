"""Offline B5 review with point-provenance checks and focused corner plots."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage

from .corner_topology import solve_vessel_openings
from .heightmap_batch import RESEARCH_CONFIG
from .raw_topview_evidence import build_topview
from .rectangle_refinement_review import _read_validation_xyz
from .run import DEFAULT_CONFIG, _atomic_json, resolve_config
from .r2h_b5_corner_forensics import _observations, _side_modes
from .scene_vessel import infer_scene_vessels
from .vessel_height_topology import build_vessel_scoped_topview


def _initial_seed_groups(proposals, config):
    """Group overlapping B2 partial openings, preserving separate basins."""
    eligible = [row for row in proposals if row.get("state") == "PARTIAL_HATCH" and
                row.get("measured_strong_side_count", 0) >= 3 and
                row.get("opening_evidence", {}).get("status") == "OPENING_LOWER" and
                not row.get("role_conflicts") and row.get("bounds_axial")]
    groups = []
    margin = config["boundary"]["corner_join_m"]
    for row in eligible:
        box = row["bounds_axial"]
        matched = []
        for index, group in enumerate(groups):
            if any(min(box[2], other["bounds_axial"][2]) + margin >=
                   max(box[0], other["bounds_axial"][0]) and
                   min(box[3], other["bounds_axial"][3]) + margin >=
                   max(box[1], other["bounds_axial"][1]) for other in group):
                matched.append(index)
        if matched:
            merged = [row]
            for index in reversed(matched):
                merged.extend(groups.pop(index))
            groups.append(merged)
        else:
            groups.append([row])
    def seed_quality(row):
        box = row["bounds_axial"]
        width, height = box[2] - box[0], box[3] - box[1]
        return (min(width, height), width * height,
                row.get("measured_strong_side_count", 0))
    return [(max(group, key=seed_quality), group) for group in groups]


def _separator_split(seed, b2, vessel, grid, config):
    """Propose two cells only around two separately measured separator faces."""
    box = seed["bounds_axial"]
    margin = config["roi"]["support_search_m"]
    roi = [box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin]
    observations = _observations(b2, vessel["vessel_hypothesis_id"],
                                 np.asarray(vessel["local_axes"]), roi)
    modes = _side_modes(observations, 0, (box[1], box[3]),
                        (box[0] + margin, box[2] - margin), config)
    minimum = config["roi"]["min_enclosure_ratio"]
    rhos = sorted(row["rho"] for row in modes if
                  min(row["primary_raw3d"], row["primary_bev"]) >= minimum)
    families = []
    for rho in rhos:
        if families and rho - families[-1][-1] <= config["boundary"]["corner_join_m"] * 2:
            families[-1].append(rho)
        else:
            families.append([rho])
    candidates = [(left, right) for left, right in zip(families[:-1], families[1:])
                  if right[0] - left[-1] <= margin and
                  box[0] + margin < np.median(left) < np.median(right) < box[2] - margin]
    # Dense B2 line sampling can fill a whole transverse beam with nearly
    # identical scores. Recover its two *measured* faces from a Raw XYZ count
    # ridge, then snap each face to a same-vessel Raw3D+BEV mode.
    u, v = grid.centers()
    along = (v[:, 0] >= box[1] + margin / 4) & (v[:, 0] <= box[3] - margin / 4)
    axial = (u[0] >= box[0] + margin) & (u[0] <= box[2] - margin)
    if along.any() and axial.sum() >= config["roi"]["min_opening_cells"]:
        density = np.mean(grid.count[along], axis=0)
        reference = float(np.median(density[axial]))
        mad = float(np.median(np.abs(density[axial] - reference)))
        high = axial & (density > reference + 3 * mad)
        high = ndimage.binary_closing(high, structure=np.ones(3, bool)) & axial
        labels, count = ndimage.label(high)
        for label in range(1, count + 1):
            columns = np.flatnonzero(labels == label)
            if len(columns) < 2 or len(columns) * grid.cell_m > margin:
                continue
            left_edge = u[0, columns[0]] - grid.cell_m / 2
            right_edge = u[0, columns[-1]] + grid.cell_m / 2
            tolerance = config["boundary"]["corner_join_m"] * 2
            left_mode = min((rho for rho in rhos if abs(rho - left_edge) <= tolerance),
                            key=lambda rho: abs(rho - left_edge), default=None)
            right_mode = min((rho for rho in rhos if abs(rho - right_edge) <= tolerance),
                             key=lambda rho: abs(rho - right_edge), default=None)
            if left_mode is not None and right_mode is not None and left_mode < right_mode:
                candidates.append(([left_mode], [right_mode]))
    if not candidates:
        if len(rhos) >= 2 and max(rhos) - min(rhos) > margin / 4:
            return [box], {"unresolved_transverse_band_m": [min(rhos), max(rhos)]}
        return [box], None
    left, right = min(candidates, key=lambda pair: abs(
        (np.median(pair[0]) + np.median(pair[1])) / 2 - (box[0] + box[2]) / 2))
    first, second = float(np.median(left)), float(np.median(right))
    return [[box[0], box[1], first, box[3]],
            [second, box[1], box[2], box[3]]], [first, second]


def _bootstrap_vessel(points, b2, private, vessel, config):
    """Start conservative opening-cell review directly from B1 and B2."""
    ident = vessel["vessel_hypothesis_id"]
    context = next((row for row in b2["per_vessel"] if
                    row["vessel_hypothesis_id"] == ident), None)
    proposals = context["hatch_hypotheses"] if context else []
    scoped = build_vessel_scoped_topview(
        points, private["vessel_point_indexes"][ident], vessel["local_axes"], config)
    grid = scoped["grids"]["fine"]
    initial_cells, rectangles = [], []
    for index, (seed, group) in enumerate(_initial_seed_groups(proposals, config)):
        boxes, separator = _separator_split(seed, b2, vessel, grid, config)
        for subindex, box in enumerate(boxes):
            cell = _bootstrap_cell(points, b2, private, vessel, config,
                                   seed, group, box, index, subindex, separator)
            initial_cells.append(cell)
        if isinstance(separator, dict):
            for cell in initial_cells[-len(boxes):]:
                cell["rectangle"] = None
                cell["status"] = "UNRESOLVED_COMPOSITE_SUSPECT"
                cell["reason"] = "BROAD_INTERIOR_TRANSVERSE_BAND_NOT_SPLIT"
        rectangles.extend(cell["rectangle"] for cell in initial_cells[-len(boxes):]
                          if cell["rectangle"] is not None)
    return dict(vessel_hypothesis_id=ident, status="B2_HEIGHT_CORNER_BOOTSTRAP_REVIEW",
                b2_hatch_proposal_count=len(proposals),
                initial_opening_cells=initial_cells, rectangles=rectangles)


def _bootstrap_cell(points, b2, private, vessel, config, seed, group,
                box, index, subindex, separator):
    ident = vessel["vessel_hypothesis_id"]
    axes = np.asarray(vessel["local_axes"])
    polygon = (np.asarray([[box[0], box[1]], [box[2], box[1]],
                           [box[2], box[3]], [box[0], box[3]]]) @ axes).tolist()
    hatch_id = f"I{index:03d}{subindex}"
    sides = {key: dict(rho_before=box[position], rho_after=box[position],
                       previous_role="UNKNOWN", candidate_modes=[],
                       selected_source_vessel_ids=[ident])
             for key, position in zip(("U0", "U1", "V0", "V1"), (0, 2, 1, 3))}
    provisional = dict(rectangles=[dict(
        hatch_id=hatch_id, vessel_hypothesis_id=ident, axes=axes.tolist(),
        polygon_before=polygon, polygon_after=polygon, sides=sides,
        hatch_hypothesis_id=seed["hatch_hypothesis_id"])])
    try:
        solved = solve_vessel_openings(
            points, b2, provisional, private["vessel_point_indexes"][ident],
            ident, config)
        candidate = solved["rectangles"][0]
        state = candidate["status"]
    except (ValueError, KeyError) as error:
        candidate = None
        state = "UNRESOLVED"
        reason = type(error).__name__ + ":" + str(error)
    else:
        reason = ("CORNER_TOPOLOGY_REVIEW" if state.startswith("PARTIAL") else
                  "INSUFFICIENT_CORNER_TOPOLOGY")
    cell = dict(seed_proposal_id=seed["hatch_hypothesis_id"],
                grouped_proposal_ids=[row["hatch_hypothesis_id"] for row in group],
                measured_separator_faces=separator,
                seed_bounds_axial=box, status=("INITIAL_" + state if
                state.startswith("PARTIAL") else "UNRESOLVED"),
                reason=reason, rectangle=candidate if state.startswith("PARTIAL") else None)
    if separator is not None and cell["rectangle"] is not None:
        cell["rectangle"]["hatch_hypothesis_id"] = None
        cell["rectangle"]["lineage_status"] = "COMPOSITE_SPLIT_UNMATCHED"
    return cell


def review_scene(name, path, b2, b4, config, research):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if b2["input_sha256"] != digest:
        raise ValueError("B2_INPUT_SHA_MISMATCH:" + name)
    points = _read_validation_xyz(path)
    scene, private = infer_scene_vessels(points, config, research)
    recorded = {row["vessel_hypothesis_id"]: row for row in b2["scene"]["vessel_hypotheses"]}
    for vessel in scene["vessel_hypotheses"]:
        ident = vessel["vessel_hypothesis_id"]
        if ident not in recorded or vessel["point_count"] != recorded[ident]["point_count"]:
            raise ValueError("B1_B2_PROVENANCE_MISMATCH:" + name + ":" + ident)
    runs = []
    for vessel in scene["vessel_hypotheses"]:
        if vessel["classification_status"] != "VESSEL_HYPOTHESIS":
            continue
        ident = vessel["vessel_hypothesis_id"]
        if b4 is None:
            runs.append(_bootstrap_vessel(points, b2, private, vessel, config))
        else:
            runs.append(solve_vessel_openings(
                points, b2, b4, private["vessel_point_indexes"][ident], ident, config))
    return dict(
        schema="ship_perception.v15r.r2h_b5_corner_topology.1",
        scene_id=name,
        input_sha256=digest,
        static_calibration_status="UNVERIFIED_NO_INDEPENDENT_INNER_EDGE_GOLDEN",
        target_vessel_status="NOT_SELECTED",
        b1_point_provenance_verified=True,
        b1_vessel_hypothesis_count=sum(
            row["classification_status"] == "VESSEL_HYPOTHESIS"
            for row in scene["vessel_hypotheses"]),
        b4_input_available=b4 is not None,
        vessels=runs,
    ), points, private


def render_holdout(report, points, private, b2, output, config):
    """Use different colors for exterior, vessel, opening and review rim."""
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    figure, ax = plt.subplots(figsize=(16, 7))
    active = {row["vessel_hypothesis_id"]: row for row in report["vessels"]}
    source_vessels = {row["vessel_hypothesis_id"]: row for row in
                      b2["scene"]["vessel_hypotheses"]}
    all_selected = np.zeros(len(points), bool)
    vessel_bounds = []
    for ident in active:
        indexes = private["vessel_point_indexes"][ident]
        all_selected[indexes] = True
        vessel_bounds.append(points[indexes, :2])
    exterior = points[~all_selected, :2]
    if len(exterior):
        step = max(1, len(exterior) // 100000)
        ax.scatter(exterior[::step, 0], exterior[::step, 1], s=.2,
                   c="#89749b", alpha=.25, rasterized=True)
    for ident, result in active.items():
        indexes = private["vessel_point_indexes"][ident]
        vessel_points = points[indexes, :2]
        step = max(1, len(vessel_points) // 100000)
        ax.scatter(vessel_points[::step, 0], vessel_points[::step, 1],
                   s=.3, c="#527aa5", alpha=.55, rasterized=True)
        for cell in result.get("initial_opening_cells", []):
            rectangle = cell["rectangle"]
            if rectangle is None:
                continue
            axes = np.asarray(source_vessels[ident]["local_axes"])
            polygon = np.asarray(rectangle["polygon_after"])
            axial = vessel_points @ axes.T
            bounds = polygon @ axes.T
            inside = ((axial[:, 0] >= bounds[:, 0].min()) &
                      (axial[:, 0] <= bounds[:, 0].max()) &
                      (axial[:, 1] >= bounds[:, 1].min()) &
                      (axial[:, 1] <= bounds[:, 1].max()))
            interior = vessel_points[inside]
            step = max(1, len(interior) // 100000)
            ax.scatter(interior[::step, 0], interior[::step, 1],
                       s=.4, c="#46b997", alpha=.6, rasterized=True)
            loop = np.vstack((polygon, polygon[0]))
            ax.plot(loop[:, 0], loop[:, 1], c="#ffbd4a", lw=1.6)
            ax.text(*polygon.mean(axis=0), rectangle["hatch_id"],
                    fontsize=7, color="black",
                    bbox=dict(facecolor="white", alpha=.75))
        for cell in result.get("initial_opening_cells", []):
            if cell["rectangle"] is not None:
                continue
            box = cell["seed_bounds_axial"]
            axes = np.asarray(source_vessels[ident]["local_axes"])
            polygon = np.asarray([[box[0], box[1]], [box[2], box[1]],
                                  [box[2], box[3]], [box[0], box[3]],
                                  [box[0], box[1]]]) @ axes
            ax.plot(polygon[:, 0], polygon[:, 1], c="#777777", lw=.8, ls="--")
    if vessel_bounds:
        xy = np.vstack(vessel_bounds)
        margin = config["roi"]["support_search_m"] * 2
        ax.set_xlim(xy[:, 0].min() - margin, xy[:, 0].max() + margin)
        ax.set_ylim(xy[:, 1].min() - margin, xy[:, 1].max() + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="Raw X (m)", ylabel="Raw Y (m)",
           title="Holdout static review: amber = partial opening candidate; dashed = unresolved seed")
    ax.legend(handles=[Line2D([], [], color=color, marker="o", linestyle="none",
                              label=label) for color, label in
                       (("#89749b", "Exterior / unowned"),
                        ("#527aa5", "Vessel outside opening"),
                        ("#46b997", "Opening interior candidate"))] +
              [Line2D([], [], color="#ffbd4a", label="Candidate rim"),
               Line2D([], [], color="#777777", ls="--", label="Unresolved seed")],
              loc="upper right", fontsize=8)
    figure.tight_layout()
    path = output / "E_semantic_regions.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [str(path)]


def render(name, report, points, private, b4, config, output):
    if b4 is None or not report["vessels"]:
        return []
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs = []
    target = next((v for v in report["vessels"] if v.get("rectangles")), None)
    if target is None:
        return []
    ident = target["vessel_hypothesis_id"]
    axes = np.asarray(next(row["axes"] for row in b4["rectangles"] if
                           row.get("vessel_hypothesis_id") == ident))
    selected = private["vessel_point_indexes"][ident]
    scoped = build_vessel_scoped_topview(points, selected, axes, config)
    masked = scoped["grids"]["coarse"]
    raw = build_topview(points, axes, config)["coarse"]
    rejected = (build_topview(points[~scoped["selected_point_mask"]], axes, config)["coarse"]
                if scoped["rejected_external_points"] else None)

    def draw_grid(ax, grid, title):
        extent = (grid.x0, grid.x0 + grid.cell_m * grid.count.shape[1],
                  grid.y0, grid.y0 + grid.cell_m * grid.count.shape[0])
        ax.imshow(np.ma.masked_invalid(grid.median), extent=extent,
                  origin="lower", aspect="equal", cmap="viridis", interpolation="nearest")
        ax.set(title=title, xlabel="Vessel U (m)", ylabel="Vessel V (m)")

    def plot_box(ax, polygon, color, width, label=None):
        xy = np.asarray(polygon) @ axes.T
        ax.plot(*np.vstack((xy, xy[0])).T, color=color, lw=width, label=label)

    envelopes = [np.asarray(row["polygon_after"]) @ axes.T for row in target["rectangles"]]
    limits = (min(x[:, 0].min() for x in envelopes) - config["roi"]["support_search_m"] * 2,
              max(x[:, 0].max() for x in envelopes) + config["roi"]["support_search_m"] * 2,
              min(x[:, 1].min() for x in envelopes) - config["roi"]["support_search_m"] * 2,
              max(x[:, 1].max() for x in envelopes) + config["roi"]["support_search_m"] * 2)

    def save(fig, axes_list, label):
        for ax in axes_list:
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])
        fig.tight_layout()
        path = output / (label + ".png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path))

    fig, ax = plt.subplots(1, 3, figsize=(21, 5))
    draw_grid(ax[0], raw, "Raw XYZ: all supports")
    draw_grid(ax[1], masked, "B1 XY footprint: all Raw Z restored")
    if rejected is not None:
        draw_grid(ax[2], rejected, "Rejected external XYZ")
    for row in target["rectangles"]:
        plot_box(ax[0], row["polygon_before"], "white", 1)
        plot_box(ax[1], row["polygon_before"], "white", 1)
    save(fig, ax, "A_raw_vs_vessel_masked_height")

    fig, ax = plt.subplots(figsize=(16, 6))
    draw_grid(ax, masked, "Persistent height contours and B2 line modes")
    summary = "V0 " + ", ".join(f"{row['rho']:.2f}" for row in target["v0_candidates"][:5])
    summary += "  |  V1 " + ", ".join(f"{row['rho']:.2f}" for row in target["v1_candidates"][:5])
    detail = ("Orange: selected pair" if target.get("selected_pair") else
              "No pair selected: B4 retained")
    ax.set_title(summary + "\n" + detail + "; dashed: alternative measured modes")
    for mode in target["height_topology"]["modes"][1]:
        ax.axhline(mode["rho"], color="yellow", alpha=.5,
                   lw=1 + mode["level_count"] / 3)
    for label, color in (("v0_candidates", "#ffb15e"),
                         ("v1_candidates", "#70d9ff")):
        for mode in target[label][:6]:
            ax.axhline(mode["rho"], color=color, alpha=.45, lw=.9, ls="--")
    if target.get("selected_pair"):
        for key in ("v0", "v1"):
            ax.axhline(target["selected_pair"][key], color="orange", lw=2)
    for mode in target["u_boundary_chain"]:
        ax.axvline(mode["rho"], color="magenta", lw=.8)
    for row in target["height_topology"]["persistent_corners"]:
        ax.scatter(*row["corner_xy"], marker="x", color="red", s=45)
    save(fig, [ax], "B_persistent_height_edges_and_candidates")

    fig, ax = plt.subplots(figsize=(16, 6))
    draw_grid(ax, masked, "Corner graph: measured / topology / unresolved")
    role_colors = {"MEASURED_INTERSECTION_CANDIDATE": "#1ce3bc",
                   "HEIGHT_TOPOLOGY_CORNER": "yellow",
                   "FOOTPRINT_CONTEXT_CORNER": "magenta",
                   "UNRESOLVED_CORNER": "red"}
    seen = set()
    for row in target["height_topology"]["persistent_corners"]:
        ax.scatter(*row["corner_xy"], marker="x", color="yellow", s=55)
    for row in target["rectangles"]:
        plot_box(ax, row["polygon_after"], "orange", 1.4)
        for corner in row["corners"]:
            key = tuple(corner["corner_xy"])
            if key in seen:
                continue
            seen.add(key)
            color = role_colors[corner["role"]]
            ax.scatter([key[0]], [key[1]], color=color, s=35, zorder=5)
            ax.text(key[0], key[1], f"C{len(seen)-1}", color="white", fontsize=8)
    save(fig, [ax], "C_corner_graph")

    fig, ax = plt.subplots(figsize=(16, 6))
    draw_grid(ax, masked, "B4 before (white) / B5 review (orange)" +
              (" — B4 retained" if target.get("selected_pair") is None else ""))
    for row in target["rectangles"]:
        plot_box(ax, row["polygon_before"], "white", 1, "B4" if row is target["rectangles"][0] else None)
        plot_box(ax, row["polygon_after"], "orange", 2, "B5" if row is target["rectangles"][0] else None)
    ax.legend()
    save(fig, [ax], "D_final_before_after")

    # Render categories from point provenance and the final review polygons.
    # This is an explanatory overlay, not a semantic steel classifier.
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    category = np.zeros(raw.count.shape, dtype=np.uint8)
    category[raw.occupancy] = 1  # external scene / quay / unowned support
    u, v = raw.centers()
    owned = scoped["points"][:, :2] @ axes.T
    rr, cc = raw.cell_indices(owned)
    live = ((rr >= 0) & (rr < category.shape[0]) &
            (cc >= 0) & (cc < category.shape[1]))
    category[rr[live], cc[live]] = 2  # vessel outside opening cells
    width = config["boundary"]["line_inlier_m"] * 2
    for row in target["rectangles"]:
        box = np.asarray(row["polygon_after"]) @ axes.T
        u0, u1 = box[:, 0].min(), box[:, 0].max()
        v0, v1 = box[:, 1].min(), box[:, 1].max()
        inside = ((u >= u0) & (u <= u1) & (v >= v0) & (v <= v1) &
                  (category == 2))
        category[inside] = 3
        rim = ((((abs(u - u0) <= width) | (abs(u - u1) <= width)) &
                (v >= v0 - width) & (v <= v1 + width)) |
               (((abs(v - v0) <= width) | (abs(v - v1) <= width)) &
                (u >= u0 - width) & (u <= u1 + width)))
        category[rim & (category >= 2)] = 4
    palette = ListedColormap(["#ffffff", "#8770a3", "#527aa5",
                              "#46b997", "#ffbd4a"])
    fig, ax = plt.subplots(figsize=(16, 6))
    extent = (raw.x0, raw.x0 + raw.cell_m * raw.count.shape[1],
              raw.y0, raw.y0 + raw.cell_m * raw.count.shape[0])
    ax.imshow(category, extent=extent, origin="lower", aspect="equal",
              cmap=palette, norm=BoundaryNorm(np.arange(-.5, 5.5), 5),
              interpolation="nearest")
    for row in target["rectangles"]:
        plot_box(ax, row["polygon_after"], "#d85b17", 1.3)
        box = np.asarray(row["polygon_after"]) @ axes.T
        ax.text(box[:, 0].mean(), box[:, 1].mean(),
                row["hatch_id"] + " " + row["status"], color="black", fontsize=8,
                ha="center", va="center", bbox=dict(facecolor="white", alpha=.8))
    ax.set(xlabel="Vessel U (m)", ylabel="Vessel V (m)",
           title="Point-provenance regions; amber rim is a review candidate, not certified steel")
    ax.legend(handles=[Patch(color=palette.colors[i], label=label) for i, label in
                       enumerate(("No return", "External / unowned", "Vessel outside opening",
                                  "Opening interior candidate", "Candidate boundary"))],
              loc="upper right", fontsize=8)
    save(fig, [ax], "E_semantic_regions")

    if target.get("section_rho_consensus") and any(
            target["section_rho_consensus"][side]["section_count"]
            for side in ("V0", "V1")):
        fig, ax = plt.subplots(figsize=(12, 5))
        for side, color in (("V0", "#e58a2b"), ("V1", "#287dc1")):
            summary = target["section_rho_consensus"][side]
            values = summary["section_rho_modes"]
            if values:
                ax.plot(np.arange(len(values)), values, "o-", color=color,
                        label=(f"{side}: median {summary['section_rho_median']:.3f} m, "
                               f"MAD {summary['section_rho_mad']:.3f} m"))
        ax.set(xlabel="Occupied U section index", ylabel="Local transition rho (m)",
               title="Independent section rho selections; diagnostic review")
        ax.legend()
        fig.tight_layout()
        path = output / "F_section_rho_modes.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(16, 6))
    draw_grid(ax, masked, "R2G (white dotted), B4 (cyan), final (amber): per-side arbitration")
    for row in target["rectangles"]:
        arbitration = row["side_stage_arbitration"]
        for field, color, style, width in (("r2g_rho", "white", ":", 1.4),
                                           ("b4_rho", "cyan", "--", 1.0),
                                           ("final_rho", "orange", "-", 1.8)):
            values = {key: decision[field] for key, decision in arbitration.items()}
            poly = np.asarray([[values["U0"], values["V0"]],
                               [values["U1"], values["V0"]],
                               [values["U1"], values["V1"]],
                               [values["U0"], values["V1"]]])
            loop = np.vstack((poly, poly[0]))
            ax.plot(loop[:, 0], loop[:, 1], color=color, ls=style, lw=width)
    save(fig, [ax], "G_stage_arbitration")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", required=True,
                        help="SCENE_ID=PCD_OR_PLY_PATH")
    parser.add_argument("--b2-root", type=Path, required=True)
    parser.add_argument("--b4-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config, _ = resolve_config(DEFAULT_CONFIG)
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf8"))
    for spec in args.scene:
        name, source = spec.split("=", 1)
        b2 = json.loads((args.b2_root / name / "hatch_proposals.json").read_text(encoding="utf8"))
        b4_path = (args.b4_root / name / "opening_semantics.json") if args.b4_root else None
        b4 = json.loads(b4_path.read_text(encoding="utf8")) if b4_path and b4_path.exists() else None
        report, points, private = review_scene(name, source, b2, b4, config, research)
        output = args.output_root / name
        output.mkdir(parents=True, exist_ok=True)
        report["png_outputs"] = (render(name, report, points, private, b4, config, output)
                                 if b4 is not None else
                                 render_holdout(report, points, private, b2, output, config))
        _atomic_json(output / "corner_topology.json", report)
        print(name, [(v["vessel_hypothesis_id"], len(v["rectangles"]),
                      v.get("selected_pair")) for v in report["vessels"]], flush=True)


if __name__ == "__main__":
    main()
