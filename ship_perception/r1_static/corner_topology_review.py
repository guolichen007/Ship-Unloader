"""Offline B5 review with point-provenance checks and focused corner plots."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .corner_topology import solve_vessel_openings
from .heightmap_batch import RESEARCH_CONFIG
from .raw_topview_evidence import build_topview
from .rectangle_refinement_review import _read_validation_xyz
from .run import DEFAULT_CONFIG, _atomic_json, resolve_config
from .scene_vessel import infer_scene_vessels
from .vessel_height_topology import build_vessel_scoped_topview


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
            scoped = build_vessel_scoped_topview(
                points, private["vessel_point_indexes"][ident], vessel["local_axes"], config)
            proposal = next((row for row in b2["per_vessel"] if
                             row["vessel_hypothesis_id"] == ident), None)
            runs.append(dict(
                vessel_hypothesis_id=ident,
                status="NO_B4_RECTANGLE_INPUT_PROPOSAL_ONLY",
                rejected_external_points=scoped["rejected_external_points"],
                restored_inside_footprint_points=scoped[
                    "restored_inside_footprint_points"],
                vessel_footprint_cells=scoped["vessel_footprint_cells"],
                b2_hatch_proposal_count=len(proposal["hatch_hypotheses"]) if proposal else 0,
                rectangles=[],
            ))
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
    for mode in target["height_topology"]["modes"][1]:
        ax.axhline(mode["rho"], color="yellow", alpha=.5,
                   lw=1 + mode["level_count"] / 3)
    for mode in target["v0_candidates"] + target["v1_candidates"]:
        if min(mode["primary_raw3d"], mode["primary_bev"]) >= config["roi"]["min_enclosure_ratio"]:
            ax.axhline(mode["rho"], color="cyan", alpha=.12, lw=.6)
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
    draw_grid(ax, masked, "B4 before (white) / B5 candidate (orange)")
    for row in target["rectangles"]:
        plot_box(ax, row["polygon_before"], "white", 1, "B4" if row is target["rectangles"][0] else None)
        plot_box(ax, row["polygon_after"], "orange", 2, "B5" if row is target["rectangles"][0] else None)
    ax.legend()
    save(fig, [ax], "D_final_before_after")
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
        report["png_outputs"] = render(name, report, points, private, b4, config, output)
        _atomic_json(output / "corner_topology.json", report)
        print(name, [(v["vessel_hypothesis_id"], len(v["rectangles"]),
                      v.get("selected_pair")) for v in report["vessels"]], flush=True)


if __name__ == "__main__":
    main()
