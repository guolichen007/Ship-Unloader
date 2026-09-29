"""Offline B3 Raw XYZ and side-mode figures for an existing two-hatch review.

This reads unlabelled scan XYZ and prior detector outputs. It never feeds
annotations, expected widths, or review geometry into the detector.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .raw_topview_evidence import build_topview, raster_modes
from .rectangle_refinement_review import _read_validation_xyz
from .run import DEFAULT_CONFIG, _atomic_json, resolve_config


COLORS = {"H000": "#00a88a", "H001": "#e69f00"}
SIDES = (("U0", (0, 3)), ("U1", (1, 2)),
         ("V0", (0, 1)), ("V1", (3, 2)))


def _polygon(ax, polygon, *, color, linestyle="-", linewidth=2, label=None):
    polygon = np.asarray(polygon)
    ax.plot(*np.vstack((polygon, polygon[0])).T, color=color,
            linestyle=linestyle, linewidth=linewidth, label=label)


def _limits(ax, xyz):
    ax.set_xlim(float(np.percentile(xyz[:, 0], 1)),
                float(np.percentile(xyz[:, 0], 99)))
    ax.set_ylim(float(np.percentile(xyz[:, 1], 1)),
                float(np.percentile(xyz[:, 1], 99)))
    ax.set_aspect("equal")
    ax.set_xlabel("Raw X (m)")
    ax.set_ylabel("Raw Y (m)")
    ax.grid(alpha=.12)


def render(scan, r2f_path, before_path, after_path, b2_path, output):
    import os

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scan = Path(scan)
    digest = hashlib.sha256(scan.read_bytes()).hexdigest()
    r2f = json.loads(Path(r2f_path).read_text(encoding="utf-8"))
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after = json.loads(Path(after_path).read_text(encoding="utf-8"))
    b2 = json.loads(Path(b2_path).read_text(encoding="utf-8"))
    if not (digest == r2f["input_pcd_sha256"] == before["input_pcd_sha256"] ==
            after["input_pcd_sha256"] == b2["input_sha256"]):
        raise ValueError("INPUT_SHA_MISMATCH")
    old_rows = {row["hatch_id"]: row for row in before["rectangles"]}
    new_rows = {row["hatch_id"]: row for row in after["rectangles"]}
    if set(old_rows) != set(new_rows):
        raise ValueError("RECTANGLE_ID_SET_CHANGED")
    points = _read_validation_xyz(scan)
    config, _ = resolve_config(DEFAULT_CONFIG)
    axes = np.asarray(after["axes"])
    grid = build_topview(points, axes, config)["fine"]
    raw_modes = raster_modes(grid, config)
    sample = points[::max(1, len(points) // 50000)]
    outputs = []

    fig, ax = plt.subplots(figsize=(15, 6), dpi=160)
    ax.scatter(sample[:, 0], sample[:, 1], c=sample[:, 2], cmap="viridis",
               s=.6, alpha=.65, rasterized=True)
    for hatch_id, row in new_rows.items():
        color = COLORS.get(hatch_id, "#00a88a")
        _polygon(ax, old_rows[hatch_id]["polygon_after"], color=color,
                 linestyle="--", linewidth=1.1, label=hatch_id + " before B3")
        _polygon(ax, row["polygon_after"], color=color, linewidth=2.1,
                 label=hatch_id + " after B3")
        ax.text(*np.asarray(row["polygon_after"]).mean(axis=0), hatch_id,
                color="white", ha="center", va="center", fontsize=9,
                bbox=dict(facecolor="#17242b", edgecolor="none", alpha=.85))
    _limits(ax, sample)
    ax.set_title("08-01 | Raw XYZ top view and two independent hatches")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    path = output / "01_raw_xyz_topview_before_after.png"
    fig.savefig(path)
    plt.close(fig)
    outputs.append(str(path))

    fig, axs = plt.subplots(2, 4, figsize=(18, 7), dpi=150, squeeze=False)
    search = config["roi"]["boundary_search_m"]
    for row_index, hatch_id in enumerate(sorted(new_rows)):
        hatch = new_rows[hatch_id]
        old = old_rows[hatch_id]
        axial = np.asarray(old["polygon_before"]) @ axes.T
        for col, (side_id, _) in enumerate(SIDES):
            ax = axs[row_index, col]
            side = hatch["sides"][side_id]
            old_side = old["sides"][side_id]
            axis = 0 if side_id.startswith("U") else 1
            span = (float(axial[:, 1-axis].min()), float(axial[:, 1-axis].max()))
            modes = [row for row in raw_modes[axis]
                     if abs(row["rho"] - old_side["rho_before"]) <= search
                     and row["along_max"] >= span[0]
                     and row["along_min"] <= span[1]]
            max_support = max((row["support_count"] for row in modes), default=1)
            ax.scatter([row["rho"] for row in modes],
                       [row["support_count"] / max_support for row in modes],
                       s=14, alpha=.55, color="#437fba", label="RawXYZ mode support")
            profile = hatch["side_refinement_profiles"][side_id]
            ax.scatter([row["rho_grid"] for row in profile],
                       [row["structural_coverage"] for row in profile],
                       s=24, marker="x", color="#8b5bb7", label="Raw3D coverage")
            ax.scatter([row["rho_grid"] for row in profile],
                       [row["raster_coverage"] for row in profile],
                       s=11, marker="s", facecolors="none", edgecolors="#2d87b5",
                       label="BEV coverage")
            ax.axvline(old_side["rho_before"], color="#777777", linewidth=1,
                       linestyle=":", label="R2F")
            ax.axvline(side["rho_after"], color=COLORS.get(hatch_id, "#00a88a"),
                       linewidth=2, label="B3 selected")
            alternatives = sorted(
                (row for row in profile
                 if abs(row["rho_grid"] - side["rho_after"]) >
                 config["geometry"]["plane_inlier_m"]),
                key=lambda row: (row["role"] == "INNER_EDGE",
                                 row["structural_coverage"],
                                 row["raster_coverage"]), reverse=True)[:2]
            for alternative in alternatives:
                ax.axvline(alternative["rho_grid"], color="#db704d", alpha=.55,
                           linestyle="--", linewidth=.8)
            ax.set(title="%s %s | conflict=%s" %
                   (hatch_id, side_id, side["mode_conflict"]),
                   xlabel="rho (m)", ylim=(-.05, 1.05))
            ax.grid(alpha=.15)
    axs[0, 0].set_ylabel("normalized support / coverage")
    axs[1, 0].set_ylabel("normalized support / coverage")
    axs[0, 0].legend(fontsize=5, loc="upper left")
    fig.tight_layout()
    path = output / "02_per_side_rho_modes.png"
    fig.savefig(path)
    plt.close(fig)
    outputs.append(str(path))

    fig, axs = plt.subplots(2, 1, figsize=(15, 8), dpi=150)
    associations = {row["old_hatch_id"]: row for row in
                    b2["prior_r2f_r2g"]["associations"]}
    per_vessel = {row["vessel_hypothesis_id"]: row for row in b2["per_vessel"]}
    for ax, hatch_id in zip(axs, sorted(new_rows)):
        row = new_rows[hatch_id]
        polygon = np.asarray(row["polygon_after"])
        lo = polygon.min(axis=0) - search * 2
        hi = polygon.max(axis=0) + search * 2
        near = sample[np.all((sample[:, :2] >= lo) & (sample[:, :2] <= hi), axis=1)]
        ax.scatter(near[:, 0], near[:, 1], s=.45, color="#a4afb4", alpha=.25,
                   rasterized=True)
        associated = associations[hatch_id]["new_hatch_hypothesis_id"]
        vessel_id = associated.split(":")[0]
        context = per_vessel[vessel_id]
        hypothesis = next(row for row in context["hatch_hypotheses"]
                          if row["hatch_hypothesis_id"] == associated)
        side_provider_ids = {provider_id for side in hypothesis["side_evidence"].values()
                             for provider_id in side["supporting_provider_ids"]}
        for evidence in context["provider_evidence"]:
            if evidence["provider_id"] not in side_provider_ids:
                continue
            if evidence["geometry_type"] != "AXIAL_LINE_INTERVAL":
                continue
            endpoints = np.asarray(evidence["source_geometry"]["endpoints_xy"])
            if not np.any(np.all((endpoints >= lo) & (endpoints <= hi), axis=1)):
                continue
            kind = evidence["provider_type"]
            color = "#2c79b8" if kind == "BEV_RECTILINEAR_PROVIDER" else "#8c5ab3"
            ax.plot(endpoints[:, 0], endpoints[:, 1], color=color, alpha=.6,
                    linewidth=1.1)
        for side_id, pair in SIDES:
            p = polygon[list(pair)]
            color = "#00a88a" if row["sides"][side_id]["role"] == "INNER_EDGE" else "#e69f00"
            ax.plot(p[:, 0], p[:, 1], color=color, linewidth=2.4)
            mid = p.mean(axis=0)
            ax.text(mid[0], mid[1], side_id, fontsize=7, color=color,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.75))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set(aspect="equal", xlabel="Raw X (m)", ylabel="Raw Y (m)",
               title=hatch_id + " | associated B2 proposal (approx.): BEV blue, Raw3D purple; B3 green/amber")
        ax.grid(alpha=.12)
    fig.tight_layout()
    path = output / "03_b2_provider_b3_side_roles.png"
    fig.savefig(path)
    plt.close(fig)
    outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(15, 6), dpi=160)
    ax.scatter(sample[:, 0], sample[:, 1], s=.55, color="#9ba9b0", alpha=.28,
               rasterized=True)
    r2f_rows = {row["hatch_id"]: row for row in r2f["rectangles"]}
    for hatch_id in sorted(new_rows):
        color = COLORS.get(hatch_id, "#00a88a")
        _polygon(ax, r2f_rows[hatch_id]["polygon_xy"], color="#697c88",
                 linestyle=":", linewidth=1.1,
                 label="R2F" if hatch_id == sorted(new_rows)[0] else None)
        _polygon(ax, old_rows[hatch_id]["polygon_after"], color=color,
                 linestyle="--", linewidth=1.2, label=hatch_id + " R2G before B3")
        _polygon(ax, new_rows[hatch_id]["polygon_after"], color=color,
                 linewidth=2.2, label=hatch_id + " R2G after B3")
        width = (new_rows[hatch_id]["sides"]["V1"]["rho_after"] -
                 new_rows[hatch_id]["sides"]["V0"]["rho_after"])
        center = np.asarray(new_rows[hatch_id]["polygon_after"]).mean(axis=0)
        ax.text(center[0], center[1], "%s %.2f m" % (hatch_id, width),
                color="white", ha="center", fontsize=9,
                bbox=dict(facecolor="#17242b", edgecolor="none", alpha=.85))
    _limits(ax, sample)
    ax.set_title("08-01 | R2F / R2G before B3 / R2G after B3")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    path = output / "04_r2f_r2g_before_after_b3.png"
    fig.savefig(path)
    plt.close(fig)
    outputs.append(str(path))

    report = dict(scene_id="08-01", input_sha256=digest,
                  png_outputs=outputs,
                  widths={hatch_id: dict(
                      r2f_m=old_rows[hatch_id]["sides"]["V1"]["rho_before"] -
                      old_rows[hatch_id]["sides"]["V0"]["rho_before"],
                      r2g_before_m=old_rows[hatch_id]["sides"]["V1"]["rho_after"] -
                      old_rows[hatch_id]["sides"]["V0"]["rho_after"],
                      r2g_after_m=new_rows[hatch_id]["sides"]["V1"]["rho_after"] -
                      new_rows[hatch_id]["sides"]["V0"]["rho_after"],
                      evidence_only_width_m=new_rows[hatch_id]["evidence_only_width_m"],
                      width_prior_used=new_rows[hatch_id]["width_prior_used"])
                          for hatch_id in new_rows})
    _atomic_json(output / "08-01_width_review.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("scan", "r2f", "before", "after", "b2", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.scan, args.r2f, args.before, args.after,
                            args.b2, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
