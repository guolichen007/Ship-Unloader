"""R0-A raw-PCD-only blind review pack for six Development scenes.

This module does not open annotation files or import proposal/recognizer output.
All pictures and reviewer templates are local ignored artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ship_perception.tools.v15_pcd import decode
from ship_perception.tools.validate_v15r_scene_manifest import read_json, DATASET

NORMAL = ("2026-08-01-05-09-32", "2026-08-08-02-33-57",
          "final_map_4_16_2d_005", "final_map_6_8_2d_005",
          "final_map_7_16_2d_005", "final_map_8_22_cut")
RENDER = {"schema": "ship_perception.r0a.blind_render.1", "source": "STRICT_RAW_XYZ_ONLY",
          "cell_m": 0.25, "min_points_per_cell": 5, "sample_max_points": 40000,
          "slice_half_width_m": 0.5, "slice_quantiles": [0.25, 0.50, 0.75],
          "z_color_percentiles": [2, 98], "layers": ["RAW_XYZ"]}
FORBIDDEN = {"CNN_CANDIDATE", "GEOMETRY_CANDIDATE", "WEAK_BOX", "ALGORITHM_EDGE", "ORACLE_OUTPUT"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_render_config(config):
    if config["source"] != "STRICT_RAW_XYZ_ONLY" or set(config["layers"]) != {"RAW_XYZ"}:
        raise ValueError("ALGORITHM_OVERLAY_LEAKAGE_MARKER")
    if FORBIDDEN & set(config["layers"]):
        raise ValueError("ALGORITHM_OVERLAY_LEAKAGE_MARKER")


def height_grid(points, cell):
    origin = np.floor(points[:, :2].min(axis=0) / cell) * cell
    ij = np.floor((points[:, :2]-origin)/cell).astype(np.int64)
    nx, ny = int(ij[:, 0].max())+1, int(ij[:, 1].max())+1
    if nx*ny > 4_000_000:
        raise ValueError("EXCESSIVE_RAW_GRID_EXTENT")
    key = ij[:, 1]*nx+ij[:, 0]
    order = np.lexsort((points[:, 2], key))
    keys = key[order]
    starts = np.r_[0, np.flatnonzero(np.diff(keys))+1]
    ends = np.r_[starts[1:], len(order)]
    middle = starts+(ends-starts-1)//2
    median = np.full(nx*ny, np.nan, dtype=np.float32)
    count = np.zeros(nx*ny, dtype=np.int32)
    median[keys[starts]] = points[order[middle], 2]
    count[keys[starts]] = ends-starts
    extent = [float(origin[0]), float(origin[0]+nx*cell),
              float(origin[1]), float(origin[1]+ny*cell)]
    return median.reshape(ny, nx), count.reshape(ny, nx), extent


def render_scan(points, output: Path, scan_id: str, config=RENDER):
    verify_render_config(config)
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    median, density, extent = height_grid(points, config["cell_m"])
    valid = density >= config["min_points_per_cell"]
    z = median[valid]
    lo, hi = np.percentile(z, config["z_color_percentiles"]) if len(z) else (0, 1)
    if hi-lo < 0.01:
        hi = lo+1
    images = []

    def save(name, fig):
        path = output / name
        fig.savefig(path, dpi=140, metadata={"Software": "R0-A blind raw XYZ renderer"})
        plt.close(fig)
        images.append({"filename": name, "sha256": sha(path)})

    for name, data, cmap, label in (
        ("top_height.png", np.ma.masked_where(~valid, median), "viridis", "raw median Z (m)"),
        ("top_density.png", np.ma.masked_where(density == 0, np.log1p(density)), "magma", "log(1+raw point count)"),
    ):
        fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
        image = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, aspect="equal", interpolation="nearest",
                          **({"vmin": lo, "vmax": hi} if name == "top_height.png" else {}))
        fig.colorbar(image, ax=ax, label=label)
        ax.set(xlabel="raw X (m)", ylabel="raw Y (m)", title=f"{scan_id}: {label}; blank = no return")
        save(name, fig)

    seed = int(hashlib.sha256(scan_id.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    selected = points[rng.choice(len(points), min(len(points), config["sample_max_points"]), replace=False)]
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(selected[:, 0], selected[:, 1], selected[:, 2], c=selected[:, 2], cmap="viridis", s=0.3, vmin=lo, vmax=hi)
    ax.view_init(elev=24, azim=-60)
    ax.set(xlabel="raw X (m)", ylabel="raw Y (m)", zlabel="raw Z (m)", title=f"{scan_id}: raw XYZ overview")
    save("xyz_overview.png", fig)

    for name, a, b in (("xz_ortho.png", 0, 2), ("yz_ortho.png", 1, 2)):
        fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
        ax.scatter(selected[:, a], selected[:, b], s=0.3, c=selected[:, 2], cmap="viridis", vmin=lo, vmax=hi)
        ax.set(xlabel=f"raw {'XYZ'[a]} (m)", ylabel="raw Z (m)", title=f"{scan_id}: raw orthographic {name[:2].upper()}")
        save(name, fig)

    for axis, quantiles in ((0, config["slice_quantiles"]), (1, [0.5])):
        for q in quantiles:
            at = float(np.quantile(points[:, axis], q))
            subset = points[np.abs(points[:, axis]-at) <= config["slice_half_width_m"]]
            fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
            remaining = 1-axis
            ax.scatter(subset[:, remaining], subset[:, 2], s=0.5, c=subset[:, 2], cmap="viridis", vmin=lo, vmax=hi)
            ax.set(xlabel=f"raw {'XYZ'[remaining]} (m)", ylabel="raw Z (m)",
                   title=f"{scan_id}: raw {'XYZ'[axis]} slice q={q:g}, center={at:.3f} m, half-width={config['slice_half_width_m']} m")
            save(f"{'xy'[axis]}_slice_{int(q*100):02d}.png", fig)
    return images


def review_template(scan, pack_index_sha, role):
    return {"schema": "ship_perception.r0a.human_review.1", "review_status": "DRAFT",
            "scan_id": scan["scan_id"], "source_pcd_sha256": scan["pcd_sha256"],
            "blind_pack_index_sha256": pack_index_sha, "reviewer_role": role,
            "reviewer_id": None, "repo_access": None, "review_a_prior_access": None,
            "algorithm_output_access": None, "raw_render_only": None,
            "vessel_roi_raw_xy": None, "fov_coverage": "UNKNOWN",
            "visible_in_fov_hatch_count": None, "whole_vessel_hatch_count": None,
            "hatches": [], "reason": "Human reviewer must fill from raw PCD/render; no algorithm overlays."}


def run(data_root: Path, output: Path, review_root: Path):
    dataset = read_json(DATASET)
    sources = {s["scan_id"]: s for s in dataset["scans"]}
    if any(sources[sid]["split_role"] != "DEVELOPMENT" for sid in NORMAL):
        raise ValueError("NON_DEVELOPMENT_SCAN_IN_BLIND_PACK")
    existing = [review_root / f"review_{role}" / f"{sid}.json"
                for role in ("A", "B") for sid in NORMAL
                if (review_root / f"review_{role}" / f"{sid}.json").exists()]
    if existing:
        raise FileExistsError(f"REVIEW_TEMPLATE_EXISTS_NO_OVERWRITE:{existing[0]}")
    index = {"schema": "ship_perception.r0a.blind_pack.1", "render_config": RENDER,
             "annotation_inputs_read": False, "algorithm_outputs_read": False, "scans": []}
    output.mkdir(parents=True, exist_ok=True)
    for sid in NORMAL:
        scan = sources[sid]
        pcd = data_root / scan["pcd_path"]
        if sha(pcd) != scan["pcd_sha256"]:
            raise ValueError(f"PCD_SHA_MISMATCH:{sid}")
        points, _ = decode(pcd)
        images = render_scan(points, output/sid, sid)
        index["scans"].append({"scan_id": sid, "pcd_sha256": scan["pcd_sha256"],
                               "point_count": int(len(points)), "renders": images})
    index_path = output / "blind_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    index_sha = sha(index_path)
    for role in ("A", "B"):
        directory = review_root / f"review_{role}"
        directory.mkdir(parents=True, exist_ok=True)
        for sid in NORMAL:
            target = directory / f"{sid}.json"
            if target.exists():
                raise FileExistsError(f"REVIEW_TEMPLATE_EXISTS_NO_OVERWRITE:{target}")
            target.write_text(json.dumps(review_template(sources[sid], index_sha, role),
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    return index_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("Ship-Unloader-Work/review/r0_a"))
    ap.add_argument("--review-root", type=Path, default=Path("Ship-Unloader-Data/annotations/v15r"))
    args = ap.parse_args()
    print(run(args.data_root, args.output, args.review_root))


if __name__ == "__main__":
    main()
