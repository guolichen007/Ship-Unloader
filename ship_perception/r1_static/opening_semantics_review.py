"""Offline B4 static opening semantics review; PNGs are never detector inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ship_perception.tools.v15_pcd import decode

from .batch import NORMAL6
from .opening_semantics import bridge_final_rectangles
from .raw_topview_evidence import build_topview
from .run import DEFAULT_CONFIG, REPO, _atomic_json, resolve_config


COLORS = {
    "HATCH_INNER_EDGE_CANDIDATE": "#1ccca5",
    "OUTER_HULL_EDGE": "#e95a56",
    "SEPARATOR_STRUCTURE": "#f0b93a",
    "UNKNOWN": "#a88be5",
}


def _loop(ax, polygon, color, width, label=None):
    if polygon is None:
        return
    xy = np.asarray(polygon, dtype=float)
    xy = np.vstack((xy, xy[0]))
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=width, label=label)


def write_review_ply(rows, config, path):
    """CloudCompare overlay of hypotheses, with side roles kept visible."""
    step = config["geometry"]["refine_voxel_m"]
    vertices = []
    for hatch in rows:
        if hatch.get("polygon_after") is None:
            continue
        polygon = np.asarray(hatch["polygon_after"], dtype=float)
        for side_id, (a, b) in zip(
            ("U0", "U1", "V0", "V1"), ((0, 3), (1, 2), (0, 1), (3, 2))
        ):
            color = COLORS[hatch["sides"][side_id]["role"]]
            rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
            start, end = polygon[[a, b]]
            count = max(2, int(np.ceil(np.linalg.norm(end - start) / step)) + 1)
            for t in np.linspace(0, 1, count):
                x, y = start + t * (end - start)
                vertices.append((x, y, 0.0, *rgb))
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            "comment B4_STATIC_REVIEW_CANDIDATE_NOT_VERIFIED_STEEL\n"
            f"element vertex {len(vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for x, y, z, red, green, blue in vertices:
            stream.write(f"{x:.6f} {y:.6f} {z:.6f} {red} {green} {blue}\n")
    return str(path)


def render(scene, points, b2, r2g, output, config, rank_ablation=None):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b4 = bridge_final_rectangles(points, r2g, b2, config)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "opening_semantics.json", b4)
    write_review_ply(b4["rectangles"], config, output / "reviewed_rectangles.ply")
    axes = np.asarray(r2g["axes"])
    grid = build_topview(points, axes, config)["coarse"]
    extent = (
        grid.x0,
        grid.x0 + grid.count.shape[1] * grid.cell_m,
        grid.y0,
        grid.y0 + grid.count.shape[0] * grid.cell_m,
    )
    outputs = []
    valid_z = np.ma.masked_invalid(grid.median)

    def base(ax, channel="height"):
        arr = valid_z if channel == "height" else np.log1p(grid.count)
        ax.imshow(
            arr,
            extent=extent,
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            aspect="equal",
        )
        ax.set(xlabel="R2G U (m)", ylabel="R2G V (m)")

    fig, ax = plt.subplots(figsize=(16, 6), dpi=140)
    base(ax)
    u, v = grid.centers()
    for h in b4["rectangles"]:
        if h["polygon_after"] is not None:
            evidence = h["opening_interior"]
            if evidence["status"] in ("LOWER_INTERIOR", "RAISED_OR_CARGO_INTERIOR"):
                projected = np.asarray(h["polygon_after"]) @ axes.T
                u0, v0 = projected.min(axis=0)
                u1, v1 = projected.max(axis=0)
                inset = config["boundary"]["side_probe_m"]
                core = (
                    (u >= u0 + inset)
                    & (u <= u1 - inset)
                    & (v >= v0 + inset)
                    & (v <= v1 - inset)
                    & grid.occupancy
                )
                ring = evidence["surrounding_ring_median_z_m"]
                drop = config["roi"]["opening_drop_m"]
                mask = core & (
                    (grid.median < ring - drop)
                    if evidence["status"] == "LOWER_INTERIOR"
                    else (grid.median > ring + drop)
                )
                ax.imshow(
                    np.ma.masked_where(~mask, mask.astype(float)),
                    extent=extent,
                    origin="lower",
                    cmap="cool",
                    alpha=0.3,
                    interpolation="nearest",
                    aspect="equal",
                )
            _loop(ax, np.asarray(h["polygon_after"]) @ axes.T, "#efb22d", 1.8)
            inside = np.asarray(h["polygon_after"]) @ axes.T
            ax.text(
                *inside.mean(axis=0),
                h["hatch_id"] + " " + h["opening_interior"]["status"],
                ha="center",
                color="white",
                fontsize=8,
            )
    ax.set_title(scene + " | raw median Z and reviewed opening interior")
    fig.tight_layout()
    path = output / "A_median_z_opening_interior.png"
    fig.savefig(path)
    plt.close(fig)
    outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(16, 6), dpi=140)
    base(ax, "density")
    for vessel in b2["scene"]["vessel_hypotheses"]:
        box = vessel["bbox_xy"]
        corners = (
            np.asarray(
                [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
            )
            @ axes.T
        )
        _loop(
            ax,
            corners,
            (
                "#eabf48"
                if vessel["classification_status"] == "VESSEL_HYPOTHESIS"
                else "#929aac"
            ),
            1,
        )
    for h in b4["rectangles"]:
        _loop(
            ax,
            np.asarray(h["polygon_after"]) @ axes.T if h["polygon_after"] else None,
            "#20bfa5",
            1.5,
        )
    ax.set_title(
        scene + " | measured density, vessel support bounds, reviewed openings"
    )
    fig.tight_layout()
    path = output / "B_vessel_support_candidate_edges.png"
    fig.savefig(path)
    plt.close(fig)
    outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(16, 6), dpi=140)
    base(ax)
    if rank_ablation is not None:
        for rank, row in enumerate(rank_ablation.get("old_top", [])[:10], 1):
            box = row["bounds_axial"]
            _loop(
                ax,
                [
                    [box[0], box[1]],
                    [box[2], box[1]],
                    [box[2], box[3]],
                    [box[0], box[3]],
                ],
                "#b7c6d0",
                0.6,
            )
            ax.text(box[0], box[3], str(rank), color="white", fontsize=7)
    ax.set_title(scene + " | top R2F accepted candidates by original rank")
    fig.tight_layout()
    path = output / "C_r2f_candidate_ranks.png"
    fig.savefig(path)
    plt.close(fig)
    outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(16, 6), dpi=140)
    base(ax)
    for h in b4["rectangles"]:
        if h["polygon_after"] is None:
            continue
        poly = np.asarray(h["polygon_after"]) @ axes.T
        for sid, (a, b) in zip(
            ("U0", "U1", "V0", "V1"), ((0, 3), (1, 2), (0, 1), (3, 2))
        ):
            side = h["sides"][sid]
            ax.plot(
                poly[[a, b], 0],
                poly[[a, b], 1],
                color=COLORS[side["role"]],
                linewidth=2.1,
            )
            mid = poly[[a, b]].mean(axis=0)
            ax.text(
                mid[0], mid[1], h["hatch_id"] + " " + sid, fontsize=6, color="white"
            )
    ax.set_title(
        scene
        + " | green opening candidate, red outer, yellow separator, purple unknown"
    )
    fig.tight_layout()
    path = output / "D_final_side_roles.png"
    fig.savefig(path)
    plt.close(fig)
    outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(16, 6), dpi=140)
    base(ax)
    for before, after in zip(r2g["rectangles"], b4["rectangles"]):
        _loop(ax, np.asarray(before["polygon_after"]) @ axes.T, "white", 0.8)
        _loop(
            ax,
            (
                np.asarray(after["polygon_after"]) @ axes.T
                if after["polygon_after"]
                else None
            ),
            "#ffb000",
            2,
        )
    ax.set_title(
        scene + " | R2G before white, B4 reviewed geometry amber; status in JSON"
    )
    fig.tight_layout()
    path = output / "E_before_after.png"
    fig.savefig(path)
    plt.close(fig)
    outputs.append(str(path))
    return b4, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", action="append", choices=tuple(NORMAL6), required=True
    )
    parser.add_argument("--r2g-root", type=Path, required=True)
    parser.add_argument("--b2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank-root", type=Path)
    args = parser.parse_args()
    config, _ = resolve_config(DEFAULT_CONFIG)
    manifest = json.loads(
        (REPO / "ship_perception/datasets/v15_dataset_manifest.json").read_text(
            encoding="utf8"
        )
    )
    scans = {row["scan_id"]: row for row in manifest["scans"]}
    for scene in args.scene:
        record = scans[NORMAL6[scene]]
        source = REPO / "Ship-Unloader-Data/legacy/hold_detector" / record["pcd_path"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != record["pcd_sha256"]:
            raise ValueError("INPUT_SHA_MISMATCH:" + scene)
        r2g = json.loads(
            (args.r2g_root / scene / "rectangle_refinement.json").read_text(
                encoding="utf8"
            )
        )
        b2 = json.loads(
            (args.b2_root / scene / "hatch_proposals.json").read_text(encoding="utf8")
        )
        if r2g["input_pcd_sha256"] != digest or b2["input_sha256"] != digest:
            raise ValueError("STAGE_INPUT_SHA_MISMATCH:" + scene)
        points, _ = decode(source)
        rank_path = (
            (args.rank_root / scene / "rank_ablation.json") if args.rank_root else None
        )
        ranking = (
            json.loads(rank_path.read_text(encoding="utf8"))
            if rank_path and rank_path.exists()
            else None
        )
        result, images = render(
            scene, points, b2, r2g, args.output_root / scene, config, ranking
        )
        print(
            scene,
            [(r["hatch_id"], r["status"]) for r in result["rectangles"]],
            len(images),
            flush=True,
        )


if __name__ == "__main__":
    main()
