"""Offline six-scan review output for provisional rectangular openings."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ship_perception.tools.v15_pcd import decode

from .batch import NORMAL6
from .boundary_topology import read_xyzrgb_ply
from .raw_topview_evidence import build_topview
from .rectangle_fusion import solve
from .run import DEFAULT_CONFIG, REPO, _atomic_json, resolve_config


SIDE_COLORS = {
    "MULTI_EVIDENCE": "#12c5a0",
    "OBSERVED_STRUCTURAL": "#3b83d4",
    "OBSERVED_RASTER_BOUNDARY": "#efa82d",
    "GEOMETRY_CONSTRAINED": "#de6a4b",
    "UNRESOLVED": "#6c7180",
}
SIDE_CODES = {
    "MULTI_EVIDENCE": "M",
    "OBSERVED_STRUCTURAL": "S",
    "OBSERVED_RASTER_BOUNDARY": "R",
    "GEOMETRY_CONSTRAINED": "G",
    "UNRESOLVED": "?",
}


def _plot_rectangles(ax, rectangles, *, annotate_sides):
    for hatch in rectangles:
        points = np.asarray(hatch["polygon_xy"])
        center = points.mean(axis=0)
        ax.text(
            center[0],
            center[1],
            hatch["hatch_id"],
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            color="white",
            bbox=dict(facecolor="#141f2a", alpha=0.85, edgecolor="none"),
        )
        for index, side in enumerate(hatch["sides"]):
            a, b = (
                (points[0], points[3]),
                (points[1], points[2]),
                (points[0], points[1]),
                (points[3], points[2]),
            )[index]
            color = SIDE_COLORS[side["evidence_level"]]
            ax.plot((a[0], b[0]), (a[1], b[1]), color=color, linewidth=1.6)
            if annotate_sides:
                mid = (a + b) / 2
                ax.text(
                    mid[0],
                    mid[1],
                    side["side_id"] + ":" + SIDE_CODES[side["evidence_level"]],
                    fontsize=6,
                    color=color,
                    bbox=dict(facecolor="#141f2a", alpha=0.75, edgecolor="none"),
                )


def render_review(result, points, ply_xyz, config, output):
    cache = Path(output) / ".mplconfig"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    rectangles = result["rectangles"]
    overview = output / "rectangular_hatch_candidates.png"
    fig, ax = plt.subplots(figsize=(17, 6), dpi=150)
    sample = (
        ply_xyz if len(ply_xyz) <= 40000 else ply_xyz[:: max(1, len(ply_xyz) // 40000)]
    )
    ax.scatter(
        sample[:, 0], sample[:, 1], s=1.5, c="#6a7f94", alpha=0.45, rasterized=True
    )
    _plot_rectangles(ax, rectangles, annotate_sides=False)
    ax.set(
        aspect="equal",
        xlabel="Raw X (m)",
        ylabel="Raw Y (m)",
        title=result["scene_id"] + " | provisional rectangles on structural review PLY",
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(overview)
    plt.close(fig)

    edges = output / "canonical_inner_edges.png"
    fig, ax = plt.subplots(figsize=(17, 6), dpi=150)
    ax.scatter(
        sample[:, 0], sample[:, 1], s=1.5, c="#6a7f94", alpha=0.3, rasterized=True
    )
    _plot_rectangles(ax, rectangles, annotate_sides=True)
    ax.set(
        aspect="equal",
        xlabel="Raw X (m)",
        ylabel="Raw Y (m)",
        title="Candidate inner edges | M=multi S=structural R=raster G=geometry",
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(edges)
    plt.close(fig)

    axes = [np.asarray(axis) for axis in result["axes"]]
    fine = build_topview(points, axes, config)["fine"]
    extent = (
        fine.x0,
        fine.x0 + fine.count.shape[1] * fine.cell_m,
        fine.y0,
        fine.y0 + fine.count.shape[0] * fine.cell_m,
    )
    panels = (
        (fine.median, "median Z (m)"),
        (np.log1p(fine.count), "log(1 + point count)"),
        (fine.maximum - fine.minimum, "vertical span (m)"),
    )
    view = output / "raw_topview_fusion.png"
    fig, axs = plt.subplots(3, 1, figsize=(17, 12), dpi=130)
    for ax, (array, label) in zip(axs, panels):
        image = ax.imshow(
            array,
            origin="lower",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
            aspect="equal",
        )
        for hatch in rectangles:
            polygon = np.asarray(hatch["polygon_xy"]) @ np.asarray(axes).T
            loop = np.vstack((polygon, polygon[0]))
            ax.plot(loop[:, 0], loop[:, 1], color="#ffb000", linewidth=1.2)
            center = polygon.mean(axis=0)
            ax.text(
                center[0],
                center[1],
                hatch["hatch_id"],
                fontsize=7,
                color="white",
                ha="center",
                va="center",
                bbox=dict(facecolor="#141f2a", alpha=0.7, edgecolor="none"),
            )
        ax.set(xlabel="Ship-axis U (m)", ylabel="Ship-axis V (m)", title=label)
        fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    fig.suptitle(result["scene_id"] + " | Raw XYZ rasters; PLY and PNG are review only")
    fig.tight_layout()
    fig.savefig(view)
    plt.close(fig)
    return [str(path) for path in (overview, edges, view)]


def write_rectangle_ply(rectangles, config, path):
    """Write derived model lines with explicit provisional provenance."""
    step = config["geometry"]["refine_voxel_m"]
    vertices = []
    colors = {
        "MULTI_EVIDENCE": (18, 197, 160),
        "OBSERVED_STRUCTURAL": (59, 131, 212),
        "OBSERVED_RASTER_BOUNDARY": (239, 168, 45),
        "GEOMETRY_CONSTRAINED": (222, 106, 75),
        "UNRESOLVED": (108, 113, 128),
    }
    for hatch in rectangles:
        polygon = np.asarray(hatch["polygon_xy"])
        for index, side in enumerate(hatch["sides"]):
            a, b = (
                (polygon[0], polygon[3]),
                (polygon[1], polygon[2]),
                (polygon[0], polygon[1]),
                (polygon[3], polygon[2]),
            )[index]
            count = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
            for t in np.linspace(0.0, 1.0, count):
                x, y = a + t * (b - a)
                vertices.append((x, y, 0.0, *colors[side["evidence_level"]]))
    path = Path(path)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "ply\nformat ascii 1.0\ncomment PROVISIONAL_RECTANGLE_MODEL_NOT_RAW_STEEL\n"
        )
        stream.write("element vertex %d\n" % len(vertices))
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for x, y, z, red, green, blue in vertices:
            stream.write("%.6f %.6f %.6f %d %d %d\n" % (x, y, z, red, green, blue))
    return str(path)


def run_scene(
    scene_id,
    pcd,
    height_root,
    structural_root,
    review_root,
    output_root,
    config,
    expected_sha256,
    recompute_structural=False,
):
    pcd = Path(pcd)
    actual_sha256 = hashlib.sha256(pcd.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("INPUT_SHA_MISMATCH:" + scene_id)
    height = json.loads(
        (Path(height_root) / scene_id / "hatch_regions.json").read_text(
            encoding="utf-8"
        )
    )
    if height["input_sha256"] != actual_sha256:
        raise ValueError("HEIGHT_BASELINE_INPUT_MISMATCH:" + scene_id)
    points, _ = decode(pcd)
    if recompute_structural:
        from .family_reference import analyze_scene_s4r1
        from .heightmap_batch import RESEARCH_CONFIG

        research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
        reference = analyze_scene_s4r1(points, config, research, height)
        segments, edges = reference["segment_resolutions"], reference["edges"]
        source = "S4R1_RECOMPUTED_FROM_VERIFIED_RAW_XYZ"
    else:
        reference = json.loads(
            (
                Path(structural_root) / scene_id / "segment_reference_resolution.json"
            ).read_text(encoding="utf-8")
        )
        structural = json.loads(
            (Path(structural_root) / scene_id / "structural_edges.json").read_text(
                encoding="utf-8"
            )
        )
        segments, edges = reference["segments"], structural["edges"]
        source = "S4R1_CACHED_RAW3D_OBSERVATIONS"
    result = solve(points, segments, edges, height["level_hierarchy"], config)
    result.update(
        scene_id=scene_id,
        input_pcd_sha256=actual_sha256,
        height_baseline_input_sha256=height["input_sha256"],
        structural_source=source,
    )
    destination = Path(output_root) / scene_id
    destination.mkdir(parents=True, exist_ok=True)
    ply = Path(review_root) / (scene_id + "_structural_lines.ply")
    xyz = read_xyzrgb_ply(ply)
    result["source_review_ply"] = str(ply)
    result["source_review_ply_sha256"] = hashlib.sha256(ply.read_bytes()).hexdigest()
    result["png_outputs"] = render_review(result, points, xyz, config, destination)
    result["model_ply_output"] = write_rectangle_ply(
        result["rectangles"], config, destination / "provisional_rectangles.ply"
    )
    _atomic_json(destination / "rectangular_hatch_candidates.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-set", choices=("normal6",), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "ship_perception/datasets/v15_dataset_manifest.json",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO / "Ship-Unloader-Data/legacy/hold_detector",
    )
    parser.add_argument(
        "--height-root",
        type=Path,
        default=REPO / "Ship-Unloader-Work/r1_heightmap/baseline_004_heightmap",
    )
    parser.add_argument(
        "--structural-root",
        type=Path,
        default=REPO / "Ship-Unloader-Work/r1_family_reference/baseline_008_s4r1",
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        default=REPO / "Ship-Unloader-Work/review/r1_structural_lines",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "Ship-Unloader-Work/review/r2f_rectangular_fusion",
    )
    parser.add_argument("--recompute-structural", action="store_true")
    parser.add_argument("--only-scene", choices=tuple(NORMAL6))
    args = parser.parse_args()
    config, _ = resolve_config(DEFAULT_CONFIG)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scans = {row["scan_id"]: row for row in manifest["scans"]}
    summary = {}
    for scene_id, scan_id in NORMAL6.items():
        if args.only_scene and scene_id != args.only_scene:
            continue
        record = scans[scan_id]
        if record["split_role"] != "DEVELOPMENT":
            raise ValueError("NON_DEVELOPMENT_INPUT:" + scan_id)
        result = run_scene(
            scene_id,
            args.data_root / record["pcd_path"],
            args.height_root,
            args.structural_root,
            args.review_root,
            args.output_root,
            config,
            record["pcd_sha256"],
            args.recompute_structural,
        )
        summary[scene_id] = dict(
            rectangle_count=len(result["rectangles"]),
            canonical_inner_edge_count=result["canonical_inner_edge_count"],
            observation_accounting_pass=result["observation_accounting_pass"],
        )
        print(scene_id, json.dumps(summary[scene_id], ensure_ascii=False), flush=True)
    _atomic_json(args.output_root / "batch_summary.json", summary)


if __name__ == "__main__":
    main()
