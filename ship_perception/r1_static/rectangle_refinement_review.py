"""Six-scene offline review of existing R2F rectangular hatch calibration."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ship_perception.tools.v15_pcd import decode

from .batch import NORMAL6
from .boundary_topology import read_xyzrgb_ply
from .rectangle_refinement import refine
from .run import DEFAULT_CONFIG, REPO, _atomic_json, resolve_config


ROLE_COLORS = {
    "INNER_EDGE": "#00bd8f",
    "AMBIGUOUS": "#edaa2e",
    "CARGO_BOUNDARY": "#d64848",
    "DISTANT_STRUCTURE": "#8052ae",
}


def _draw_rectangles(ax, rows, annotate=False):
    for hatch in rows:
        before = np.asarray(hatch["polygon_before"])
        after = np.asarray(hatch["polygon_after"])
        ax.plot(
            *np.vstack((before, before[0])).T, color="#94a4b1", linewidth=0.8, alpha=0.8
        )
        pairs = ((0, 3), (1, 2), (0, 1), (3, 2))
        for side_id, (a, b) in zip(("U0", "U1", "V0", "V1"), pairs):
            side = hatch["sides"][side_id]
            color = ROLE_COLORS[side["role"]]
            ax.plot(after[[a, b], 0], after[[a, b], 1], color=color, linewidth=2.2)
            if annotate:
                mid = after[[a, b]].mean(axis=0)
                ax.text(
                    mid[0],
                    mid[1],
                    "%s %+.2fm" % (side_id, side["delta_m"]),
                    fontsize=6,
                    color=color,
                    bbox=dict(facecolor="#101820", edgecolor="none", alpha=0.8),
                )
        center = after.mean(axis=0)
        ax.text(
            center[0],
            center[1],
            hatch["hatch_id"],
            color="white",
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            bbox=dict(facecolor="#101820", edgecolor="none", alpha=0.8),
        )


def _save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path)


def render_review(result, points, ply_xyz, config, destination):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(destination / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs = []
    sample = ply_xyz[:: max(1, len(ply_xyz) // 40000)]
    for filename, annotated in (
        ("rectangular_hatch_before_after.png", True),
        ("canonical_inner_edges.png", False),
    ):
        fig, ax = plt.subplots(figsize=(17, 6), dpi=150)
        ax.scatter(
            sample[:, 0], sample[:, 1], s=1.5, c="#8495a2", alpha=0.4, rasterized=True
        )
        _draw_rectangles(ax, result["rectangles"], annotate=annotated)
        ax.set(
            aspect="equal",
            xlabel="Raw X (m)",
            ylabel="Raw Y (m)",
            title=result["scene_id"] + " | R2F thin gray, R2G thick role-colored",
        )
        ax.grid(alpha=0.15)
        outputs.append(_save_fig(fig, destination / filename))

    rows = result["rectangles"]
    fig, axs = plt.subplots(
        max(1, len(rows)),
        4,
        figsize=(16, max(3, 2.7 * len(rows))),
        dpi=130,
        squeeze=False,
    )
    for row_index, hatch in enumerate(rows):
        for col, side_id in enumerate(("U0", "U1", "V0", "V1")):
            ax = axs[row_index, col]
            candidates = hatch["side_refinement_profiles"][side_id]
            rhos = [candidate["rho_grid"] for candidate in candidates]
            ax.scatter(
                rhos,
                [candidate["structural_coverage"] for candidate in candidates],
                s=12,
                label="Raw3D",
            )
            ax.scatter(
                rhos,
                [candidate["raster_coverage"] for candidate in candidates],
                s=12,
                label="RawXYZ",
            )
            side = hatch["sides"][side_id]
            ax.axvline(
                side["rho_before"], color="#94a4b1", linewidth=0.8, label="before"
            )
            ax.axvline(
                side["rho_after"],
                color=ROLE_COLORS[side["role"]],
                linewidth=2,
                label="after",
            )
            ax.set(
                title=hatch["hatch_id"] + " " + side_id,
                xlabel="rho (m)",
                ylabel="coverage",
                ylim=(-0.04, 1.04),
            )
            ax.grid(alpha=0.15)
            if row_index == 0 and col == 0:
                ax.legend(fontsize=6)
    outputs.append(_save_fig(fig, destination / "side_refinement_profiles.png"))

    axes = np.asarray(result["axes"])
    from .raw_topview_evidence import build_topview

    grid = build_topview(points, axes, config)["fine"]
    extent = (
        grid.x0,
        grid.x0 + grid.count.shape[1] * grid.cell_m,
        grid.y0,
        grid.y0 + grid.count.shape[0] * grid.cell_m,
    )
    panels = (
        (grid.median, "median Z (m)"),
        (np.log1p(grid.count), "log(1 + count)"),
        (grid.maximum - grid.minimum, "vertical span (m)"),
    )
    fig, axs = plt.subplots(3, 1, figsize=(17, 12), dpi=130)
    for ax, (array, title) in zip(axs, panels):
        image = ax.imshow(
            array,
            origin="lower",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
            aspect="equal",
        )
        for hatch in rows:
            before = np.asarray(hatch["polygon_before"]) @ axes.T
            after = np.asarray(hatch["polygon_after"]) @ axes.T
            ax.plot(*np.vstack((before, before[0])).T, color="#d6d6d6", linewidth=0.8)
            ax.plot(*np.vstack((after, after[0])).T, color="#ffb000", linewidth=1.8)
        ax.set(xlabel="Ship-axis U (m)", ylabel="Ship-axis V (m)", title=title)
        fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    fig.suptitle(result["scene_id"] + " | measured Raw XYZ, before white, after amber")
    outputs.append(_save_fig(fig, destination / "raw_topview_fusion.png"))
    return outputs


def write_model_ply(rows, config, path):
    step = config["geometry"]["refine_voxel_m"]
    vertices = []
    for hatch in rows:
        polygon = np.asarray(hatch["polygon_after"])
        for side_id, pair in zip(
            ("U0", "U1", "V0", "V1"), ((0, 3), (1, 2), (0, 1), (3, 2))
        ):
            a, b = polygon[list(pair)]
            count = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
            color = (
                (0, 189, 143)
                if hatch["sides"][side_id]["role"] == "INNER_EDGE"
                else (237, 170, 46)
            )
            for t in np.linspace(0, 1, count):
                x, y = a + t * (b - a)
                vertices.append((x, y, 0.0, *color))
    with Path(path).open("w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "ply\nformat ascii 1.0\ncomment R2G_REFINED_CANDIDATE_NOT_VERIFIED_STEEL\n"
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
    expected_sha,
    height_root,
    initial_root,
    structural_root,
    review_root,
    output_root,
    config,
    recompute_structural,
):
    pcd = Path(pcd)
    actual_sha = hashlib.sha256(pcd.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("INPUT_SHA_MISMATCH:" + scene_id)
    height = json.loads(
        (Path(height_root) / scene_id / "hatch_regions.json").read_text(
            encoding="utf-8"
        )
    )
    initial = json.loads(
        (Path(initial_root) / scene_id / "rectangular_hatch_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        height["input_sha256"] != actual_sha
        or initial["input_pcd_sha256"] != actual_sha
    ):
        raise ValueError("BASELINE_INPUT_MISMATCH:" + scene_id)
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
    result = refine(points, segments, edges, initial, config)
    result.update(
        scene_id=scene_id,
        input_pcd_sha256=actual_sha,
        structural_source=source,
        r2f_initial_sha256=hashlib.sha256(
            (
                Path(initial_root) / scene_id / "rectangular_hatch_candidates.json"
            ).read_bytes()
        ).hexdigest(),
    )
    destination = Path(output_root) / scene_id
    destination.mkdir(parents=True, exist_ok=True)
    ply = Path(review_root) / (scene_id + "_structural_lines.ply")
    ply_xyz = read_xyzrgb_ply(ply)
    result["source_review_ply"] = str(ply)
    result["png_outputs"] = render_review(result, points, ply_xyz, config, destination)
    result["model_ply_output"] = write_model_ply(
        result["rectangles"], config, destination / "refined_rectangles.ply"
    )
    _atomic_json(destination / "axis_diagnostics.json", result["axis_diagnostics"])
    _atomic_json(destination / "rectangle_refinement.json", result)
    return result


def _read_validation_xyz(path):
    """Decode copied map inputs without accessing annotations or split labels."""
    if path.suffix.lower() == ".pcd":
        points, _ = decode(path)
        return points
    if path.suffix.lower() != ".ply":
        raise ValueError("UNSUPPORTED_VALIDATION_INPUT:" + path.name)
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line or len(header) > 64:
                raise ValueError("INVALID_VALIDATION_PLY_HEADER:" + path.name)
            header.append(line.decode("utf-8", errors="replace").strip())
            if header[-1] == "end_header":
                break
        if header[:2] != ["ply", "format ascii 1.0"]:
            raise ValueError("UNSUPPORTED_VALIDATION_PLY_FORMAT:" + path.name)
        vertex = [row for row in header if row.startswith("element vertex ")]
        properties = [row for row in header if row.startswith("property ")]
        if len(vertex) != 1 or properties != [
            "property float x",
            "property float y",
            "property float z",
        ]:
            raise ValueError("UNSUPPORTED_VALIDATION_PLY_SCHEMA:" + path.name)
        count = int(vertex[0].split()[-1])
        points = np.loadtxt(stream, dtype=float, max_rows=count)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError("INVALID_VALIDATION_PLY_POINTS:" + path.name)
    return points


def run_validation_file(path, output_root, config):
    """Run the frozen upstream stages and R2G on one copied unlabeled scan."""
    from .family_reference import analyze_scene_s4r1
    from .heightmap_batch import RESEARCH_CONFIG
    from .heightmap_topology import detect_regions
    from .rectangle_fusion import solve

    path = Path(path)
    points = _read_validation_xyz(path)
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    height, _ = detect_regions(points, config, research)
    reference = analyze_scene_s4r1(points, config, research, height)
    initial = solve(
        points,
        reference["segment_resolutions"],
        reference["edges"],
        height["level_hierarchy"],
        config,
    )
    if "axes" in initial:
        result = refine(
            points,
            reference["segment_resolutions"],
            reference["edges"],
            initial,
            config,
        )
    else:
        result = dict(
            status="NO_REFINEMENT_STRUCTURAL_AXES_UNRESOLVED",
            static_calibration_status="UNVERIFIED_NO_INDEPENDENT_INNER_EDGE_GOLDEN",
            rectangle_count_before=len(initial["rectangles"]),
            rectangle_count_after=0,
            rectangles=[],
            axis_diagnostics=dict(status="AXIS_UNRESOLVED"),
            reason=initial["status"],
        )
    result.update(
        scene_id=path.stem,
        input_file=path.name,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        input_point_count=len(points),
        validation_role="UNLABELED_MAP_SCAN_NO_GT_USED",
        upstream_region_count=height["region_count"],
        upstream_rectangle_count=len(initial["rectangles"]),
        upstream_observation_accounting_pass=initial.get("observation_accounting_pass"),
        upstream_structural_edge_count=len(reference["edges"]),
    )
    destination = Path(output_root) / path.stem
    destination.mkdir(parents=True, exist_ok=True)
    if "axes" in initial:
        result["png_outputs"] = render_review(
            result, points, points, config, destination
        )
    else:
        import os
        import matplotlib

        os.environ.setdefault("MPLCONFIGDIR", str(destination / ".mplconfig"))
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample = points[:: max(1, len(points) // 40000)]
        fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
        ax.scatter(sample[:, 0], sample[:, 1], s=1.0, c=sample[:, 2], rasterized=True)
        ax.set(aspect="equal", title=path.stem + " | " + initial["status"])
        result["png_outputs"] = [
            _save_fig(fig, destination / "raw_topview_unresolved.png")
        ]
    result["model_ply_output"] = write_model_ply(
        result["rectangles"], config, destination / "refined_rectangles.ply"
    )
    _atomic_json(destination / "axis_diagnostics.json", result["axis_diagnostics"])
    _atomic_json(destination / "rectangle_refinement.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-set", choices=("normal6", "map-validation"), required=True
    )
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
        "--initial-root",
        type=Path,
        default=REPO / "Ship-Unloader-Work/review/r2f_rectangular_fusion_final",
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
        default=REPO / "Ship-Unloader-Work/review/r2g_rectangular_refinement",
    )
    parser.add_argument("--recompute-structural", action="store_true")
    parser.add_argument("--only-scene")
    parser.add_argument("--validation-input-dir", type=Path)
    args = parser.parse_args()
    config, _ = resolve_config(DEFAULT_CONFIG)
    if args.scene_set == "map-validation":
        if args.validation_input_dir is None:
            parser.error("--validation-input-dir is required for map-validation")
        files = sorted(
            path
            for path in args.validation_input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in (".pcd", ".ply")
            and (args.only_scene is None or path.stem == args.only_scene)
        )
        if not files:
            parser.error("No matching validation files")
        summary = {}
        for path in files:
            result = run_validation_file(path, args.output_root, config)
            summary[path.stem] = dict(
                input_file=path.name,
                input_sha256=result["input_sha256"],
                region_count=result["upstream_region_count"],
                rectangle_count=result["rectangle_count_after"],
                status=result["status"],
            )
            print(
                path.stem,
                json.dumps(summary[path.stem], ensure_ascii=False),
                flush=True,
            )
        _atomic_json(args.output_root / "batch_summary.json", summary)
        return
    if args.only_scene is not None and args.only_scene not in NORMAL6:
        parser.error("Unknown normal6 scene: " + args.only_scene)
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
            record["pcd_sha256"],
            args.height_root,
            args.initial_root,
            args.structural_root,
            args.review_root,
            args.output_root,
            config,
            args.recompute_structural,
        )
        summary[scene_id] = dict(
            before=result["rectangle_count_before"],
            after=result["rectangle_count_after"],
            axis_status=result["axis_diagnostics"]["status"],
            axis_delta_deg=result["axis_diagnostics"]["axis_delta_deg"],
        )
        print(scene_id, json.dumps(summary[scene_id], ensure_ascii=False), flush=True)
    _atomic_json(args.output_root / "batch_summary.json", summary)


if __name__ == "__main__":
    main()
