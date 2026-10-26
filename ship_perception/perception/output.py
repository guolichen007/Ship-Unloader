"""Atomic scene review and four-corner output, with a separate failure contract."""

import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.intensity_evidence import IntensityEvidence, AUX_WEIGHT
from ship_perception.r1_static.run import _atomic_json
from .pipeline import _json_default
from .review_labels import FIGURE_TEXT, annotate_measurements, chinese_plot_style


def auxiliary_review(payload, frame, ownership_indexes):
    result = copy.deepcopy(payload["geometry_only"])
    stats = payload["intensity_status"]
    result.update(intensity_used=False, intensity_fusion_status=stats["status"])
    if payload.get("production_geometry_mode") == "XYZ_ONLY":
        result["intensity_fusion_status"] = "DISABLED_PRODUCTION_XYZ_ONLY"
        return result
    if stats["status"] != "AVAILABLE":
        return result
    from ship_perception.r1_static.run import resolve_config

    config, _ = resolve_config()
    scorers = {}
    for hatch in result["hatches"]:
        ident = hatch["vessel_hypothesis_id"]
        axes = np.asarray(hatch["vessel_local_axes"])
        key = (ident, tuple(axes.ravel()))
        if key not in scorers:
            owned = ownership_indexes[ident]
            scorers[key] = IntensityEvidence(
                frame.xyz[owned],
                frame.intensity[owned],
                axes,
                config["geometry"]["candidate_voxel_m"],
            )
        scorer = scorers[key]
        uv = np.asarray(hatch["boundary_xy"]) @ axes.T
        low, high = uv.min(0), uv.max(0)
        sides = {
            name: scorer.edge(
                axis, (low if end == 0 else high)[axis], (low[1 - axis], high[1 - axis])
            )
            for name, axis, end in (
                ("U0", 0, 0),
                ("U1", 0, 1),
                ("V0", 1, 0),
                ("V1", 1, 1),
            )
        }
        score = float(np.mean([s["intensity_edge_score"] for s in sides.values()]))
        hatch["intensity_evidence"] = dict(
            sides=sides,
            auxiliary_score_bonus=AUX_WEIGHT * score,
            source_calibration="UNKNOWN",
            geometry_changed=False,
            role="AUXILIARY_ONLY",
        )
        result["intensity_used"] |= any(s["intensity_used"] for s in sides.values())
    return result


def render(payload, frame, path, *, fusion=False, grid=None):
    import matplotlib

    matplotlib.use("Agg")
    with matplotlib.rc_context(chinese_plot_style()):
        _render_scene(payload, frame, path, fusion=fusion, grid=grid)


def _render_scene(payload, frame, path, *, fusion=False, grid=None):
    import matplotlib.pyplot as plt

    grid = R1HeightGrid.from_points(frame.xyz, 0.5) if grid is None else grid
    step = max(1, int(np.ceil(max(grid.count.shape) / 1000)))
    display_aspect = grid.count.shape[0] / grid.count.shape[1]
    figure, ax = plt.subplots(figsize=(20, max(6, min(12, 20 * display_aspect + 1))))
    masked = np.ma.masked_where(~grid.occupancy, grid.median)
    image = ax.imshow(
        masked[::step, ::step],
        origin="lower",
        interpolation="nearest",
        extent=(
            grid.x0,
            grid.x0 + grid.count.shape[1] * grid.cell_m,
            grid.y0,
            grid.y0 + grid.count.shape[0] * grid.cell_m,
        ),
        cmap="viridis",
        aspect="equal",
    )
    figure.colorbar(image, ax=ax, label=FIGURE_TEXT["colorbar"])
    # Leave room for corner labels while retaining every measured scene cell.
    y_low, y_high = ax.get_ylim()
    ax.set_ylim(y_low - 0.1 * (y_high - y_low), y_high + 0.04 * (y_high - y_low))
    result = payload["intensity_fusion" if fusion else "geometry_only"]
    for hatch in result["hatches"]:
        poly = np.asarray(hatch["boundary_xy"])
        coverage = np.asarray(hatch["review_coverage_boundary_xy"])
        ax.plot(*np.vstack((poly, poly[0])).T, color="#ffbb33", lw=1.6)
        ax.plot(
            *np.vstack((coverage, coverage[0])).T,
            color="#ffbb33",
            lw=0.7,
            ls=":",
            alpha=0.6,
        )
    title = FIGURE_TEXT["geometry_title"]
    if fusion:
        title = (
            FIGURE_TEXT["intensity_unavailable"]
            if payload["intensity_status"]["status"] != "AVAILABLE"
            else FIGURE_TEXT["intensity_available"]
        )
        if payload.get("production_geometry_mode") == "XYZ_ONLY":
            title = "强度判定已关闭：复用纯几何复核结果"
    ax.set(title=title, xlabel=FIGURE_TEXT["xlabel"], ylabel=FIGURE_TEXT["ylabel"])
    ax.text(
        0.01,
        -0.12,
        FIGURE_TEXT["legend"],
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85),
    )
    figure.tight_layout()
    annotate_measurements(ax, result["hatches"])
    figure.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def save_output(payload, frame, output_dir, save_snapshot, timing):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ownership = payload.pop("_ownership_indexes", {})
    payload["intensity_fusion"] = auxiliary_review(payload, frame, ownership)
    payload["intensity_status"]["used"] = payload["intensity_fusion"]["intensity_used"]
    with tempfile.TemporaryDirectory(
        prefix=".ship_hatch_", dir=str(output_dir)
    ) as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        import os

        with tempfile.TemporaryDirectory(prefix="ship_hatch_mpl_") as mpl_cache:
            previous = os.environ.get("MPLCONFIGDIR")
            os.environ["MPLCONFIGDIR"] = mpl_cache
            try:
                with timing.stage("RENDER_PREPARE"):
                    display_grid = R1HeightGrid.from_points(frame.xyz, 0.5)
                for fusion, filename in (
                    (False, "01_geometry_only.png"),
                    (True, "02_intensity_fusion.png"),
                ):
                    with timing.stage("RENDER_" + filename[:2]):
                        render(
                            payload,
                            frame,
                            staged / filename,
                            fusion=fusion,
                            grid=display_grid,
                        )
                    payload["intensity_fusion" if fusion else "geometry_only"][
                        "image_path"
                    ] = str(final / filename)
            finally:
                if previous is None:
                    os.environ.pop("MPLCONFIGDIR", None)
                else:
                    os.environ["MPLCONFIGDIR"] = previous
        if save_snapshot:
            with timing.stage("SAVE_INPUT_SNAPSHOT"):
                np.savez_compressed(
                    staged / "input_snapshot.npz",
                    xyz=frame.xyz,
                    intensity=(
                        np.array([]) if frame.intensity is None else frame.intensity
                    ),
                    metadata=json.dumps(payload["input"], allow_nan=False),
                )
        with timing.stage("SAVE_RESULTS"):
            if "performance_budget" in payload:
                report = payload["performance_budget"]
                elapsed = sum(
                    value
                    for key, value in timing.elapsed.items()
                    if key.startswith("RENDER")
                )
                report["stages"].append(
                    dict(
                        stage="RENDER",
                        elapsed_s=elapsed,
                        budget_s=5.0,
                        status=(
                            "PERFORMANCE_BUDGET_EXCEEDED"
                            if elapsed > 5.0
                            else "WITHIN_BUDGET"
                        ),
                    )
                )
                report["total_elapsed_s"] += elapsed
                report["total_budget_exceeded"] = (
                    report["total_elapsed_s"] > report["total_budget_s"]
                )
            payload["stage_elapsed"].update(timing.elapsed)
            if timing.progress is not None:
                payload["stage_elapsed"].update(timing.progress.elapsed_stages)
            clean = json.loads(
                json.dumps(payload, default=_json_default, allow_nan=False)
            )
            from jsonschema import Draft202012Validator

            schema = (
                Path(__file__).resolve().parents[1] / "config/hatch_once.schema.json"
            )
            Draft202012Validator(json.loads(schema.read_text())).validate(clean)
            _atomic_json(staged / "03_results.json", clean)
            from .coordinates import corner_coordinates

            _atomic_json(
                staged / "05_hatch_corners.json",
                corner_coordinates(
                    clean,
                    hashlib.sha256(
                        (staged / "03_results.json").read_bytes()
                    ).hexdigest(),
                ),
            )
            (staged / "04_input_summary.txt").write_text(
                "\n".join(
                    key.upper() + "=" + json.dumps(value, allow_nan=False)
                    for key, value in {
                        **payload["input"],
                        "intensity_status": payload["intensity_status"],
                        "pipeline_health": payload["pipeline_health"],
                    }.items()
                )
                + "\n",
                encoding="utf-8",
            )
        staged.rename(final)
    return final


def save_failure(output_dir, error, stage=None):
    directory = Path(output_dir).resolve() / (
        "failure_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    directory.mkdir(parents=True)
    _atomic_json(
        directory / "failure.json",
        dict(
            status="FAILED",
            exception_type=type(error).__name__,
            error_code=str(error),
            first_failure_stage=stage,
            diagnostics=getattr(error, "diagnostics", {}),
        ),
    )
    return directory
