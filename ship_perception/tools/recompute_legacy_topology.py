"""Offline Raw replay of frozen topology; never the production hatch entry."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ship_perception.io.pcd_source import read_pcd
from ship_perception.perception.context import SceneEvidenceContext
from ship_perception.perception.legacy_topology import FrozenLegacyTopologyProvider
from ship_perception.perception.pipeline import (
    software_identity,
    source_identity,
    numerical_environment,
)
from ship_perception.r1_static.height_grid import R1HeightGrid
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG
from ship_perception.r1_static.run import _atomic_json, resolve_config
from ship_perception.tools.stage_progress import StageProgress


def render_regions(payload, frame, path):
    """Show complete Raw background and conservative XY; missing Z stays null."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ship_perception.perception.review_labels import (
        chinese_plot_style,
        annotate_measurements,
    )

    grid = R1HeightGrid.from_points(frame.xyz, 0.5)
    with matplotlib.rc_context(chinese_plot_style()):
        fig, ax = plt.subplots(figsize=(22, 8))
        image = ax.imshow(
            np.ma.masked_where(~grid.occupancy, grid.median),
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            aspect="equal",
            extent=(
                grid.x0,
                grid.x0 + grid.count.shape[1] * grid.cell_m,
                grid.y0,
                grid.y0 + grid.count.shape[0] * grid.cell_m,
            ),
        )
        fig.colorbar(image, ax=ax, label="原始点云 Z 中位高度（米）")
        hatches = []
        for index, region in enumerate(payload["regions"]):
            poly = np.asarray(region["conservative_boundary_xy"])
            ax.plot(*np.vstack((poly, poly[0])).T, color="#ffbb33", lw=1.6, ls="--")
            hatches.append(
                dict(
                    hatch_id="R%02d" % index,
                    status="UNRESOLVED",
                    boundary_xy=poly,
                    candidate_type="UNRESOLVED_OPENING_GROUP",
                )
            )
        ax.set(
            title="冻结历史拓扑诊断：保守候选区域（召回门待审；非钢边标定）",
            xlabel="输入坐标 X（米）",
            ylabel="输入坐标 Y（米）",
        )
        ax.text(
            0.01,
            -0.16,
            "虚线：当前 Raw 重算的保守区域｜四角为区域几何交点｜Z 无直接观测，保持未观测｜数量不代表真实召回通过",
            transform=ax.transAxes,
            fontsize=9,
        )
        fig.tight_layout()
        annotate_measurements(ax, hatches)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)


def run(pcd, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    progress = StageProgress()
    start = time.perf_counter()
    frame = read_pcd(pcd)
    config, config_sha = resolve_config()
    research = json.loads(RESEARCH_CONFIG.read_bytes())
    scene = SceneEvidenceContext.build(
        frame, config, research, config_sha, software_identity()
    )
    batch = FrozenLegacyTopologyProvider().recompute(
        frame, scene, config, research, progress
    )
    core_elapsed = time.perf_counter() - start
    payload = dict(
        batch.audit(),
        input_sha256=frame.provenance.get("wire_sha256", frame.sha256),
        input_semantic_sha256=frame.sha256,
        config_sha256=config_sha,
        source_identity=identity,
        numerical_environment=numerical_environment(),
        scene_ownership=scene.audit(),
        core_elapsed_s=core_elapsed,
        selected_vessel_runtime_integration="NOT_ENABLED_RECALL_GATE_PENDING",
    )
    current = source_identity()
    if current != identity:
        raise ValueError("SOURCE_CHANGED_DURING_HISTORICAL_REPLAY")
    render_regions(payload, frame, output_dir / "01_topology_regions.png")
    payload["elapsed_s"] = time.perf_counter() - start
    payload["render_and_export_elapsed_s"] = payload["elapsed_s"] - core_elapsed
    _atomic_json(output_dir / "topology.json", payload)
    progress.finish(True)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.pcd, args.output_dir)
    print(
        "TOPOLOGY_AUDIT_ONLY regions={} elapsed_s={:.3f}".format(
            len(result["regions"]), result["elapsed_s"]
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
