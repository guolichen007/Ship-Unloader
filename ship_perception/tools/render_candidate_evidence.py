"""Render full-scene unconfirmed search evidence for offline review."""

import argparse
import json
from pathlib import Path
import numpy as np


def render(result, frame, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ship_perception.r1_static.height_grid import R1HeightGrid
    from ship_perception.perception.review_labels import chinese_plot_style
    from ship_perception.perception.final_boundary import polygon
    from ship_perception.perception.line_geometry import SIDE_SPECS

    grid = R1HeightGrid.from_points(frame.xyz, 0.5)
    axes = {
        v["vessel_id"]: np.asarray(v["local_axes"])
        for v in result["scene_context"]["vessels"]
    }
    with matplotlib.rc_context(chinese_plot_style()):
        figure, ax = plt.subplots(figsize=(20, 9))
        ax.imshow(
            np.ma.masked_where(~grid.occupancy, grid.median),
            origin="lower",
            extent=(
                grid.x0,
                grid.x0 + grid.count.shape[1] * grid.cell_m,
                grid.y0,
                grid.y0 + grid.count.shape[0] * grid.cell_m,
            ),
            cmap="viridis",
            aspect="equal",
        )
        ranked = sorted(
            result.get("physical_candidate_registry", []),
            key=lambda q: q.get("raw_probe", {}).get(
                "raw_probe_support_side_count", -1
            ),
        )
        for query in ranked:
            if query["state"] == "REJECTED":
                continue
            transform = axes[query["vessel_hypothesis_id"]]
            box = query["bounds_axial"]
            probe = query.get("raw_probe", {})
            if not probe:
                points = polygon(box) @ transform
                ax.plot(
                    *np.vstack((points, points[0])).T,
                    color="#dddddd",
                    alpha=0.10,
                    lw=0.3,
                    ls=":",
                )
                continue
            for name, axis, position, sign in SIDE_SPECS:
                side = probe["sides"].get(name, {})
                rho = side.get("rho", box[position])
                start, end = np.zeros(2), np.zeros(2)
                start[axis] = end[axis] = rho
                start[1 - axis], end[1 - axis] = box[1 - axis], box[3 - axis]
                points = np.array([start, end]) @ transform
                strong = side.get("status") == "STRONG_UNIQUE_EDGE"
                ax.plot(
                    *points.T,
                    color="#80ff9f" if strong else "#f7b267",
                    alpha=0.8 if strong else 0.25,
                    lw=1.1 if strong else 0.5,
                    ls="--",
                )
            if probe.get("raw_probe_support_side_count", 0) >= 2:
                center = (
                    np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) @ transform
                )
                ax.text(
                    *center,
                    query["query_id"][:8]
                    + " / "
                    + str(probe["raw_probe_support_side_count"]),
                    fontsize=6,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.55),
                )
        ax.set(
            title="原始点云边证据复核：候选范围不代表已确认船舱",
            xlabel="输入 X（米）",
            ylabel="输入 Y（米）",
        )
        ax.text(
            0.01,
            -0.08,
            "绿色：附近存在 Raw 边证据；橙色：歧义或缺测；虚线：待精修位置。完整逐边状态见 JSON。",
            transform=ax.transAxes,
            fontsize=9,
        )
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(figure)


def main():
    from ship_perception.io.pcd_source import read_pcd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_bytes())
    frame = read_pcd(args.pcd, result["input"]["frame_id"])
    if frame.sha256 != result["input"]["semantic_sha256"]:
        raise ValueError("DIAGNOSTIC_INPUT_IDENTITY_MISMATCH")
    render(result, frame, args.output)


if __name__ == "__main__":
    main()
