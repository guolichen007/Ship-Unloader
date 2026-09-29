"""Offline Chinese static-calibration review figures and sampled color PLY.

The categories describe point provenance and polygon location. They are not
training labels, steel classifications, or independent boundary ground truth.
"""

import json
import os
from pathlib import Path

import numpy as np

from .raw_topview_evidence import build_topview


COLORS = np.asarray([
    [255, 255, 255],  # no point; never written to PLY
    [139, 113, 162],  # external / unowned
    [75, 126, 174],   # vessel outside opening candidates
    [47, 180, 143],   # points inside candidate opening
    [255, 177, 72],   # point near candidate boundary
], dtype=np.uint8)


def classify_review_points(points, private, report, axes, config):
    """Classify measured points for display without changing detection."""
    category = np.ones(len(points), dtype=np.uint8)
    width = config["boundary"]["line_inlier_m"] * 2
    for vessel in report["vessels"]:
        ident = vessel["vessel_hypothesis_id"]
        indexes = private["vessel_point_indexes"][ident]
        category[indexes] = 2
        axial = points[indexes, :2] @ axes.T
        for row in vessel["rectangles"]:
            polygon = np.asarray(row["polygon_after"]) @ axes.T
            u0, u1 = polygon[:, 0].min(), polygon[:, 0].max()
            v0, v1 = polygon[:, 1].min(), polygon[:, 1].max()
            interior = ((axial[:, 0] >= u0) & (axial[:, 0] <= u1) &
                        (axial[:, 1] >= v0) & (axial[:, 1] <= v1))
            rim = ((((abs(axial[:, 0] - u0) <= width) |
                     (abs(axial[:, 0] - u1) <= width)) &
                    (axial[:, 1] >= v0 - width) & (axial[:, 1] <= v1 + width)) |
                   (((abs(axial[:, 1] - v0) <= width) |
                     (abs(axial[:, 1] - v1) <= width)) &
                    (axial[:, 0] >= u0 - width) & (axial[:, 0] <= u1 + width)))
            category[indexes[interior]] = 3
            category[indexes[rim]] = 4
    return category


def _write_colored_ply(path, xyz, category):
    colors = COLORS[category]
    vertex = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"),
                                      ("z", "<f4"), ("red", "u1"),
                                      ("green", "u1"), ("blue", "u1")])
    for column, field in enumerate(("x", "y", "z")):
        vertex[field] = xyz[:, column]
    for column, field in enumerate(("red", "green", "blue")):
        vertex[field] = colors[:, column]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertex)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\n"
              "property uchar blue\nend_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(vertex.tobytes())


def render_showcase(report, points, private, b2, config, output):
    """Write one raw-height frame, one colored point plot, and provenance data."""
    active = [row for row in report["vessels"] if row["rectangles"]]
    if not active:
        return []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    ident = active[0]["vessel_hypothesis_id"]
    axes = np.asarray(next(row["local_axes"] for row in
                           b2["scene"]["vessel_hypotheses"] if
                           row["vessel_hypothesis_id"] == ident))
    axial = points[:, :2] @ axes.T
    category = classify_review_points(points, private, report, axes, config)
    grid = build_topview(points, axes, config)["coarse"]
    polygons = [(row, np.asarray(row["polygon_after"]) @ axes.T)
                for vessel in active for row in vessel["rectangles"]]
    boxes = np.vstack([polygon for _, polygon in polygons])
    margin = config["roi"]["support_search_m"] * 2
    limits = (boxes[:, 0].min() - margin, boxes[:, 0].max() + margin,
              boxes[:, 1].min() - margin, boxes[:, 1].max() + margin)

    def frame(title):
        fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("船体纵向 U（米）", fontproperties=font)
        ax.set_ylabel("船体横向 V（米）", fontproperties=font)
        ax.set_title(title, fontproperties=font, fontsize=16)
        return fig, ax

    def boxes_on(ax):
        for row, polygon in polygons:
            loop = np.vstack((polygon, polygon[0]))
            ax.plot(loop[:, 0], loop[:, 1], color="#ffac32", lw=2.3,
                    ls="-" if row["status"].startswith("PARTIAL") else "--")
            center = polygon.mean(axis=0)
            state = "局部候选" if row["status"].startswith("PARTIAL") else "待复核"
            ax.text(center[0], center[1], row["hatch_id"] + " · " + state,
                    fontproperties=font, fontsize=10, color="#202020",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="#ffac32", alpha=.85))

    fig, ax = frame(report["scene_id"] + "｜原始 XYZ 高度图与船舱开口候选")
    extent = (grid.x0, grid.x0 + grid.cell_m * grid.count.shape[1],
              grid.y0, grid.y0 + grid.cell_m * grid.count.shape[0])
    im = ax.imshow(np.ma.masked_invalid(grid.median), extent=extent,
                   origin="lower", aspect="equal", cmap="viridis",
                   interpolation="nearest")
    boxes_on(ax)
    cbar = fig.colorbar(im, ax=ax, shrink=.78, pad=.02)
    cbar.set_label("原始点云中位高度（米）", fontproperties=font)
    ax.text(.01, .02, "琥珀线：待复核的开口边界", transform=ax.transAxes,
            fontproperties=font, fontsize=9,
            bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
    fig.tight_layout()
    raw_path = output / "原始XYZ高度_船舱候选单图.png"
    fig.savefig(raw_path, dpi=180)
    plt.close(fig)

    fig, ax = frame(report["scene_id"] + "｜原始点云按船舱区域分色")
    names = ((1, "船体外／未归属点"), (2, "船体内、开口外"),
             (3, "候选船舱内"), (4, "候选围挡附近"))
    for index, label in names:
        selected = np.flatnonzero(category == index)
        if not len(selected):
            continue
        selected = selected[::max(1, len(selected) // 90000)]
        ax.scatter(axial[selected, 0], axial[selected, 1],
                   s=.35 if index != 4 else .75,
                   color=COLORS[index] / 255.0, alpha=.52 if index != 4 else .9,
                   rasterized=True)
    boxes_on(ax)
    handles = [Line2D([], [], marker="o", ls="none", color=COLORS[index] / 255.0,
                      label=label, markersize=7) for index, label in names]
    handles.append(Line2D([], [], color="#ffac32", lw=2.3, label="候选矩形"))
    ax.legend(handles=handles, prop=font, loc="upper right", fontsize=8,
              framealpha=.92)
    ax.text(.01, .02, "分色表示点的位置与来源；围挡颜色不等于钢边真值",
            transform=ax.transAxes, fontproperties=font, fontsize=9,
            bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    fig.tight_layout()
    color_path = output / "点云分区_舱内围挡船外.png"
    fig.savefig(color_path, dpi=180)
    plt.close(fig)

    count = len(points)
    sampled = np.arange(0, count, max(1, int(np.ceil(count / 400000))))
    ply_path = output / "点云分区_采样彩色.ply"
    _write_colored_ply(ply_path, points[sampled, :3], category[sampled])
    manifest = dict(
        场景=report["scene_id"], 原始输入SHA256=report["input_sha256"],
        彩色PLY坐标系="原始XYZ（米）",
        静态标定状态=report["static_calibration_status"],
        目标船状态=report["target_vessel_status"],
        分区说明={name: label for name, label in names},
        注意="颜色为原始点归属和候选几何位置，不是钢材/物料语义真值。",
        原始点数=count, 彩色PLY采样点数=len(sampled),
        船舱=[dict(编号=row["hatch_id"], 状态=row["status"],
                 矩形角点XY=row["polygon_after"],
                 B2舱假设=row.get("hatch_hypothesis_id"),
                 四角=[dict(位置=corner["corner_xy"], 角色=corner["role"])
                     for corner in row.get("corners", [])])
              for vessel in active for row in vessel["rectangles"]],
    )
    data_path = output / "标定结果与证据.json"
    data_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    return [str(raw_path), str(color_path), str(ply_path), str(data_path)]
