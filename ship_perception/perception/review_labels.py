"""Display final review coordinates and dimensions without changing geometry."""

import numpy as np

from .coordinates import boundary_dimensions


FIGURE_TEXT = dict(
    geometry_title="原始点云高度与船舱边界（几何复核）",
    intensity_unavailable="强度不可用：复用几何识别结果",
    intensity_available="几何与强度辅助复核（强度源标定未知）",
    xlabel="输入坐标 X（米）",
    ylabel="输入坐标 Y（米）",
    colorbar="原始点云 Z 中位高度（米）",
    legend="琥珀实线：复核边界｜点线：不确定范围｜钢边精度尚未验证",
)
STATUS_LABELS = {
    "HATCH_REVIEW_CANDIDATE": "船舱候选",
    "PARTIAL_HATCH": "部分边界待复核",
    "UNRESOLVED": "待复核",
    "UNRESOLVED_VESSEL_REVIEW": "船体归属待定",
}


def chinese_plot_style():
    """Choose an installed CJK font with coverage for every displayed label."""
    from matplotlib import font_manager, ft2font

    required = {
        ord(character)
        for character in (
            "".join(FIGURE_TEXT.values())
            + "".join(STATUS_LABELS.values())
            + "开口组舱数未定状态待核对角点长宽米"
        )
        if ord(character) > 127
    }
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        # Some variable Noto SC files expose only their Thin default to
        # Matplotlib. Prefer installed static CJK families when available.
        "Noto Sans SC",
    ):
        if family not in available:
            continue
        path = font_manager.findfont(
            font_manager.FontProperties(family=family), fallback_to_default=False
        )
        if required.issubset(ft2font.FT2Font(path).get_charmap()):
            return {"font.family": family, "axes.unicode_minus": False}
    raise ValueError("CHINESE_PLOT_FONT_NOT_FOUND: install fonts-noto-cjk")


def annotate_measurements(ax, hatches):
    """Label every corner in the original input frame and every review object.

    Corner text extends toward its own hatch horizontally. Nearby labels are
    staggered vertically in display space and remain linked to their corner.
    This changes text placement only; all geometry stays in input coordinates.
    """
    centers = []
    for hatch in hatches:
        polygon = np.asarray(hatch["boundary_xy"], dtype=float)
        dimensions = boundary_dimensions(polygon)
        status = (
            "开口组（舱数未定）"
            if hatch.get("candidate_type") == "UNRESOLVED_OPENING_GROUP"
            else STATUS_LABELS.get(hatch["status"], "状态待核对")
        )
        centers.append(
            ax.text(
                *polygon.mean(0),
                hatch["hatch_id"]
                + " | "
                + status
                + "\n长={length_m:.3f}米｜宽={width_m:.3f}米".format(**dimensions),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="#101528", alpha=0.8, edgecolor="#ffbb33"),
                zorder=5,
            )
        )
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    occupied = [
        text.get_window_extent(renderer).expanded(1.04, 1.1) for text in centers
    ]
    annotations = []
    # IDs and corner order are stable; input ordering must not affect staggering.
    for hatch in sorted(hatches, key=lambda row: row["hatch_id"]):
        polygon = np.asarray(hatch["boundary_xy"], dtype=float)
        center = ax.transData.transform(polygon.mean(0))
        for index, point in enumerate(polygon):
            position = ax.transData.transform(point)
            left = position[0] <= center[0]
            below = position[1] <= center[1]
            sign_y = -1 if below else 1
            x_offset = 4 if left else -4
            label = ax.annotate(
                "{} 角点P{}\nX={:.3f}米\nY={:.3f}米".format(
                    hatch["hatch_id"], index + 1, point[0], point[1]
                ),
                xy=point,
                xytext=(x_offset, sign_y * 9),
                textcoords="offset points",
                ha="left" if left else "right",
                va="top" if below else "bottom",
                fontsize=7,
                color="white",
                bbox=dict(facecolor="#101528", alpha=0.8, edgecolor="none", pad=2),
                arrowprops=dict(arrowstyle="-", color="#ffbb33", lw=0.6),
                annotation_clip=False,
                zorder=6,
            )
            # Text alone (rather than the connecting arrow) participates in
            # collision detection. Include the final padded box in tight bounds.
            for stagger in range(25):
                label.set_position((x_offset, sign_y * (9 + stagger * 12)))
                label.update_positions(renderer)
                label.update_bbox_position_size(renderer)
                bounds = label.get_bbox_patch().get_window_extent(renderer)
                if not any(bounds.overlaps(other) for other in occupied):
                    break
            occupied.append(bounds.expanded(1.02, 1.04))
            annotations.append(label)
        ax.scatter(*polygon.T, s=10, color="#ffbb33", zorder=7)
    return centers, annotations
