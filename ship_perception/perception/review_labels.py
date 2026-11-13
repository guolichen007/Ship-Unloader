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
    legend="琥珀实线：复核边界｜点线：不确定范围｜Z：局部围板/甲板上沿实测估计，缺测不补值｜钢边精度尚未验证",
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
            + "开口组舱数未定状态待核对角点长宽米未观测第严格矩形可解未决钢边认证疑似船舷人工确认坐标显示保留原始浮点精度厘米仅格式外侧栏部分高度延期复测"
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


def ordered_reviews(hatches):
    """Display ordering only: vessel identity, longitudinal position, identity."""

    def key(row):
        polygon = np.asarray(row["boundary_xy"])
        axes = np.asarray(row.get("vessel_local_axes", np.eye(2)))
        return (
            row.get("vessel_hypothesis_id", ""),
            float((polygon @ axes.T)[:, 0].mean()),
            row.get("opening_identity", row.get("hatch_id", "")),
        )

    return sorted(hatches, key=key)


def review_objects(result):
    """Borrow immutable geometry from every strict and unresolved cell."""
    strict = {
        h.get("opening_identity", h.get("hatch_id")): h for h in result["hatches"]
    }
    objects = []
    cells = result.get("selected_opening_cells", [])
    if not cells:
        return ordered_reviews(
            result["hatches"] + list(result.get("coarse_opening_reviews", []))
        )
    for cell in cells:
        if cell.get("geometry") is None:
            continue
        obj = dict(strict.get(cell["primary_region_id"], cell["geometry"]))
        obj["strict_rectangle_state"] = cell["strict_rectangle_state"]
        obj["rim_error_recovery"] = cell.get("rim_error_recovery", {})
        review = cell.get("review_geometry", {})
        if review.get("adopted_for_review"):
            obj.update(review)
            obj["corner_heights"] = [
                dict(z_m=point[2], z_state="UNOBSERVED")
                for point in review["corners_xyz"]
            ]
        objects.append(obj)
    return ordered_reviews(objects)


def review_status(hatch):
    if hatch.get("adopted_for_review"):
        strict = (
            "严格矩形未决"
            if hatch.get("strict_rectangle_state") != "FEASIBLE"
            else "严格结果另列"
        )
        return "{}｜{}级｜{}\n待人工确认｜非认证，禁止控制".format(
            "多解待审" if hatch.get("state") == "REVIEW_AMBIGUOUS" else "复核估计",
            hatch.get("grade", "C"), strict
        )
    state = hatch.get("strict_rectangle_state")
    strict = state == "FEASIBLE" or (
        state is None and "rectangle_constraint_conflict_count" in hatch
    )
    status = "严格矩形可解" if strict else "矩形未决"
    if hatch.get("rim_error_recovery", {}).get("candidates"):
        status += "｜疑似船舷，待人工确认"
    if hatch.get("z_measurement_audit", {}).get("status") == "DEFERRED_PERFORMANCE":
        status += "\n部分高度延期，待复测"
    return status + "｜钢边未认证"


def display_number(value):
    if value is None:
        return "—"
    if not np.isfinite(value):
        raise ValueError("NONFINITE_DISPLAY_COORDINATE")
    text = "{:.2f}".format(value)
    return "0.00" if text == "-0.00" else text


def corner_texts(hatch):
    result = []
    heights = hatch.get("corner_heights", [])
    xyz = hatch.get("corners_xyz", [])
    for index, (x, y) in enumerate(hatch["boundary_xy"]):
        z = (
            heights[index].get("z_m")
            if len(heights) == 4
            else xyz[index][2] if len(xyz) == 4 else None
        )
        result.append(
            "P{}（X={}，Y={}，Z={}）".format(
                index + 1, display_number(x), display_number(y), display_number(z)
            )
        )
    return result


def pixel_segments(ax):
    """Every original plotted line, including adjacent boxes and uncertainty."""
    segments = []
    for line in ax.lines:
        xy = line.get_transform().transform(line.get_xydata())
        for first, second in zip(xy[:-1], xy[1:]):
            if np.all(np.isfinite([first, second])):
                segments.append(np.array([first, second]))
    return segments


def segment_hits_box(segment, bounds):
    # Liang-Barsky clipping, including contact with the padded label boundary.
    first, last = segment
    delta = last - first
    lo, hi = 0.0, 1.0
    for axis in (0, 1):
        low, high = (bounds.x0, bounds.x1) if axis == 0 else (bounds.y0, bounds.y1)
        if abs(delta[axis]) < np.finfo(float).eps:
            if first[axis] < low or first[axis] > high:
                return False
        else:
            a, b = sorted(
                ((low - first[axis]) / delta[axis], (high - first[axis]) / delta[axis])
            )
            lo, hi = max(lo, a), min(hi, b)
            if lo > hi:
                return False
    return True


def annotate_measurements(ax, hatches, draw_corner_points=False, *, fontsize=8):
    """Place actual pixel bboxes outside frames; overflow goes to a new column.

    Display artists alone are changed. No dots, leaders, coordinate rounding in
    data, or geometry mutation. Render at the figure's native DPI after this.
    """
    from matplotlib.transforms import Bbox, IdentityTransform
    from matplotlib.path import Path as PlotPath

    figure = ax.figure
    hatches = ordered_reviews(hatches)
    centers, labels = [], []
    for number, hatch in enumerate(hatches, 1):
        polygon = np.asarray(hatch["boundary_xy"])
        size = boundary_dimensions(polygon)
        centers.append(
            ax.text(
                *polygon.mean(0),
                "第{}舱｜长{}米 × 宽{}米\n{}".format(
                    number,
                    display_number(size["length_m"]),
                    display_number(size["width_m"]),
                    review_status(hatch).replace(
                        "｜疑似船舷，待人工确认", "\n疑似船舷｜待人工确认"
                    ),
                ),
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="normal",
                color="white",
                bbox=dict(facecolor="#101528", alpha=0.85, edgecolor="none", pad=3),
                zorder=6,
            )
        )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    obstacles = [
        t.get_window_extent(renderer).padded(5)
        for other in figure.axes
        for t in list(other.texts) + [other.title, other.xaxis.label, other.yaxis.label]
        if t.get_visible() and t.get_text()
    ]
    if ax.get_legend() is not None:
        obstacles.append(ax.get_legend().get_window_extent(renderer).padded(5))
    segments = pixel_segments(ax)
    paths = [
        PlotPath(
            np.vstack(
                (
                    ax.transData.transform(h["boundary_xy"]),
                    ax.transData.transform(h["boundary_xy"])[0],
                )
            ),
            closed=True,
        )
        for h in hatches
    ]
    boundary = ax.get_window_extent(renderer).padded(-8)
    external = []
    for number, hatch in enumerate(hatches, 1):
        polygon = ax.transData.transform(hatch["boundary_xy"])
        center = polygon.mean(0)
        placed, boxes = [], []
        for corner, text in zip(polygon, corner_texts(hatch)):
            test = ax.text(0, 0, text, fontsize=fontsize, transform=IdentityTransform())
            box = test.get_window_extent(renderer)
            width, height = box.width, box.height
            test.remove()
            radial = corner - center
            radial /= max(1.0, np.linalg.norm(radial))
            choices = []
            for distance in (12, 26, 44, 70):
                for direction in (radial, [0, 1], [0, -1], [-1, 0], [1, 0]):
                    anchor = corner + distance * np.asarray(direction)
                    x = anchor[0] - (
                        width
                        if direction[0] < -0.2
                        else width / 2 if abs(direction[0]) <= 0.2 else 0
                    )
                    y = anchor[1] - (
                        height
                        if direction[1] < -0.2
                        else height / 2 if abs(direction[1]) <= 0.2 else 0
                    )
                    candidate = Bbox.from_bounds(x, y, width, height)
                    padded = candidate.padded(6)
                    if not (
                        boundary.contains(*padded.p0) and boundary.contains(*padded.p1)
                    ):
                        continue
                    if any(padded.overlaps(b) for b in obstacles + boxes):
                        continue
                    if any(segment_hits_box(line, padded) for line in segments):
                        continue
                    if any(path.intersects_bbox(padded, filled=True) for path in paths):
                        continue
                    # Prefer available blank background over occupied point cloud.
                    if ax.images:
                        image = ax.images[0]
                        data = np.ma.asarray(image.get_array())
                        extent = image.get_extent()
                        low, high = ax.transData.inverted().transform(
                            [padded.p0, padded.p1]
                        )
                        ix = np.clip(
                            np.floor(
                                (np.array([low[0], high[0]]) - extent[0])
                                / (extent[1] - extent[0])
                                * data.shape[1]
                            ).astype(int),
                            0,
                            data.shape[1] - 1,
                        )
                        iy = np.clip(
                            np.floor(
                                (np.array([low[1], high[1]]) - extent[2])
                                / (extent[3] - extent[2])
                                * data.shape[0]
                            ).astype(int),
                            0,
                            data.shape[0] - 1,
                        )
                        block = data[min(iy) : max(iy) + 1, min(ix) : max(ix) + 1]
                        if block.size and np.mean(~np.ma.getmaskarray(block)) > 0.1:
                            continue
                    choices.append(
                        (
                            np.linalg.norm(candidate.get_points().mean(0) - corner),
                            candidate,
                        )
                    )
            if not choices:
                break
            box = min(choices, key=lambda row: row[0])[1]
            placed.append((text, box))
            boxes.append(box.padded(6))
        if len(placed) != 4:
            external.append((number, hatch))
            continue
        obstacles.extend(boxes)
        for text, box in placed:
            labels.append(
                ax.text(
                    box.x0,
                    box.y0,
                    text,
                    transform=IdentityTransform(),
                    va="bottom",
                    fontsize=fontsize,
                    fontweight="normal",
                    color="#15243c",
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2),
                    clip_on=False,
                    zorder=7,
                )
            )
    if external:
        # Expand the canvas while preserving EVERY existing axes in pixel units.
        old_w, old_h = figure.get_size_inches() * figure.dpi
        line_h = fontsize * figure.dpi / 72 * 1.65
        block_h = line_h * 6.5
        per_column = max(1, int((old_h - 40) // block_h))
        columns = int(np.ceil(len(external) / per_column))
        texts = [text for _, h in external for text in corner_texts(h)]
        sample = ax.text(
            0, 0, max(texts, key=len), fontsize=fontsize, transform=IdentityTransform()
        )
        col_w = sample.get_window_extent(renderer).width + 32
        sample.remove()
        positions = [(a, a.get_position().frozen()) for a in figure.axes]
        new_w = old_w + columns * col_w + 20
        if hasattr(figure, "set_layout_engine"):
            figure.set_layout_engine(None)
        figure.set_size_inches(new_w / figure.dpi, old_h / figure.dpi, forward=True)
        for a, pos in positions:
            a.set_position(
                [pos.x0 * old_w / new_w, pos.y0, pos.width * old_w / new_w, pos.height]
            )
        for index, (number, hatch) in enumerate(external):
            column, row = divmod(index, per_column)
            x = old_w + 16 + column * col_w
            y = old_h - 20 - row * block_h
            lines = (
                ["第{}舱｜四角原始坐标".format(number)]
                + corner_texts(hatch)
                + ["两位小数为显示格式，钢边未认证"]
            )
            for i, text in enumerate(lines):
                artist = figure.text(
                    x / new_w,
                    (y - i * line_h) / old_h,
                    text,
                    fontsize=fontsize,
                    fontweight="normal",
                    va="top",
                    color="#15243c",
                )
                if 1 <= i <= 4:
                    labels.append(artist)
    figure.canvas.draw()
    # Expose only a rendering audit on the axes, never in geometry JSON.
    renderer = figure.canvas.get_renderer()
    ax._coordinate_layout_audit = dict(
        external_hatch_count=len(external),
        label_count=len(labels),
        label_bboxes_px=[t.get_window_extent(renderer).bounds for t in labels],
        original_line_segments_px=[s.tolist() for s in pixel_segments(ax)],
        font_size=fontsize,
        display_decimals=2,
    )
    return centers, labels
