"""Display final review coordinates and dimensions without changing geometry."""

import numpy as np

from .coordinates import boundary_dimensions


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
        centers.append(
            ax.text(
                *polygon.mean(0),
                hatch["hatch_id"]
                + " | "
                + hatch["status"]
                + "\nL={length_m:.3f} m | W={width_m:.3f} m".format(**dimensions),
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
                "{} P{}\nX={:.3f} m\nY={:.3f} m".format(
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
