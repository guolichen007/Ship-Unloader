import copy
import unittest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from ship_perception.perception.review_labels import (
    annotate_measurements,
    chinese_plot_style,
    segment_hits_box,
    corner_texts,
)


class ChineseCoordinates(unittest.TestCase):
    def test_actual_pixel_bounds_avoid_lines_neighbors_and_each_other_at_multiple_scales(
        self,
    ):
        source = []
        for i in range(3):
            x = -85 + i * 20
            source.append(
                dict(
                    hatch_id=str(i),
                    opening_identity=str(i),
                    strict_rectangle_state="FEASIBLE",
                    boundary_xy=[[x, -23], [x + 18, -23], [x + 18, -9], [x, -9]],
                    corner_heights=[
                        dict(z_m=z) for z in (-5.712345, None, -0.0001, -6.13)
                    ],
                )
            )
        before = copy.deepcopy(source)
        with plt.rc_context(chinese_plot_style()):
            for size, dpi, font in [
                ((8, 3), 100, 8),
                ((11, 5), 150, 9),
                ((5, 3), 150, 7),
            ]:
                fig, ax = plt.subplots(figsize=size, dpi=dpi)
                for h in source:
                    p = np.asarray(h["boundary_xy"])
                    ax.plot(*np.vstack((p, p[0])).T)
                ax.set(
                    xlim=(-90, -20),
                    ylim=(-30, 0),
                    title="本轮船舱标定结果",
                    xlabel="输入坐标X（米）",
                )
                fig.tight_layout()
                _, texts = annotate_measurements(ax, source, fontsize=font)
                self.assertEqual(len(texts), 12)
                renderer = fig.canvas.get_renderer()
                boxes = [t.get_window_extent(renderer).padded(2) for t in texts]
                for i, b in enumerate(boxes):
                    self.assertTrue(
                        fig.bbox.contains(*b.p0) and fig.bbox.contains(*b.p1)
                    )
                    self.assertFalse(any(b.overlaps(other) for other in boxes[i + 1 :]))
                    self.assertFalse(
                        any(
                            segment_hits_box(np.asarray(s), b)
                            for s in ax._coordinate_layout_audit[
                                "original_line_segments_px"
                            ]
                        )
                    )
                self.assertFalse(ax.collections)
                self.assertTrue(all(t.get_fontweight() == "normal" for t in texts))
                plt.close(fig)
        self.assertEqual(source, before)
        self.assertEqual(
            corner_texts(source[0])[0], "P1（X=-85.00，Y=-23.00，Z=-5.71）"
        )
        self.assertIn("Z=—", corner_texts(source[0])[1])
        self.assertIn("Z=0.00", corner_texts(source[0])[2])


if __name__ == "__main__":
    unittest.main()
