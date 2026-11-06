"""Chinese operator artifacts; read-only views of persisted geometry and Raw IDs."""

import copy
import csv
import html
import json
from pathlib import Path
import numpy as np
from .r3h_geometry import dimensions
from .review_labels import (
    annotate_measurements,
    chinese_plot_style,
    review_objects,
    display_number,
    review_status,
    corner_texts,
)

DECISIONS = {
    "OBSERVED_INNER_RIM": "观测到内沿，仍待钢边认证",
    "PARTIAL_INNER_RIM_SUPPORT": "有部分内沿支持，未采用",
    "GEOMETRIC_TRANSITION_ONLY": "只有过渡几何，未确认钢边",
    "FAST_PATH_RETAINED": "未发现跨舱宽度异常，保留快速路径",
    "OWNED_INTERIOR_ANCHOR_REQUIRES_MATERIAL_CONFIRMATION": "舱内锚点物理归属未确认",
    "INTERIOR_SEARCH_COMPLETED": "已搜索舱内横向原始证据",
    "COMMITTED_RAW_ERROR_RECOVERY": "本舱原始证据及成员保护验证通过，采用新边",
    "CANDIDATE_NOT_ADOPTED": "疑似船舷、候选修正未采用",
}


def side_changes(payload):
    axes = {
        v["vessel_id"]: np.asarray(v["local_axes"])
        for v in payload["scene_context"]["vessels"]
    }
    output = []
    for cell in payload["geometry_only"].get("selected_opening_cells", []):
        if cell.get("geometry") is None:
            continue
        recovery = cell.get("rim_error_recovery", {})
        after = np.asarray(cell["geometry"]["boundary_xy"]) @ axes[cell["vessel_id"]].T
        shadow = recovery.get("shadow_ab", {})
        before = np.asarray(shadow.get("old_boundary_uv", after))
        for side, indexes in (
            ("V0", [0, 1]),
            ("U1", [1, 2]),
            ("V1", [3, 2]),
            ("U0", [0, 3]),
        ):
            evidence = recovery.get("candidates", {}).get(side, {})
            candidates = evidence.get("candidates", [])
            output.append(
                dict(
                    cell_id=cell["cell_id"],
                    vessel_id=cell["vessel_id"],
                    side=side,
                    old_endpoints_uv=before[indexes].tolist(),
                    new_endpoints_uv=after[indexes].tolist(),
                    max_corner_move_m=float(
                        np.max(np.linalg.norm(after - before, axis=1))
                    ),
                    actual_side_move_m=float(
                        np.max(np.linalg.norm(after[indexes] - before[indexes], axis=1))
                    ),
                    adopted_mode_id=evidence.get("adopted_mode_id"),
                    decision=(
                        DECISIONS.get(shadow.get("status"), "保留原版正式复核框")
                        if candidates
                        else "原边保留"
                    ),
                    candidates=candidates,
                    section_profiles=evidence.get("sections", []),
                    original_local_raw=cell.get("local_rim_evidence", {}).get(side, {}),
                    shadow_ab=shadow,
                    steel_certified=False,
                    human_review="PENDING",
                )
            )
    return output


def render_comparison(
    payload, frame, output, grid, *, scene_label="输入场景", previous=None
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    result = payload["geometry_only"]
    current = review_objects(result)
    axes = {
        v["vessel_id"]: np.asarray(v["local_axes"])
        for v in payload["scene_context"]["vessels"]
    }
    old = (
        review_objects(previous["geometry_only"])
        if previous is not None
        else copy.deepcopy(current)
    )
    if previous is None:
        for obj in old:
            before = (
                obj.get("rim_error_recovery", {})
                .get("shadow_ab", {})
                .get("old_boundary_uv")
            )
            if before is not None:
                obj["boundary_xy"] = (
                    np.asarray(before) @ axes[obj["vessel_hypothesis_id"]]
                ).tolist()
    extent = (
        grid.x0,
        grid.x0 + grid.count.shape[1] * grid.cell_m,
        grid.y0,
        grid.y0 + grid.count.shape[0] * grid.cell_m,
    )
    step = max(1, int(np.ceil(max(grid.count.shape) / 1000)))
    data = np.ma.masked_where(~grid.occupancy, grid.median)[::step, ::step]
    with plt.rc_context(chinese_plot_style()):
        height = max(
            3.5, min(8.0, 8.0 * grid.count.shape[0] / grid.count.shape[1] + 1.5)
        )
        fig, panels = plt.subplots(1, 4, figsize=(32, height), dpi=145)
        for panel, title in zip(
            panels,
            [
                "原始点云高度图",
                "上一版船舱复核框",
                "新增内沿候选及原始证据",
                "本轮船舱标定结果",
            ],
        ):
            panel.imshow(
                data,
                origin="lower",
                interpolation="nearest",
                extent=extent,
                cmap="viridis",
                aspect="equal",
            )
            panel.set(
                title=scene_label + "｜" + title,
                xlabel="输入坐标 X（米）",
                ylabel="输入坐标 Y（米）",
            )
        for panel, objects, color in [
            (panels[1], old, "#ffbb33"),
            (panels[2], old, "#a1a7b3"),
            (panels[3], current, "#ffbb33"),
        ]:
            for number, obj in enumerate(objects, 1):
                p = np.asarray(obj["boundary_xy"])
                panel.plot(*np.vstack((p, p[0])).T, color=color, lw=1.3)
                if panel is panels[1]:
                    d = dimensions(p.tolist())
                    panel.text(
                        *p.mean(0),
                        "第{}舱\n长{}米 × 宽{}米".format(
                            number,
                            display_number(d["length_m"]),
                            display_number(d["width_m"]),
                        ),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                        bbox=dict(facecolor="#101528", alpha=0.8, edgecolor="none"),
                    )
        for cell in result.get("selected_opening_cells", []):
            rotation = axes[cell["vessel_id"]]
            for entry in (
                cell.get("rim_error_recovery", {}).get("candidates", {}).values()
            ):
                modes = entry.get("candidates", [])
                ranked = sorted(
                    modes,
                    key=lambda m: (
                        -m["qualified_for_recovery"],
                        -m["independently_observed_face_sections"],
                        -m["profile_section_coverage"],
                        m["uncertainty_m"],
                    ),
                )
                for mode in ranked[:4]:
                    p = np.asarray(mode["predicted_endpoints_uv"]) @ rotation
                    physical = bool(mode["source_point_indexes"])
                    panels[2].plot(
                        *p.T,
                        color="#ff5c8a" if physical else "#33cfff",
                        lw=1.4,
                        ls="--",
                    )
                    ids = np.asarray(
                        (
                            mode["source_point_indexes"]
                            if physical
                            else mode["profile_source_point_indexes"]
                        ),
                        dtype=np.int64,
                    )
                    if len(ids):
                        # Display sampling only; the complete original IDs stay in JSON.
                        ids = ids[:: max(1, int(np.ceil(len(ids) / 16000)))]
                        q = frame.xyz[ids]
                        panels[2].scatter(
                            q[:, 0],
                            q[:, 1],
                            s=1,
                            color="#ff5c8a" if physical else "#33cfff",
                            alpha=0.4,
                            rasterized=True,
                        )
        panels[2].legend(
            handles=[
                Line2D([], [], color="#a1a7b3", label="旧版边界"),
                Line2D([], [], color="#ff5c8a", ls="--", label="有立面原始支持的候选"),
                Line2D(
                    [], [], color="#33cfff", ls="--", label="仅高度过渡候选，未认证"
                ),
            ],
            loc="upper center",
            fontsize=7,
        )
        fig.suptitle(
            "原始证据对照｜虚线候选不代表已认证钢边｜全部待人工复核", fontsize=13
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        annotate_measurements(panels[3], current, fontsize=7)
        # Position disclaimer inside the ORIGINAL plotting area, not the appended coordinate columns.
        fig.text(
            0.015,
            0.025,
            "角点坐标保持原始精度；图片两位小数仅为显示格式。缺测高度以 — 显示，JSON 保留 null。",
            fontsize=9,
        )
        fig.savefig(output, dpi=145, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)


def write_bundle(
    payload, frame, directory, grid, *, scene_label="输入场景", previous=None
):
    directory = Path(directory)
    changes = side_changes(payload)
    (directory / "逐边变动与原始证据.json").write_text(
        json.dumps(changes, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    render_comparison(
        payload,
        frame,
        directory / "新旧候选四栏对照.png",
        grid,
        scene_label=scene_label,
        previous=previous,
    )
    records = review_objects(payload["geometry_only"])
    cards = []
    for number, row in enumerate(records, 1):
        size = dimensions(row["boundary_xy"])
        recovery = row.get("rim_error_recovery", {})
        details = []
        for side, entry in recovery.get("candidates", {}).items():
            modes = entry.get("candidates", [])
            if not modes:
                details.append("{}：未找到可辨内沿，保留原框".format(side))
            for mode in sorted(
                modes,
                key=lambda m: (
                    -m["independently_observed_face_sections"],
                    -m["profile_section_coverage"],
                ),
            )[:4]:
                details.append(
                    "{}：{}；立面断面{}个，完整物理链断面{}个，过渡覆盖{}%；候选位置不确定性{}米".format(
                        side,
                        DECISIONS[mode["observation_state"]],
                        mode["independently_observed_face_sections"],
                        mode["independent_raw_sections"],
                        display_number(100 * mode["profile_section_coverage"]),
                        display_number(mode["uncertainty_m"]),
                    )
                )
        cards.append(
            "<article><h2>第{}舱｜长{}米 × 宽{}米</h2><p>{}</p><pre>{}</pre><p>{}</p><p>{}</p></article>".format(
                number,
                display_number(size["length_m"]),
                display_number(size["width_m"]),
                html.escape(review_status(row)),
                html.escape("\n".join(corner_texts(row))),
                html.escape(DECISIONS.get(recovery.get("status"), "保留原版复核几何")),
                "<br>".join(html.escape(s) for s in details),
            )
        )
    with (directory / "船舱长宽与四角_中文.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        fields = ["船舱编号", "长度（米）", "宽度（米）", "复核状态"] + [
            "P{}{}（米）".format(i, a) for i in range(1, 5) for a in "XYZ"
        ]
        writer = csv.writer(stream)
        writer.writerow(fields)
        for number, row in enumerate(records, 1):
            size = dimensions(row["boundary_xy"])
            heights = row.get("corner_heights", [])
            xyz = row.get("corners_xyz", [])
            values = []
            for i, (x, y) in enumerate(row["boundary_xy"]):
                z = (
                    heights[i].get("z_m")
                    if len(heights) == 4
                    else xyz[i][2] if len(xyz) == 4 else None
                )
                values.extend([x, y, z])
            writer.writerow(
                [
                    "第{}舱".format(number),
                    size["length_m"],
                    size["width_m"],
                    review_status(row),
                ]
                + values
            )
    document = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>船舱标定复核</title><style>body{font:16px/1.7 system-ui,sans-serif;margin:24px;background:#f4f6fa;color:#17263a}img{width:100%;height:auto}article{display:inline-block;vertical-align:top;background:white;padding:16px;margin:12px}h2{font-size:20px}pre{font:14px/1.8 monospace}a{color:#154b9c}</style><h1>'
        + html.escape(scene_label)
        + '｜船舱标定复核结果</h1><p>全部钢边精度未认证，人工审核待签。严格矩形可解仅表示几何约束可解。候选修正未采用时保留旧版正式复核框。坐标两位小数不代表厘米级测量精度。</p><p><a href="hatches_compact.json">每舱原始四角及长宽</a> · <a href="船舱长宽与四角_中文.csv">长宽与坐标表</a> · <a href="逐边变动与原始证据.json">逐边原始证据与变动</a> · <a href="runtime_io_metrics.json">阶段性能报告</a></p><img src="新旧候选四栏对照.png" alt="原始高度、旧框、内沿候选及最终复核四栏">'
        + "".join(cards)
        + "</html>"
    )
    (directory / "船舱复核总览.html").write_text(document, encoding="utf-8")
