"""Operator-only estimated geometry; strict machine exports remain separate."""

import json
from pathlib import Path

import numpy as np

from .r3h_geometry import dimensions


def write_review_layers(payload, directory):
    records = []
    changes = []
    for cell in payload["geometry_only"].get("selected_opening_cells", []):
        if cell.get("geometry") is None:
            continue
        layer = cell.get("review_geometry", {})
        geometry = layer if layer.get("adopted_for_review") else cell["geometry"]
        records.append(
            dict(
                cell_id=cell["cell_id"],
                vessel_id=cell["vessel_id"],
                boundary_xy=geometry["boundary_xy"],
                corners_xyz=geometry["corners_xyz"],
                **dimensions(geometry["boundary_xy"]),
                state=layer.get("state", "RETAINED_BASELINE"),
                grade=layer.get("grade", "UNRESOLVED"),
                adopted_for_review=layer.get("adopted_for_review", False),
                baseline_boundary_xy=cell["geometry"]["boundary_xy"],
                strict_rectangle_state=cell["strict_rectangle_state"],
                control_eligible=False,
                steel_certified=False,
                human_review="NOT_VERIFIED",
                review_gates=cell.get("review_gates", {}),
                max_corner_move_m=layer.get("max_corner_move_m", 0.0),
                per_side_intervals=layer.get("per_side_intervals", {}),
                source_family=layer.get("source_family", "RETAINED_STRICT_BASELINE"),
            )
        )
        for side, evidence in layer.get("side_evidence", {}).items():
            changes.append(
                dict(
                    cell_id=cell["cell_id"],
                    side=side,
                    adopted_for_review=layer.get("adopted_for_review", False),
                    selected_mode_id=layer.get("selected_modes", {}).get(side),
                    reason=layer.get("reason"),
                    evidence=evidence,
                )
            )
    directory = Path(directory)
    for name, value in (
        ("review_hatches_compact.json", records),
        ("复核层逐边留出证据.json", changes),
    ):
        (directory / name).write_text(
            json.dumps(
                value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            + "\n",
            encoding="utf-8",
        )


def draw_review_evidence(panel, cell, frame, rotation):
    layer = cell.get("review_geometry", {})
    if not layer.get("side_evidence"):
        return False
    for entry in layer["side_evidence"].values():
        selected = entry.get("review_selected_mode_id")
        ranked = sorted(
            entry["candidates"],
            key=lambda m: (
                m["mode_id"] != selected,
                not m["qualified_for_review"],
                -m["score"],
            ),
        )
        for mode in ranked[:2]:
            p = np.asarray(mode["predicted_endpoints_uv"]) @ rotation
            panel.plot(*p.T, color="#33cfff", lw=1.4, ls="--")
            train = mode["nomination_raw_indexes"]
            heldout = [
                i
                for row in mode["holdout_evidence"]
                if row["passed"]
                for i in entry["source_supports"][row["raw_support_ref"]]
            ]
            for ids, color in ((train, "#33cfff"), (heldout, "#ff9060")):
                ids = np.asarray(ids, dtype=np.int64)
                if len(ids):
                    ids = ids[:: max(1, int(np.ceil(len(ids) / 12000)))]
                    q = frame.xyz[ids]
                    panel.scatter(
                        q[:, 0], q[:, 1], s=1, color=color, alpha=0.35, rasterized=True
                    )
    return True


def review_evidence_text(row):
    if not row.get("adopted_for_review"):
        return []
    details = ["复核估计采用独立物理断面留出验证；原严格结果另存；钢边精度未认证。"]
    for side, entry in row.get("side_evidence", {}).items():
        selected = entry.get("review_selected_mode_id")
        for mode in entry["candidates"]:
            if mode["mode_id"] == selected:
                details.append(
                    "{}：训练断面{}个，留出通过{}个；留出覆盖{:.0f}%；位置观测区间[{:.2f}, {:.2f}]米；{}级估计".format(
                        side,
                        len(mode["train_section_ids"]),
                        len(mode["holdout_section_ids"]),
                        100 * mode["holdout_coverage"],
                        *mode["position_interval_m"],
                        mode["grade"],
                    )
                )
    details.append("变更角点高度全部缺测，未移用旧位置高度；禁止用于生产控制。")
    return details
