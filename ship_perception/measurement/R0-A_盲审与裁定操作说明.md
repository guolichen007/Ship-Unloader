# R0-A Development 真值盲审与裁定

本阶段只建立测量真值。六份正常 Development 扫描的九舱数量来自先前 Review A 计数，**尚非双人裁定真值**。旧 CNN/最小几何候选数也不能证明实际召回。未经独立 A/B 审核、人工裁定与文件哈希绑定，Proposal/Oracle 指标必须为 `null`。

## 资产与边界

盲审包生成命令（仓库根目录）：

```powershell
python -m ship_perception.measurement.blind_review_pack --data-root 'Ship-Unloader-Data/legacy/hold_detector' --output 'Ship-Unloader-Work/review/r0_a' --review-root 'Ship-Unloader-Data/annotations/v15r'
```

程序严格读取原始 PCD XYZ 和数据 manifest 的 PCD SHA，生成每帧九张原始图：顶视中位高度、点密度、三维概览、XZ/YZ 正交视图、X 三切片、Y 一切片。`blind_index.json` 记录渲染参数及逐图 SHA。无 intensity 字段时以点密度代替。程序不读旧 weak box、CNN/Geometry candidate、产品预测边或 Oracle 输出；已存在的审阅模板不会被覆盖。图片和填好的坐标文件留在 Git 忽略的 `Ship-Unloader-Work/`、`Ship-Unloader-Data/`。

Reviewer A/B 分别填写 `Ship-Unloader-Data/annotations/v15r/review_A|B/<scan_id>.json`，参照同目录模板及 `r0a_review.schema.json`。将 `review_status` 改为 `COMPLETE`，真实记录 `reviewer_id`、是否看过仓库、Review A、算法输出，以及本次是否只依据原始渲染。Review A 旧结论已公开，不能宣称严格 sealed。Review B 若看过仓库、Review A 或算法输出，则须如实记录；工具会阻止以“独立 B”身份评分。

每份 review 需要目标船 `vessel_roi_raw_xy`、视野内舱数、逐舱粗 `rough_polygon_raw_xy`、`observation_status` 和理由。粗 polygon 仅用于 Proposal Recall，不是钢边。每条 **实际可见** 的 opening-side 物理边另填 `observed_support` 原始 XY 折线、`visibility`、`opening_side_status` 和理由。只认顶板向舱内下降的开口侧结构转折/真实内侧结构面；舱底坡脚、Cargo 边、围板外侧、空白像素均不是实测钢边。看不清时填 `AMBIGUOUS_TRANSITION`/`NOT_OBSERVABLE`、`PARTIAL`/`UNCERTAIN`，支持线用 `null`，不可补框或按候选修正。不要因为要凑四边而推断不可见边。

## 差异审计与人工签署

A/B 独立完成后运行：

```powershell
python -m ship_perception.measurement.adjudicate_scene_manifest --review-root 'Ship-Unloader-Data/annotations/v15r' --blind-index 'Ship-Unloader-Work/review/r0_a/blind_index.json' --output 'Ship-Unloader-Work/review/r0_a/disagreement.json'
```

脚本校验扫描、图片及 A/B 文件绑定，报告舱数差异、粗 polygon 一对一**建议**及各实测边的双向采样差异。它同时生成 `human_decision_DRAFT.json` 与 `scene_manifest_DRAFT.json`，不会自动选择物理含义、改写 A/B 或把建议视为裁决。人工检查每个舱的关联、目标船 ROI、局部 ID、opening-side 支持，再把两份 DRAFT 复制到忽略目录中的正式文件，填上裁定人和逐项 `human_confirmed`。A1 可以将尚未审理的 `edge_links` 留空，仅发布粗 ROI；A2 再增加逐边关联。

正式 `scene_manifest.2` 需有六场景、九个逐舱非空粗 ROI、目标船 ROI、A/B 文件 SHA、`annotation_status=ADJUDICATED` 以及每帧 `scene_review_status=ADJUDICATED`。完整格式由 `v15r_scene_manifest.schema.json` 和原有 validator 校验。不要把 `whole_vessel_hatch_count` 当作视野内计数；艏艉/舱号不确定时不补造。

完成后执行审计并生成与最终 manifest 相邻的 `adjudication_audit.json`：

```powershell
python -m ship_perception.measurement.adjudicate_scene_manifest --review-root 'Ship-Unloader-Data/annotations/v15r' --blind-index 'Ship-Unloader-Work/review/r0_a/blind_index.json' --output 'Ship-Unloader-Work/review/r0_a/disagreement.json' --final-manifest 'Ship-Unloader-Data/annotations/v15r/scene_manifest.json' --decision-file 'Ship-Unloader-Data/annotations/v15r/human_decision.json'
```

Proposal 只在最终粗 ROI 与审计均有效时运行；`run_proposals --scene-manifest` 会再次校验 manifest、decision、A/B review 和盲审包 SHA。Oracle 只对两名 reviewer 均确认 opening-side、均有实测支持、双向边差异 P95 ≤ **0.05 m** 且人工确认的边计算物理误差；其余边 `EDGE_GT_UNRESOLVED`，没有合格边的舱 `PHYSICAL_EDGE_UNSCORED`。这个 5 cm 是 15 cm 门禁的真值能力检查，不能通过事后移动标线达成。粗 ROI 永不替代钢边。真实真值缺失时不运行评分，也不填零分。

## 停止点

有正式签署后才运行 `run_proposals` 与 `run_oracle`，汇总 CNN 前/后 NMS、Geometry、Union 的逐舱召回、合并/重复和 Oracle 的 `LOCAL_DECK/BOUNDARY/TOPOLOGY/INSUFFICIENT_OBSERVATION`、逐边匹配及误差。随后判定主阻塞，不进入 R1，也不改 Proposal、局部几何或 V1.4。提交 Git 的只能是脚本、schema、测试和去坐标统计；原始图、review、裁定 manifest、日志和构建产物均不得提交。
