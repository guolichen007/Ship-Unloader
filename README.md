# Ship-Unloader

双激光雷达卸船机船舱识别与船体稳定跟踪系统。

本分支是 **V2.6 S4 因果修复实验 checkpoint**，目标环境为 Ubuntu 22.04 / ROS2 Humble。逐段平面、边线与连续结构轴的观测稳定性检查通过，7-16 仍未形成完整 OpeningRegion 身份，真实召回未通过。本轮允许修改的 S4 文件仍在旧 56 文件冻结清单内，默认 V2 入口当前会报 `LEGACY_EXPERT_BEHAVIOR_CHANGED`，两项入口集成测试错误保留，未绕过校验。以下入口说明属于原有接口，不表示此 checkpoint 可替代现场基线。main 与 V1.4 冻结源保持不变，不用于自动控制。

## 两个现场入口

首次安装依赖见 [ROS2 现场说明](docs/02_ROS2现场一键识别运行说明.md)。在仓库根目录运行：

```bash
source /opt/ros/humble/setup.bash
source .local/ros2-venv/bin/activate

# Offline PCD: no ROS runtime is required by this script.
./scripts/run_ship_hatch_pcd.sh --pcd /data/map.pcd --save-snapshot

# Capture exactly one ROS2 map message, then run the same algorithm.
./scripts/run_ship_hatch_once.sh --save-snapshot
```

脚本名、参数和日志使用英文，输出图的说明使用中文。`x/y/z` 必需，支持 ASCII、binary 与无损 binary_compressed PCD。V2 的 intensity 始终只记录状态，不建立强度 scorer，不执行第二套融合几何；第二张图标记“强度判定已关闭：复用纯几何复核结果”。

## 靠岸作业船

多船场景需要同一输入坐标系中的现场码头参考：修改 `config/site/berth_reference.yaml` 的码头直线、朝水方向与作业范围后启用，或传入经测量的外部配置。配置不得包含船尺寸、舱数或舱角坐标。离线坐标系可用 `--frame-id map` 显式指定。

```bash
./scripts/run_ship_hatch_pcd.sh --pcd /data/map.pcd --frame-id map \
  --berth-reference /data/site/berth_reference.yaml
./scripts/run_ship_hatch_once.sh --berth-reference /data/site/berth_reference.yaml
```

仓库模板默认关闭。自动码头检测需要可信的 `observer_xy`，不会把 map 原点当作雷达位置。无可靠参考且有多艘船时返回 `MULTI_VESSEL_BERTH_AMBIGUOUS`，不猜选、不合船；仍输出整幅场景和诊断。只有一个已分辨船体时允许未认证的单船复核，不据此声称靠岸选船已通过现场验收。

地图采集默认 `--topic /ship_scan/map --qos auto --timeout-sec 30`。AUTO 读取实际发布者 QoS；TransientLocal 地图支持晚加入订阅。Volatile 地图需要发布者再次发布，日志会说明等待原因。多个发布者、错误消息类型或超时生成独立失败诊断。

```bash
./scripts/run_ship_hatch_once.sh --topic /ship_scan/map --qos auto \
  --timeout-sec 60 --output-dir ./output --save-snapshot
```

## 输出与进度

```text
output/YYYYMMDD_HHMMSS_microseconds/
├── 01_geometry_only.png
├── 02_intensity_fusion.png
├── 03_results.json
├── 04_input_summary.txt
├── 05_hatch_corners.json    # four XY corners for every final review candidate
└── input_snapshot.npz       # only with --save-snapshot
```

终端逐阶段显示 `START / RUNNING / DONE / FAILED`、阶段耗时与总耗时，长阶段每 15 秒报告仍在运行。正式目录只在五份产物全部写完后发布；失败另存 `failure_*/failure.json`。

`05_hatch_corners.json` 单列整幅场景中所有最终复核对象的四角坐标，使用 `ship_perception.hatch_corners.2`，坐标为原始输入 frame 的 XYZ、单位米。XY 来自最终四线交点；Z 按用户选择，独立测量局部围板/甲板上沿，含 `z_status`、`z_uncertainty_m` 与 Raw 来源。无唯一局部上沿或相邻边支撑不足时，`z_m=null`、`z_status=UNOBSERVED`，不补值。它与两张全局图共用同一份坐标，也不把 `UNRESOLVED` 包围组当作单个已确认船舱。

两张全船图均在角点附近标注 `P1～P4` 的输入 XYZ 坐标（Z 缺测明确写“未观测”），并在每舱中央显示“长 / 宽”。JSON 的 `C0～C3` 依次对应图中 `P1～P4`，`dimensions` 同时保存长、宽及四条边的实长。长度为两条 U 向边实长的平均，宽度为两条 V 向边实长的平均；不使用旋转后的 XY 包围框尺寸。显示保留三位小数只是数值格式，不代表毫米级精度认证。

琥珀实线为最佳复核边界，点线为不确定性覆盖范围。`PARTIAL_HATCH`、`UNRESOLVED_VESSEL_REVIEW` 不能当作已确认船舱；矩形约束和缺测推断不能当作实测钢边。完整原始场景保留在背景中，避免漏检区域被绘图裁掉。

## 当前技术路线与边界

`geometry_only.opening_regions` 单列当前区域身份、保守边界、搜索范围和逐边测量状态，与最终 `hatches` 分开。缺少三条实测边、精修失败或预算中断不会清空区域注册表；扩大搜索范围本身也不能证明片段已经合成一个完整舱口。结构线族只提供同船搜索证据，不复制最终边 rho，不把缺测段当作负证据。

- V2 主链为一次 B1 → 靠岸船选择 → 原始索引工作区 → Legacy / Height Basin / Structural Line / Local Enclosure / Loaded Hold / Large Sand Hold 候选联合与 post-B2 局部证据 → 名义候选编译 → 统一验证 → 廉价 Raw 探测和逐区域调查 → 一个共享四边精修器。Layout Recovery / Multi Hold 仅搜索和排序，不创造边。
- Legacy 冻结 B2 候选及原始证据，旧 R2F/R2G/B4/B5 只保留在 `--pipeline v1-diagnostic` 历史诊断路径，不在 V2 中重复执行整套最终精修。历史旧路径的召回效果尚未全部迁入 V2，见验收说明。
- `UNKNOWN_CONTEXT` 不作为强船外种子；缺少高度、一边缺测、装载高度不明和布局不一致不作为硬拒绝。只对明确物理矛盾拒绝；证据不足的候选保存为未决记录。
- 最终 Raw 不降采样，保持原有 0.05 m 分辨率。共享排序空间索引、剖面、边模式、逐相位 Raw 测量与按需法向；冻结 Legacy provider 自身的法向资格计算仍保留。
- 精修阶段设 20 秒协作式预算，核心预留绘图时间；超时保留已完成结果和全部原始提案。预算超限逐项记录，不代表识别成功或所有场景已经达到耗时目标。
- 没有独立钢内沿 Golden，不能声明厘米级精度、商业泛化或现场自动作业通过。未开舱船需人工确认范围，当前没有自动开闭舱分类。

当前 Normal6 与 V2 单调回归门均未通过。程序退出成功、复核对象数量和较快耗时不能代替完整开口覆盖与水面误检复核。开发回归与正式验收、Windows 与 Ubuntu、Synthetic 与 SITE 分别记录，见 [测试与验收](docs/04_测试与验收说明.md)。旧 C++ V1.5 Quick 的既有失败继续保留。

V2.3 的离线拓扑审计入口如下；它从当前 Raw 重算冻结 `prepare_geometry / review_points`，输出区域及逐边证据，保留空角点的未决区域。虚线图为保守候选区域，四角 Z 缺测保持 `null`。整套历史重放仍有昂贵组合搜索，且 7-16 尚未恢复完整舱区，因此不接入默认主链、不作为提速方案，召回门通过前暂停新的精修性能优化。

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m ship_perception.tools.recompute_legacy_topology \
  --pcd /data/input.pcd --output-dir /data/review/topology
```

## ROS2 安装入口

混合 C++ / Python 包采用 `ament_cmake` 与 `ament_cmake_python`，保留 standalone C++ 构建。完整依赖和构建命令见 [现场说明](docs/02_ROS2现场一键识别运行说明.md)。构建后支持：

```bash
source install/setup.bash
ros2 run ship_perception ship_hatch_once --save-snapshot
ros2 run ship_perception ship_hatch_pcd --pcd /data/map.pcd
```

仓库脚本可直接使用源码，不要求先编译 C++。`ros2 run` 和 launch 入口需要先构建安装。

## 新工程师阅读顺序

1. [架构与技术路线](docs/00_系统架构与技术路线.md)：职责、主链、坐标和证据身份。
2. [ROS2 现场运行](docs/02_ROS2现场一键识别运行说明.md)：依赖、两个脚本、QoS、故障处理。
3. [算法实现](docs/01_船舱识别算法实现说明.md)：逐船召回、船外拒绝和四边精修。
4. [数据与证据](docs/03_数据与证据规范.md)：原始身份、GT 隔离、admission 和 Golden。
5. [测试与验收](docs/04_测试与验收说明.md)：开发回归、反证测试与放行边界。
6. [开发与 Git 协作](docs/05_开发与Git协作规范.md)：分支、提交及代理执行规则。
7. [V1.4 跟踪实现](docs/V1.4_船体稳定跟踪实现说明.md)和 [Ubuntu20 历史验证](docs/V1.4_Ubuntu20验证记录.md)。

```text
scripts/                        # English field launchers
ship_perception/io/             # ROS-free PCD and PointCloud2 decoding
ship_perception/perception/     # transport-neutral review pipeline
ship_perception/ros2/           # graph/QoS discovery and one-frame capture
ship_perception/r1_static/      # reused measured geometry primitives and legacy comparison
ship_perception/legacy/ros1/    # historical comparison only
ship_perception/evaluation/     # independent annotation evaluation; never imported by runtime
ship_perception/src/            # C++ recognition and frozen V1.4 core
Ship-Unloader-Data/              # private raw data; never pushed
Ship-Unloader-Work/              # builds, logs and validation; never pushed
```

正式历史跟踪 Tag `baseline-v1.4-ubuntu20-20260919` 保留。`archive/*` 只用于追溯；开发、合入和历史保全遵守[开发与 Git 协作规范](docs/05_开发与Git协作规范.md)。
