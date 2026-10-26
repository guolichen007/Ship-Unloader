# Ship-Unloader

双激光雷达卸船机船舱识别与船体稳定跟踪系统。

当前现场入口是 **Ubuntu 22.04 / ROS2 Humble 的单帧开舱候选与人工复核工具**。读取 PCD 或 `/ship_scan/map` 后，使用同一条几何链输出边界、逐边证据、不确定性和两张复核图。结果不用于自动控制；V1.4 船体跟踪源码继续冻结保留，尚未与此入口联动。

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

脚本名、参数和日志使用英文，输出图的说明使用中文。`x/y/z` 必需，`intensity` 可选。缺失或近似恒定的强度明确回退几何，第二张图标记“强度不可用：复用几何识别结果”，不重复运行整套几何复核。

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

`05_hatch_corners.json` 单列整幅场景中所有最终复核对象的四角坐标，使用原始输入 frame 的 XY、单位米；每角含 `x_m`、`y_m`、相交边名称及不确定性。它与全局图使用同一份最终几何，不补造 Z，也不把 `UNRESOLVED` 包围组当作单个已确认船舱。

两张全船图均在角点附近标注 `P1～P4` 的输入 XY 坐标，并在每舱中央显示“长 / 宽”。JSON 的 `C0～C3` 依次对应图中 `P1～P4`，`dimensions` 同时保存长、宽及四条边的实长。长度为两条 U 向边实长的平均，宽度为两条 V 向边实长的平均；不使用旋转后的 XY 包围框尺寸。显示保留三位小数只是数值格式，不代表毫米级精度认证。

琥珀实线为最佳复核边界，点线为不确定性覆盖范围。`PARTIAL_HATCH`、`UNRESOLVED_VESSEL_REVIEW` 不能当作已确认船舱；矩形约束和缺测推断不能当作实测钢边。完整原始场景保留在背景中，避免漏检区域被绘图裁掉。

## 当前技术路线与边界

- 主链为输入检查 → 一次 B1 船体归属 → 冻结证据上下文 → 逐船 UVZ 召回 → 船外低区与包围验证 → 四边 Raw XYZ 精修 → 角点重建与防内裁。
- 全局 Height/R2F/R2G 保留为可关闭的比较证据，不能限制逐船候选是否存在。`--no-legacy-evidence` 用于独立链验证；禁用后不代表所有真实船已完成召回验收。
- 不通过固定水面 Z、船型尺寸或指定舱数补框。外部连通检查只经过实测低格，空白不充当通道。三边或裁剪不明时保留局部未决证据。
- 强度仍为 `AUX_WEIGHT=0.05` 的辅助；没有双雷达源标识时校准状态为 `UNKNOWN`，不靠强度判钢。
- 没有独立钢内沿 Golden，不能声明厘米级精度、商业泛化或现场自动作业通过。未开舱船需人工确认范围，当前没有自动开闭舱分类。

当前 Normal6 召回门尚未通过。开发回归与正式验收、Windows 与 Ubuntu、Synthetic 与 SITE 的结论分别记录，见 [测试与验收](docs/04_测试与验收说明.md)。旧 C++ V1.5 Quick 的已知失败继续保留，Python 工具运行成功不覆盖该验收事实。

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
