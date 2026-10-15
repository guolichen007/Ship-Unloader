# ROS 现场一键识别运行说明

> 状态：REVIEWED
> 适用版本：当前现场复核基线；V1.4 历史事实单列
> 最后更新：2026-10-15
> Owner：Ship-Unloader Engineering

本工具适用于 Ubuntu 20.04、Python 3.8、ROS 1 Noetic，接收拼接完成的 `sensor_msgs/PointCloud2`。启动雷达/建图发布器后，在仓库根目录执行。脚本名、参数、终端提示和输出文件名均为英文；展示图保持统一中文标题与坐标轴。

## 首次准备环境

已有兼容依赖可跳过。首次使用先安装 `python3-venv`：`sudo apt-get install python3-venv`。

```bash
source /opt/ros/noetic/setup.bash
# Source your mapping workspace's devel/setup.bash if needed.
python3 -m venv --system-site-packages .local/ros-venv
source .local/ros-venv/bin/activate
python3 -m pip install -r requirements-ros.txt
# Install a CJK font once if the machine has none:
sudo apt-get install fonts-noto-cjk
```

## 每次运行

```bash
source /opt/ros/noetic/setup.bash
source .local/ros-venv/bin/activate
./scripts/run_ship_hatch_once.sh --save_snapshot
```

脚本可从任意工作目录调用；默认输出根目录为仓库内 `output/`。Python 入口的相对输出路径以当前工作目录为准：

```bash
python3 -m ship_perception.ros.ship_scan_once \
  --topic /ship_scan/map --output_dir ./output --timeout_sec 30 --save_snapshot
```

也支持 `python3 ship_perception/ros/ship_scan_once.py`。参数 `--disable_intensity_fusion` 禁用强度辅助，两张图仍输出，第二张明确标注 `DISABLED`；省略 `--save_snapshot` 时不写快照。可用 `PYTHON_BIN` 为 shell 入口指定解释器。远程 ROS master 按 ROS 标准设置 `ROS_MASTER_URI`、`ROS_IP`/`ROS_HOSTNAME`。

## 输出与结果解释

每次成功运行仅发布一个完整目录：

```text
output/YYYYMMDD_HHMMSS_microseconds/
  01_geometry_only.png
  02_intensity_fusion.png
  03_results.json
  04_input_summary.txt
  input_snapshot.npz       # only with --save_snapshot
```

无旧版图、调试图、颜色分区图、单独强度图或中间报告。中途失败不会留下看似成功的正式结果目录。JSON 中保留同一输入哈希、时间戳、原始/有效点数、字段、配置哈希、每船/每舱边界与状态、逐边和角点强度证据及原有诊断。NPZ 用 `numpy.load(path, allow_pickle=False)` 读取 `xyz`、`intensity` 和字符串 `metadata`，无强度时 intensity 为空数组。快照中的 XYZ 已过滤非有限坐标，intensity 始终与其一一对应。

读取器支持 PointField 数值类型、字段偏移、大小端、point_step、组织点云的 row_step 填充与额外字段。XYZ 必须存在；强度字段可缺省，非有限强度不丢弃有效 XYZ。摘要记录额外字段名（如 sensor_id）。缺失/全无效强度返回 `UNAVAILABLE`；逐船 P5/P95 无对比或无船体时返回 `UNAVAILABLE_NO_VESSEL_CONTRAST`，几何照常运行。超时、错误类型、损坏布局或无有效 XYZ 返回非零退出码与英文错误信息。

几何 providers 完全复用当前 PCD 静态主链。基于同一帧共享 XYZ providers，再分别运行几何与强度辅助边界复核。强度按 B1 船体归属点计算 P5/P50/P95 和 MAD，以归一化强度的中位网格计算 U/V 梯度、两侧差异、沿边符号一致性与占据支持率。当前辅助权重为 0.05，仅影响 V0/V1 几何候选排序中的连续评分尾项；可行性、边角色、独立角点数量、货物多模式拒绝和验收门保持原样。U 端边与角点增加辅助证据供复核，不借强度单独生成舱或认证钢边。相同的两张几何框可能是合理结果，不承诺强度一定改善。

JSON `boundary_xy` 保留原 B5 几何，`display_boundary_xy` 对应高度复核后的图示框；两者均在输入消息的 XY 米制坐标系中，逐船 UV 轴记录在诊断中。无可信目标船 seed 时保持 `TARGET_VESSEL_UNRESOLVED`；不自动选择目标船、不输出控制模型。`UNVERIFIED_NO_INDEPENDENT_INNER_EDGE_GOLDEN` 保留，不表示真实钢内沿精度已验收。

字体可通过 `SHIP_SHOWCASE_CJK_FONT` 指定本机字体路径。Windows 上可以运行纯 Python 解码/合成测试和离线算法，但现场订阅需要上述 ROS 1 环境；ROS 2 不属于此入口支持范围。

## 阶段进度与耗时

Shell 先打印英文 ENV 依赖预检查及其耗时。Python 从 CAPTURE_POINTCLOUD 开始依次报告配置、Height、结构 providers、融合/精修、Scene/Vessel、opening-side、几何复核、强度复核或 fallback、两图、JSON、可选快照和发布。

```text
[01] CAPTURE_POINTCLOUD START total_elapsed=0.00s
[01] CAPTURE_POINTCLOUD DONE stage_elapsed=0.42s total_elapsed=0.42s
[04] STRUCTURAL_PROVIDERS RUNNING stage_elapsed=15.00s total_elapsed=16.08s
TOTAL status=COMPLETED elapsed=93.42s
```

以上时间只是格式示例。编号按实际执行递增，不是百分比；每个长阶段每 15 秒打印 RUNNING，异常为 FAILED。阶段/累计时间使用单调时钟，TOTAL 包含 Python 消息接收与计算，shell 预检查耗时另列。库函数默认静默。观察日志不参与算法，不增加结果文件。

`--timeout_sec` 约束等待消息，不约束整套算法计算。点数、范围、候选和机器决定计算耗时，不承诺实时控制频率。依赖问题核对解释器、ROS source 和 requirements；超时先用 `rostopic type /ship_scan/map` 与 `rostopic hz /ship_scan/map` 查发布器、消息类型及网络。UNRESOLVED 或零候选不等于崩溃，执行完成也不等于钢内沿通过。

本轮评估已开舱场景，未开舱船由现场确认后登记范围；程序尚未自动排除闭舱。人工说明不能进入运行时评分或改写原检测结果。文件型 PCD 的支持范围和转换追溯见[数据规范](03_数据与证据规范.md)。
