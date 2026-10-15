# Ship-Unloader

双激光雷达卸船机船舱识别与船体稳定跟踪系统。

当前现场交付是 **ROS 1 单帧开舱候选识别与人工复核工具**。直接读取 `/ship_scan/map` 的拼接点云，对同一帧生成几何主算法和强度辅助两种结果。V1.4 刚体跟踪保留在仓库中，尚未与这个现场入口集成。

## 当前现场入口

已经准备好 ROS 和 Python 依赖时，在仓库根目录运行：

```bash
source /opt/ros/noetic/setup.bash
./scripts/run_ship_hatch_once.sh --save_snapshot
```

默认消息类型 `sensor_msgs/PointCloud2`，`x/y/z` 必需，`intensity` 可选。强度缺失或无效时明确回退到几何结果。脚本名、参数和终端提示均为英文，展示图保持中文。

首次使用 Ubuntu 20.04 / Python 3.8 / ROS 1 Noetic：

```bash
git clone git@github.com:guolichen007/Ship-Unloader.git
cd Ship-Unloader
source /opt/ros/noetic/setup.bash
sudo apt-get install python3-venv fonts-noto-cjk
python3 -m venv --system-site-packages .local/ros-venv
source .local/ros-venv/bin/activate
python3 -m pip install -r requirements-ros.txt
./scripts/run_ship_hatch_once.sh --save_snapshot
```

建图发布器需要另行启动；如需工作空间环境，先 source 其 `devel/setup.bash`。后续继续激活同一 Python 环境。远程 ROS 网络、参数与错误处理见[现场运行说明](docs/02_ROS现场一键识别运行说明.md)。

## 输出与进度

```text
output/YYYYMMDD_HHMMSS_microseconds/
├── 01_geometry_only.png
├── 02_intensity_fusion.png
├── 03_results.json
├── 04_input_summary.txt
└── input_snapshot.npz       # only with --save_snapshot
```

每次只生成这两张展示图，诊断信息放入 JSON。终端按阶段显示 `START / RUNNING / DONE / FAILED`、阶段耗时和累计耗时，长阶段每 15 秒报告运行状态。只有完整结果才发布到正式目录；执行成功表示计算完成，不表示船舱已经确认。

```bash
./scripts/run_ship_hatch_once.sh \
  --topic /ship_scan/map --timeout_sec 60 --output_dir ./output --save_snapshot
```

## 当前真实能力

| 能力 | 当前状态 |
| --- | --- |
| PointCloud2、XYZ/intensity 对齐和强度缺失回退 | 已实现；真实 ROS 发布器连接仍需现场确认 |
| 几何与强度辅助双结果、两图及结果追溯 | 已实现，使用同一份输入点云 |
| Scene/Vessel 与逐船多源候选 | 已实现；多船最终开口召回仍有缺口，不自动选择作业船 |
| 强度辅助 | 逐船 robust normalization，弱辅助评分，不单独生成舱或认证钢边 |
| 开口边界 | 输出 PARTIAL/UNRESOLVED 等审查状态；候选数不等于确认舱数 |
| V1.4 船体跟踪 | 保留冻结代码及 Ubuntu20 历史合成验证；识别联动与现场为 SITE_PENDING |
| 钢内沿厘米级精度、商业泛化、自动控制 | 尚未验收；无 PLC 或自动作业输出 |

本轮面向已开舱场景。未开舱船由现场人员确认并排除本轮开舱评估，程序尚无独立自动开/闭舱分类能力；人工范围说明不进入算法评分。

正常六场景当前候选数为 `2/1/1/1/1/3`，只是开发回归覆盖。没有独立 opening inner-edge Golden，不能据此声明钢边精度通过。旧 C++ V1.5 Quick 的 real-smoke/integration 仍有失败，详见[测试说明](docs/04_测试与验收说明.md)。

## 新工程师阅读顺序

1. [系统架构与技术路线](docs/00_系统架构与技术路线.md)：职责、当前链路、坐标和证据。
2. [ROS 现场运行](docs/02_ROS现场一键识别运行说明.md)：安装、启动、参数、输出、故障处理。
3. [识别算法实现](docs/01_船舱识别算法实现说明.md)：模块及有效技术结论。
4. [数据与证据规范](docs/03_数据与证据规范.md)：追溯、GT 隔离和数据暴露。
5. [测试与验收](docs/04_测试与验收说明.md)：开发回归与正式放行。
6. [开发与 Git 协作](docs/05_开发与Git协作规范.md)：提交、分支、PR、Tag 和代理规则。
7. [V1.4 跟踪实现](docs/V1.4_船体稳定跟踪实现说明.md)及[Ubuntu20 验证记录](docs/V1.4_Ubuntu20验证记录.md)：历史技术基线。

## 代码与数据位置

```text
scripts/run_ship_hatch_once.sh   # field launcher
ship_perception/ros/            # one-frame ROS input and orchestration
ship_perception/r1_static/      # geometry, intensity evidence, review and rendering
ship_perception/src/            # C++ recognition and frozen V1.4 tracking
ship_perception/config/         # configuration and schemas
ship_perception/tests/          # unit, contract and synthetic gates
Ship-Unloader-Data/             # private raw data; never pushed
Ship-Unloader-Work/             # build, validation, logs; never pushed
output/                        # field outputs; never pushed
```

现场 Python 入口不需要编译 C++。C++ 构建与正式复验见[工程构建入口](ship_perception/README.md)。客户数据不随 clone 提供；Git 只保存源码、配置、测试、小型元数据和当前文档。

`main` 是当前工程与现场复核入口。正式历史跟踪 Tag 为 `baseline-v1.4-ubuntu20-20260919`；`archive/*` 仅用于追溯，不作为新开发起点。历史过程文档可从 Git 历史读取，HEAD 不再保存过程副本。
