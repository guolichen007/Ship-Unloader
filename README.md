# 卸船机船体识别与稳定跟踪

V1.4 已完成 Ubuntu20 独立验证，当前功能分支正在实现 V1.5 船舱结构识别。目标环境为 Ubuntu 20.04、GCC9、CMake3.16、Eigen3.3.7、PCL1.10、C++17。

**当前产品优先级：V1.5-R Recognition First。当前阶段：R0 Data Contract / Baseline。** 生产架构为 `Proposal → 1/2/3 舱假设 → Local Deck → opening-side 3D 边界 → Hatch model`；Global Self-Bootstrap 仅作研究支线。场景契约见 `docs/V1.5-R数据契约_V2场景语义.md`，工程规范见 `docs/工程执行规范.md`。

原始数据通过 `SHIP_UNLOADER_DATA_ROOT` 提供，构建/验证输出通过 `SHIP_UNLOADER_WORK_ROOT` 提供，均位于仓库外，不写本机绝对路径。

V1.5 开发起点为 `d586b93fbf676e585e77286b7b10d7076a64d3b7`；冻结的 V1.4 技术 SHA 为 `2fdeb5055ae6ad6d84513453b5c6eff2d742a5d1`。V1.4 的正式通过报告已归档；V1.5 尚未达到整版放行条件。原始 M0 正式通过证据不补造。

- [构建与运行说明](ship_perception/README.md)
- [V1.4 实现说明](docs/V1.4版本_船体稳定跟踪核心版实现说明.md)
- [V1.4 Ubuntu20 正式独立验证报告](docs/validation/V1.4阶段_Ubuntu20正式独立验证报告.md)
- [V1.5 数据集与标注契约](docs/V1.5版本_数据集与标注说明.md)
- [V1.5 实现状态与技术说明](docs/V1.5版本_实现说明.md)
- [V1.5 计划符合性审查与未完成项](docs/V1.5版本_计划符合性审查.md)
- [V1.5 阶段性进展与未通过门禁（2026-09-21）](docs/validation/V1.5阶段性进展_2026-09-21.md)
- [V1.5 Ubuntu20 验证交接准备](docs/V1.5版本_Ubuntu20统一验证交接.md)
- [Ubuntu20 统一验证交接](docs/V1.4版本_Ubuntu20验证交接说明.md)
- [协作与提交规范](docs/GitHub企业级协作与提交规范.md)
- [M0 历史验证交接](docs/M0阶段_验证交接说明.md)

新实现位于 `ship_perception/`。本地 `hold_detector/` 是只读冻结资产，不导入此仓库。V1.5 真实数据验证通过 `--data-root` 使用外部副本；Synthetic 与核心单元测试不依赖这些资产。第三方源码及原始许可证保留原文。

Synthetic 验收不能代替现场点密度、遮挡、同步和扫描失真验证，现场状态持续为 `SITE_PENDING`。V1.5 输出的多边形只是未审查结构候选，不是控制地图或 Canonical Model；不实现 EKF、最终可观性/协方差标定、完整 LOST/重定位、AI 或 PLC 控制。
