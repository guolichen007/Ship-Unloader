# 卸船机船体识别与稳定跟踪

当前功能分支实现 V1.4：真实 GICP/VGICP 配准、固定船体坐标系、参考地图生命周期与闭环 Synthetic 验收。目标环境为 Ubuntu 20.04、GCC9、CMake3.16、Eigen3.3.7、PCL1.10、C++17。

开发起点为 `06af2d66262afd828edf2eef72b6ce1bc6969797`。本分支等待统一代码审查和同一精确 SHA 的 Ubuntu20 独立验证；尚未合并或固化 V1.4 基线。原始 M0 正式通过证据仍待归档。

- [构建与运行说明](ship_perception/README.md)
- [V1.4 实现说明](docs/V1.4版本_船体稳定跟踪核心版实现说明.md)
- [Ubuntu20 统一验证交接](docs/V1.4版本_Ubuntu20验证交接说明.md)
- [协作与提交规范](docs/GitHub企业级协作与提交规范.md)
- [M0 历史验证交接](docs/M0阶段_验证交接说明.md)

新实现位于 `ship_perception/`。本地 `hold_detector/` 是只读冻结资产，不导入此仓库；测试不依赖其数据。第三方源码及原始许可证保留原文。

Synthetic 验收不能代替现场点密度、遮挡、同步和扫描失真验证，现场状态持续为 `SITE_PENDING`。本版本不实现 EKF、最终可观性/协方差标定、完整 LOST/重定位、Polygon、控制地图、AI 或 PLC 控制。
