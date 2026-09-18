# M0 Ubuntu20 独立验证反馈与修复范围

来源：用户于 2026-09-18 转交的 Ubuntu 20.04 ClaudeCLI 结构化验证报告。
本文件记录反馈摘要，不代替 Ubuntu 上的原始日志，也不代表修复后的提交已通过。

```text
VALIDATION_SHA=da36c5164d4296fa1d8b559f281ab8bec46826ce
BASE_SHA=NONE
WORKTREE=CLEAN（验证开始时）
OS=Ubuntu 20.04
GCC=9.4.0
CMAKE=3.16.3（系统）/4.3.2（PATH 中 pip 版本）
PCL=1.10.0
EIGEN=3.3.7（dpkg）/3.4.0（pkg-config）
FIRST_FAIL=configure
FAIL_TYPE=CODE_FAIL（ClaudeCLI 独立复核；原脚本误归 ENV_FAIL）
PRODUCT_CODE_MODIFIED_BY_CLAUDE=NO
CLAUDE_PATCH_SHA=NONE
M0_CODE=FAIL
M0_SYNTHETIC=PENDING
M0_LINUX20_BUILD=FAIL
M0_SITE=PENDING
BASELINE_M0_SHA=PENDING
```

正式 PCL-ON 通道 configure 失败，G1–G7 未运行。PCL-OFF 补充诊断 clean build
及 9/9 CTest 通过，仅证明该诊断通道的测试通过，不能代替正式 PCL-ON 验证。

根因是硬编码无版本 pkg-config 模块 `pcl_common`/`pcl_io`。验证主机安装的是
`pcl_common-1.10.pc`/`pcl_io-1.10.pc`。ClaudeCLI 已用系统 CMake 3.16.3 复现失败，
并确认 `find_package(PCL 1.10 REQUIRED COMPONENTS common io)` 在该主机可解析。

原始日志（Ubuntu 主机）位于：
`validation/da36c5164d4296fa1d8b559f281ab8bec46826ce/20260918T040824Z-4ca25597/`。

Codex 修复范围：PCL CMake 包集成；验证脚本明确工具路径；记录实际解析的依赖；
澄清失败分类须独立复核。不改核心算法、阈值或场景矩阵。
新 SHA 必须由 Ubuntu20 ClaudeCLI 在 PCL-ON 通道重新 clean build，运行
`validation_boundary`、`replay_selfcheck`、`pcd_selfcheck`、G1–G7 及 PCD 回放导出。
未经此次完整复验，不固化基线。

`hold_detector/` 的 270 个本地文件未修改、未导入 Git；此前没有推送这些数据。
